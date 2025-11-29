"\"\"\"Training utilities: seeding, trainer factory, single-recipe runner.\"\"\""
from __future__ import annotations

import gc
import json
import os
import random
import sys
from typing import Dict, Optional

import lightning as L
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .callbacks import PhaseValMetricsCallback
from .data import CIFAR10CatDogDM, IMAGENET_MEAN, IMAGENET_STD, STL10CatDog, tf_train
from .models import LitBinaryClassifier, LitBinaryClassifierBrevitas
from .attacks import _build_attack, _to01, _tonorm


def set_seed(seed: int = 42):
    """Fix Python and torch random seeds so repeated runs give the same results."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)
L.seed_everything(42, workers=True)
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass
torch.backends.cudnn.benchmark = False


def _get_peak_memory_bytes() -> Optional[int]:
    """Best-effort process peak memory (GPU if available, else CPU RSS)."""
    if torch.cuda.is_available():
        try:
            return int(torch.cuda.max_memory_reserved())
        except Exception:
            try:
                return int(torch.cuda.max_memory_allocated())
            except Exception:
                return None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if rss is None:
            return None
        # Linux reports KB, macOS reports bytes.
        return int(rss if sys.platform == "darwin" else rss * 1024)
    except Exception:
        return None


def make_trainer(
    root: str = "./runs",
    epochs: int = 10,
    precision_override: Optional[str] = None,
    callbacks: Optional[list] = None,
) -> L.Trainer:
    """Build a Lightning trainer that runs for the given epochs and uses the provided callbacks."""
    torch.set_float32_matmul_precision("medium")
    use_cuda = torch.cuda.is_available()
    use_mps = (
        not use_cuda
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )
    accelerator = "gpu" if use_cuda else "mps" if use_mps else "cpu"

    if precision_override is None:
        if use_cuda:
            bf16_ok = torch.cuda.is_bf16_supported()
            precision = "bf16-mixed" if bf16_ok else "16-mixed"
        else:
            precision = "32-true"
    else:
        precision = precision_override
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        precision=precision,
        max_epochs=int(epochs),
        min_epochs=int(epochs),
        default_root_dir=root,
        enable_checkpointing=False,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        callbacks=callbacks or [],
    )
    return trainer


def _build_phaseB_loader(cfg: dict) -> DataLoader:
    """Return the DataLoader for Phase B online training (defaults to STL10 cats/dogs)."""
    dataset_name = cfg.get("dataset", "stl10").lower()
    if dataset_name == "stl10":
        root = cfg.get("root", "./data")
        batch_size = cfg.get("batch_size", 128)
        num_workers = cfg.get("num_workers", 0)
        img_size = cfg.get("img_size", 224)
        pin_memory = cfg.get("pin_memory", False)
        stl_split = cfg.get("split", "train")

        ds = STL10CatDog(
            root=root,
            split=stl_split,
            img_size=img_size,
            transform=tf_train(img_size),
        )
        loader_args = dict(
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            persistent_workers=False,
        )
        if num_workers > 0:
            loader_args["prefetch_factor"] = 2
        return DataLoader(ds, **loader_args)
    if dataset_name == "cifar10":
        dm = CIFAR10CatDogDM(
            root=cfg.get("root", "./data"),
            batch_size=cfg.get("batch_size", 128),
            num_workers=cfg.get("num_workers", 0),
            img_size=cfg.get("img_size", 224),
            val_ratio=cfg.get("val_ratio", 0.0),
            pin_memory=cfg.get("pin_memory", False),
        )
        dm.setup()
        return dm.train_dataloader()
    raise ValueError(f"Unsupported Phase B dataset: {dataset_name}")


def _prepare_attacked_batch(model, x, y, adv_cfg):
    """Mirror the attack augmentation logic but force Phase B behavior."""
    cfg = adv_cfg or {}
    if not cfg.get("enabled", False):
        return x, y

    freq = int(cfg.get("every_n", 0))
    if freq <= 0:
        return x, y

    idx = torch.arange(x.size(0), device=x.device)
    target_pos = (freq - 1) % freq
    mask = (idx % freq) == target_pos
    if not torch.any(mask):
        return x, y

    if not hasattr(model, "mean_buf") or not hasattr(model, "std_buf"):
        raise RuntimeError("Model missing normalization buffers for adversarial augmentation.")
    attack = _build_attack(model, cfg, model.mean_buf, model.std_buf, device=x.device)
    x_sel = x[mask]
    y_sel = y[mask]
    x01 = _to01(x_sel, model.mean_buf, model.std_buf).clamp_(0.0, 1.0)

    was_training = model.training
    model.eval()
    adv01 = attack(x01, y_sel)
    if was_training:
        model.train()

    adv_norm = _tonorm(adv01.detach(), model.mean_buf, model.std_buf)
    x_aug = torch.cat([x, adv_norm], dim=0)
    y_aug = torch.cat([y, y_sel], dim=0)
    return x_aug, y_aug


def run_phaseB_online_training(
    model,
    loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    adv_cfg: dict,
) -> dict:
    """Online Phase B: evaluate attacked batches for pseudo-labels, then train on them."""
    if epochs <= 0:
        return {}

    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    criterion = getattr(model, "criterion", torch.nn.CrossEntropyLoss())

    total_steps = 0
    total_loss = 0.0
    total_eval_acc = 0.0
    total_gt_acc = 0.0
        
    for epoch in tqdm(range(epochs), desc="Epochs"):
        for batch in tqdm(loader, desc=f"Epoch {epoch+1} - batches", leave=False):
            total_steps += 1
            x, y_true = batch
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)

            x_aug, y_aug = _prepare_attacked_batch(model, x, y_true, adv_cfg)

            with torch.no_grad():
                model.eval()
                logits_eval = model(x_aug)
                pseudo_labels = torch.argmax(logits_eval, dim=1)
                eval_acc = (pseudo_labels == y_aug).float().mean().item()
                del logits_eval
            model.train()

            optimizer.zero_grad(set_to_none=True)
            logits = model(x_aug)
            loss = criterion(logits, pseudo_labels)
            loss.backward()
            optimizer.step()

            total_eval_acc += eval_acc

            preds_vs_true = torch.argmax(logits.detach(), dim=1)
            gt_acc = (preds_vs_true == y_aug).float().mean().item()
            total_gt_acc += gt_acc
            total_loss += float(loss.detach().cpu())

            del (
                x,
                y_true,
                x_aug,
                y_aug,
                pseudo_labels,
                logits,
                loss,
                preds_vs_true,
            )
            torch.cuda.empty_cache()

    if total_steps == 0:
        return {}

    return {
        "phaseB_eval_acc": total_eval_acc / total_steps,
        "phaseB_train_acc": total_gt_acc / total_steps,
        "phaseB_train_loss": total_loss / total_steps,
        "phaseB_steps": total_steps,
        "phaseB_epochs": epochs,
    }


def train_and_attack(recipe: dict, out_dir: Optional[str] = None) -> Dict[str, dict]:
    """Run Phase A clean training, Phase B online attacks, and Phase C eval for one recipe."""
    d = recipe.get("data", {})
    dm = CIFAR10CatDogDM(
        root=d.get("root", "./data"),
        batch_size=d.get("batch_size", 128),
        num_workers=d.get("num_workers", 0),
        img_size=d.get("img_size", 224),
        val_ratio=d.get("val_ratio", 0.10),
        pin_memory=d.get("pin_memory", False),
    )
    dm.setup()
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    m = recipe.get("model", {})
    adv = recipe.get("attack_injection", recipe.get("adv_training", {"enabled": False}))
    phaseA_epochs = int(m.get("phaseA_epochs", 0))

    # model = LitBinaryClassifier(
    #     lr=recipe.get("opt", {}).get("lr", 3e-4),
    #     weight_decay=recipe.get("opt", {}).get("wd", 1e-4),
    #     pretrained=m.get("pretrained", True),
    #     freeze_backbone=m.get("freeze_backbone", False),
    #     dropout_p=m.get("dropout_p", 0.2),
    #     adv_cfg=adv,
    #     phaseA_epochs=phaseA_epochs,
    #     mean=IMAGENET_MEAN,
    #     std=IMAGENET_STD,
    # )
    model = LitBinaryClassifierBrevitas(
        lr=recipe.get("opt", {}).get("lr", 3e-4),
        weight_decay=recipe.get("opt", {}).get("wd", 1e-4),
        pretrained=m.get("pretrained", True),
        freeze_backbone=m.get("freeze_backbone", False),
        dropout_p=m.get("dropout_p", 0.2),
        adv_cfg=adv,
        phaseA_epochs=phaseA_epochs,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
    )

    t = recipe.get("trainer", {})
    total_epochs = int(t.get("epochs", 10))
    phaseB_epochs = max(0, total_epochs - phaseA_epochs)

    phase_cb = PhaseValMetricsCallback(phaseA_epochs=phaseA_epochs)
    trainer = make_trainer(
        root=t.get("root", "./runs/online"),
        epochs=max(phaseA_epochs, 1),
        callbacks=[phase_cb],
    )

    print(
        f"[Orchestrator] Phase A epochs: {phaseA_epochs}, "
        f"Phase B epochs: {phaseB_epochs}"
    )
    if phaseA_epochs > 0:
        trainer.fit(model, datamodule=dm, ckpt_path=None)
    else:
        print("[Orchestrator] Skipping Phase A training (0 epochs requested).")

    phaseB_metrics = {}
    if phaseB_epochs > 0:
        dataB_cfg = recipe.get(
            "phaseB_data",
            {
                "root": "./data",
                "batch_size": d.get("batch_size", 128),
                "num_workers": d.get("num_workers", 0),
                "img_size": d.get("img_size", 224),
                "pin_memory": d.get("pin_memory", False),
                "dataset": "stl10",
                "split": "train",
            },
        )
        loaderB = _build_phaseB_loader(dataB_cfg)
        opt_cfg = recipe.get("opt", {})
        print("[Orchestrator] Phase B online adaptation (pseudo-labeling) begins.")
        phaseB_metrics = run_phaseB_online_training(
            model=model,
            loader=loaderB,
            epochs=phaseB_epochs,
            lr=opt_cfg.get("lr", 3e-4),
            weight_decay=opt_cfg.get("wd", 1e-4),
            adv_cfg=adv,
        )
        del loaderB
        print("[Orchestrator] Phase B metrics:", phaseB_metrics)

    print("[Orchestrator] Phase C: final validation/test on CIFAR10 cats/dogs (no updates).")
    val_metrics = trainer.validate(model, datamodule=dm, ckpt_path=None)
    test_metrics = trainer.test(model, datamodule=dm, ckpt_path=None)

    phase_metrics = phase_cb.compute_phase_metrics()
    phase_metrics.update(phaseB_metrics)
    mem_peak_bytes = _get_peak_memory_bytes()
    metrics: Dict[str, dict] = {
        "val": val_metrics[0] if val_metrics else {},
        "test": test_metrics[0] if test_metrics else {},
        "phases": phase_metrics,
        "mem_peak_bytes": mem_peak_bytes,
        "mem_peak_GiB": float(mem_peak_bytes) / (1024 ** 3) if mem_peak_bytes is not None else None,
    }

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "recipe.json"), "w", encoding="utf-8") as f:
            json.dump(recipe, f, ensure_ascii=False, indent=2)
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    del model
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return metrics


__all__ = ["set_seed", "make_trainer", "train_and_attack"]
