"\"\"\"Training utilities: seeding, trainer factory, single-recipe runner.\"\"\""
from __future__ import annotations

import gc
import json
import os
import random
from typing import Dict, Optional

import lightning as L
import torch

from .callbacks import PhaseValMetricsCallback
from .data import CIFAR10CatDogDM, IMAGENET_MEAN, IMAGENET_STD
from .models import LitBinaryClassifier


def set_seed(seed: int = 42):
    """Fix Python and torch random seeds so repeated runs give the same results."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)
L.seed_everything(42, workers=True)
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass
torch.backends.cudnn.benchmark = False


def make_trainer(
    root: str = "./runs",
    epochs: int = 10,
    precision_override: Optional[str] = None,
    callbacks: Optional[list] = None,
) -> L.Trainer:
    """Build a Lightning trainer that runs for the given epochs and uses the provided callbacks."""
    torch.set_float32_matmul_precision("medium")
    use_cuda = torch.cuda.is_available()
    if precision_override is None:
        bf16_ok = use_cuda and torch.cuda.is_bf16_supported()
        precision = "bf16-mixed" if bf16_ok else ("16-mixed" if use_cuda else "32-true")
    else:
        precision = precision_override
    trainer = L.Trainer(
        accelerator="gpu" if use_cuda else "cpu",
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

    m = recipe.get("model", {})
    adv = recipe.get("attack_injection", recipe.get("adv_training", {"enabled": False}))
    phaseA_epochs = int(m.get("phaseA_epochs", 0))

    model = LitBinaryClassifier(
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

    phase_cb = PhaseValMetricsCallback(phaseA_epochs=phaseA_epochs)
    trainer = make_trainer(
        root=t.get("root", "./runs/online"),
        epochs=total_epochs,
        callbacks=[phase_cb],
    )

    print(
        f"[Orchestrator] Phase A epochs: {phaseA_epochs}, "
        f"Phase B epochs: {total_epochs - phaseA_epochs}"
    )
    trainer.fit(model, datamodule=dm, ckpt_path=None)

    print("[Orchestrator] Phase C: final validation/test on CIFAR10 cats/dogs (no updates).")
    val_metrics = trainer.validate(model, datamodule=dm, ckpt_path=None)
    test_metrics = trainer.test(model, datamodule=dm, ckpt_path=None)

    phase_metrics = phase_cb.compute_phase_metrics()
    metrics: Dict[str, dict] = {
        "val": val_metrics[0] if val_metrics else {},
        "test": test_metrics[0] if test_metrics else {},
        "phases": phase_metrics,
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
