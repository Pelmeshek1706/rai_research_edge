"""
PGD-aware benchmarking of torchattacks across the best grid-search recipes.

This script reuses the up-to-date training components from ``training_pipeline_def``
and the curated ``best_recipes`` collection defined in ``train_pgd_old``.
It mirrors the behaviour of ``torch_attacks_base.py`` while ensuring that
adversarial (PGD) training stays enabled for the evaluated configurations.
"""

from __future__ import annotations

import gc
import json
import os
from copy import deepcopy
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from rai_research.train_pgd_old import best_recipes
from rai_research.training_pipeline_def import (
    CollectMetricsCallback,
    LitBinaryClassifier,
    TwoClassDataModule,
    get_epoch_metrics_df,
    make_trainer,
    replace_activations_for_double_backward,
)


class NormalizedModel(nn.Module):
    """
    Wraps a classifier so that it expects inputs in the [0, 1] range and applies
    the training-time mean/std normalisation internally before forwarding.
    """

    def __init__(self, base: nn.Module, mean: List[float], std: List[float], device: torch.device):
        super().__init__()
        self.base = base
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean_t, persistent=False)
        self.register_buffer("std", std_t, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.base(x)


class UnnormalizeView(Dataset):
    """Converts normalised tensors back to the [0, 1] image space on the fly."""

    def __init__(self, base: Dataset, mean: List[float], std: List[float]):
        self.base = base
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]
        if isinstance(x, torch.Tensor):
            x = (x * self.std + self.mean).clamp_(0.0, 1.0)
        return x, y


def make_torchattacks_suite(
    model: nn.Module,
    device: torch.device,
    eps: float = 8 / 255,
    steps: int = 10,
    alpha: Optional[float] = None,
):
    """Constructs the set of attacks used in previous experiments."""
    try:
        import torchattacks as ta
    except ImportError as exc:  # pragma: no cover - helpful message when dependency is missing
        raise RuntimeError("Install `torchattacks` to run the adversarial benchmarks.") from exc

    if alpha is None:
        alpha = (eps / max(steps, 1)) * 1.25
    return {
        "FGSM": ta.FGSM(model, eps=eps),
        "PGD": ta.PGD(model, eps=eps, alpha=alpha, steps=steps),
        "MIFGSM": ta.MIFGSM(model, eps=eps, steps=steps, decay=1.0),
        "DIFGSM": ta.DIFGSM(model, eps=eps, steps=steps, alpha=alpha),
        "TIFGSM": ta.TIFGSM(model, eps=eps, steps=steps, alpha=alpha),
        "APGD": ta.APGD(model, eps=eps, steps=steps),
    }


def get_mean_std_from_dm(dm) -> Tuple[List[float], List[float]]:
    if hasattr(dm, "CIFAR10_MEAN") and hasattr(dm, "CIFAR10_STD"):
        return list(dm.CIFAR10_MEAN), list(dm.CIFAR10_STD)
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _build_split_loader_for_attacks(
    dm, split: str, mean: List[float], std: List[float], *, batch_size: Optional[int] = None
) -> DataLoader:
    base_ds = {"train": dm.train_ds, "val": dm.val_ds, "test": dm.test_ds}[split]
    wrapped = UnnormalizeView(base_ds, mean, std)
    return DataLoader(
        wrapped,
        batch_size=batch_size or getattr(dm, "batch_size", 64),
        shuffle=False,
        num_workers=getattr(dm, "num_workers", 0),
        pin_memory=getattr(dm, "_pin_memory_flag", lambda: False)(),
    )


@torch.no_grad()
def _clean_acc(norm_model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    norm_model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = norm_model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def _adv_acc_single_attack(attack, norm_model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    norm_model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(x, y)
        with torch.no_grad():
            pred = norm_model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def train_model_with_recipe(recipe: dict, force_fp32_for_jacobian: bool = True):
    """
    Trains a model according to a recipe and returns the fitted (eval-ready) model,
    the prepared data module, and the collected metrics history.
    """
    dcfg = recipe.get("data", {})
    dm = TwoClassDataModule(
        workdir=dcfg.get("workdir", "./workdir"),
        batch_size=dcfg.get("batch_size", 128),
        num_workers=dcfg.get("num_workers", 0),
        img_size=dcfg.get("img_size", 224),
        val_ratio_per_class=dcfg.get("val_ratio", 0.1),
        pin_memory=dcfg.get("pin_memory", False),
        train_aug=dcfg.get("train_aug", None),
    )
    dm.setup()

    mcfg = recipe.get("model", {})
    ocfg = recipe.get("optimizer", {})
    acfg = recipe.get("augment", {})
    advcfg = recipe.get("adv_training", {})

    model = LitBinaryClassifier(
        lr=ocfg.get("lr", 3e-4),
        weight_decay=ocfg.get("wd", 1e-4),
        pretrained=mcfg.get("pretrained", True),
        freeze_backbone=mcfg.get("freeze_backbone", False),
        dropout_p=mcfg.get("dropout_p", 0.25),
        spectral_norm=mcfg.get("spectral_norm", True),
        label_smoothing=mcfg.get("label_smoothing", 0.0),
        use_custom_optimizer=ocfg.get("use_custom", True),
        layer_decay=ocfg.get("layer_decay", 0.75),
        mean=mcfg.get("mean", getattr(dm, "CIFAR10_MEAN", (0.4914, 0.4822, 0.4465))),
        std=mcfg.get("std", getattr(dm, "CIFAR10_STD", (0.2470, 0.2435, 0.2616))),
        adv_training=advcfg,
        augment_cfg=acfg,
    )

    rcfg = recipe.get("regularizers", {}).get("jacobian", {})
    jac_enabled = rcfg.get("enabled", False)
    if jac_enabled:
        replace_activations_for_double_backward(model)

    metrics_cb = CollectMetricsCallback()
    callbacks = [metrics_cb]

    tcfg = recipe.get("trainer", {})
    adv_enabled = advcfg.get("enabled", False)
    precision_override = "32-true" if (adv_enabled or (jac_enabled and force_fp32_for_jacobian)) else None
    # trainer = make_trainer(
    #     root_dir=tcfg.get("root_dir", "./runs/attacks"),
    #     epochs=tcfg.get("epochs", 40),
    #     callbacks=callbacks,
    #     precision_override=precision_override,
    #     num_sanity_val_steps=tcfg.get("num_sanity_val_steps", 0),
    # )
    trainer = make_trainer(
    root_dir=tcfg.get("root_dir", "./runs/attacks"),
    epochs=tcfg.get("epochs", 40),
    callbacks=callbacks,
    precision_override=precision_override,
    num_sanity_val_steps=tcfg.get("num_sanity_val_steps", 0),
    inference_mode=False,  # <-- important
)


    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    history = get_epoch_metrics_df(metrics_cb)
    return model, dm, history


def eval_attacks_for_model(
    model: nn.Module,
    dm,
    attacks: List[str],
    eps: float = 8 / 255,
    steps: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Computes clean accuracy and adversarial accuracy per split for each attack.
    Returns a flat dict mirroring the original summary table.
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()

    mean, std = get_mean_std_from_dm(dm)
    norm_model = NormalizedModel(model, mean, std, device).to(device).eval()

    loaders = {
        split: _build_split_loader_for_attacks(dm, split, mean, std, batch_size=getattr(dm, "batch_size", 64))
        for split in ("train", "val", "test")
    }

    results: Dict[str, float] = {}
    for split, loader in loaders.items():
        results[f"clean/{split}"] = _clean_acc(norm_model, loader, device)

    suite = make_torchattacks_suite(norm_model, device, eps=eps, steps=steps)
    for name in attacks:
        atk = suite[name]
        for split, loader in loaders.items():
            results[f"{name}/{split}"] = _adv_acc_single_attack(atk, norm_model, loader, device)

    del loaders, suite
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    return results


def _normalize_config_collection(cfgs: Iterable) -> Dict[str, dict]:
    if isinstance(cfgs, dict):
        return deepcopy(cfgs)
    norm: Dict[str, dict] = {}
    for item in cfgs:
        if isinstance(item, dict) and len(item) == 1:
            name = next(iter(item))
            norm[name] = deepcopy(item[name])
        else:
            raise ValueError("Expected items shaped as {name: recipe}.")
    return norm


def benchmark_configs_with_attacks(
    configs,
    attacks: List[str],
    eps: float = 8 / 255,
    steps: int = 10,
    out_dir: str = "./bench_attacks",
) -> pd.DataFrame:
    """
    Trains each recipe, evaluates the torchattacks suite and stores CSV artefacts.
    Returns the wide summary table (one row per recipe).
    """
    os.makedirs(out_dir, exist_ok=True)
    cfgs = _normalize_config_collection(configs)

    rows: List[Dict[str, float]] = []
    for name, recipe in cfgs.items():
        print(f"[benchmark] config={name}")
        model, dm, history = train_model_with_recipe(recipe)
        history.to_csv(os.path.join(out_dir, f"{name}.history.csv"), index=False)
        with open(os.path.join(out_dir, f"{name}.recipe.json"), "w", encoding="utf-8") as f:
            json.dump(recipe, f, ensure_ascii=False, indent=2)

        accs = eval_attacks_for_model(model, dm, attacks, eps=eps, steps=steps)
        rows.append({"config": name, **accs})

        del model, dm, history
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

    summary = pd.DataFrame(rows).sort_values("config").reset_index(drop=True)
    summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return summary


# if __name__ == "__main__":
#     attack_names = ["FGSM", "PGD", "MIFGSM", "DIFGSM", "TIFGSM", "APGD"]
#     summary_df = benchmark_configs_with_attacks(best_recipes, attacks=attack_names, eps=8 / 255, steps=10)
#     print(summary_df)
