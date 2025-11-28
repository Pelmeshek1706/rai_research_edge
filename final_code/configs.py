"\"\"\"Recipe builders that were previously defined inside the notebook.\"\"\""
from __future__ import annotations

from copy import deepcopy
from typing import Dict, List


def deep_update(base: dict, patch: dict) -> dict:
    """Copy values from patch into base, merging nested dicts."""
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


BASE_ONLINE_ATTACK_RECIPE = {
    "data": {
        "root": "./data",
        "batch_size": 256,
        "num_workers": 0,
        "img_size": 224,
        "val_ratio": 0.10,
        "pin_memory": False,
    },
    "phaseB_data": {
        "root": "./data",
        "batch_size": 256,
        "num_workers": 0,
        "img_size": 224,
        "pin_memory": False,
        "dataset": "stl10",
        "split": "train",
    },
    "model": {
        "pretrained": True,
        "freeze_backbone": False,
        "dropout_p": 0.2,
        "label_smoothing": 0.0,
    },
    "opt": {
        "lr": 3e-4,
        "wd": 1e-4,
        "layer_decay": None,
    },
    "trainer": {
        "root": "./runs/online",
    },
    "attack_injection": {
        "enabled": False,
        "method": "pgd",
        "every_n": 4,
        "attack_kwargs": {},
    },
    "augment": {
        "mixup": False,
        "cutmix": False,
        "p": 0.0,
        "mixup_alpha": 0.2,
        "cutmix_alpha": 1.0,
    },
    "regularizers": {
        "jacobian": {"enabled": False, "lam": 1e-4, "subbatch": 8},
    },
    "callbacks": {
        "ema": False,
        "ema_decay": 0.999,
        "prune": {"enabled": False, "amount": 0.3, "start_epoch": 1},
    },
    "eval": {
        "noise_avg": {"enabled": False, "n": 8, "sigma": 0.05},
        "temp_scaling": False,
    },
    "adv_training": {
        "enabled": False,
        "eps": 8 / 255,
        "alpha": 2 / 255,
        "steps": 10,
        "random_start": True,
    },
}


def method_patch_online(method: str) -> dict:
    """Return the small patch that turns on the given regularization method."""

    patches = {
        "baseline": {},
        "mixup": {"augment": {"mixup": True, "cutmix": False, "p": 1.0}},
        "cutmix": {"augment": {"cutmix": True, "mixup": False, "p": 1.0}},
        "jacobian": {"regularizers": {"jacobian": {"enabled": True}}},
        "ema": {"callbacks": {"ema": True}},
        "prune": {"callbacks": {"prune": {"enabled": True}}},
        "temp": {"eval": {"temp_scaling": True}},
        "noiseavg": {"eval": {"noise_avg": {"enabled": True}}},
        "spectral": {"model": {"spectral_norm": True}},
        "labelsmooth": {"model": {"label_smoothing": 0.1}},
        "layerdecay": {"opt": {"layer_decay": 0.75}, "model": {"freeze_backbone": False}},
        "pgd_at": {"adv_training": {"enabled": True}},
    }
    if method not in patches:
        raise ValueError(f"Unknown method: {method}")
    return patches[method]


def _grid_patches_for_method_online(method: str, variants_per_method: int = 5) -> List[dict]:
    """Return a list of hyperparameter patches for the selected method."""

    sweeps = {
        "mixup": [
            {"augment": {"mixup_alpha": v, "p": 1.0}}
            for v in [0.1, 0.2, 0.3, 0.4, 0.5]
        ],
        "cutmix": [
            {"augment": {"cutmix_alpha": v, "p": 1.0}}
            for v in [0.5, 1.0, 1.5, 2.0, 0.25]
        ],
        "jacobian": [
            {"regularizers": {"jacobian": {"lam": v, "subbatch": 8}}}
            for v in [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
        ],
        "ema": [
            {"callbacks": {"ema_decay": v}}
            for v in [0.99, 0.995, 0.997, 0.999, 0.9995]
        ],
        "prune": [
            {"callbacks": {"prune": {"amount": v, "start_epoch": 1}}}
            for v in [0.1, 0.2, 0.3, 0.4, 0.5]
        ],
        "labelsmooth": [
            {"model": {"label_smoothing": v}}
            for v in [0.05, 0.10, 0.15, 0.20, 0.0]
        ],
        "layerdecay": [
            {"opt": {"layer_decay": v}, "model": {"freeze_backbone": False}}
            for v in [0.6, 0.7, 0.75, 0.8, 0.85]
        ],
        "noiseavg": [
            {"eval": {"noise_avg": {"enabled": True, "n": n, "sigma": s}}}
            for (n, s) in [(4, 0.03), (8, 0.05), (8, 0.10), (16, 0.05), (16, 0.10)]
        ],
        "pgd_at": [
            {
                "adv_training": {
                    "enabled": True,
                    "eps": e,
                    "alpha": 2 / 255,
                    "steps": k,
                    "random_start": True,
                }
            }
            for (e, k) in [(8 / 255, 5), (8 / 255, 10), (8 / 255, 20), (6 / 255, 10), (10 / 255, 10)]
        ],
        "baseline": [{}],
    }
    grid = sweeps.get(method)
    if grid is None:
        raise ValueError(f"Unknown method: {method}")
    return grid[: max(1, int(variants_per_method))]


def phaseB_dataset_patch(dataset: str = "stl10", **phaseB_kwargs) -> Dict[str, dict]:
    """Return a small patch that switches Phase B to the requested cat/dog dataset."""
    dataset = dataset.lower()
    allowed = {"stl10", "cifar10"}
    if dataset not in allowed:
        raise ValueError(f"Unsupported Phase B dataset: {dataset}")
    patch = {"phaseB_data": {"dataset": dataset}}
    patch["phaseB_data"].update(phaseB_kwargs)
    return patch


def get_recipes_for_benchmark_online_attack(
    every_n: int = 4,
    epochs_phaseA: int = 10,
    epochs_phaseB: int = 2,
    variants_per_method: int = 3,
) -> Dict[str, dict]:
    """Build the recipe grid by pairing each regularization with attack types."""

    base = deepcopy(BASE_ONLINE_ATTACK_RECIPE)
    total_epochs = int(epochs_phaseA + epochs_phaseB) 
    base["trainer"]["epochs"] = total_epochs
    base["model"]["phaseA_epochs"] = int(epochs_phaseA)

    reg_methods = [
        "baseline",
        "mixup",
        "cutmix",
        "noiseavg",
    ]
    attack_methods = ["fgsm", "pgd", "mifgsm", "difgsm", "tifgsm", "apgd"]

    recipes: Dict[str, dict] = {}

    clean = deepcopy(base)
    clean["attack_injection"] = {
        "enabled": False,
        "method": "pgd",
        "every_n": every_n,
        "attack_kwargs": {},
    }
    recipes["clean"] = clean

    for reg_method in reg_methods:
        base_with_method = deepcopy(base)
        deep_update(base_with_method, method_patch_online(reg_method))

        hp_grid = _grid_patches_for_method_online(
            reg_method, variants_per_method=variants_per_method
        )

        for hp_idx, hp_patch in enumerate(hp_grid, start=1):
            reg_cfg = deepcopy(base_with_method)
            deep_update(reg_cfg, hp_patch)

            for atk_method in attack_methods:
                cfg = deepcopy(reg_cfg)
                cfg["attack_injection"] = {
                    "enabled": True,
                    "method": atk_method,
                    "every_n": int(every_n),
                    "attack_kwargs": {},
                }

                if reg_method == "baseline" and len(hp_grid) == 1:
                    reg_name = "baseline"
                elif reg_method == "baseline":
                    reg_name = f"baseline#v{hp_idx}"
                else:
                    reg_name = f"{reg_method}#v{hp_idx}"

                atk_name = f"{atk_method}_n{every_n}"
                name = f"{reg_name}__{atk_name}"
                recipes[name] = cfg

    return recipes


__all__ = [
    "BASE_ONLINE_ATTACK_RECIPE",
    "deep_update",
    "method_patch_online",
    "get_recipes_for_benchmark_online_attack",
    "phaseB_dataset_patch",
]
