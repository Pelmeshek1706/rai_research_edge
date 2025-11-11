# online_recipe_builders.py
# ============================================================
# Config builders adapted for *online* learning:
#   Phase A  -> base training on CIFAR-10 (cats/dogs)
#   Phase B  -> fixed-epochs online-style adaptation on Oxford-IIIT Pet
#   Phase C  -> third-dataset check (STL-10 cats/dogs)
#
# Includes the same regularization knobs you used before (mixup, cutmix,
# jacobian, ema, prune, temp, noiseavg, spectral, labelsmooth, layerdecay, dropout),
# but *scoped into Phase A* where they apply during base training.
#
# Default config already contains Phase B ("online") steps so *every* config has them.
# ============================================================

import json
from copy import deepcopy

def deep_update(base: dict, patch: dict):
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def build_online_base_recipe_for_search(
    epochs_phaseA: int = 10,
    epochs_phaseB: int = 1,
    adv_pgd_at_enabled: bool = False,
) -> dict:
    """
    Returns the DEFAULT recipe with Phase A/B/C + all common slots.
    Phase B (online) steps are *always present* by default.
    """
    return {
        # -------- Phase A: base training on CIFAR-10 cats/dogs --------
        "phaseA": {
            "data": {
                "root": "./data",
                "batch_size": 128,
                "num_workers": 0,
                "img_size": 224,
                "val_ratio": 0.10,
                "pin_memory": False,
                # augment slots live here for Phase A (train-time)
                "train_aug": {
                    "random_resized_crop": {"enabled": True, "scale": [0.6, 1.0], "ratio": [0.75, 1.3333]},
                    "horizontal_flip_p": 0.5,
                    "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02},
                },
            },
            "model": {
                "pretrained": True,
                "freeze_backbone": False,
                "dropout_p": 0.2,
                "spectral_norm": False,
                "label_smoothing": 0.0,
            },
            "optimizer": {"use_custom": True, "layer_decay": None, "wd": 1e-4, "lr": 3e-4},
            "augment": {"mixup": False, "cutmix": False, "p": 0.0, "mixup_alpha": 0.2, "cutmix_alpha": 1.0},
            "regularizers": {"jacobian": {"enabled": False, "lam": 1e-4, "subbatch": 8}},
            "callbacks": {"ema": False, "ema_decay": 0.999, "prune": {"enabled": False, "amount": 0.3, "start_epoch": 1}},
            "trainer": {"root": "./runs/phaseA", "epochs": int(epochs_phaseA), "num_sanity_val_steps": 0},
            "adv_training": {  # PGD-AT toggle for Phase A
                "enabled": bool(adv_pgd_at_enabled),
                "eps": 8/255, "alpha": 2/255, "steps": 10, "random_start": True,
                "mix_clean_adv": 0.0, "eval_steps": 5
            },
            "eval": {"temp_scaling": False, "noise_avg": {"enabled": False, "n": 8, "sigma": 0.05}},
        },

        # -------- Phase B: online-style adaptation on Oxford-IIIT Pet --------
        "phaseB": {
            "root": "./data/oxford-iiit-pet",
            "batch_size": 64,
            "num_workers": 0,
            "img_size": 224,
            "split": "trainval",     # Pet supports "trainval" / "test"
            "pin_memory": False,
            # fixed number of epochs for the online pass — DEFAULT present for every config
            "epochs": int(epochs_phaseB),
            "lr": 3e-5,
            "wd": 0.0,
            "trainable": "classifier",   # "classifier" | "last_n" | "all"
            "last_n": 0,
            "limit_batches_per_epoch": None,
            "max_steps": None,
        },

        # -------- Phase C: verification on third dataset (STL-10 cats/dogs) --------
        "phaseC": {
            "root": "./data",
            "batch_size": 128,
            "num_workers": 0,
            "img_size": 224,
            "split": "test",
            "pin_memory": False,
        },
    }

def method_patch_online(method: str) -> dict:
    """
    Regularization/augmentation patches adapted to Phase A scope.
    """
    patches = {
        "baseline":    {},
        "mixup":       {"phaseA": {"augment": {"mixup": True,  "cutmix": False, "p": 1.0}}},
        "cutmix":      {"phaseA": {"augment": {"cutmix": True, "mixup": False,  "p": 1.0}}},
        "jacobian":    {"phaseA": {"regularizers": {"jacobian": {"enabled": True, "lam": 1e-4, "subbatch": 8}}}},
        "ema":         {"phaseA": {"callbacks": {"ema": True, "ema_decay": 0.999}}},
        "prune":       {"phaseA": {"callbacks": {"prune": {"enabled": True, "amount": 0.3, "start_epoch": 1}}}},
        "temp":        {"phaseA": {"eval": {"temp_scaling": True}}},
        "noiseavg":    {"phaseA": {"eval": {"noise_avg": {"enabled": True, "n": 8, "sigma": 0.05}}}},
        "spectral":    {"phaseA": {"model": {"spectral_norm": True}}},
        "labelsmooth": {"phaseA": {"model": {"label_smoothing": 0.1}}},
        "layerdecay":  {"phaseA": {"optimizer": {"layer_decay": 0.75}, "model": {"freeze_backbone": False}}},
        "dropout":     {"phaseA": {"model": {"dropout_p": 0.30}}},
        # Optional: enable PGD-AT as a "method"
        "pgd_at":      {"phaseA": {"adv_training": {"enabled": True}}},
    }
    if method not in patches:
        raise ValueError(f"Unknown method: {method}")
    return patches[method]

def _grid_patches_for_method_online(method: str, variants_per_method: int = 5) -> list[dict]:
    """
    Hyper-param grids mapped to Phase A (base training).
    """
    sweeps = {
        "mixup": [
            {"phaseA": {"augment": {"mixup_alpha": v, "p": 1.0}}}
            for v in [0.1, 0.2, 0.3, 0.4, 0.5]
        ],
        "cutmix": [
            {"phaseA": {"augment": {"cutmix_alpha": v, "p": 1.0}}}
            for v in [0.5, 1.0, 1.5, 2.0, 0.25]
        ],
        "jacobian": [
            {"phaseA": {"regularizers": {"jacobian": {"lam": v, "subbatch": 8}}}}
            for v in [5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
        ],
        "ema": [
            {"phaseA": {"callbacks": {"ema_decay": v}}}
            for v in [0.99, 0.995, 0.997, 0.999, 0.9995]
        ],
        "prune": [
            {"phaseA": {"callbacks": {"prune": {"amount": v, "start_epoch": 1}}}}
            for v in [0.1, 0.2, 0.3, 0.4, 0.5]
        ],
        "labelsmooth": [
            {"phaseA": {"model": {"label_smoothing": v}}}
            for v in [0.05, 0.10, 0.15, 0.20, 0.0]
        ],
        "layerdecay": [
            {"phaseA": {"optimizer": {"layer_decay": v}, "model": {"freeze_backbone": False}}}
            for v in [0.6, 0.7, 0.75, 0.8, 0.85]
        ],
        "noiseavg": [
            {"phaseA": {"eval": {"noise_avg": {"enabled": True, "n": n, "sigma": s}}}}
            for (n, s) in [(4, 0.03), (8, 0.05), (8, 0.10), (16, 0.05), (16, 0.10)]
        ],
        # Optional sweep for PGD-AT steps/eps if you enable "pgd_at"
        "pgd_at": [
            {"phaseA": {"adv_training": {"enabled": True, "eps": e, "alpha": 2/255, "steps": k, "random_start": True}}}
            for (e, k) in [(8/255, 5), (8/255, 10), (8/255, 20), (6/255, 10), (10/255, 10)]
        ],
        "baseline": [ {} ],
    }
    grid = sweeps.get(method)
    if grid is None:
        raise ValueError(f"Unknown method: {method}")
    return grid[:max(1, int(variants_per_method))]

def build_online_base_recipes_for_methods(
    methods: list[str],
    base_recipe: dict | None = None,
    epochs_phaseA: int = 10,
    epochs_phaseB: int = 1,
    adv_pgd_at_enabled: bool = False,
) -> dict[str, dict]:
    """
    Build a dict of recipes (one per method) starting from the *online* base.
    """
    base = dict(base_recipe) if base_recipe is not None else build_online_base_recipe_for_search(
        epochs_phaseA=epochs_phaseA,
        epochs_phaseB=epochs_phaseB,
        adv_pgd_at_enabled=adv_pgd_at_enabled,
    )
    out = {}
    for m in methods:
        rec = json.loads(json.dumps(base))  # deepcopy-safe
        deep_update(rec, method_patch_online(m))
        out[m] = rec
    return out

def build_online_hparam_grid_recipes(
    methods: list[str],
    base_recipe: dict | None = None,
    epochs_phaseA: int = 10,
    epochs_phaseB: int = 1,
    adv_pgd_at_enabled: bool = False,
) -> dict[str, dict]:
    """
    Build a dict of recipes with HP-grid variants per method (online-aware).
    """
    base = dict(base_recipe) if base_recipe is not None else build_online_base_recipe_for_search(
        epochs_phaseA=epochs_phaseA,
        epochs_phaseB=epochs_phaseB,
        adv_pgd_at_enabled=adv_pgd_at_enabled,
    )
    out = {}
    for m in methods:
        grid = _grid_patches_for_method_online(m, variants_per_method=5)
        for i, hp_patch in enumerate(grid, 1):
            rec = deepcopy(base)
            deep_update(rec, method_patch_online(m))   # turn on the method
            deep_update(rec, hp_patch)                 # apply method-specific HPs
            name = m if (m == "baseline" and len(grid) == 1) else f"{m}#v{i}"
            out[name] = rec
    return out

# ---- Example usage:
# methods = ['baseline','cutmix','mixup','labelsmooth','layerdecay','pgd_at']
# base = build_online_base_recipe_for_search(epochs_phaseA=10, epochs_phaseB=1, adv_pgd_at_enabled=False)
# grid = build_online_hparam_grid_recipes(methods, base_recipe=base, epochs_phaseA=10, epochs_phaseB=1)


