# ============================
# TRAIN + EVAL FOR MULTIPLE ATTACK CONFIGS
# (uses your TwoClassDataModule/LitBinaryClassifier from the other file)
# ============================
import os, json, gc, time, math, contextlib
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Iterable
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# ---------- if these imports/classes already exist in your file, feel free to drop duplicates ----------
# (Only needed when they are not included elsewhere)
# from torchvision import transforms, datasets, models
# import lightning as L
# from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
# from your_file import TwoClassDataModule, LitBinaryClassifier, CollectMetricsCallback, \
#                      build_base_recipe, method_patch, deep_update, replace_activations_for_double_backward, \
#                      JacobianRegularizer, BatchAugmentor, build_optimizer, make_trainer

# ------------------ auxiliary wrappers for attacks (as before) ------------------
class NormalizedModel(nn.Module):
    """Wraps a model: expects x in [0,1] and applies mean/std before passing to the base model."""
    def __init__(self, base: nn.Module, mean: List[float], std: List[float], device: torch.device):
        super().__init__()
        self.base = base
        mean_t = torch.tensor(mean, dtype=torch.float32, device=device).view(1,3,1,1)
        std_t  = torch.tensor(std,  dtype=torch.float32, device=device).view(1,3,1,1)
        self.register_buffer("mean", mean_t, persistent=False)
        self.register_buffer("std",  std_t,  persistent=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.base(x)

class UnnormalizeView(Dataset):
    """Takes a dataset with normalized images and returns x in [0,1] on the fly."""
    def __init__(self, base: Dataset, mean: List[float], std: List[float]):
        self.base = base
        self.mean = torch.tensor(mean).view(3,1,1)
        self.std  = torch.tensor(std).view(3,1,1)
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x,y = self.base[idx]
        if isinstance(x, torch.Tensor):
            x = x.clone()
            x = x * self.std + self.mean
            x = x.clamp(0.0, 1.0)
        return x,y

def make_torchattacks_suite(model: nn.Module,
                            device: torch.device,
                            eps: float = 8/255,
                            steps: int = 10,
                            alpha: Optional[float] = None):
    """Build a collection of attacks. The model here is the normalizing wrapper (NormalizedModel)."""
    import torchattacks as ta
    if alpha is None:
        alpha = (eps / max(steps,1)) * 1.25
    return {
        "FGSM":   ta.FGSM(model, eps=eps),
        "PGD":    ta.PGD(model, eps=eps, alpha=alpha, steps=steps),
        "MIFGSM": ta.MIFGSM(model, eps=eps, steps=steps, decay=1.0),
        "DIFGSM": ta.DIFGSM(model, eps=eps, steps=steps, alpha=alpha),
        "TIFGSM": ta.TIFGSM(model, eps=eps, steps=steps, alpha=alpha),
        "APGD":   ta.APGD(model, eps=eps, steps=steps),
    }

# ------------------ utility: fetch mean/std from your DataModule ------------------
def get_mean_std_from_dm(dm) -> Tuple[List[float], List[float]]:
    # The DM from your file stores CIFAR constants on the class
    if hasattr(dm, "CIFAR10_MEAN") and hasattr(dm, "CIFAR10_STD"):
        return list(dm.CIFAR10_MEAN), list(dm.CIFAR10_STD)
    # fallback — ImageNet
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def _build_split_loader_for_attacks(dm, split: str, mean: List[float], std: List[float]) -> DataLoader:
    base_ds = {"train": dm.train_ds, "val": dm.val_ds, "test": dm.test_ds}[split]
    wrapped = UnnormalizeView(base_ds, mean, std)
    return DataLoader(
        wrapped,
        batch_size=getattr(dm, "batch_size", 64),
        shuffle=False,
        num_workers=getattr(dm, "num_workers", 4),
        pin_memory=True
    )

@torch.no_grad()
def _clean_acc(norm_model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    norm_model.eval()
    correct = total = 0
    for x,y in loader:
        x, y = x.to(device), y.to(device)
        pred = norm_model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return correct / max(total, 1)

def _adv_acc_single_attack(attack, norm_model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    norm_model.eval()
    correct = total = 0
    for x,y in loader:
        x, y = x.to(device), y.to(device)
        x_adv = attack(x, y)
        with torch.no_grad():
            pred = norm_model(x_adv).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return correct / max(total, 1)

# ------------------ training via recipe (copy of your run_experiment logic, but returns model+dm) ------------------
def train_from_recipe(recipe: dict, force_fp32_for_jacobian: bool = True):
    # --- Data ---
    dcfg = recipe.get("data", {})
    dm = TwoClassDataModule(
        workdir=dcfg.get("workdir","./workdir"),
        batch_size=dcfg.get("batch_size",128),
        num_workers=dcfg.get("num_workers",0),
        img_size=dcfg.get("img_size",224),
        val_ratio_per_class=dcfg.get("val_ratio",0.1)
    )
    dm.setup()

    # --- Model ---
    mcfg, ocfg = recipe.get("model", {}), recipe.get("optimizer", {})
    model = LitBinaryClassifier(
        lr=ocfg.get("lr",3e-4),
        weight_decay=ocfg.get("wd",1e-4),
        pretrained=mcfg.get("pretrained",True),
        freeze_backbone=mcfg.get("freeze_backbone",True),
        dropout_p=mcfg.get("dropout_p",0.2),
        spectral_norm=mcfg.get("spectral_norm",False),
        feature_noise_sigma=mcfg.get("feature_noise_sigma",0.0),
        label_smoothing=mcfg.get("label_smoothing",0.0),
        consistency_lambda=mcfg.get("consistency_lambda",0.0),
        consistency_noise_sigma=mcfg.get("consistency_noise_sigma",0.0),
        use_custom_optimizer=recipe.get("optimizer",{}).get("use_custom",True),
        layer_decay=recipe.get("optimizer",{}).get("layer_decay",None),
    )

    # --- Augmentor ---
    acfg = recipe.get("augment", {})
    if acfg.get("mixup", False) or acfg.get("cutmix", False):
        p = acfg.get("p", 1.0)
        priors = (1.0, 0.0) if (acfg.get("mixup", False) and not acfg.get("cutmix", False)) else \
                 (0.0, 1.0) if (acfg.get("cutmix", False) and not acfg.get("mixup", False)) else \
                 (0.5, 0.5)
        model.batch_aug = BatchAugmentor(
            mixup_alpha=acfg.get("mixup_alpha",0.2),
            cutmix_alpha=acfg.get("cutmix_alpha",1.0),
            p=p, mode_prior=priors, apply_on_train_only=True
        )

    # --- Jacobian + fix ---
    rcfg = recipe.get("regularizers", {}).get("jacobian", {})
    jac_enabled = rcfg.get("enabled", False)
    if jac_enabled:
        replace_activations_for_double_backward(model)
        model.jac_reg = JacobianRegularizer(lam=rcfg.get("lam",1e-4), subbatch=rcfg.get("subbatch",8))

    # --- Callbacks & Trainer ---
    metrics_cb = CollectMetricsCallback()
    cbs = [metrics_cb]
    if recipe.get("callbacks", {}).get("ema", False):
        cbs.append(EMAWeightsCallback(decay=recipe["callbacks"].get("ema_decay",0.999)))
    pcfg = recipe.get("callbacks", {}).get("prune", {})
    if pcfg.get("enabled", False):
        cbs.append(PruningCallback(amount=pcfg.get("amount",0.3),
                                   start_epoch=pcfg.get("start_epoch",1),
                                   reapply_each_epoch=pcfg.get("reapply",False)))
    tcfg = recipe.get("trainer", {})
    precision_override = "32-true" if (jac_enabled and force_fp32_for_jacobian) else None
    trainer = make_trainer(
        root_dir=tcfg.get("root_dir","./runs/auto"),
        epochs=tcfg.get("epochs",5),
        callbacks=cbs,
        precision_override=precision_override,
        num_sanity_val_steps=tcfg.get("num_sanity_val_steps", 0)
    )

    # --- Fit/Val/Test ---
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    df_hist = pd.DataFrame(metrics_cb.history)
    return model, dm, df_hist

# ------------------ attack evaluation across splits ------------------
def eval_attacks_for_model(model: nn.Module,
                           dm,
                           attacks: List[str],
                           eps: float = 8/255,
                           steps: int = 10,
                           device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Returns a dict with keys:
      'clean/train', 'clean/val', 'clean/test',
      '{ATTACK}/train', '{ATTACK}/val', '{ATTACK}/test'
    """
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model = model.to(device).eval()

    mean, std = get_mean_std_from_dm(dm)
    # normalizing wrapper for attacks and clean evaluation
    norm_model = NormalizedModel(model, mean, std, device).to(device).eval()

    # loaders (x in [0,1])
    loaders = {
        split: _build_split_loader_for_attacks(dm, split, mean, std)
        for split in ("train","val","test")
    }

    # baseline clean accuracies
    out: Dict[str, float] = {}
    for split, loader in loaders.items():
        out[f"clean/{split}"] = _clean_acc(norm_model, loader, device)

    # attack suite
    suite = make_torchattacks_suite(norm_model, device, eps=eps, steps=steps)

    # iterate over attacks and splits
    print("Now test atacks")
    for name in tqdm(attacks):
        atk = suite[name]
        for split, loader in loaders.items():
            adv_acc = _adv_acc_single_attack(atk, norm_model, loader, device)
            out[f"{name}/{split}"] = adv_acc

    # light memory cleanup
    del loaders, suite; torch.cuda.empty_cache(); gc.collect()
    return out

# ------------------ aggregation across multiple configs ------------------
def _normalize_config_collection(cfgs: Iterable) -> Dict[str, dict]:
    """
    Supports two formats:
      [{'cutmix': {...}}, {'mixup': {...}}, ...]  OR  {'cutmix': {...}, 'mixup': {...}}
    """
    if isinstance(cfgs, dict):
        return cfgs
    norm: Dict[str, dict] = {}
    for item in cfgs:
        if isinstance(item, dict) and len(item) == 1:
            name = next(iter(item.keys()))
            norm[name] = item[name]
        else:
            raise ValueError("Expected entries of the form {name: recipe}")
    return norm

def benchmark_configs_with_attacks(configs,
                                   attacks: List[str] = ("FGSM","PGD","MIFGSM","DIFGSM","TIFGSM","APGD"),
                                   eps: float = 8/255,
                                   steps: int = 10,
                                   out_dir: str = "./bench_attacks") -> pd.DataFrame:
    """
    Main routine:
      - train with each config,
      - compute clean and adversarial accuracy on train/val/test for the requested attacks,
      - return a wide table (columns: clean/* and ATTACK/*).
    """
    os.makedirs(out_dir, exist_ok=True)
    cfgs = _normalize_config_collection(configs)

    rows = []
    for name, recipe in cfgs.items():
        print(f"[run] config={name}")
        model, dm, hist = train_from_recipe(recipe)
        # optionally save history
        hist.to_csv(os.path.join(out_dir, f"{name}.history.csv"), index=False)
        with open(os.path.join(out_dir, f"{name}.recipe.json"), "w", encoding="utf-8") as f:
            json.dump(recipe, f, ensure_ascii=False, indent=2)
        # attacks
        accs = eval_attacks_for_model(model, dm, list(attacks), eps=eps, steps=steps)
        row = {"config": name, **accs}
        rows.append(row)

        # gently clean up memory between configs
        del model, dm, hist; torch.cuda.empty_cache(); gc.collect()

    df = pd.DataFrame(rows).sort_values("config").reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return df


# your list of best configs (sample from the message):
t_cfgs = [
    {"cutmix": {
        "data": {"batch_size": 128, "img_size": 224, "num_workers": 0, "workdir": "./workdir"},
        "model": {"pretrained": True, "freeze_backbone": True, "dropout_p": 0.2,
                  "spectral_norm": False, "feature_noise_sigma": 0.0,
                  "label_smoothing": 0.0, "consistency_lambda": 0.0, "consistency_noise_sigma": 0.0},
        "optimizer": {"use_custom": True, "layer_decay": None, "wd": 1e-4, "lr": 3e-4},
        "augment": {"mixup": False, "cutmix": True, "p": 1.0, "mixup_alpha": 0.2, "cutmix_alpha": 1.0},
        "regularizers": {"jacobian": {"enabled": False, "lam": 1e-4, "subbatch": 8}},
        "callbacks": {"ema": False, "ema_decay": 0.999, "prune": {"enabled": False, "amount": 0.3, "start_epoch": 1}},
        "trainer": {"epochs": 3, "root_dir": "./runs/auto", "num_sanity_val_steps": 0},
        "eval": {"temp_scaling": False, "noise_avg": {"enabled": False, "n": 8, "sigma": 0.05}}
    }},
    # add more {name: recipe} entries here
]

attacks = ["FGSM","PGD","MIFGSM","DIFGSM","TIFGSM","APGD"]
summary = benchmark_configs_with_attacks(best_recipes, attacks=attacks, eps=8/255, steps=10, out_dir="./bench_attacks")


# The columns will look like this:
# ['config', 'clean/train','clean/val','clean/test',
#  'FGSM/train','FGSM/val','FGSM/test', 'PGD/train', ... , 'APGD/test']
