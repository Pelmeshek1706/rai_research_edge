# pgd_at_training_with_torchattacks.py
# ------------------------------------------------------------
# MobileNetV3-Small on CIFAR-10 (cat/dog) + PGD Adversarial Training
# Benchmark robustness with torchattacks (FGSM/PGD/MIFGSM/DIFGSM/TIFGSM/APGD)
# Returns a pandas DataFrame per config with clean/robust acc and "tips"
# ------------------------------------------------------------

from __future__ import annotations
import os, math, random, gc, json
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable, List

# ---- Core deps
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd

# ---- Vision
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

# ---- Lightning
import lightning as L
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

# ---- torchattacks (explicit imports for full functionality)
from torchattacks import FGSM, PGD, MIFGSM, DIFGSM, TIFGSM, APGD  # pip install torchattacks

# ---- import your configs from a DIFFERENT method/module
# Replace this with your own provider returning a dict[str, dict] of recipes
# Example signature: def get_recipes_for_benchmark() -> Dict[str, dict]
from my_configs import get_recipes_for_benchmark  # <-- YOU provide this module/function


# =========================
# Repro & backend safety
# =========================
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42); L.seed_everything(42, workers=True)
try:
    torch.multiprocessing.set_sharing_strategy("file_system")
except Exception:
    pass
torch.backends.cudnn.benchmark = False


# =========================
# Normalization (ImageNet)
# =========================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def tf_train(img_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def tf_eval(img_size: int = 224, normalize: bool = True):
    t = [
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ]
    if normalize:
        t.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(t)


# ============================================================
# CIFAR-10 (cat/dog) datamodule (Lightning-style)
# ============================================================
class CIFAR10CatDogDM(L.LightningDataModule):
    CAT, DOG = 3, 5

    def __init__(self,
                 root: str = "./data",
                 batch_size: int = 128,
                 num_workers: int = 0,
                 img_size: int = 224,
                 val_ratio: float = 0.10,
                 pin_memory: bool = False):
        super().__init__()
        self.root = root
        self.bs = batch_size
        self.nw = int(num_workers)
        self.img = img_size
        self.val_ratio = float(val_ratio)
        self.pin = bool(pin_memory)
        self.train_ds = self.val_ds = self.test_ds = None

    @staticmethod
    def _remap(y: int) -> int:
        return 0 if y == CIFAR10CatDogDM.CAT else 1

    def setup(self, stage=None):
        tr = datasets.CIFAR10(self.root, train=True,  download=True)
        te = datasets.CIFAR10(self.root, train=False, download=True)

        def split_one(ds, label, ratio):
            idx = [i for i, (_, y) in enumerate(ds) if y == label]
            random.shuffle(idx)
            k = max(1, int(len(idx)*ratio))
            return idx[k:], idx[:k]

        cat_tr, cat_val = split_one(tr, self.CAT, self.val_ratio)
        dog_tr, dog_val = split_one(tr, self.DOG, self.val_ratio)

        class _SubsetRemap(Dataset):
            def __init__(self, base, idxs, tf):
                self.base, self.idxs, self.tf = base, idxs, tf
            def __len__(self): return len(self.idxs)
            def __getitem__(self, i):
                x, y = self.base[self.idxs[i]]
                x = self.tf(x)
                return x, CIFAR10CatDogDM._remap(y)

        self.train_ds = _SubsetRemap(tr, cat_tr + dog_tr, tf_train(self.img))
        self.val_ds   = _SubsetRemap(tr, cat_val + dog_val, tf_eval(self.img, normalize=True))
        te_idx = [i for i, (_, y) in enumerate(te) if y in (self.CAT, self.DOG)]
        self.test_ds  = _SubsetRemap(te, te_idx, tf_eval(self.img, normalize=True))

    def _dl(self, ds, shuffle):
        args = dict(batch_size=self.bs, shuffle=shuffle, num_workers=self.nw,
                    pin_memory=self.pin, persistent_workers=False)
        if self.nw > 0:
            args["prefetch_factor"] = 2
        return DataLoader(ds, **args)

    def train_dataloader(self): return self._dl(self.train_ds, True)
    def val_dataloader(self):   return self._dl(self.val_ds,   False)
    def test_dataloader(self):  return self._dl(self.test_ds,  False)

    # For torchattacks: provide RAW (non-normalized) test loader in [0,1]
    def raw_test_dataloader(self):
        te = datasets.CIFAR10(self.root, train=False, download=True)
        te_idx = [i for i, (_, y) in enumerate(te) if y in (self.CAT, self.DOG)]

        class _SubsetRemapRaw(Dataset):
            def __init__(self, base, idxs, tf):
                self.base, self.idxs, self.tf = base, idxs, tf
            def __len__(self): return len(self.idxs)
            def __getitem__(self, i):
                x, y = self.base[self.idxs[i]]
                x = self.tf(x)  # ToTensor() only (0..1), resize+crop
                y = CIFAR10CatDogDM._remap(y)
                return x, y

        raw_tf = tf_eval(self.img, normalize=False)  # NO Normalize here
        raw_ds = _SubsetRemapRaw(te, te_idx, raw_tf)
        return DataLoader(raw_ds, batch_size=self.bs, shuffle=False,
                          num_workers=self.nw, pin_memory=self.pin,
                          persistent_workers=False)


# ============================================================
# PGD (ℓ∞) in [0,1] space for training (your PGD-AT)
# ============================================================
def _to01(x_norm, mean, std): return x_norm * std + mean
def _tonorm(x01, mean, std):  return (x01 - mean) / std

@torch.enable_grad()
def linf_pgd_attack_train(model: nn.Module,
                          x_norm: torch.Tensor,
                          y: torch.Tensor,
                          mean: torch.Tensor,
                          std: torch.Tensor,
                          eps: float = 8/255,
                          alpha: float = 2/255,
                          steps: int = 10,
                          random_start: bool = True) -> torch.Tensor:
    x01 = _to01(x_norm, mean, std).clamp_(0.0, 1.0)
    x_adv01 = (x01 + torch.empty_like(x01).uniform_(-eps, eps)).clamp_(0.0, 1.0).detach() if random_start else x01.clone().detach()
    for _ in range(int(steps)):
        x_adv01.requires_grad_(True)
        logits = model(_tonorm(x_adv01, mean, std))
        loss = F.cross_entropy(logits, y)
        (grad,) = torch.autograd.grad(loss, x_adv01, retain_graph=False, create_graph=False, only_inputs=True)
        x_adv01 = x_adv01.detach() + alpha * grad.detach().sign()
        delta = torch.clamp(x_adv01 - x01, min=-eps, max=eps)
        x_adv01 = (x01 + delta).clamp_(0.0, 1.0)
    return _tonorm(x_adv01.detach(), mean, std)


# ============================================================
# Model (LightningModule) + PGD-AT in training_step
# ============================================================
class LitBinaryClassifier(L.LightningModule):
    def __init__(self,
                 lr: float = 3e-4,
                 weight_decay: float = 1e-4,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 dropout_p: float = 0.2,
                 adv_cfg: Optional[dict] = None,
                 mean: Tuple[float,float,float] = IMAGENET_MEAN,
                 std:  Tuple[float,float,float] = IMAGENET_STD):
        super().__init__()
        self.save_hyperparameters()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mb = models.mobilenet_v3_small(weights=weights)
        in_features = mb.classifier[3].in_features
        mb.classifier[3] = nn.Linear(in_features, 2)
        if isinstance(mb.classifier[2], nn.Dropout):
            mb.classifier[2].p = float(dropout_p)
        if freeze_backbone:
            for p in mb.features.parameters(): p.requires_grad = False
        self.backbone = mb

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc   = BinaryAccuracy(); self.val_f1  = BinaryF1Score()
        self.test_acc  = BinaryAccuracy(); self.test_f1 = BinaryF1Score()

        self.register_buffer("mean_buf", torch.tensor(mean).view(1,3,1,1), persistent=False)
        self.register_buffer("std_buf",  torch.tensor(std).view(1,3,1,1),  persistent=False)

        self.adv_cfg = adv_cfg or {
            "enabled": False,
            "eps": 8/255, "alpha": 2/255, "steps": 10, "random_start": True,
            "mix_clean_adv": 0.0
        }

    def forward(self, x): return self.backbone(x)
    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],
                                 lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def training_step(self, batch, _):
        x, y = batch
        logits_clean = self(x)
        loss_clean = self.criterion(logits_clean, y)
        loss = loss_clean

        if self.adv_cfg.get("enabled", False):
            x_adv = linf_pgd_attack_train(
                self, x, y, self.mean_buf, self.std_buf,
                eps=self.adv_cfg.get("eps", 8/255),
                alpha=self.adv_cfg.get("alpha", 2/255),
                steps=self.adv_cfg.get("steps", 10),
                random_start=self.adv_cfg.get("random_start", True)
            )
            logits_adv = self(x_adv)
            loss_adv = self.criterion(logits_adv, y)
            mix = float(self.adv_cfg.get("mix_clean_adv", 0.0))
            loss = mix * loss_clean + (1.0 - mix) * loss_adv

        preds = torch.argmax(logits_clean.detach(), dim=1)
        self.train_acc.update(preds, y)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc",  self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y); self.val_f1.update(preds, y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.log("val/f1",  self.val_f1.compute(),  prog_bar=False)
        self.val_acc.reset(); self.val_f1.reset()

    def test_step(self, batch, _):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y); self.test_f1.update(preds, y)
        self.log("test/loss", loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        self.log("test/acc", self.test_acc.compute(), prog_bar=False)
        self.log("test/f1",  self.test_f1.compute(),  prog_bar=False)
        self.test_acc.reset(); self.test_f1.reset()


# ============================================================
# Trainer factory (explicit epochs; no resume side-effects)
# ============================================================
def make_trainer(root: str = "./runs", epochs: int = 10, precision_override: Optional[str] = None) -> L.Trainer:
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
    )
    return trainer


# ============================================================
# Torchattacks evaluation (on raw [0,1] inputs)
# ============================================================
class NormalizeWrapper(nn.Module):
    """Wraps a classifier that expects normalized input so that it can
    accept [0,1] images; used for torchattacks which perturb in [0,1]."""
    def __init__(self, model: nn.Module, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.model = model.eval()
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(std).view(1,3,1,1))
    def forward(self, x01):
        x = (x01 - self.mean) / self.std
        return self.model(x)

@torch.no_grad()
def _acc(model: nn.Module, loader: DataLoader, device=None) -> float:
    device = device or next(model.parameters()).device
    correct = 0; total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item(); total += y.numel()
    return (correct / total) if total else 0.0

def _attack_tips(clean_acc: float, atk_acc: float, name: str, eps: float, steps: int) -> str:
    drop = clean_acc - atk_acc
    tips: List[str] = []
    if drop >= 0.5 * max(1e-6, clean_acc) or atk_acc <= 0.05:
        tips.append("Severe drop: enable PGD-AT (or TRADES), verify eps/steps (L∞ 8/255), and consider stronger aug.")
    if name in {"PGD", "APGD"} and steps < 10:
        tips.append("Increase steps (≥10–20) for stable PGD/APGD eval.")
    if name in {"MIFGSM", "DIFGSM", "TIFGSM"}:
        tips.append("Momentum/diversity/translation attacks expose texture bias; AT + stronger crops often help.")
    if not tips:
        tips.append("Looks ok; cross-check with AutoAttack for sanity.")
    return " ".join(tips)

@torch.no_grad()
def benchmark_with_torchattacks(model_norm: nn.Module,
                                raw_loader: DataLoader,
                                device=None,
                                eps: float = 8/255,
                                steps: int = 10) -> pd.DataFrame:
    """Return per-attack clean/robust acc + tips."""
    device = device or next(model_norm.parameters()).device
    model_norm.to(device).eval()
    clean_acc = _acc(model_norm, raw_loader, device=device)

    # instantiate attacks (defaults are sensible; we override eps/steps where relevant)
    attacks = {
        "FGSM":  FGSM(model_norm, eps=eps),
        "PGD":   PGD(model_norm,  eps=eps, alpha=2/255, steps=steps, random_start=True),
        "MIFGSM":MIFGSM(model_norm, eps=eps, alpha=2/255, steps=steps),
        "DIFGSM":DIFGSM(model_norm, eps=eps, alpha=2/255, steps=steps),
        "TIFGSM":TIFGSM(model_norm, eps=eps, alpha=2/255, steps=steps),
        "APGD":  APGD(model_norm, norm='Linf', eps=eps, steps=steps, n_restarts=1, verbose=False),
    }

    rows = []
    for name, atk in attacks.items():
        correct = 0; total = 0
        for x01, y in raw_loader:
            x01, y = x01.to(device), y.to(device)
            x_adv = atk(x01, y)
            pred = model_norm(x_adv).argmax(1)
            correct += (pred == y).sum().item(); total += y.numel()
        atk_acc = (correct / total) if total else 0.0
        rows.append({
            "attack": name,
            "eps": float(eps),
            "steps": int(steps),
            "clean_acc": float(clean_acc),
            "attack_acc": float(atk_acc),
            "drop_vs_clean": float(clean_acc - atk_acc),
            "tip": _attack_tips(clean_acc, atk_acc, name, eps, steps),
        })
    return pd.DataFrame(rows).sort_values(by=["attack_acc"], ascending=False).reset_index(drop=True)


# ============================================================
# Orchestrator: one recipe -> train + validate + test + attacks
# Expects recipe to include 'data', 'model', 'opt', 'trainer', 'adv_training'
# ============================================================
def train_and_attack(recipe: dict,
                     out_dir: str | None = None,
                     atk_eps: float = 8/255,
                     atk_steps: int = 10) -> pd.DataFrame:
    # Data
    d = recipe.get("data", {})
    dm = CIFAR10CatDogDM(root=d.get("root", "./data"),
                         batch_size=d.get("batch_size", 128),
                         num_workers=d.get("num_workers", 0),
                         img_size=d.get("img_size", 224),
                         val_ratio=d.get("val_ratio", 0.10),
                         pin_memory=d.get("pin_memory", False))
    dm.setup()

    # Model
    m = recipe.get("model", {})
    adv = recipe.get("adv_training", {"enabled": False})
    model = LitBinaryClassifier(
        lr=recipe.get("opt", {}).get("lr", 3e-4),
        weight_decay=recipe.get("opt", {}).get("wd", 1e-4),
        pretrained=m.get("pretrained", True),
        freeze_backbone=m.get("freeze_backbone", False),
        dropout_p=m.get("dropout_p", 0.2),
        adv_cfg=adv,
        mean=IMAGENET_MEAN, std=IMAGENET_STD,
    )

    # Train/Val/Test
    t = recipe.get("trainer", {})
    trainer = make_trainer(root=t.get("root", "./runs/pgd_at"), epochs=t.get("epochs", 10))
    trainer.fit(model, datamodule=dm, ckpt_path=None)
    trainer.validate(model, datamodule=dm, ckpt_path=None)
    trainer.test(model, datamodule=dm, ckpt_path=None)

    # Torchattacks benchmarking on RAW [0,1] loader + Normalize wrapper
    raw_test_loader = dm.raw_test_dataloader()
    model_norm = NormalizeWrapper(model)

    atk_df = benchmark_with_torchattacks(model_norm, raw_test_loader,
                                         eps=float(atk_eps), steps=int(atk_steps))

    # Optional: persist artifacts
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        atk_df.to_csv(os.path.join(out_dir, f"attacks_{t.get('epochs','?')}e.csv"), index=False)
        with open(os.path.join(out_dir, "recipe.json"), "w", encoding="utf-8") as f:
            json.dump(recipe, f, ensure_ascii=False, indent=2)

    # Cleanup
    del model, model_norm; gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    return atk_df


# ============================================================
# (Optional) Multi-recipe runner collecting one DF per config
# ============================================================
def run_recipes_and_collect(recipes: Dict[str, dict],
                            out_dir: str = "./grid_results",
                            atk_eps: float = 8/255,
                            atk_steps: int = 10) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, recipe in recipes.items():
        print(f"[run] recipe={name}")
        df = train_and_attack(recipe, out_dir=os.path.join(out_dir, name),
                              atk_eps=atk_eps, atk_steps=atk_steps)
        # aggregate per attack into a single line per recipe (keep the details inline)
        clean = float(df["clean_acc"].iloc[0]) if not df.empty else None
        row = {
            "recipe": name,
            "k": 1,
            "methods": [name],
            "accuracy": clean,
            "mem_peak_bytes": None,
            "mem_peak_GiB": None,
            "config": recipe,
            # flatten per-attack scores (optional)
            **{f"acc_{r.attack.iloc[0]}": float(r.attack_acc.iloc[0]) for _, r in df.groupby("attack")},
            "attacks_table_json": df.to_json(orient="records"),
        }
        rows.append(row)
        # hygiene
        del df; gc.collect()
        try: torch.cuda.empty_cache()
        except Exception: pass
    return pd.DataFrame(rows)


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    # You provide this function in my_configs.py (outside this file)
    # It should return: Dict[str, dict] with keys: data/model/opt/trainer/adv_training
    recipes = get_recipes_for_benchmark()

    # Single run:
    # df_attacks = train_and_attack(recipes["baseline_pgd_at"], out_dir="./runs_single", atk_eps=8/255, atk_steps=10)
    # print(df_attacks)

    # Or batch:
    summary = run_recipes_and_collect(recipes, out_dir="./grid_results", atk_eps=8/255, atk_steps=10)
    print(summary.head())
