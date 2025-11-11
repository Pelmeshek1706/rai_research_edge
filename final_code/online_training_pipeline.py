# online_training_pipeline.py
# ------------------------------------------------------------
# Phase A: Train on CIFAR-10 (cat/dog)
# Phase B: Online-style supervised adaptation on Oxford-IIIT Pet (binary cat/dog)
# Phase C: Verify on STL-10 (cat/dog), no updates
# Returns a DataFrame with per-config accuracies (base/online_pre/online_post/third)
# ------------------------------------------------------------

from __future__ import annotations
import os, json, math, random, gc
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pandas as pd

from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V3_Small_Weights

import lightning as L
from torchmetrics.classification import BinaryAccuracy

# ===== Repro =====
def set_seed(seed: int = 42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42); L.seed_everything(42, workers=True)
try: torch.multiprocessing.set_sharing_strategy("file_system")
except Exception: pass
torch.backends.cudnn.benchmark = False

# ===== Normalization =====
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def make_tf_train(img_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2,0.2,0.2,0.02),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def make_tf_eval(img_size=224, normalize=True):
    t = [transforms.Resize(256), transforms.CenterCrop(img_size), transforms.ToTensor()]
    if normalize: t.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose(t)

# ===== CIFAR-10 (cat/dog) DM =====
class CIFAR10CatsDogs(L.LightningDataModule):
    CAT, DOG = 3, 5
    def __init__(self, root="./data", batch_size=128, num_workers=0, img_size=224, val_ratio=0.10, pin_memory=False):
        super().__init__()
        self.root=root; self.bs=batch_size; self.nw=int(num_workers)
        self.img=img_size; self.val_ratio=float(val_ratio); self.pin=bool(pin_memory)
        self.train_ds=self.val_ds=self.test_ds=None

    @staticmethod
    def _remap(y): return 0 if y==CIFAR10CatsDogs.CAT else 1

    def setup(self, stage=None):
        tr = datasets.CIFAR10(self.root, train=True,  download=True)
        te = datasets.CIFAR10(self.root, train=False, download=True)

        def split_idxs(ds, label, r):
            idx = [i for i,(_,y) in enumerate(ds) if y==label]; random.shuffle(idx)
            k=max(1,int(len(idx)*r)); return idx[k:], idx[:k]

        cat_tr, cat_val = split_idxs(tr, self.CAT, self.val_ratio)
        dog_tr, dog_val = split_idxs(tr, self.DOG, self.val_ratio)
        te_idx = [i for i,(_,y) in enumerate(te) if y in (self.CAT,self.DOG)]

        class _Subset(Dataset):
            def __init__(self, base, idxs, tf): self.base, self.idxs, self.tf = base, idxs, tf
            def __len__(self): return len(self.idxs)
            def __getitem__(self, i):
                x,y = self.base[self.idxs[i]]
                x = self.tf(x); y = CIFAR10CatsDogs._remap(y); return x,y

        self.train_ds = _Subset(tr, cat_tr+dog_tr, make_tf_train(self.img))
        self.val_ds   = _Subset(tr, cat_val+dog_val, make_tf_eval(self.img, True))
        self.test_ds  = _Subset(te, te_idx,          make_tf_eval(self.img, True))

    def _dl(self, ds, shuffle):
        args=dict(batch_size=self.bs, shuffle=shuffle, num_workers=self.nw,
                  pin_memory=self.pin, persistent_workers=False)
        if self.nw>0: args["prefetch_factor"]=2
        return DataLoader(ds, **args)

    def train_dataloader(self): return self._dl(self.train_ds, True)
    def val_dataloader(self):   return self._dl(self.val_ds,   False)
    def test_dataloader(self):  return self._dl(self.test_ds,  False)

# ===== Oxford-IIIT Pet stream loader (binary cat/dog) =====
def make_pets_stream_loader(root="./data/oxford-iiit-pet", batch_size=64, num_workers=0, img_size=224,
                            split="trainval", pin_memory=False):
    ds = datasets.OxfordIIITPet(
        root, split=split, target_types="binary-category",
        transform=make_tf_eval(img_size, True), download=True
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=int(num_workers),
                      pin_memory=bool(pin_memory), persistent_workers=False)

# ===== STL-10 (filter to cat/dog) eval loader =====
def make_stl10_eval_loader(root="./data", batch_size=128, num_workers=0, img_size=224,
                           split="test", pin_memory=False):
    ds = datasets.STL10(root, split=split, transform=make_tf_eval(img_size, True), download=True)
    # robust mapping using class names
    name_to_idx = {name:i for i,name in enumerate(getattr(ds, "classes", []))}
    keep_set = {name_to_idx.get("cat"), name_to_idx.get("dog")}
    keep_set.discard(None)
    idxs = [i for i,(_,y) in enumerate(ds) if y in keep_set]
    class _Subset(Dataset):
        def __init__(self, base, idxs): self.base, self.idxs = base, idxs
        def __len__(self): return len(self.idxs)
        def __getitem__(self, i): return self.base[self.idxs[i]]
    sub = _Subset(ds, idxs)
    return DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=int(num_workers),
                      pin_memory=bool(pin_memory), persistent_workers=False)

# ===== PGD-AT core (for training) =====
def _to01(x, m, s): return x*s + m
def _tonorm(x, m, s): return (x - m) / s

@torch.enable_grad()
def linf_pgd_attack_train(model, x_norm, y, mean, std, eps=8/255, alpha=2/255, steps=10, random_start=True):
    x01 = _to01(x_norm, mean, std).clamp_(0,1)
    x_adv01 = (x01 + torch.empty_like(x01).uniform_(-eps, eps)).clamp_(0,1).detach() if random_start else x01.clone().detach()
    for _ in range(int(steps)):
        x_adv01.requires_grad_(True)
        logits = model(_tonorm(x_adv01, mean, std))
        loss = F.cross_entropy(logits, y)
        (grad,) = torch.autograd.grad(loss, x_adv01, retain_graph=False, create_graph=False, only_inputs=True)
        x_adv01 = x_adv01.detach() + alpha * grad.detach().sign()
        delta = torch.clamp(x_adv01 - x01, -eps, eps)
        x_adv01 = (x01 + delta).clamp_(0,1)
    return _tonorm(x_adv01.detach(), mean, std)

# ===== Model =====
class LitBinaryClassifier(L.LightningModule):
    def __init__(self, lr=3e-4, weight_decay=1e-4, pretrained=True, freeze_backbone=False, dropout_p=0.2,
                 adv_cfg: Optional[dict]=None, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__()
        self.save_hyperparameters()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mb = models.mobilenet_v3_small(weights=weights)
        in_f = mb.classifier[3].in_features
        mb.classifier[3] = nn.Linear(in_f, 2)
        if isinstance(mb.classifier[2], nn.Dropout): mb.classifier[2].p = float(dropout_p)
        if freeze_backbone:
            for p in mb.features.parameters(): p.requires_grad=False
        self.backbone = mb
        self.crit = nn.CrossEntropyLoss()
        self.train_acc = BinaryAccuracy(); self.val_acc = BinaryAccuracy()
        self.register_buffer("mean_buf", torch.tensor(mean).view(1,3,1,1), persistent=False)
        self.register_buffer("std_buf",  torch.tensor(std).view(1,3,1,1),  persistent=False)
        self.adv_cfg = adv_cfg or {"enabled": False, "eps":8/255, "alpha":2/255, "steps":10, "random_start":True, "mix_clean_adv":0.0}

    def forward(self,x): return self.backbone(x)
    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad],
                                 lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def training_step(self, batch, _):
        x,y = batch
        logits_clean = self(x); loss_clean = self.crit(logits_clean, y)
        loss = loss_clean
        if self.adv_cfg.get("enabled", False):
            x_adv = linf_pgd_attack_train(self, x, y, self.mean_buf, self.std_buf,
                                          eps=self.adv_cfg.get("eps",8/255),
                                          alpha=self.adv_cfg.get("alpha",2/255),
                                          steps=self.adv_cfg.get("steps",10),
                                          random_start=self.adv_cfg.get("random_start",True))
            logits_adv = self(x_adv); loss_adv = self.crit(logits_adv, y)
            mix = float(self.adv_cfg.get("mix_clean_adv", 0.0))
            loss = mix*loss_clean + (1.0-mix)*loss_adv
        self.train_acc.update(logits_clean.argmax(1).detach(), y)
        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        x,y = batch; logits=self(x); loss=self.crit(logits,y)
        self.val_acc.update(logits.argmax(1), y)
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute(), prog_bar=True); self.val_acc.reset()

# ===== Trainer =====
def make_trainer(root="./runs", epochs=10, precision_override: Optional[str]=None):
    torch.set_float32_matmul_precision("medium")
    use_cuda = torch.cuda.is_available()
    if precision_override is None:
        bf16_ok = use_cuda and torch.cuda.is_bf16_supported()
        precision = "bf16-mixed" if bf16_ok else ("16-mixed" if use_cuda else "32-true")
    else:
        precision = precision_override
    return L.Trainer(accelerator="gpu" if use_cuda else "cpu",
                     devices=1, precision=precision,
                     max_epochs=int(epochs), min_epochs=int(epochs),
                     default_root_dir=root,
                     enable_checkpointing=False, log_every_n_steps=10,
                     num_sanity_val_steps=0, enable_progress_bar=True)

# ===== Eval helper =====
@torch.no_grad()
def evaluate_acc(model: nn.Module, loader: DataLoader, device=None) -> float:
    device = device or next(model.parameters()).device
    model.eval().to(device)
    correct=0; total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred==y).sum().item(); total += y.numel()
    return (correct/total) if total else 0.0

# ===== Online adaptation =====
def online_adapt_supervised(model: nn.Module, stream_loader: DataLoader,
                            epochs: int = 1, lr: float = 3e-5, weight_decay: float = 0.0,
                            trainable: str = "classifier", last_n: int = 0,
                            limit_batches_per_epoch: Optional[int] = None,
                            max_steps: Optional[int] = None) -> dict:
    params=[]
    if trainable=="classifier":
        for p in model.backbone.features.parameters(): p.requires_grad=False
        for p in model.backbone.classifier.parameters(): p.requires_grad=True
        params = [p for p in model.backbone.classifier.parameters() if p.requires_grad]
    elif trainable=="last_n":
        for p in model.parameters(): p.requires_grad=False
        layers=list(model.backbone.features)
        for layer in layers[-int(max(1,last_n)):]:
            for p in layer.parameters(): p.requires_grad=True
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        for p in model.parameters(): p.requires_grad=True
        params = [p for p in model.parameters() if p.requires_grad]

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()
    device = next(model.parameters()).device
    model.train()
    step=0
    for ep in range(int(epochs)):
        for b,(x,y) in enumerate(stream_loader):
            if (limit_batches_per_epoch is not None) and (b>=limit_batches_per_epoch): break
            if (max_steps is not None) and (step>=max_steps): break
            x,y=x.to(device), y.to(device)
            logits=model(x); loss=crit(logits,y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            step+=1
        if (max_steps is not None) and (step>=max_steps): break
    return {"epochs": int(epochs), "steps": step}

# ===== One-config runner that also returns model + loaders (used by File 2) =====
def train_base_and_online(recipe: dict):
    # Phase A
    A = recipe["phaseA"]
    dm = CIFAR10CatsDogs(root=A["data"].get("root","./data"),
                         batch_size=A["data"].get("batch_size",128),
                         num_workers=A["data"].get("num_workers",0),
                         img_size=A["data"].get("img_size",224),
                         val_ratio=A["data"].get("val_ratio",0.10),
                         pin_memory=A["data"].get("pin_memory",False))
    dm.setup()
    model = LitBinaryClassifier(
        lr=A["opt"].get("lr",3e-4), weight_decay=A["opt"].get("wd",1e-4),
        pretrained=A["model"].get("pretrained",True),
        freeze_backbone=A["model"].get("freeze_backbone",False),
        dropout_p=A["model"].get("dropout_p",0.2),
        adv_cfg=A.get("adv_training", {"enabled": False})
    )
    trainer = make_trainer(root=A["trainer"].get("root","./runs/phaseA"),
                           epochs=A["trainer"].get("epochs",10))
    trainer.fit(model, datamodule=dm, ckpt_path=None)
    base_test_acc = evaluate_acc(model, dm.test_dataloader())

    # Phase B (online)
    B = recipe["phaseB"]
    pets_loader = make_pets_stream_loader(root=B.get("root","./data/oxford-iiit-pet"),
                                          batch_size=B.get("batch_size",64),
                                          num_workers=B.get("num_workers",0),
                                          img_size=B.get("img_size",224),
                                          split=B.get("split","trainval"),
                                          pin_memory=B.get("pin_memory",False))
    online_pre_acc = evaluate_acc(model, pets_loader)
    online_adapt_supervised(
        model, pets_loader,
        epochs=B.get("epochs",1), lr=B.get("lr",3e-5), weight_decay=B.get("wd",0.0),
        trainable=B.get("trainable","classifier"), last_n=B.get("last_n",0),
        limit_batches_per_epoch=B.get("limit_batches_per_epoch", None),
        max_steps=B.get("max_steps", None),
    )
    online_post_acc = evaluate_acc(model, pets_loader)

    # Phase C (third dataset)
    C = recipe["phaseC"]
    stl_loader = make_stl10_eval_loader(root=C.get("root","./data"),
                                        batch_size=C.get("batch_size",128),
                                        num_workers=C.get("num_workers",0),
                                        img_size=C.get("img_size",224),
                                        split=C.get("split","test"),
                                        pin_memory=C.get("pin_memory",False))
    third_dataset_acc = evaluate_acc(model, stl_loader)

    extras = {
        "base_test_acc": base_test_acc,
        "online_pre_acc": online_pre_acc,
        "online_post_acc": online_post_acc,
        "third_dataset_acc": third_dataset_acc,
        "pets_cfg": {k:B.get(k) for k in ["root","batch_size","num_workers","img_size","split","pin_memory"]},
        "stl_cfg":  {k:C.get(k) for k in ["root","batch_size","num_workers","img_size","split","pin_memory"]},
    }
    return model, pets_loader, stl_loader, extras

# ===== Multi-config API (returns DataFrame) =====
def run_online_recipes_and_summarize(recipes: Dict[str, dict], out_dir: str="./online_results") -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    rows=[]
    for name, recipe in recipes.items():
        print(f"[online] {name}")
        model, pets_loader, stl_loader, extras = train_base_and_online(recipe)
        row = {
            "recipe": name,
            "config": recipe,
            "base_test_acc": extras["base_test_acc"],
            "online_pre_acc": extras["online_pre_acc"],
            "online_post_acc": extras["online_post_acc"],
            "third_dataset_acc": extras["third_dataset_acc"],
        }
        rows.append(row)
        # persist
        with open(os.path.join(out_dir, f"{name}.recipe.json"), "w", encoding="utf-8") as f:
            json.dump(recipe, f, ensure_ascii=False, indent=2)
        # hygiene
        del model, pets_loader, stl_loader; gc.collect()
        try: torch.cuda.empty_cache()
        except Exception: pass
    return pd.DataFrame(rows)

# ===== Config builder (kept INSIDE this file as requested) =====
def build_online_recipes(epochs_phaseA=10, epochs_phaseB=1) -> Dict[str, dict]:
    base = {
        "phaseA": {
            "data": {"root":"./data","batch_size":128,"num_workers":0,"img_size":224,"val_ratio":0.10,"pin_memory":False},
            "model":{"pretrained":True,"freeze_backbone":False,"dropout_p":0.2},
            "opt":{"lr":3e-4,"wd":1e-4},
            "trainer":{"root":"./runs/phaseA","epochs":int(epochs_phaseA)},
            "adv_training":{"enabled":False}
        },
        "phaseB": { # online
            "root":"./data/oxford-iiit-pet","batch_size":64,"num_workers":0,"img_size":224,
            "split":"trainval","pin_memory":False,
            "epochs":int(epochs_phaseB),"lr":3e-5,"wd":0.0,"trainable":"classifier","last_n":0
        },
        "phaseC": {
            "root":"./data","batch_size":128,"num_workers":0,"img_size":224,"split":"test","pin_memory":False
        }
    }
    return {"baseline_online": base}

if __name__ == "__main__":
    recipes = build_online_recipes(epochs_phaseA=5, epochs_phaseB=1)
    df = run_online_recipes_and_summarize(recipes, out_dir="./online_results")
    print(df)
