# online_experiments.py
# ============================================================
# CIFAR-10 (cat/dog) baseline + optional PGD-AT + grid verification
# + Online ("train-on-inference") adaptation on Oxford-IIIT Pets (binary)
# + Third-dataset generalization check on STL-10 (cat/dog)
# ============================================================

import os, math, random, json, gc, io
from pathlib import Path
from copy import deepcopy
from contextlib import nullcontext
from typing import Iterable, Tuple, Optional

# --- PyTorch / Lightning ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets, models
from torch.nn.utils import spectral_norm as apply_sn

import lightning as L
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
import pandas as pd

# -----------------------------
# Repro & system safety knobs
# -----------------------------
def set_seed(seed: int = 42):
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

# -----------------------------
# CIFAR-10 constants
# (common, widely used exact stats)
# -----------------------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)
CIFAR10_LABELS = {"cat": 3, "dog": 5}

# ============================================================
# DataModule: CIFAR-10 → binary classification cat vs dog
# ============================================================
class TwoClassDataModule(L.LightningDataModule):
    CIFAR10_LABELS = CIFAR10_LABELS
    CIFAR10_MEAN   = CIFAR10_MEAN
    CIFAR10_STD    = CIFAR10_STD

    def __init__(self,
                 workdir: str = "./workdir",
                 batch_size: int = 128,
                 num_workers: int = 0,
                 img_size: int = 224,
                 val_ratio_per_class: float = 0.1,
                 pin_memory: bool = False,
                 train_aug: dict | None = None):
        super().__init__()
        self.workdir = Path(workdir); self.workdir.mkdir(parents=True, exist_ok=True)
        self.batch_size, self.num_workers = batch_size, max(0, int(num_workers))
        self.img_size, self.val_ratio = img_size, float(val_ratio_per_class)
        self._pin_memory_cfg = bool(pin_memory)
        self._train_aug_cfg = train_aug or {}

        # ----- TRAIN transforms -----
        rrc_cfg = self._train_aug_cfg.get("random_resized_crop",
                                          {"enabled": True, "scale": [0.6, 1.0], "ratio": [0.75, 1.3333]})
        flip_p  = float(self._train_aug_cfg.get("horizontal_flip_p", 0.5))
        cj_cfg  = self._train_aug_cfg.get("color_jitter",
                                          {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02})

        train_ops = []
        if rrc_cfg.get("enabled", True):
            train_ops.append(transforms.RandomResizedCrop(
                self.img_size,
                scale=tuple(rrc_cfg.get("scale", [0.6, 1.0])),
                ratio=tuple(rrc_cfg.get("ratio", [0.75, 1.3333]))
            ))
        else:
            train_ops.extend([transforms.Resize(256), transforms.CenterCrop(self.img_size)])
        if flip_p > 0.0:
            train_ops.append(transforms.RandomHorizontalFlip(p=flip_p))
        if cj_cfg:
            train_ops.append(transforms.ColorJitter(**cj_cfg))
        train_ops.extend([transforms.ToTensor(), transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD)])
        self.tf_train = transforms.Compose(train_ops)

        # ----- EVAL transforms -----
        self.tf_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(self.CIFAR10_MEAN, self.CIFAR10_STD),
        ])

        self.train_ds = self.val_ds = self.test_ds = None
        self.class_names = ["cat", "dog"]

    def setup(self, stage=None):
        tr_base = datasets.CIFAR10(root=str(self.workdir/"cifar10"), train=True,  download=True, transform=None)
        te_base = datasets.CIFAR10(root=str(self.workdir/"cifar10"), train=False, download=True, transform=None)

        def idx_by_label(ds, label): return [i for i,(_,y) in enumerate(ds) if y==label]
        cat_idx = idx_by_label(tr_base, self.CIFAR10_LABELS["cat"])
        dog_idx = idx_by_label(tr_base, self.CIFAR10_LABELS["dog"])

        def split_idx(idxs, ratio):
            idxs = idxs[:]; random.shuffle(idxs)
            k = max(1, int(len(idxs)*ratio))
            return idxs[k:], idxs[:k]

        cat_tr, cat_val = split_idx(cat_idx, self.val_ratio)
        dog_tr, dog_val = split_idx(dog_idx, self.val_ratio)

        def remap(y): return 0 if y==self.CIFAR10_LABELS["cat"] else 1

        class _SubsetRemap(Dataset):
            def __init__(self, base, indices, remap, tf=None):
                self.base, self.indices, self.remap, self.tf = base, indices, remap, tf
            def __len__(self): return len(self.indices)
            def __getitem__(self, i):
                x, y = self.base[self.indices[i]]  # PIL.Image
                if self.tf is not None: x = self.tf(x)
                return x, self.remap(y)

        self.train_ds = _SubsetRemap(tr_base, cat_tr+dog_tr, remap, tf=self.tf_train)
        self.val_ds   = _SubsetRemap(tr_base, cat_val+dog_val, remap, tf=self.tf_eval)

        keep = {self.CIFAR10_LABELS["cat"], self.CIFAR10_LABELS["dog"]}
        te_idx = [i for i,(_,y) in enumerate(te_base) if y in keep]
        self.test_ds  = _SubsetRemap(te_base, te_idx, remap, tf=self.tf_eval)

    def _dl_common_args(self, shuffle: bool):
        args = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=bool(self._pin_memory_cfg),
            persistent_workers=False,
        )
        if self.num_workers > 0:
            args["prefetch_factor"] = 2
        return args

    def train_dataloader(self):
        return DataLoader(self.train_ds, **self._dl_common_args(shuffle=True))
    def val_dataloader(self):
        return DataLoader(self.val_ds,   **self._dl_common_args(shuffle=False))
    def test_dataloader(self):
        return DataLoader(self.test_ds,  **self._dl_common_args(shuffle=False))

# ============================================================
# Oxford-IIIT Pets → Binary stream (cat=0, dog=1)
# (robust to folder layout; uses annotations/list.txt + split files)
# ============================================================
class PetsBinaryDataset(Dataset):
    """
    Builds a binary (cat=0, dog=1) dataset from Oxford-IIIT Pet:
    - Expects either:
      root/oxford-iiit-pet/{images,annotations}
      or
      root/{images,annotations}
    - Uses annotations/list.txt to map filename -> species (1=Cat, 2=Dog)
    - Uses annotations/{trainval.txt|test.txt} to build splits
    """
    def __init__(self, root: str, split: str = "trainval", transform=None):
        super().__init__()
        assert split in {"trainval", "test"}
        self.transform = transform

        base = Path(root)
        if (base/"oxford-iiit-pet").exists():
            base = base/"oxford-iiit-pet"
        images_dir = base/"images"
        ann_dir    = base/"annotations"
        if not images_dir.exists() or not ann_dir.exists():
            raise FileNotFoundError(f"Expect images/ and annotations/ under {base}")

        # Parse species from list.txt
        list_txt = ann_dir/"list.txt"
        if not list_txt.exists():
            # some mirrors place list.txt inside annotations/ (already)
            raise FileNotFoundError(f"Missing {list_txt}")
        species_by_name = {}
        with open(list_txt, "r", encoding="utf-8") as f:
            for line in f:
                # Lines look like: "<name> <class_id> <species> <breed_id>"
                if line.startswith("#") or len(line.strip().split()) < 2:
                    continue
                toks = line.strip().split()
                name = toks[0]  # no extension
                if len(toks) >= 3:
                    try:
                        species = int(toks[2])  # 1=Cat, 2=Dog
                    except Exception:
                        continue
                    species_by_name[name] = 0 if species == 1 else 1  # 0=cat,1=dog

        split_file = ann_dir/(f"{split}.txt")
        if not split_file.exists():
            raise FileNotFoundError(f"Missing split file: {split_file}")

        samples = []
        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                # some files have "<name> <id>" format; take first token
                name = line.strip().split()[0]
                label = species_by_name.get(name, None)
                if label is None:
                    # try fallback: strip suffixes
                    label = species_by_name.get(Path(name).stem, None)
                if label is None:
                    continue
                img_path = images_dir / f"{name}.jpg"
                if not img_path.exists():
                    # some splits already contain ".jpg"
                    cand = images_dir / name
                    if cand.exists(): img_path = cand
                    else: continue
                samples.append((str(img_path), int(label)))

        if not samples:
            raise RuntimeError("No samples built for PetsBinaryDataset. Check folder layout & lists.")
        self._samples = samples

    def __len__(self): return len(self._samples)
    def __getitem__(self, i):
        from PIL import Image
        p, y = self._samples[i]
        x = Image.open(p).convert("RGB")
        if self.transform is not None:
            x = self.transform(x)
        return x, y

class OxfordPetsBinaryDataModule(L.LightningDataModule):
    def __init__(self, root="./workdir/pets", batch_size=64, num_workers=0, img_size=224, split_stream="trainval", pin_memory=False):
        super().__init__()
        self.root = root; self.bs = batch_size; self.nw = int(num_workers)
        self.img_size = img_size; self.split_stream = split_stream; self.pin = bool(pin_memory)
        self.tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(self.img_size),
            transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        self._stream = None

    def setup(self, stage=None):
        self._stream = PetsBinaryDataset(self.root, split=self.split_stream, transform=self.tf)

    def stream_dataloader(self):
        return DataLoader(self._stream, batch_size=self.bs, shuffle=True,
                          num_workers=self.nw, pin_memory=self.pin, persistent_workers=False)

# ============================================================
# STL-10 (third dataset) → keep only {cat,dog} and remap to {0,1}
# ============================================================
class STL10CatsDogsDataModule(L.LightningDataModule):
    def __init__(self, root="./workdir/stl10", batch_size=128, num_workers=0, img_size=224, split="test", pin_memory=False):
        super().__init__()
        self.root = root; self.bs = batch_size; self.nw = int(num_workers)
        self.img_size = img_size; self.split = split; self.pin = bool(pin_memory)
        self.tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(self.img_size),
            transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        self._eval = None

    def setup(self, stage=None):
        base = datasets.STL10(self.root, split=self.split, transform=self.tf, download=True)
        # map names to ids and keep cat,dog
        name_to_id = {name: i for i, name in enumerate(getattr(base, "classes", []))}
        cat_id = name_to_id.get("cat"); dog_id = name_to_id.get("dog")
        if cat_id is None or dog_id is None:
            raise RuntimeError("STL10 classes must include 'cat' and 'dog'.")
        keep = {cat_id, dog_id}

        class _SubsetRelabel(Dataset):
            def __init__(self, ds, keep_ids, cat_id, dog_id):
                self.ds = ds
                self.keep_idx = [i for i, (_, y) in enumerate(ds) if y in keep_ids]
                self.cat_id, self.dog_id = cat_id, dog_id
            def __len__(self): return len(self.keep_idx)
            def __getitem__(self, i):
                x, y = self.ds[self.keep_idx[i]]
                y_bin = 0 if y == self.cat_id else 1
                return x, y_bin

        self._eval = _SubsetRelabel(base, keep, cat_id, dog_id)

    def test_dataloader(self):
        return DataLoader(self._eval, batch_size=self.bs, shuffle=False,
                          num_workers=self.nw, pin_memory=self.pin, persistent_workers=False)

# ============================================================
# Utilities: optimizer groups (layer-decay)
# ============================================================
def _is_norm_module(m):
    return isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
                          nn.GroupNorm, nn.LayerNorm, nn.InstanceNorm1d,
                          nn.InstanceNorm2d, nn.InstanceNorm3d))

def parameter_groups(model, base_lr: float, weight_decay: float, layer_decay: float | None = None):
    no_decay, decay = [], []
    for name, module in model.named_modules():
        for pname, p in module.named_parameters(recurse=False):
            if not p.requires_grad: continue
            if pname.endswith("bias") or _is_norm_module(module):
                no_decay.append(p)
            else:
                decay.append(p)
    if layer_decay is None or not hasattr(model, "backbone"):
        return [
            {"params": decay, "lr": base_lr, "weight_decay": weight_decay},
            {"params": no_decay, "lr": base_lr, "weight_decay": 0.0},
        ]
    groups = []
    layers = list(model.backbone.features)
    depth = len(layers)
    for i, layer in enumerate(layers):
        lrd = layer_decay ** (depth - 1 - i)
        d_params = [p for p in layer.parameters() if p.requires_grad]
        if d_params:
            groups.append({"params": d_params, "lr": base_lr * lrd, "weight_decay": weight_decay})
    cls_decay, cls_nodecay = [], []
    for n, p in model.backbone.classifier.named_parameters():
        if not p.requires_grad: continue
        (cls_nodecay if n.endswith("bias") else cls_decay).append(p)
    if cls_decay:
        groups.append({"params": cls_decay, "lr": base_lr, "weight_decay": weight_decay})
    if cls_nodecay:
        groups.append({"params": cls_nodecay, "lr": base_lr, "weight_decay": 0.0})
    return groups

def build_optimizer(model, lr: float = 3e-4, weight_decay: float = 1e-4, layer_decay: float | None = None):
    groups = parameter_groups(model, lr, weight_decay, layer_decay)
    from torch.optim import AdamW
    return AdamW(groups, lr=lr, weight_decay=0.0)  # WD set in groups

# ============================================================
# PGD attack in [0,1] with per-channel mean/std
# ============================================================
def _to01(x_norm, mean, std):
    return x_norm * std + mean

def _tonorm(x01, mean, std):
    return (x01 - mean) / std

@torch.enable_grad()
def linf_pgd_attack(model: nn.Module,
                    x_norm: torch.Tensor,
                    y: torch.Tensor,
                    mean: torch.Tensor,
                    std: torch.Tensor,
                    eps: float = 8/255,
                    alpha: float = 2/255,
                    steps: int = 10,
                    random_start: bool = True) -> torch.Tensor:
    x01 = _to01(x_norm, mean, std).clamp_(0.0, 1.0)
    if random_start:
        x_adv01 = (x01 + torch.empty_like(x01).uniform_(-eps, eps)).clamp_(0.0, 1.0).detach()
    else:
        x_adv01 = x01.clone().detach()
    for _ in range(steps):
        x_adv01.requires_grad_(True)
        logits = model(_tonorm(x_adv01, mean, std))
        loss = F.cross_entropy(logits, y)
        (grad,) = torch.autograd.grad(loss, x_adv01, retain_graph=False, create_graph=False, only_inputs=True)
        x_adv01 = x_adv01.detach() + alpha * grad.detach().sign()
        delta   = torch.clamp(x_adv01 - x01, min=-eps, max=eps)
        x_adv01 = (x01 + delta).clamp_(0.0, 1.0)
    return _tonorm(x_adv01.detach(), mean, std)

# ============================================================
# Optional tensor-level MixUp/CutMix
# ============================================================
class BatchAugmentor:
    def __init__(self, mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0, p: float = 0.3,
                 mode_prior: tuple[float,float] = (0.5, 0.5), apply_on_train_only: bool = True):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.p = p
        self.mode_prior = mode_prior
        self.apply_on_train_only = apply_on_train_only

    @staticmethod
    def _rand_bbox(W, H, lam):
        cut_rat = math.sqrt(1. - lam)
        cut_w = int(W * cut_rat); cut_h = int(H * cut_rat)
        cx = random.randint(0, W); cy = random.randint(0, H)
        x1 = max(cx - cut_w // 2, 0); y1 = max(cy - cut_w // 2, 0)
        x2 = min(cx + cut_w // 2, W); y2 = min(cy + cut_h // 2, H)
        return x1, y1, x2, y2

    def _mixup(self, x, y):
        lam = torch.distributions.Beta(self.mixup_alpha, self.mixup_alpha).sample().item()
        idx = torch.randperm(x.size(0), device=x.device)
        x_mix = lam * x + (1 - lam) * x[idx]
        return x_mix, y, y[idx], lam

    def _cutmix(self, x, y):
        lam = torch.distributions.Beta(self.cutmix_alpha, self.cutmix_alpha).sample().item()
        idx = torch.randperm(x.size(0), device=x.device)
        B, C, H, W = x.size()
        x1, y1, x2, y2 = self._rand_bbox(W, H, lam)
        x_aug = x.clone(); x_aug[:, :, y1:y2, x1:x2] = x[idx][:, :, y1:y2, x1:x2]
        lam_adj = 1.0 - ((x2 - x1) * (y2 - y1) / (W * H))
        return x_aug, y, y[idx], lam_adj

    def __call__(self, x, y, *, training: bool = True):
        if self.apply_on_train_only and not training:
            return x, y, None, None, "none"
        if random.random() > self.p:
            return x, y, None, None, "none"
        if random.random() < self.mode_prior[0]:
            x_aug, y1, y2, lam = self._mixup(x, y); mode = "mixup"
        else:
            x_aug, y1, y2, lam = self._cutmix(x, y); mode = "cutmix"
        return x_aug, y1, y2, lam, mode

def mixup_criterion(ce_loss_fn, logits, y, y_perm=None, lam: float | None = None):
    if y_perm is None or lam is None:
        return ce_loss_fn(logits, y)
    return lam * ce_loss_fn(logits, y) + (1.0 - lam) * ce_loss_fn(logits, y_perm)

# ============================================================
# (Optional) FIX for double-backward if you enable Jacobian reg
# ============================================================
def replace_activations_for_double_backward(model: nn.Module):
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Hardswish):
                setattr(parent, name, nn.SiLU(inplace=getattr(child, "inplace", False)))
            elif isinstance(child, nn.Hardsigmoid):
                setattr(parent, name, nn.Sigmoid())
    return model

# ============================================================
# Model: MobileNetV3-Small → 2 classes (+ optional PGD-AT)
# ============================================================
class LitBinaryClassifier(L.LightningModule):
    def __init__(self,
                 lr: float = 3e-4,
                 weight_decay: float = 1e-4,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 dropout_p: float = 0.25,
                 spectral_norm: bool = True,
                 label_smoothing: float = 0.0,
                 use_custom_optimizer: bool = True,
                 layer_decay: float | None = 0.75,
                 mean: Tuple[float,float,float] = CIFAR10_MEAN,
                 std: Tuple[float,float,float] = CIFAR10_STD,
                 adv_training: Optional[dict] = None,
                 augment_cfg: Optional[dict] = None):
        super().__init__()
        self.save_hyperparameters()

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        mb = models.mobilenet_v3_small(weights=weights)
        in_features = mb.classifier[3].in_features
        mb.classifier[3] = nn.Linear(in_features, 2)
        if isinstance(mb.classifier[2], nn.Dropout):
            mb.classifier[2].p = dropout_p
        else:
            mb.classifier = nn.Sequential(*list(mb.classifier[:-1]), nn.Dropout(dropout_p), nn.Linear(in_features, 2))
        if spectral_norm:
            mb.classifier[3] = apply_sn(mb.classifier[3])

        if freeze_backbone:
            for p in mb.features.parameters(): p.requires_grad = False
            for p in mb.classifier.parameters(): p.requires_grad = True

        self.backbone = mb
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.use_custom_optimizer = use_custom_optimizer
        self.layer_decay = layer_decay

        self.register_buffer("mean_buf", torch.tensor(mean).view(1,3,1,1), persistent=False)
        self.register_buffer("std_buf",  torch.tensor(std).view(1,3,1,1),  persistent=False)

        self.batch_aug = None
        if augment_cfg is not None and (augment_cfg.get("mixup") or augment_cfg.get("cutmix")):
            self.batch_aug = BatchAugmentor(
                mixup_alpha=augment_cfg.get("mixup_alpha", 0.2),
                cutmix_alpha=augment_cfg.get("cutmix_alpha", 1.0),
                p=augment_cfg.get("p", 0.3),
                mode_prior=augment_cfg.get("mode_prior", (0.5, 0.5)),
                apply_on_train_only=True
            )

        self.adv_cfg = adv_training or {
            "enabled": False,
            "eps": 8/255, "alpha": 2/255, "steps": 10, "random_start": True,
            "mix_clean_adv": 0.0,
            "eval_steps": 5
        }

        # metrics
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy(); self.val_f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy(); self.test_f1 = BinaryF1Score()

    def _forward_features(self, x):
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        feats = self._forward_features(x)
        return self.backbone.classifier(feats)

    def configure_optimizers(self):
        trainable = [p for p in self.parameters() if p.requires_grad]
        if len(trainable) == 0:
            raise RuntimeError("No trainable parameters — check freeze_backbone.")
        if self.use_custom_optimizer:
            return build_optimizer(self, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay, layer_decay=self.layer_decay)
        else:
            return torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def _autocast_disabled(self):
        device = getattr(self, "device", torch.device("cpu"))
        device_type = device.type
        if device_type == "cuda" and torch.cuda.is_available():
            return torch.cuda.amp.autocast(enabled=False)
        if device_type == "cpu" and hasattr(torch, "cpu") and hasattr(torch.cpu, "amp") and hasattr(torch.cpu.amp, "autocast"):
            return torch.cpu.amp.autocast(enabled=False)
        return nullcontext()

    def _maybe_pgd_adv(self, x_norm, y):
        cfg = self.adv_cfg
        if not cfg.get("enabled", False):
            return None
        was_training = self.training
        self.eval()
        with self._autocast_disabled():
            x_adv = linf_pgd_attack(
                model=self, x_norm=x_norm.float(), y=y,
                mean=self.mean_buf, std=self.std_buf,
                eps=cfg.get("eps", 8/255), alpha=cfg.get("alpha", 2/255),
                steps=cfg.get("steps", 10), random_start=cfg.get("random_start", True)
            )
        if was_training:
            self.train()
        return x_adv.detach()

    def training_step(self, batch, batch_idx):
        x_norm, y = batch
        x_adv_norm = self._maybe_pgd_adv(x_norm, y)
        if self.batch_aug is not None:
            x_aug, y1, y2, lam, _ = self.batch_aug(x_norm, y, training=True)
        else:
            x_aug, y1, y2, lam = x_norm, y, None, None

        logits_clean = self(x_aug)
        loss_clean = mixup_criterion(self.criterion, logits_clean, y1, y2, lam)
        loss = loss_clean

        if x_adv_norm is not None:
            logits_adv = self(x_adv_norm)
            loss_adv = self.criterion(logits_adv, y)
            mix = float(self.adv_cfg.get("mix_clean_adv", 0.0))
            loss = mix * loss_clean + (1.0 - mix) * loss_adv

        preds = torch.argmax(logits_clean.detach(), dim=1)
        self.train_acc.update(preds, y)
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train/acc",  self.train_acc, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y); self.val_f1.update(preds, y)
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.log("val/f1",  self.val_f1.compute(),  prog_bar=False)
        self.val_acc.reset(); self.val_f1.reset()

        if not self.adv_cfg.get("enabled", False):
            return

        trainer = getattr(self, "trainer", None)
        if trainer is None or trainer.sanity_checking:
            return
        loaders = getattr(trainer, "val_dataloaders", None)
        val_loader = loaders[0] if isinstance(loaders, (list, tuple)) else loaders
        if val_loader is None: return

        was_training = self.training
        self.eval()
        total = 0; correct = 0; device = self.device

        with torch.enable_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch[:2]
                else:
                    x, y = batch
                x = x.to(device).float(); y = y.to(device)
                with self._autocast_disabled():
                    x_adv = linf_pgd_attack(
                        model=self, x_norm=x, y=y,
                        mean=self.mean_buf, std=self.std_buf,
                        eps=self.adv_cfg.get("eval_eps", 8/255),
                        alpha=self.adv_cfg.get("eval_alpha", 2/255),
                        steps=self.adv_cfg.get("eval_steps", 5),
                        random_start=True,
                    )
                logits = self(x_adv)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y).sum().item()
                total   += y.numel()
        if was_training:
            self.train()
        if total > 0:
            self.log("val/pgd5_acc", correct / total, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y); self.test_f1.update(preds, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def on_test_epoch_end(self):
        self.log("test/acc", self.test_acc.compute(), prog_bar=False)
        self.log("test/f1",  self.test_f1.compute(),  prog_bar=False)
        self.test_acc.reset(); self.test_f1.reset()

# ============================================================
# Temp scaling (optional) + noise-averaged inference
# ============================================================
class TemperatureScaler(nn.Module):
    def __init__(self): super().__init__(); self.log_T = nn.Parameter(torch.zeros(()))
    @property
    def T(self): return self.log_T.exp()
    def forward(self, logits): return logits / self.T
    @torch.no_grad()
    def fit(self, model, val_loader, device=None):
        device = device or next(model.parameters()).device
        self.to(device)
        opt = torch.optim.LBFGS([self.log_T], lr=0.01, max_iter=50, line_search_fn="strong_wolfe")
        nll = nn.CrossEntropyLoss()
        def _eval_loss():
            total, count = 0.0, 0
            for x,y in val_loader:
                x,y = x.to(device), y.to(device)
                logits = model(x)
                total += float(nll(self.forward(logits), y).item()) * x.size(0)
                count += x.size(0)
            return total / max(count,1)
        def closure():
            opt.zero_grad(set_to_none=True)
            loss = torch.tensor(_eval_loss(), device=device, requires_grad=True)
            loss.backward(); return loss
        opt.step(closure); return float(self.T.item())

@torch.no_grad()
def predict_with_noise_avg(model, x, n_samples: int = 8, sigma: float = 0.05):
    device = next(model.parameters()).device
    x = x.to(device)
    logits_sum = None
    for _ in range(n_samples):
        eps = torch.randn_like(x) * sigma
        logits = model(x + eps)
        logits_sum = logits if logits_sum is None else (logits_sum + logits)
    return logits_sum / float(n_samples)

# ============================================================
# Callback: collects per-epoch metrics + memory
# ============================================================
try:
    import psutil
    _ps = psutil.Process()
except Exception:
    _ps = None

class CollectMetricsCallback(L.Callback):
    def __init__(self):
        super().__init__()
        self.history = []
        self._mem_samples = []; self._epoch_rec = None

    def on_sanity_check_start(self, trainer, pl_module):
        self._epoch_rec = None; self._mem_samples = []

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_rec = {"epoch": int(trainer.current_epoch)}
        self._mem_samples = []
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def _sample_mem(self):
        if torch.cuda.is_available():
            return int(torch.cuda.memory_allocated())
        elif _ps is not None:
            return int(_ps.memory_info().rss)
        else:
            return 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._mem_samples.append(self._sample_mem())
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._mem_samples.append(self._sample_mem())

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        if self._epoch_rec is None:
            self._epoch_rec = {"epoch": int(trainer.current_epoch)}
        for k, v in trainer.callback_metrics.items():
            if isinstance(k, str) and (k.startswith("train/") or k.startswith("val/")):
                if isinstance(v, torch.Tensor): v = float(v.detach().cpu().item())
                else:
                    try: v = float(v)
                    except: continue
                self._epoch_rec[k] = v
        if self._mem_samples:
            self._epoch_rec["mem/min"] = int(min(self._mem_samples))
            self._epoch_rec["mem/avg"] = int(sum(self._mem_samples)/len(self._mem_samples))
            self._epoch_rec["mem/max"] = int(max(self._mem_samples))
        if torch.cuda.is_available():
            self._epoch_rec["mem/peak"] = int(torch.cuda.max_memory_allocated())
        self.history.append(self._epoch_rec)
        self._epoch_rec = None; self._mem_samples = []

    def on_test_epoch_end(self, trainer, pl_module):
        rec = {"epoch": "test"}
        for k, v in trainer.callback_metrics.items():
            if isinstance(k, str) and k.startswith("test/"):
                if isinstance(v, torch.Tensor): v = float(v.detach().cpu().item())
                else:
                    try: v = float(v)
                    except: continue
                rec[k] = v
        self.history.append(rec)

def get_epoch_metrics_df(metrics_cb: CollectMetricsCallback):
    if not metrics_cb.history:
        raise RuntimeError("Empty metrics history.")
    return pd.DataFrame(metrics_cb.history)

# ============================================================
# Trainer factory (allows inference_mode=False for PGD eval)
# ============================================================
def make_trainer(
    root_dir: str = "./runs",
    epochs: int = 40,
    callbacks=None,
    precision_override: str | None = None,
    num_sanity_val_steps: int = 0,
    inference_mode: bool = False,
):
    torch.set_float32_matmul_precision("medium")
    use_cuda = torch.cuda.is_available()
    if precision_override is None:
        bf16_ok = use_cuda and torch.cuda.is_bf16_supported()
        precision = "bf16-mixed" if bf16_ok else ("16-mixed" if use_cuda else "32-true")
    else:
        precision = precision_override
    try:
        return L.Trainer(
            accelerator="gpu" if use_cuda else "cpu",
            devices=1,
            precision=precision,
            max_epochs=epochs,
            log_every_n_steps=10,
            default_root_dir=root_dir,
            callbacks=callbacks or [],
            num_sanity_val_steps=num_sanity_val_steps,
            inference_mode=inference_mode,
        )
    except TypeError:
        return L.Trainer(
            accelerator="gpu" if use_cuda else "cpu",
            devices=1,
            precision=precision,
            max_epochs=epochs,
            log_every_n_steps=10,
            default_root_dir=root_dir,
            callbacks=callbacks or [],
            num_sanity_val_steps=num_sanity_val_steps,
        )

# ============================================================
# Simple eval + Online supervised adaptation
# ============================================================
@torch.no_grad()
def evaluate_simple(model: nn.Module, loader: DataLoader) -> float:
    device = next(model.parameters()).device
    model.eval()
    tot = 0; correct = 0
    for x, y in loader:
        x, y = x.to(device), torch.as_tensor(y, device=device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item(); tot += y.numel()
    return correct / max(1, tot)

def _set_trainable_layers_for_adapt(model: nn.Module, mode: str = "classifier", last_n: int = 0):
    for p in model.parameters(): p.requires_grad = False
    if mode == "classifier":
        for p in model.backbone.classifier.parameters(): p.requires_grad = True
    elif mode == "last_n":
        assert isinstance(last_n, int) and last_n > 0
        layers = list(model.backbone.features)
        for layer in layers[-last_n:]:
            for p in layer.parameters(): p.requires_grad = True
        for p in model.backbone.classifier.parameters(): p.requires_grad = True
    else:
        for p in model.parameters(): p.requires_grad = True

# def online_adapt_supervised(model: nn.Module,
#                             stream_loader: DataLoader,
#                             epochs: int = 1,
#                             lr: float = 3e-5,
#                             weight_decay: float = 0.0,
#                             trainable: str = "classifier",
#                             last_n: int = 0,
#                             max_steps: int | None = None) -> dict:
#     device = next(model.parameters()).device
#     _set_trainable_layers_for_adapt(model, trainable, last_n)
#     optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
#                               lr=lr, weight_decay=weight_decay)
#     ce = nn.CrossEntropyLoss()
#     model.train()
#     seen = 0; steps = 0; losses = []
#     for ep in range(max(1, int(epochs))):
#         for x, y in stream_loader:
#             x, y = x.to(device), torch.as_tensor(y, device=device)
#             optim.zero_grad(set_to_none=True)
#             logits = model(x)
#             loss = ce(logits, y)
#             loss.backward()
#             optim.step()
#             steps += 1; seen += y.numel(); losses.append(float(loss.detach().cpu()))
#             if max_steps and steps >= max_steps:
#                 break
#         if max_steps and steps >= max_steps:
#             break
#     model.eval()
#     return {"steps": steps, "samples_seen": seen, "loss_avg": (sum(losses) / max(1, len(losses)))}
def online_adapt_supervised(
    model: nn.Module,
    stream_loader: torch.utils.data.DataLoader,
    epochs: int = 1,                    
    lr: float = 3e-5,
    weight_decay: float = 0.0,
    trainable: str = "classifier",      # "classifier" | "last_n" | "all"
    last_n: int = 0,                    
    max_steps: int | None = None,       
    limit_batches_per_epoch: int | None = None 
) -> dict:
    device = next(model.parameters()).device
    _set_trainable_layers_for_adapt(model, trainable, last_n)

    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                              lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    model.train()

    seen = 0
    steps = 0
    losses = []

    for ep in range(max(1, int(epochs))):
        print(f"[ONLINE] epoch {ep+1}/{epochs}")
        for b_idx, (x, y) in enumerate(stream_loader):
            if limit_batches_per_epoch is not None and b_idx >= int(limit_batches_per_epoch):
                break
            if max_steps is not None and steps >= int(max_steps):
                break

            x, y = x.to(device), torch.as_tensor(y, device=device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            optim.step()

            steps += 1
            seen  += y.numel()
            losses.append(float(loss.detach().cpu()))

        if max_steps is not None and steps >= int(max_steps):
            break

    model.eval()
    return {"steps": steps, "samples_seen": seen, "loss_avg": (sum(losses) / max(1, len(losses)))}


# ============================================================
# Experiment runner (returns df and extras dict)
# ============================================================
def run_experiment(recipe: dict, force_fp32_for_jacobian: bool = True):
    # --- Data ---
    dcfg = recipe.get("data", {})
    dm = TwoClassDataModule(
        workdir=dcfg.get("workdir","./workdir"),
        batch_size=dcfg.get("batch_size",128),
        num_workers=dcfg.get("num_workers",0),
        img_size=dcfg.get("img_size",224),
        val_ratio_per_class=dcfg.get("val_ratio",0.1),
        pin_memory=dcfg.get("pin_memory", False),
        train_aug=dcfg.get("train_aug", None)
    )
    dm.setup()

    # --- Model ---
    mcfg   = recipe.get("model", {})
    ocfg   = recipe.get("optimizer", {})
    acfg   = recipe.get("augment", {})
    advcfg = recipe.get("adv_training", {})

    model = LitBinaryClassifier(
        lr=ocfg.get("lr",3e-4),
        weight_decay=ocfg.get("wd",1e-4),
        pretrained=mcfg.get("pretrained",True),
        freeze_backbone=mcfg.get("freeze_backbone",False),
        dropout_p=mcfg.get("dropout_p",0.25),
        spectral_norm=mcfg.get("spectral_norm",True),
        label_smoothing=mcfg.get("label_smoothing",0.0),
        use_custom_optimizer=ocfg.get("use_custom",True),
        layer_decay=ocfg.get("layer_decay",0.75),
        mean=CIFAR10_MEAN, std=CIFAR10_STD,
        adv_training=advcfg,
        augment_cfg=acfg
    )

    # --- Jacobian reg FIX (if enabled) ---
    rcfg = recipe.get("regularizers", {}).get("jacobian", {})
    jac_enabled = rcfg.get("enabled", False)
    if jac_enabled:
        replace_activations_for_double_backward(model)

    # --- Callbacks (collect metrics) ---
    metrics_cb = CollectMetricsCallback()
    cbs = [metrics_cb]

    # --- Trainer ---
    tcfg = recipe.get("trainer", {})
    adv_enabled = advcfg.get("enabled", False)
    precision_override = "32-true" if (adv_enabled or (jac_enabled and force_fp32_for_jacobian)) else None
    trainer = make_trainer(root_dir=tcfg.get("root_dir","./runs/pgd_at"),
                           epochs=tcfg.get("epochs",40),
                           callbacks=cbs,
                           precision_override=precision_override,
                           num_sanity_val_steps=tcfg.get("num_sanity_val_steps", 0))

    # --- Fit/Validate/Test ---
    trainer.fit(model, datamodule=dm)
    trainer.validate(model, datamodule=dm)
    trainer.test(model, datamodule=dm)

    # --- Optional post-hoc calibration ---
    evalcfg = recipe.get("eval", {})
    extras = {}
    if evalcfg.get("temp_scaling", False):
        T = TemperatureScaler(); T.fit(model, dm.val_dataloader())
        extras["temperature_T"] = T.T.item()

    # --- Optional noise-avg (test-time) ---
    if evalcfg.get("noise_avg", {}).get("enabled", False):
        n = evalcfg["noise_avg"].get("n", 8); sigma = evalcfg["noise_avg"].get("sigma", 0.05)
        device = next(model.parameters()).device
        correct=0; total=0
        with torch.no_grad():
            for x,y in dm.test_dataloader():
                x,y = x.to(device), y.to(device)
                logits = predict_with_noise_avg(model, x, n_samples=n, sigma=sigma)
                pred = logits.argmax(1)
                correct += (pred==y).sum().item(); total += y.numel()
        extras["test_acc_noise_avg"] = correct/total if total>0 else None

    # --- Online “train-on-inference” (optional) ---
    onlinecfg = recipe.get("online", {})
    if onlinecfg.get("enabled", False):
        pets = OxfordPetsBinaryDataModule(
            root=onlinecfg.get("root", "./workdir/pets"),
            batch_size=onlinecfg.get("batch_size", 64),
            num_workers=onlinecfg.get("num_workers", 0),
            img_size=dcfg.get("img_size", 224),
            split_stream=onlinecfg.get("split_stream", "trainval"),
            pin_memory=dcfg.get("pin_memory", False),
        )
        pets.setup()
        stream_loader = pets.stream_dataloader()

        pre_acc = evaluate_simple(model, stream_loader)

        adapt_stats = online_adapt_supervised(
            model, stream_loader,
            epochs=onlinecfg.get("epochs", 1),                   
            lr=onlinecfg.get("lr", 3e-5),
            weight_decay=onlinecfg.get("wd", 0.0),
            trainable=onlinecfg.get("trainable", "classifier"),
            last_n=onlinecfg.get("last_n", 0),
            max_steps=onlinecfg.get("max_steps", None),  
            limit_batches_per_epoch=onlinecfg.get("limit_batches_per_epoch", None),
        )

        post_acc = evaluate_simple(model, stream_loader)
        extras["online_pre_acc"]  = pre_acc
        extras["online_post_acc"] = post_acc
        extras["online_stats"]    = adapt_stats


    # --- Third-dataset eval (no updates) ---
    thirdcfg = recipe.get("third_eval", {"enabled": True})
    if thirdcfg.get("enabled", True):
        stl = STL10CatsDogsDataModule(
            root=thirdcfg.get("root", "./workdir/stl10"),
            batch_size=thirdcfg.get("batch_size", 128),
            num_workers=thirdcfg.get("num_workers", 0),
            img_size=dcfg.get("img_size", 224),
            split=thirdcfg.get("split", "test"),
            pin_memory=dcfg.get("pin_memory", False),
        )
        stl.setup()
        third_acc = evaluate_simple(model, stl.test_dataloader())
        extras["third_eval_acc"] = third_acc

    # --- RETURN df history + extras ---
    df = get_epoch_metrics_df(metrics_cb)

    del model; gc.collect()
    try: torch.cuda.empty_cache()
    except Exception: pass

    return df, extras

# ============================================================
# Helpers: deep_update / metrics extraction
# ============================================================
def deep_update(base: dict, patch: dict):
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base

def bytes_to_gib(x: int | float | None):
    if x is None: return None
    return float(x) / (1024 ** 3)

def extract_acc_and_mem(df: pd.DataFrame, extras: dict):
    acc = extras.get("test_acc_noise_avg", None)
    if acc is None:
        test_rows = df[df["epoch"] == "test"] if "epoch" in df.columns else pd.DataFrame()
        if not test_rows.empty and "test/acc" in test_rows.columns:
            acc = float(test_rows["test/acc"].iloc[0])
    if acc is None and "val/acc" in df.columns:
        mask_num = pd.to_numeric(df["epoch"], errors="coerce").notna()
        val_rows = df[mask_num]
        if not val_rows.empty:
            acc = float(val_rows["val/acc"].max())
    mem_peak = None
    if "mem/peak" in df.columns:
        epoch_mask = df["epoch"] != "test"
        mem_peak = pd.to_numeric(df.loc[epoch_mask, "mem/peak"], errors="coerce").max()
        mem_peak = None if pd.isna(mem_peak) else int(mem_peak)
    elif "mem/max" in df.columns:
        epoch_mask = df["epoch"] != "test"
        mem_peak = pd.to_numeric(df.loc[epoch_mask, "mem/max"], errors="coerce").max()
        mem_peak = None if pd.isna(mem_peak) else int(mem_peak)
    return acc, mem_peak

# ============================================================
# Base & grid recipes
# ============================================================
def build_base_recipe_for_search(epochs: int = 10) -> dict:
    return {
        "data": {
            "batch_size": 128, "img_size": 224, "num_workers": 0, "pin_memory": False,
            "workdir": "./workdir", "val_ratio": 0.10,
            "train_aug": {
                "random_resized_crop": {"enabled": True, "scale": [0.6, 1.0], "ratio": [0.75, 1.3333]},
                "horizontal_flip_p": 0.5,
                "color_jitter": {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.02}
            }
        },
        "model": {
            "pretrained": True, "freeze_backbone": True,
            "dropout_p": 0.2, "spectral_norm": False,
            "label_smoothing": 0.0
        },
        "optimizer": {"use_custom": True, "layer_decay": None, "wd": 1e-4, "lr": 3e-4},
        "augment": {"mixup": False, "cutmix": False, "p": 1.0, "mixup_alpha": 0.2, "cutmix_alpha": 1.0},
        "adv_training": { "enabled": False },
        "regularizers": {"jacobian": {"enabled": False, "lam": 1e-4, "subbatch": 8}},
        "callbacks": {"ema": False, "ema_decay": 0.999, "prune": {"enabled": False, "amount": 0.3, "start_epoch": 1}},
        "trainer": {"epochs": 1, "root_dir": "./runs/auto", "num_sanity_val_steps": 0},
        "eval": {"temp_scaling": False, "noise_avg": {"enabled": False, "n": 8, "sigma": 0.05}},
        # NEW:
        "online": {
            "enabled": True,
            "root": "/content/data/pets",        # <- Colab default from your download
            "split_stream": "trainval",
            "batch_size": 64,
            "num_workers": 0,
            "epochs": 1,
            "lr": 3e-5,
            "wd": 0.0,
            "trainable": "classifier",           # or "last_n" with "last_n": K
            "last_n": 0,
            "max_steps": 500,                    # cap for quick simulation
        },
        "third_eval": {
            "enabled": True,
            "root": "/content/data/stl10",       # <- Colab default from your download
            "split": "test",
            "batch_size": 128,
            "num_workers": 0
        },
    }

def method_patch(method: str) -> dict:
    patches = {
        "baseline":    {},
        "mixup":       {"augment": {"mixup": True,  "cutmix": False, "p": 1.0}},
        "cutmix":      {"augment": {"cutmix": True, "mixup": False,  "p": 1.0}},
        "jacobian":    {"regularizers": {"jacobian": {"enabled": True, "lam": 1e-4, "subbatch": 8}}},
        "ema":         {"callbacks": {"ema": True, "ema_decay": 0.999}},
        "prune":       {"callbacks": {"prune": {"enabled": True, "amount": 0.3, "start_epoch": 1}}},
        "temp":        {"eval": {"temp_scaling": True}},
        "noiseavg":    {"eval": {"noise_avg": {"enabled": True, "n": 8, "sigma": 0.05}}},
        "spectral":    {"model": {"spectral_norm": True}},
        "labelsmooth": {"model": {"label_smoothing": 0.1}},
        "layerdecay":  {"optimizer": {"layer_decay": 0.75}, "model": {"freeze_backbone": False}},
        "dropout":     {"model": {"dropout_p": 0.30}},
    }
    if method not in patches:
        raise ValueError(f"Unknown method: {method}")
    return patches[method]

def _grid_patches_for_method(method: str, variants_per_method: int = 5) -> list[dict]:
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
            {"optimizer": {"layer_decay": v}, "model": {"freeze_backbone": False}}
            for v in [0.6, 0.7, 0.75, 0.8, 0.85]
        ],
        "noiseavg": [
            {"eval": {"noise_avg": {"enabled": True, "n": n, "sigma": s}}}
            for (n, s) in [(4, 0.03), (8, 0.05), (8, 0.10), (16, 0.05), (16, 0.10)]
        ],
        "baseline": [ {} ],
    }
    grid = sweeps.get(method)
    if grid is None:
        raise ValueError(f"Unknown method: {method}")
    return grid[:max(1, int(variants_per_method))]

def build_base_recipes_for_methods(methods: list[str], base_recipe: dict | None = None, epochs: int = 10) -> dict[str, dict]:
    base = dict(base_recipe) if base_recipe is not None else build_base_recipe_for_search(epochs=epochs)
    out = {}
    for m in methods:
        rec = json.loads(json.dumps(base))  # deepcopy-safe
        deep_update(rec, method_patch(m))
        out[m] = rec
    return out

def build_hparam_grid_recipes(methods: list[str], base_recipe: dict | None = None, epochs: int = 10) -> dict[str, dict]:
    base = dict(base_recipe) if base_recipe is not None else build_base_recipe_for_search(epochs=epochs)
    out = {}
    for m in methods:
        grid = _grid_patches_for_method(m, variants_per_method=5)
        for i, hp_patch in enumerate(grid, 1):
            rec = deepcopy(base)
            deep_update(rec, method_patch(m))
            deep_update(rec, hp_patch)
            name = m if (m == "baseline" and len(grid) == 1) else f"{m}#v{i}"
            out[name] = rec
    return out

# ============================================================
# Run a set of recipes and summarize (config verification)
# ============================================================
def run_recipes_and_summarize(recipes: dict[str, dict],
                              run_fn,  # usually run_experiment
                              out_dir: str = "./grid_results",
                              save_per_run: bool = True,
                              save_summary: bool = True,
                              print_progress: bool = True) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      ['recipe', 'k', 'methods', 'accuracy', 'mem_peak_bytes', 'mem_peak_GiB', 'config']
    + saves CSV/JSON artifacts for each run and summary.csv.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, recipe in recipes.items():
        if print_progress:
            print(f"[run] {name}")
        df, extras = run_fn(recipe)

        if save_per_run:
            df.to_csv(os.path.join(out_dir, f"{name}.csv"), index=False)
            with open(os.path.join(out_dir, f"{name}.extras.json"), "w", encoding="utf-8") as f:
                json.dump(extras, f, ensure_ascii=False, indent=2)
            with open(os.path.join(out_dir, f"{name}.recipe.json"), "w", encoding="utf-8") as f:
                json.dump(recipe, f, ensure_ascii=False, indent=2)

        acc, mem_peak = extract_acc_and_mem(df, extras)
        rows.append({
            "recipe": name,
            "k": 1,
            "methods": [name],
            "accuracy": acc,
            "mem_peak_bytes": mem_peak,
            "mem_peak_GiB": bytes_to_gib(mem_peak),
            "config": recipe
        })

        del df; gc.collect()
        try: torch.cuda.empty_cache()
        except Exception: pass

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(by=["accuracy", "mem_peak_bytes"], ascending=[False, True]).reset_index(drop=True)
    if save_summary:
        summary.to_csv(os.path.join(out_dir, "summary.csv"), index=False)
    return summary



if __name__ == "__main__":
    methods = ['baseline','cutmix','mixup','labelsmooth','layerdecay']
    base = build_base_recipe_for_search(epochs=1)
    grid = build_hparam_grid_recipes(methods, base_recipe=base, epochs=1)
    df_summary = run_recipes_and_summarize(grid, run_experiment,
                                           out_dir="./grid_results_demo",
                                           save_per_run=True,
                                           save_summary=True,
                                           print_progress=True)
    df_summary.to_csv("./grid_results_demo/summary.csv", index=False)
# ============================================================