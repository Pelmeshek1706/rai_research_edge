"""Lightning modules for online adversarial experiments."""
from __future__ import annotations

from typing import Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights

from .attacks import _build_attack, _to01, _tonorm
from .data import IMAGENET_MEAN, IMAGENET_STD


class LitBinaryClassifier(L.LightningModule):
    """MobileNetV3-Small binary classifier with optional online attacks."""

    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_p: float = 0.2,
        adv_cfg: Optional[dict] = None,
        phaseA_epochs: int = 0,
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
    ) -> None:
        """Set up the backbone, metrics, and attack options before training starts."""
        super().__init__()
        self.save_hyperparameters()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)
        in_features = backbone.classifier[3].in_features
        backbone.classifier[3] = nn.Linear(in_features, 2)
        if isinstance(backbone.classifier[2], nn.Dropout):
            backbone.classifier[2].p = float(dropout_p)
        if freeze_backbone:
            for p in backbone.features.parameters():
                p.requires_grad = False
        self.backbone = backbone

        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()

        self.register_buffer("mean_buf", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std_buf", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        self.adv_cfg = adv_cfg or {
            "enabled": False,
            "method": "pgd",
            "every_n": 4,
            "attack_kwargs": {},
        }
        self.phaseA_epochs = int(phaseA_epochs)

    def on_train_epoch_start(self):  # type: ignore[override]
        """Print which phase (A or B) we are entering at the start of each epoch."""
        if self.current_epoch == 0:
            self.print(
                f"[Phase A] Clean training for {self.phaseA_epochs} epochs (no torchattacks)."
            )
        if self.current_epoch == self.phaseA_epochs:
            max_epochs = getattr(self.trainer, "max_epochs", None)
            if max_epochs is not None:
                phaseB_epochs = max_epochs - self.phaseA_epochs
                self.print(
                    f"[Phase B] Online adversarial training starts for {phaseB_epochs} epochs."
                )
            else:
                self.print("[Phase B] Online adversarial training starts (torchattacks enabled).")

    def on_train_end(self):  # type: ignore[override]
        """Print that training is done and Phase C (final eval) starts."""
        self.print("[Phase C] Training finished. Proceeding to final evaluation.")

    def forward(self, x):
        """Return logits from the backbone for the given batch."""
        return self.backbone(x)

    def configure_optimizers(self):
        """Return AdamW optimizer built from the stored lr and weight decay."""
        params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    def _augment_with_torchattacks(self, x: torch.Tensor, y: torch.Tensor):
        """When Phase B is active, build an attack, craft extra examples, and add them to the batch."""
        if getattr(self, "current_epoch", 0) < self.phaseA_epochs:
            return x, y

        cfg = self.adv_cfg or {}
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

        attack = _build_attack(self, cfg, self.mean_buf, self.std_buf, device=x.device)
        x_sel = x[mask]
        y_sel = y[mask]
        x01 = _to01(x_sel, self.mean_buf, self.std_buf).clamp_(0.0, 1.0)

        was_training = self.training
        self.eval()
        adv01 = attack(x01, y_sel)
        if was_training:
            self.train()

        adv_norm = _tonorm(adv01.detach(), self.mean_buf, self.std_buf)
        x_aug = torch.cat([x, adv_norm], dim=0)
        y_aug = torch.cat([y, y_sel], dim=0)
        return x_aug, y_aug

    def training_step(self, batch, _):  # type: ignore[override]
        """Run a training batch, mixing in adversarial samples, then log loss and train accuracy."""
        x, y = batch
        x_aug, y_aug = self._augment_with_torchattacks(x, y)
        logits = self(x_aug)
        loss = self.criterion(logits, y_aug)
        preds = torch.argmax(logits.detach(), dim=1)
        self.train_acc.update(preds, y_aug)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):  # type: ignore[override]
        """Run a validation batch, log loss, and update val accuracy/F1."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):  # type: ignore[override]
        """Log val accuracy and F1, then reset metrics for the next epoch."""
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute(), prog_bar=False)
        self.val_acc.reset()
        self.val_f1.reset()

    def test_step(self, batch, _):  # type: ignore[override]
        """Run a test batch, log test loss, and update test metrics."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)
        self.log("test/loss", loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self):  # type: ignore[override]
        """Log test accuracy/F1 and reset metrics so future runs are clean."""
        self.log("test/acc", self.test_acc.compute(), prog_bar=False)
        self.log("test/f1", self.test_f1.compute(), prog_bar=False)
        self.test_acc.reset()
        self.test_f1.reset()

from typing import Optional, Tuple

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score

from .attacks import _build_attack, _to01, _tonorm
from .data import IMAGENET_MEAN, IMAGENET_STD
from .brevitas_model.MobileNetV3Brevitas import mobilenetv3_small_quantized


def apply_sn(module: nn.Module) -> nn.Module:
    """Apply spectral norm to a linear layer (helper)."""
    return nn.utils.spectral_norm(module)


class LitBinaryClassifierBrevitas(L.LightningModule):
    """Quantized MobileNetV3-Small binary classifier with optional online attacks and regularization."""

    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        pretrained: bool = True,          # kept for API symmetry; currently unused
        freeze_backbone: bool = True,
        dropout_p: float = 0.2,
        spectral_norm: bool = False,
        feature_noise_sigma: float = 0.0,
        label_smoothing: float = 0.1,
        consistency_lambda: float = 0.0,
        consistency_noise_sigma: float = 0.0,
        adv_cfg: Optional[dict] = None,
        phaseA_epochs: int = 0,
        mean: Tuple[float, float, float] = IMAGENET_MEAN,
        std: Tuple[float, float, float] = IMAGENET_STD,
        quant_ckpt_path: str = "./final_code/brevitas_model/trained_models/MobileNetV3Small_INT4_CIFAR-10.pth",
    ) -> None:
        """Set up quantized backbone, metrics, regularizations, and optional online attacks."""
        super().__init__()
        self.save_hyperparameters()

        # -------------------------
        # Backbone: Brevitas MobileNetV3-Small (quantized)
        # -------------------------
        mb = mobilenetv3_small_quantized()
        state = torch.load(quant_ckpt_path, map_location="cpu")
        mb.load_state_dict(state)

        in_features = mb.classifier[3].in_features
        # replace final layer for 2 classes
        mb.classifier[3] = nn.Linear(in_features, 2)

        # Dropout in the head
        if isinstance(mb.classifier[2], nn.Dropout):
            mb.classifier[2].p = float(dropout_p)
        else:
            mb.classifier = nn.Sequential(
                *list(mb.classifier[:-1]),
                nn.Dropout(dropout_p),
                nn.Linear(in_features, 2),
            )

        # spectral norm on final Linear
        if spectral_norm:
            mb.classifier[3] = apply_sn(mb.classifier[3])

        # optionally freeze features, train classifier only
        if freeze_backbone:
            for p in mb.features.parameters():
                p.requires_grad = False
            for p in mb.classifier.parameters():
                p.requires_grad = True

        self.backbone = mb

        # -------------------------
        # Losses & regularization
        # -------------------------
        self.feature_noise_sigma = float(feature_noise_sigma)
        self.consistency_lambda = float(consistency_lambda)
        self.consistency_noise_sigma = float(consistency_noise_sigma)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # -------------------------
        # Metrics (same pattern as upper model)
        # -------------------------
        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.val_f1 = BinaryF1Score()
        self.test_acc = BinaryAccuracy()
        self.test_f1 = BinaryF1Score()

        # -------------------------
        # Normalization buffers (for attacks)
        # -------------------------
        self.register_buffer("mean_buf", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std_buf", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # -------------------------
        # Online adversarial config
        # -------------------------
        self.adv_cfg = adv_cfg or {
            "enabled": False,
            "method": "pgd",
            "every_n": 4,
            "attack_kwargs": {},
        }
        self.phaseA_epochs = int(phaseA_epochs)

    # -------------------------------------------------------------------------
    # Phase A / B logging (same as upper)
    # -------------------------------------------------------------------------
    def on_train_epoch_start(self):  # type: ignore[override]
        """Print which phase (A or B) we are entering at the start of each epoch."""
        if self.current_epoch == 0:
            self.print(
                f"[Phase A] Clean training for {self.phaseA_epochs} epochs (no torchattacks)."
            )
        if self.current_epoch == self.phaseA_epochs:
            max_epochs = getattr(self.trainer, "max_epochs", None)
            if max_epochs is not None:
                phaseB_epochs = max_epochs - self.phaseA_epochs
                self.print(
                    f"[Phase B] Online adversarial training starts for {phaseB_epochs} epochs."
                )
            else:
                self.print("[Phase B] Online adversarial training starts (torchattacks enabled).")

    def on_train_end(self):  # type: ignore[override]
        """Print that training is done and Phase C (final eval) starts."""
        self.print("[Phase C] Training finished. Proceeding to final evaluation.")

    # -------------------------------------------------------------------------
    # Forward path with optional feature noise (for regularization)
    # -------------------------------------------------------------------------
    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened feature embeddings from the quantized backbone."""
        x = self.backbone.features(x)
        x = self.backbone.conv(x)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)

        # feature noise only during training
        if self.training and self.feature_noise_sigma > 0.0:
            x = x + torch.randn_like(x) * self.feature_noise_sigma
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits from the quantized backbone for the given batch."""
        feats = self._forward_features(x)
        return self.backbone.classifier(feats)

    # -------------------------------------------------------------------------
    # Optimizer (trainable params only, same pattern as upper)
    # -------------------------------------------------------------------------
    def configure_optimizers(self):
        """Return AdamW optimizer built from the stored lr and weight decay."""
        params = [p for p in self.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError(
                "No trainable parameters: all requires_grad=False. "
                "Set freeze_backbone=False or check freeze logic."
            )
        return torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)

    # -------------------------------------------------------------------------
    # Online adversarial augmentation (copied from upper + reused here)
    # -------------------------------------------------------------------------
    def _augment_with_torchattacks(self, x: torch.Tensor, y: torch.Tensor):
        """When Phase B is active, build an attack, craft extra examples, and add them to the batch."""
        if getattr(self, "current_epoch", 0) < self.phaseA_epochs:
            return x, y

        cfg = self.adv_cfg or {}
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

        attack = _build_attack(self, cfg, self.mean_buf, self.std_buf, device=x.device)
        x_sel = x[mask]
        y_sel = y[mask]
        x01 = _to01(x_sel, self.mean_buf, self.std_buf).clamp_(0.0, 1.0)

        was_training = self.training
        self.eval()
        adv01 = attack(x01, y_sel)
        if was_training:
            self.train()

        adv_norm = _tonorm(adv01.detach(), self.mean_buf, self.std_buf)
        x_aug = torch.cat([x, adv_norm], dim=0)
        y_aug = torch.cat([y, y_sel], dim=0)
        return x_aug, y_aug

    # -------------------------------------------------------------------------
    # Lightning steps (mirroring the upper class, + consistency regularization)
    # -------------------------------------------------------------------------
    def training_step(self, batch, _):  # type: ignore[override]
        """Run a training batch, mix in adversarial samples, then apply regularizations and log metrics."""
        x, y = batch  # y ∈ {0,1}

        # online adversarial augmentation (Phase B)
        x_aug, y_aug = self._augment_with_torchattacks(x, y)

        logits = self(x_aug)
        loss = self.criterion(logits, y_aug)

        # Consistency: KL( logits(x) || logits(x+ε) )
        if self.consistency_lambda > 0.0 and self.consistency_noise_sigma > 0.0:
            with torch.no_grad():
                x_eps = x_aug + torch.randn_like(x_aug) * self.consistency_noise_sigma
            logits_eps = self(x_eps)
            p = F.log_softmax(logits, dim=1)
            q = F.softmax(logits_eps, dim=1)
            kl = F.kl_div(p, q, reduction="batchmean")
            loss = loss + self.consistency_lambda * kl

        preds = torch.argmax(logits.detach(), dim=1)
        self.train_acc.update(preds, y_aug)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", self.train_acc, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, _):  # type: ignore[override]
        """Run a validation batch, log loss, and update val accuracy/F1."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, y)
        self.val_f1.update(preds, y)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):  # type: ignore[override]
        """Log val accuracy and F1, then reset metrics for the next epoch."""
        self.log("val/acc", self.val_acc.compute(), prog_bar=True)
        self.log("val/f1", self.val_f1.compute(), prog_bar=False)
        self.val_acc.reset()
        self.val_f1.reset()

    def test_step(self, batch, _):  # type: ignore[override]
        """Run a test batch, log test loss, and update test metrics."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc.update(preds, y)
        self.test_f1.update(preds, y)
        self.log("test/loss", loss, on_epoch=True)
        return loss

    def on_test_epoch_end(self):  # type: ignore[override]
        """Log test accuracy/F1 and reset metrics so future runs are clean."""
        self.log("test/acc", self.test_acc.compute(), prog_bar=False)
        self.log("test/f1", self.test_f1.compute(), prog_bar=False)
        self.test_acc.reset()
        self.test_f1.reset()

__all__ = ["LitBinaryClassifier", "LitBinaryClassifierBrevitas"]
