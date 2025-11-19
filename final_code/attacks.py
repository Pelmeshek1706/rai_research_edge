"\"\"\"Wrappers and helpers around torchattacks for online adversarial training.\"\"\""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torchattacks import APGD, DIFGSM, FGSM, MIFGSM, PGD, TIFGSM

from .data import IMAGENET_MEAN, IMAGENET_STD


class NormalizeWrapper(nn.Module):
    """Wrap the model so attacks stay in 0-1 space while the model sees normalized pixels."""

    def __init__(
        self,
        model: nn.Module,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        set_eval: bool = True,
    ):
        super().__init__()
        self.model = model.eval() if set_eval else model

        def _as_tensor(value):
            if isinstance(value, torch.Tensor):
                t = value.detach().clone()
            else:
                t = torch.tensor(value, dtype=torch.float32)
            return t.view(1, -1, 1, 1)

        self.register_buffer("mean", _as_tensor(mean))
        self.register_buffer("std", _as_tensor(std))

    def forward(self, x01):
        """Scale the 0-1 image to normalized space and run the wrapped model."""
        x = (x01 - self.mean) / self.std
        return self.model(x)


ATTACK_REGISTRY = {
    "fgsm": FGSM,
    "pgd": PGD,
    "mifgsm": MIFGSM,
    "difgsm": DIFGSM,
    "tifgsm": TIFGSM,
    "apgd": APGD,
}

ATTACK_DEFAULTS: Dict[str, dict] = {
    "fgsm": {"eps": 8 / 255},
    "pgd": {"eps": 8 / 255, "alpha": 2 / 255, "steps": 10, "random_start": True},
    "mifgsm": {"eps": 8 / 255, "alpha": 2 / 255, "steps": 10},
    "difgsm": {"eps": 8 / 255, "alpha": 2 / 255, "steps": 10},
    "tifgsm": {"eps": 8 / 255, "alpha": 2 / 255, "steps": 10},
    "apgd": {"norm": "Linf", "eps": 8 / 255, "steps": 10, "n_restarts": 1, "verbose": False},
}


def _to01(x_norm, mean, std):
    """Convert normalized inputs back to 0-1 range for torchattacks (used by `_augment_with_torchattacks`)."""
    return x_norm * std + mean


def _tonorm(x01, mean, std):
    """Re-normalize torchattack outputs so they can be concatenated with clean examples."""
    return (x01 - mean) / std


@torch.no_grad()
def _build_attack(
    model: nn.Module,
    attack_cfg: dict,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
) -> nn.Module:
    """Create the attack with default settings and the shared normalization wrapper."""
    method = str(attack_cfg.get("method", "pgd")).lower()
    if method not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown torchattack method: {method}")
    attack_cls = ATTACK_REGISTRY[method]
    defaults = {**ATTACK_DEFAULTS.get(method, {})}
    overrides = attack_cfg.get("attack_kwargs", {})
    defaults.update(overrides)
    wrapper = NormalizeWrapper(model, mean=mean, std=std, set_eval=False).to(device)
    attack = attack_cls(wrapper, **defaults)
    if hasattr(attack, "set_device"):
        attack.set_device(device)
    elif hasattr(attack, "to"):
        attack.to(device)
    return attack


__all__ = [
    "NormalizeWrapper",
    "ATTACK_REGISTRY",
    "ATTACK_DEFAULTS",
    "_to01",
    "_tonorm",
    "_build_attack",
]
