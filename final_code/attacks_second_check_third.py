# attacks_second_check_third.py
# ------------------------------------------------------------
# Uses training from File 1.
# After training + online adaptation:
#   - Run torchattacks on the SECOND dataset (Oxford-IIIT Pet, binary cat/dog) using RAW [0,1] loader.
#   - Check clean accuracy on the THIRD dataset (STL-10 cat/dog) without attacks.
# Returns a DataFrame per config with per-attack and clean metrics.
# ------------------------------------------------------------

from __future__ import annotations
import os, json, gc
from typing import Dict, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import pandas as pd
from torchvision import datasets, transforms

# import the training + loaders from File 1
from online_training_pipeline import (
    train_base_and_online, make_stl10_eval_loader, IMAGENET_MEAN, IMAGENET_STD
)

# torchattacks (FGSM/PGD/MIFGSM/DIFGSM/TIFGSM/APGD)
from torchattacks import FGSM, PGD, MIFGSM, DIFGSM, TIFGSM, APGD  # pip install torchattacks

# ===== RAW (unnormalized) Pet loader for attacks =====
def make_pets_raw_loader(root="./data/oxford-iiit-pet", batch_size=64, num_workers=0, img_size=224,
                         split="trainval", pin_memory=False) -> DataLoader:
    tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(img_size), transforms.ToTensor()])
    ds = datasets.OxfordIIITPet(root, split=split, target_types="binary-category",
                                transform=tf, download=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=int(num_workers),
                      pin_memory=bool(pin_memory), persistent_workers=False)

# ===== Normalize wrapper (model expects normalized inputs) =====
class NormalizeWrapper(nn.Module):
    def __init__(self, model: nn.Module, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        super().__init__(); self.model=model.eval()
        self.register_buffer("mean", torch.tensor(mean).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor(std).view(1,3,1,1))
    def forward(self, x01): return self.model((x01 - self.mean)/self.std)

@torch.no_grad()
def _acc01(model01: nn.Module, loader: DataLoader, device=None) -> float:
    device = device or next(model01.parameters()).device
    correct=0; total=0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        pred = model01(x).argmax(1); correct += (pred==y).sum().item(); total += y.numel()
    return (correct/total) if total else 0.0

def _attack_tips(clean_acc: float, atk_acc: float, name: str, eps: float, steps: int) -> str:
    drop = clean_acc - atk_acc; tips: List[str] = []
    if drop >= 0.5*max(1e-6, clean_acc) or atk_acc <= 0.05:
        tips.append("Large drop: enable/strengthen AT (PGD/TRADES) and verify eps/steps.")
    if name in {"PGD","APGD"} and steps < 10:
        tips.append("Increase steps (>=10–20) for PGD/APGD.")
    if name in {"MIFGSM","DIFGSM","TIFGSM"}:
        tips.append("Momentum/diversity/translation attacks: consider stronger crops & AT.")
    return " ".join(tips) or "Looks ok; sanity-check with AutoAttack."

# ===== Benchmark attacks on 2nd dataset; clean on 3rd =====
def run_attacks_on_second_and_check_third(recipes: Dict[str, dict],
                                          out_dir: str="./attack_results",
                                          eps: float=8/255, steps: int=10) -> pd.DataFrame:
    os.makedirs(out_dir, exist_ok=True)
    all_rows=[]
    for name, recipe in recipes.items():
        print(f"[attack] {name}")
        # Train + online (from File 1)
        model, pets_loader_norm, stl_loader_norm, extras = train_base_and_online(recipe)
        device = next(model.parameters()).device

        # Build RAW Pets loader from same B-phase cfg
        B = recipe["phaseB"]
        pets_raw = make_pets_raw_loader(root=B.get("root","./data/oxford-iiit-pet"),
                                        batch_size=B.get("batch_size",64),
                                        num_workers=B.get("num_workers",0),
                                        img_size=B.get("img_size",224),
                                        split=B.get("split","trainval"),
                                        pin_memory=B.get("pin_memory",False))

        # Wrap model so attacks perturb [0,1] space
        model01 = NormalizeWrapper(model).to(device).eval()

        # Clean acc on second dataset (raw wrapper)
        clean_second = _acc01(model01, pets_raw, device=device)

        # Instantiate attacks
        attacks = {
            "FGSM":  FGSM(model01, eps=eps),
            "PGD":   PGD(model01,  eps=eps, alpha=2/255, steps=int(steps), random_start=True),
            "MIFGSM":MIFGSM(model01, eps=eps, alpha=2/255, steps=int(steps)),
            "DIFGSM":DIFGSM(model01, eps=eps, alpha=2/255, steps=int(steps)),
            "TIFGSM":TIFGSM(model01, eps=eps, alpha=2/255, steps=int(steps)),
            "APGD":  APGD(model01, norm="Linf", eps=eps, steps=int(steps), n_restarts=1, verbose=False),
        }

        # Run attacks on second dataset
        for atk_name, atk in attacks.items():
            correct=0; total=0
            for x01,y in pets_raw:
                x01,y = x01.to(device), y.to(device)
                x_adv = atk(x01, y)
                pred = model01(x_adv).argmax(1)
                correct += (pred==y).sum().item(); total += y.numel()
            atk_acc = (correct/total) if total else 0.0
            all_rows.append({
                "recipe": name,
                "attack": atk_name,
                "eps": float(eps),
                "steps": int(steps),
                "second_clean_acc": float(clean_second),
                "second_attack_acc": float(atk_acc),
                "drop_vs_second_clean": float(clean_second - atk_acc),
                "third_clean_acc": float(extras["third_dataset_acc"]),
                "tips": _attack_tips(clean_second, atk_acc, atk_name, eps, steps),
                "config": recipe,
            })

        # persist
        with open(os.path.join(out_dir, f"{name}.recipe.json"), "w", encoding="utf-8") as f:
            json.dump(recipe, f, ensure_ascii=False, indent=2)

        # hygiene
        del model, model01, pets_raw, pets_loader_norm, stl_loader_norm; gc.collect()
        try: torch.cuda.empty_cache()
        except Exception: pass

    df = pd.DataFrame(all_rows).sort_values(by=["recipe","second_attack_acc"], ascending=[True, False]).reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "attacks_summary.csv"), index=False)
    return df

if __name__ == "__main__":
    # Example: reuse builder from File 1
    from online_training_pipeline import build_online_recipes
    recipes = build_online_recipes(epochs_phaseA=5, epochs_phaseB=1)
    df = run_attacks_on_second_and_check_third(recipes, out_dir="./attack_results", eps=8/255, steps=10)
    print(df.head())


