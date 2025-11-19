"\"\"\"High-level runner to execute many recipes and collect metrics.\"\"\""
from __future__ import annotations

import gc
import json
import os
from typing import Dict

import pandas as pd
import torch

from .training import train_and_attack


def run_recipes_and_collect(recipes: Dict[str, dict], out_dir: str = "./grid_results") -> pd.DataFrame:
    """Run each recipe with `train_and_attack` and put the metrics into a DataFrame."""
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for name, recipe in recipes.items():
        print(f"[run] recipe={name}")
        metrics = train_and_attack(recipe, out_dir=os.path.join(out_dir, name))

        val_metrics = metrics.get("val", {})
        test_metrics = metrics.get("test", {})
        phase_metrics = metrics.get("phases", {}) or {}

        test_acc = test_metrics.get("test/acc")
        val_acc_final = val_metrics.get("val/acc")
        phaseB_best = phase_metrics.get("phaseB_val_acc_best")
        phaseB_last = phase_metrics.get("phaseB_val_acc_last")

        row = {
            "recipe": name,
            "k": 1,
            "methods": [name],
            "accuracy": float(test_acc) if test_acc is not None else None,
            "val_acc": float(val_acc_final) if val_acc_final is not None else None,
            "phaseB_val_acc_best": float(phaseB_best) if phaseB_best is not None else None,
            "phaseB_val_acc_last": float(phaseB_last) if phaseB_last is not None else None,
            "mem_peak_bytes": None,
            "mem_peak_GiB": None,
            "config": recipe,
            "metrics_json": json.dumps(metrics),
        }
        rows.append(row)

        del metrics
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return pd.DataFrame(rows)


__all__ = ["run_recipes_and_collect"]
