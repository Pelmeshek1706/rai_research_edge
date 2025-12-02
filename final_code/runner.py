"\"\"\"High-level runner to execute many recipes and collect metrics.\"\"\""
from __future__ import annotations

import gc
import json
import os
from datetime import datetime
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
        # Fallbacks: online Phase B may only expose eval accuracy.
        if phaseB_best is None:
            phaseB_best = phase_metrics.get("phaseB_eval_acc")
        if phaseB_last is None:
            phaseB_last = phase_metrics.get("phaseB_eval_acc")

        mem_peak_bytes = metrics.get("mem_peak_bytes")
        mem_peak_gib = metrics.get("mem_peak_GiB")
        if mem_peak_gib is None and mem_peak_bytes is not None:
            mem_peak_gib = float(mem_peak_bytes) / (1024 ** 3)

        timestamp = datetime.now().strftime("%M-%H-%d-%m-%Y")
        row = {
            "recipe": name,
            "k": 1,
            "methods": [name],
            "accuracy": float(test_acc) if test_acc is not None else None,
            "val_acc": float(val_acc_final) if val_acc_final is not None else None,
            "phaseB_val_acc_best": float(phaseB_best) if phaseB_best is not None else None,
            "phaseB_val_acc_last": float(phaseB_last) if phaseB_last is not None else None,
            "mem_peak_bytes": int(mem_peak_bytes) if mem_peak_bytes is not None else None,
            "mem_peak_GiB": float(mem_peak_gib) if mem_peak_gib is not None else None,
            "config": recipe,
            "metrics_json": json.dumps(metrics),
            "timestamp": timestamp,
        }
        rows.append(row)

        recipe_dir = os.path.join(out_dir, name)
        os.makedirs(recipe_dir, exist_ok=True)
        recipe_df = pd.DataFrame([row])
        recipe_df.to_csv(os.path.join(recipe_dir, f"metrics_{timestamp}.csv"), index=False)

        del metrics
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    return pd.DataFrame(rows)


__all__ = ["run_recipes_and_collect"]
