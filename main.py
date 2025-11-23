"""Entry point for running the online adversarial benchmark."""
from __future__ import annotations

from final_code.configs import get_recipes_for_benchmark_online_attack
from final_code.runner import run_recipes_and_collect


def main():
    recipes = get_recipes_for_benchmark_online_attack(
        every_n=4,
        epochs_phaseA=1,
        epochs_phaseB=1,
        variants_per_method=3,
    )
    summary = run_recipes_and_collect(recipes, out_dir="./grid_results")
    print(summary.head())

if __name__ == "__main__":
    main()
