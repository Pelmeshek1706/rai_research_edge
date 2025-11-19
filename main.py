from online_adv.configs import get_recipes_for_benchmark_online_attack
from online_adv.runner import run_recipes_and_collect


def main():
    recipes = get_recipes_for_benchmark_online_attack(
        every_n=4,
        epochs_phaseA=10,
        epochs_phaseB=2,
        variants_per_method=3,
    )
    summary = run_recipes_and_collect(recipes, out_dir="./grid_results")
    print(summary.head())


if __name__ == "__main__":
    main()
"""Entry point for running the online adversarial benchmark."""
from __future__ import annotations

from online_adv.configs import get_recipes_for_benchmark_online_attack
from online_adv.runner import run_recipes_and_collect


def main():
    recipes = get_recipes_for_benchmark_online_attack(
        every_n=4,
        epochs_phaseA=10,
        epochs_phaseB=2,
        variants_per_method=3,
    )
    summary = run_recipes_and_collect(recipes, out_dir="./grid_results")
    print(summary.head())


if __name__ == "__main__":
    main()
