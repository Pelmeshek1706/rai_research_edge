"""Entry point for running the online adversarial benchmark."""
from __future__ import annotations

from final_code.configs import (
    deep_update,
    get_recipes_for_benchmark_online_attack,
    phaseB_dataset_patch,
)
from final_code.runner import run_recipes_and_collect

# Choose dataset for Phase B online adaptation: "stl10" or "cifar10"
PHASEB_DATASET = "cifar10"


def main():
    recipes = get_recipes_for_benchmark_online_attack(
        every_n=4,
        epochs_phaseA=1,
        epochs_phaseB=1,
        variants_per_method=1,
    )

    if PHASEB_DATASET != "stl10":
        patch = phaseB_dataset_patch(PHASEB_DATASET)
        for recipe in recipes.values():
            deep_update(recipe, patch)

    summary = run_recipes_and_collect(recipes, out_dir="./grid_results")
    print(summary.head())
    summary.to_csv("summary_online_attack_brevitas.csv", index=False)


if __name__ == "__main__":
    main()
