# Online Adversarial Training Benchmark

This repository contains a modular, reproducible pipeline for running online adversarial learning experiments on the CIFAR-10 cats vs. dogs subset using PyTorch Lightning and torchattacks.

The `online_adv` package provides datasets, model, attacks, callbacks, training/runner plumbing, and recipe builders, while `main.py` wires them together for benchmarks.

---

## Layout and Entry Point

- **`main.py`** – Entry point. Builds the recipe grid using `online_adv.configs.get_recipes_for_benchmark_online_attack`, runs `online_adv.runner.run_recipes_and_collect`, and prints a summary DataFrame.
- **`online_adv/configs.py`** – Base recipe definitions, regularization/attack patch utilities, and recipe grid generation.
- **`online_adv/runner.py`** – Iterates over recipes, trains each one, collects metrics/configs into a pandas DataFrame.
- **`online_adv/training.py`** – Handles reproducibility, Lightning `Trainer` instantiation, and the Phase A/B/C training loop.
- **`online_adv/data.py`** – ImageNet-normalized transforms and `CIFAR10CatDogDM` (DataModule for cats/dogs).
- **`online_adv/models.py`** – `LitBinaryClassifier`: MobileNetV3-Small binary classifier with online attack augmentation.
- **`online_adv/attacks.py`** – Torchattacks wrappers, attack registry, normalization helpers, and attack builders.
- **`online_adv/callbacks.py`** – Phase metrics callback for tracking and reporting per-phase validation scores.

---

## Online Learning Mechanism

Training consists of three distinct phases:

1. **Phase A – Clean Training:**  
   Initial `phaseA_epochs` epochs, no adversarial injection. Simulates warm-start on clean data.

2. **Phase B – Online Adversarial Training:**  
   After Phase A, adversarial examples (via `_build_attack`) are periodically injected (every `every_n` batches as specified by the recipe). Adversarial samples are generated on-the-fly, concatenated with the clean batch, and used jointly for loss calculation.

3. **Phase C – Evaluation:**  
   After training, `trainer.validate` and `trainer.test` compute final metrics. Detailed Phase A/B accuracy is reported by the callback.

Adversarial injection is governed by the `attack_injection` key in each recipe, and all samples are normalized for correct model input.

---

## Attacks

`online_adv/attacks.py` exposes five torchattacks methods via `ATTACK_REGISTRY`:

- FGSM
- PGD
- MIFGSM
- DIFGSM
- TIFGSM
- APGD

Default attack parameters (epsilon, alpha, steps) are in `ATTACK_DEFAULTS`.  
The `NormalizeWrapper` ensures attacks operate in pixel space and are re-normalized for model input.

Attack parameters can be overridden per recipe (`attack_kwargs`).  
Validation/test accuracy is logged for each attack.

---

## PGD Training

PGD-style adversarial training is implemented by setting `attack_injection` with `"method": "pgd"` and configuring `"every_n"`.  
During training, the model augments selected batches with PGD-generated adversarial samples (normalized in/out), so each batch contains both clean and adversarial data for loss calculation.

---

## Configurations and Regularization Methods

- **Base recipe:**  
  Dataset parameters, model toggles (dropout, etc.), optimizer defaults, root path, attack settings, and regularization placeholders.

- **Regularization methods:**  
  Enabled by `method_patch_online` and includes: mixup, cutmix, jacobian smoothing, EMA, pruning, temperature scaling, noise averaging, spectral norm, label smoothing, layer decay, etc.  
  Each method is combined with its hyperparameter sweep as defined in `_grid_patches_for_method_online`.

- **Recipe generation:**  
  `get_recipes_for_benchmark_online_attack` iterates over selected regularization methods and attack types, overlaying them to produce a recipe dictionary.  
  A "clean" (no-attack) recipe is also included for reference.

---

## Running the Experiments

1. **Install dependencies:**

   ```sh
   pip install lightning torch torchvision torchmetrics torchattacks pandas