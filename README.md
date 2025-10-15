# CIFAR-10 Cat vs Dog — MobileNetV3-Small (Lightning)

# Base pipeline

A minimal baseline that reformulates CIFAR-10 as a binary classification task (cat = 0, dog = 1), training a lightweight MobileNetV3-Small head using PyTorch Lightning.

---

## TL;DR

- **Backbone:** `torchvision.models.mobilenet_v3_small` (ImageNet weights, features frozen)
- **Head:** Final Linear layer → 2 classes, with configurable Dropout (`p=0.3`) and optional SpectralNorm
- **Loss/Regularization:** Cross-Entropy with label smoothing (0.1), optional feature noise & consistency loss
- **Optimizer:** AdamW (`lr=3e-4`, `weight_decay=1e-4`)
- **Precision:** Auto (bf16/AMP on GPU, fp32 on CPU)
- **Params:**
  - 592 K trainable
  - 927 K non-trainable
  - **1.5 M total**
  - ~6.080 MB estimated size

> **Note:** Although training uses standard mini-batches, the process simulates online learning—data arrive in small, shuffled batches—but with the caveat that all images are drawn from the single stationary distribution (CIFAR-10 cats/dogs).

---

## Data Pipeline

- **Dataset:** CIFAR-10 (filtered to only `cat` (3) and `dog` (5) labels)
- **Splits:**
  - From the train split, select a stratified 10% per class as validation
  - Test set: Official test split, filtered to cat/dog only
- **Label Remapping:** `{cat: 0, dog: 1}`
- **Transforms (applied once at subset level):**
  - `Resize(256)` → `CenterCrop(img_size=224)` → `ToTensor` → `Normalize` (CIFAR-10 mean/std)

#### TwoClassDataModule Details

- `pin_memory=True` only if CUDA is available
- `num_workers` capped at 2 (safe for laptops/Colab)
- Avoids double-transforming by building base datasets with `transform=None`, then wrapping subsets with transforms

---

## Model Architecture

`LitBinaryClassifier` wraps MobileNetV3-Small:

- Replace the classifier’s last layer with `Linear(in_features, 2)`
- Dropout in head: `dropout_p=0.3`
- Optional SpectralNorm on final Linear layer (`spectral_norm=False` by default)
- Backbone features are frozen (`freeze_backbone=True`); only classifier is trained
- Cross-entropy with label smoothing (`label_smoothing=0.1`)
- Optional feature noise in embedding space during `train()`
- Optional consistency regularization (KL-divergence between logits of `x` and `x + ε`)

**Metrics:**  
BinaryAccuracy and BinaryF1Score for validation/test (logged per epoch).  
A `CollectMetricsCallback` also tracks memory usage and compiles a per-epoch history.

---

⸻

Training setup
```python
dm = TwoClassDataModule(workdir="./workdir", batch_size=128, num_workers=4, img_size=224)
dm.setup()

model = LitBinaryClassifier(
    lr=3e-4, weight_decay=1e-4,
    pretrained=True, freeze_backbone=True,
    dropout_p=0.3,
    spectral_norm=False,
    feature_noise_sigma=0.0,
    label_smoothing=0.1,
    consistency_lambda=0.0,
    consistency_noise_sigma=0.0
)

metrics_cb = CollectMetricsCallback()
trainer = make_trainer(root_dir="./runs/cifar10_catdog", epochs=6, callbacks=[metrics_cb])
fit_and_validate(model, dm, trainer)
```

Precision:
  •  GPU + bf16 support → bf16-mixed
  •  GPU (no bf16) → 16-mixed
  •  CPU → 32-true

⸻

## Baseline Results (5 epochs)

Primary focus: accuracy/F1 (see callback CSV/logs for full loss and memory stats).

| epoch | val/acc | val/f1  |
|-------|---------|---------|
| 3     | 0.732   | 0.6608  |
| 4     | 0.685   | 0.5619  |
| 5     | 0.669   | 0.5278  |

**Additional logged values:**
- `val/loss` ~0.59–0.61 (epochs 3–5)
- `test/loss` ≈ 0.5854

**Best val/acc:** 0.732 (epoch 3)

> _Lightning’s validation logic and sanity checks may affect which epochs persist. The table reflects the relevant log slice._

---

## Reproducing & Environment

- **Python ≥ 3.10**
- **Install:** <code/> pip install torch torchvision lightning torchmetrics pandas matplotlib psutil </code>

> **Determinism:**  
> This baseline uses `random.shuffle` without a fixed seed; set seeds if you require repeatable splits.

---

## Notes & Next Steps

- This is a frozen-backbone baseline for fast iteration. Unfreezing or fine-tuning later layers usually improves accuracy.
- No data augmentation beyond resize/crop; adding flips, color jitter, RandAugment, or MixUp/CutMix can further boost performance.
- Consistency loss and feature noise are implemented but **disabled**; enable them for robustness experiments.
- As data come from one distribution and are shuffled into mini-batches, this baseline fits an online-style training pattern, but without distribution shift or concept drift.


# Part 2: Robustness-Oriented Training Pipeline (CIFAR-10 Cat/Dog)

A concise summary of the modular pipeline for online-style training on CIFAR-10 cats vs dogs using MobileNetV3-Small and robust regularization techniques.

---

## What This Repo Does

- **Task:** Reformulates CIFAR-10 as a binary classification problem (cat=0, dog=1) using a MobileNetV3-Small backbone with a modular robustness stack.
- **Data:** Stratified validation split from train (10% per class); test set is the official CIFAR-10 test filtered to cat/dog.
- **Transforms:** Applied once at the subset level: `Resize(256) → CenterCrop(224) → ToTensor → Normalize(CIFAR10 mean/std)`.
- **Model:** `mobilenet_v3_small` with final Linear → 2 classes, optional Dropout, SpectralNorm, feature-noise, and consistency loss. Features frozen by default (classifier head only is trained: ~592K trainable / ~1.5M total params).
- **Metrics:** `BinaryAccuracy`, `BinaryF1Score`, and a callback logging per-epoch memory (min/avg/max/peak).

---

## Key Modules & Their Purpose

### **BatchAugmentor (Train-Time)**
- **MixUp (`mixup_alpha`)**: Combines two images/labels; mixes loss:  
  `L = λ·CE(logits, y) + (1−λ)·CE(logits, y_perm)`
- **CutMix (`cutmix_alpha`)**: Patches from another image; loss weighted by kept area.
- **Control**: Probability `p` to apply, and `mode_prior` to select MixUp or CutMix.

### **Optimizer Builder**
- Parameter groups: no weight decay for norms/bias, decay for the rest.
- Optional layer-wise LR decay (`layer_decay`) if unfreezing the backbone.
- Default: AdamW over the constructed groups.

### **Jacobian Regularizer (+ "double-backward" fix)**
- Penalizes the input-gradient norm:  
  `||∂maxlogit/∂x||²`, computed in sub-batches to save memory.
- Replaces Hardswish→SiLU and Hardsigmoid→Sigmoid for 2nd-derivative stability.
- Option to force precision=`32-true` for robustness.

### **EMA of Weights (Callback)**
- Tracks an exponential moving average of parameters; swaps EMA weights in for validation/test, then restores live weights.

### **Pruning (Callback)**
- L1 unstructured pruning of the final Linear (classifier head) at `start_epoch`, with optional reapply per epoch. Removes reparam after fit.

### **Post-Hoc Calibration**
- Temperature scaling on validation using LBFGS optimizer (optimizes CE); evaluation divides logits by learned temperature.

### **Noise-Averaged Inference**
- Test-time input noise sampling (multiple runs with σ) and logit averaging to improve robustness.

### **Other Regularizers**
- Label smoothing (CE)
- Dropout in head
- Feature-space noise (Gaussian, applied during train)
- Optional consistency loss (`KL(logits(x) || logits(x+ε))`)

---

## Pipeline Execution Steps

1. **Build recipe/config dict:** Specifies data, model, optimizer, augment, regularizers, callbacks, trainer, eval.
2. **run_experiment(recipe):**
    - Instantiates DataModule & Model
    - Applies augmentor/Jacobian reg (and activation fix)
    - Registers callbacks (EMA, Pruning, Metrics)
    - Creates Lightning Trainer (auto-precision unless Jacobian forces fp32)
    - Runs fit → validate → test
    - (Optional) Temperature scaling on val; (Optional) Noise-averaged inference on test
    - Returns per-epoch metrics dataframe + extras (e.g., calibrated temp, noise-avg test acc)
3. **Grid utilities:**
    - `configure_recipes(...)` builds baseline + 8 methods × 5 sweeps
    - `evaluate_recipes(...)` runs them, saves per-run CSVs & `summary.csv`

---

## Brief Method Cheat-Sheet

| Method      | What It Changes                        | Expected Effect                                                  |
|-------------|----------------------------------------|------------------------------------------------------------------|
| mixup       | Convex image/label mixing              | Reduces overfitting, improves calibration, robustness            |
| cutmix      | Patch replacement + mixed targets      | Stronger locality bias, can improve generalization               |
| jacobian    | Penalizes input-gradient norm          | Smoother boundary, adversarial robustness (higher compute/mem)   |
| ema         | Averages weights over steps            | Lower val loss, more stable eval                                 |
| prune       | Sparsifies final linear layer          | Potential regularization, minor speed-ups, too much hurts acc    |
| labelsmooth | Softens targets                        | Combats overconfidence, better calibration                       |
| layerdecay  | Smaller LR for early layers            | Useful when unfreezing backbone                                  |
| noiseavg    | Monte-Carlo noise TTA (eval)           | Boosts robustness on test                                        |
| spectral    | Spectral normalization on final Linear | Limits Lipschitz, stabilizes head                                |
| featnoise   | Gaussian noise in embedding space      | Mild regularizer                                                 |
| consistency | KL(logits(x)‖logits(x+ε))              | Encourages invariance to small input noise                       |
| temp        | Temperature scaling (eval)             | Better calibration without changing accuracy                     |
| dropout     | Standard head-level regularization     | Regularizes head, reduces overfitting                            |

---
Run it (single recipe)
```python
base = build_base_recipe()
df, extras = run_experiment(base)  # baseline
```

Run a single method
```python
rec = deepcopy(build_base_recipe())
deep_update(rec, method_patch("mixup"))     # or "cutmix", "ema", ...
df, extras = run_experiment(rec)
```

Run the curated sweeps (and save a comparison)
```
recipes = configure_recipes(include_baseline=True)  # baseline + 8 methods × up to 5 variants each
summary = evaluate_recipes(recipes, run_experiment, out_dir="./grid_results")
print(summary.head())
# Files written:
#   ./grid_results/<recipe>.csv, <recipe>.extras.json, <recipe>.recipe.json
#   ./grid_results/summary.csv   ← aggregated leaderboard
```

---

## What's in `summary.csv`

- **recipe:** method or method#vi (variant index)
- **k:** method count (1 unless methods are composed)
- **methods:** list of enabled methods (for combos)
- **accuracy:** best available score (noise-avg test acc → test/acc → best val/acc)
- **mem_peak_bytes, mem_peak_GiB:** Peak memory usage (GPU/CPU RSS) during epochs (excludes "test" row)

---

## 📊 Methods Results Comparison

| Rank | Methods        | Accuracy | Peak Mem (GiB) |
|------|---------------|----------|----------------|
| 1    | [noiseavg]    | 0.8300   | 0.254          |
| 2    | [layerdecay]  | 0.7910   | 0.254          |
| 3    | [prune]       | 0.8010   | 0.254          |
| 4    | [jacobian]    | 0.8010   | 5.629          |
| 5    | [mixup]       | 0.7980   | 0.369          |
| 6    | [labelsmooth] | 0.7900   | 0.254          |
| 7    | [cutmix]      | 0.7720   | 0.345          |
| 8    | [ema]         | 0.6330   | 0.229          

---

> **Usage:**  
> Wire up `configure_recipes(...)` + `evaluate_recipes(...)`, then drop the produced `summary.csv` into the comparison table above.
