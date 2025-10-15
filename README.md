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
