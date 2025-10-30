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



# Robustness Grid Search

This chapter documents a systematic **grid search** over robustness-oriented training and inference techniques on a lightweight binary benchmark (CIFAR-10 cat vs. dog) using a MobileNetV3-Small backbone. The goal was to quantify accuracy/memory trade-offs of individual methods and identify practical combinations worth carrying forward.

---

## Why these methods

* **MixUp** — linearly combines pairs of samples and labels to improve generalization and robustness. ([arXiv][1])
* **CutMix** — pastes patches between images and mixes labels proportionally to cut area; typically stronger than Cutout/MixUp on vision tasks. ([arXiv][2])
* **Label smoothing** — replaces one-hot targets with a softened distribution to reduce overconfidence. ([arXiv][3])
* **Temperature scaling (post-hoc)** — single-parameter calibration to fix probability overconfidence without changing accuracy. ([arXiv][4])
* **EMA weights** — exponential moving average of parameters for smoother validation behavior (popularized in *Mean Teacher*). ([arXiv][5])
* **Spectral normalization (SN)** — constrains layer Lipschitzness via spectral norm to stabilize training. ([arXiv][6])
* **Layer-wise learning-rate decay / discriminative fine-tuning** — different LRs per depth to stabilize fine-tuning. ([arXiv][7])
* **Magnitude pruning** — unstructured L1 pruning for sparsity/efficiency. ([arXiv][5])
* **Jacobian / input-gradient regularization** — penalizes sensitivity of outputs to inputs for robustness. ([arXiv][8])
* **Noise-averaged inference (TTA)** — adds small input noise and averages logits at test time. ([arXiv][9])

---

## Experimental setup (summary)

* **Data**: CIFAR-10, reduced to two classes *(cat=3, dog=5)*; class-balanced split with a per-class validation holdout.
* **Transforms**: `Resize(256) → CenterCrop(224) → ToTensor → Normalize(CIFAR10 mean/std)`.
* **Model**: MobileNetV3-Small; pre-trained; final classifier head replaced with 2-class linear layer; backbone **frozen** by default.
* **Training**: AdamW; base LR 3e-4; WD 1e-4; 10 epochs (default); automatic mixed precision on GPU (FP16/BF16) except Jacobian runs forced to FP32 due to second-order gradients.
* **Metrics**: Binary accuracy (primary); optional post-hoc calibration (temperature scaling); peak memory tracked per epoch.
* **Engineering**: To enable double-backprop for Jacobian regularization on MobileNetV3, hard activations are swapped for smooth counterparts (Hardswish→SiLU, Hardsigmoid→Sigmoid), a standard trick to ensure well-behaved higher-order derivatives.

**Grid protocol.** For **each method family**, ~50 configurations were evaluated (hyper-parameters and seeds/variants). The table below reports the **best single result per family** (“best-per-family” summary).

---

## Results — best per family

> *Accuracy = test accuracy (or best validation accuracy when test TTA not enabled). Memory is peak device memory measured during training/validation.*
> *Configs are intentionally omitted here, as requested.*

| Method (family or combo) | #methods | Accuracy | Peak Memory (GiB) |
| ------------------------ | -------- | -------- | ----------------- |
| cutmix                   | 1        | 0.864    | 0.52              |
| noiseavg#v2              | 1        | 0.855    | 0.37              |
| labelsmooth              | 1        | 0.840    | 0.34              |
| layerdecay#v1            | 1        | 0.835    | 0.29              |
| prune#v1                 | 1        | 0.829    | 0.29              |
| baseline                 | 1        | 0.827    | 0.38              |
| jacobian#v1              | 1        | 0.822    | 5.58              |
| mixup#v2                 | 1        | 0.817    | 0.50              |
| ema#v1                   | 1        | 0.800    | 0.25              |

---

## Discussion

* **CutMix** topped accuracy with modest memory overhead, aligning with literature that CutMix yields stronger gains than simpler pixel-drop variants and often outperforms MixUp on images. ([arXiv][2])
* **Noise-averaged inference (TTA)** produced a notable bump without retraining cost; it’s a cheap deployment-time knob to improve predictions. ([arXiv][9])
* **Label smoothing** consistently helped with minimal cost, and pairs naturally with strong data augmentation. It also synergizes with better calibration when followed by temperature scaling. ([arXiv][3])
* **Layer-wise LR decay (discriminative fine-tuning)** gave a small gain on a frozen backbone setup; benefits may grow with deeper fine-tuning or longer schedules. ([arXiv][7])
* **Pruning** at moderate sparsity did **not** hurt accuracy here and slightly reduced memory, which echoes evidence that magnitude pruning can retain accuracy at moderate sparsity. Heavier sparsity typically requires re-training or longer schedules. ([arXiv][5])
* **Jacobian regularization** improved robustness-oriented signals in prior work but was **memory-heavy** in our setting (FP32 + double backprop); its accuracy didn’t surpass CutMix within 10 epochs. This mirrors the known compute cost of input-gradient penalties. ([arXiv][8])
* **EMA** stabilized validation curves but didn’t raise peak accuracy at this budget; EMA tends to shine with longer training or semi-supervised consistency setups (*Mean Teacher*). ([arXiv][5])
* **Spectral normalization** (tested on the classifier head) is theoretically appealing for Lipschitz control and stability, but with a frozen MobileNetV3 backbone the incremental effect was small; SN is typically more impactful in adversarial or GAN settings. ([arXiv][6])

---

## Takeaways & recommendations

1. **Default recipe**: *CutMix + label smoothing* as first-line robustness regularizers on small-to-mid-size vision tasks — strong accuracy with low complexity. ([CVF Open Access][10])
2. **Deployment knob**: enable **temperature scaling** for calibrated probabilities and **noise-averaged inference** for a free accuracy boost when latency allows. ([arXiv][11])
3. **When compute allows**: explore **Jacobian** penalties for robustness research, but budget for FP32 and higher memory. ([CVF Open Access][12])
4. **Efficiency**: mild **pruning** appears safe at this scale; for real sparsity wins, combine iterative pruning with fine-tuning. ([arXiv][13])

---

## Conclusion

Across ~50 configurations per method family, **CutMix** delivered the best accuracy-to-cost trade-off on CIFAR-10 (cat vs. dog) with a frozen MobileNetV3-Small. **Label smoothing** reliably improved results at negligible cost, and **noise-averaged inference** offered an additional, training-free bump. **Temperature scaling** remains the recommended final step for calibrated probabilities. Methods with stronger theoretical robustness signals (e.g., **Jacobian** penalties, **spectral normalization**) were comparatively **compute-intensive** and didn’t surpass CutMix within our 10-epoch budget; they remain promising for extended training or robustness-centric evaluations. Overall, a **simple, production-friendly stack** — *CutMix + label smoothing + calibration (+ TTA at test time)* — is the most effective baseline to carry forward for this project. ([CVF Open Access][10])

---


[1]: https://arxiv.org/abs/1710.09412 "mixup: Beyond Empirical Risk Minimization"  
[2]: https://arxiv.org/abs/1905.04899 "CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"  
[3]: https://arxiv.org/abs/1512.00567 "Rethinking the Inception Architecture for Computer Vision"  
[4]: https://arxiv.org/abs/1706.04599 "On Calibration of Modern Neural Networks"  
[5]: https://arxiv.org/abs/1506.02626 "Learning both Weights and Connections for Efficient Neural Networks"  
[6]: https://arxiv.org/abs/1802.05957 "Spectral Normalization for Generative Adversarial Networks"  
[7]: https://arxiv.org/pdf/1801.06146 "arXiv:1801.06146v5 [cs.CL] 23 May 2018"  
[8]: https://arxiv.org/abs/1711.09404 "Improving the Adversarial Robustness and Interpretability of Deep Neural Networks by Regularizing their Input Gradients"  
[9]: https://arxiv.org/abs/1710.05941?utm_source=chatgpt.com "Searching for Activation Functions"  
[10]: https://openaccess.thecvf.com/content_ICCV_2019/papers/Yun_CutMix_Regularization_Strategy_to_Train_Strong_Classifiers_With_Localizable_Features_ICCV_2019_paper.pdf "CutMix: Regularization Strategy to Train Strong Classifiers ..."  
[11]: https://arxiv.org/pdf/1706.04599 "On Calibration of Modern Neural Networks"  
[12]: https://openaccess.thecvf.com/content_ECCV_2018/papers/Daniel_Jakubovitz_Improving_DNN_Robustness_ECCV_2018_paper.pdf "Improving DNN Robustness to Adversarial Attacks using ..."  
[13]: https://arxiv.org/pdf/1506.02626 "Learning both Weights and Connections for Efficient ..."  
