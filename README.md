# Diabetic Retinopathy Grading

A deep learning pipeline for **automated grading of Diabetic Retinopathy (DR)** from fundus photographs using a hybrid EfficientNet-B5 + Lesion U-Net architecture with CBAM attention. The model classifies retinal images into 5 severity grades on the International Clinical Diabetic Retinopathy scale.

---

## Table of Contents

- [Overview](#overview)
- [DR Grading Scale](#dr-grading-scale)
- [Model Architecture](#model-architecture)
- [Datasets](#datasets)
- [Results](#results)
- [Training Setup](#training-setup)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)

---

## Overview

Diabetic Retinopathy is a leading cause of blindness worldwide. This project trains a multi-source, 5-fold cross-validated deep learning model that:

- Classifies fundus images into 5 DR severity grades
- Uses a **hybrid architecture** fusing high-level CNN features with explicit lesion detection (microaneurysms, haemorrhages, exudates, neovascularization)
- Applies a **composite loss** directly optimizing the Quadratic Weighted Kappa (QWK) metric — the competition standard for DR grading
- Employs **Test-Time Augmentation (TTA)** and **ordinal threshold optimization** for robust inference

The final ensemble achieves a **QWK of 0.8422 ± 0.0046** across 5 folds, surpassing the LAOT baseline (QWK 0.8296).

---

## DR Grading Scale

| Grade | Label | Description |
|:-----:|-------|-------------|
| 0 | **No DR** | No signs of diabetic retinopathy |
| 1 | **Mild NPDR** | Microaneurysms only |
| 2 | **Moderate NPDR** | More than just microaneurysms but less than severe NPDR |
| 3 | **Severe NPDR** | Extensive intraretinal hemorrhages, venous beading, IRMA |
| 4 | **Proliferative DR** | Neovascularization or vitreous/pre-retinal hemorrhage |

---

## Model Architecture

The model (`HybridModel`) fuses two complementary branches:

```
Input (384x384x3)
+-- EfficientNet-B5 Backbone (pretrained, timm)
|     +-- CBAM Attention (Channel + Spatial)
|           +-- AdaptiveAvgPool --> 2048-d feature vector
+-- Lesion U-Net
      +-- 4-channel lesion maps (MA / HE / EX / NV)
            +-- AdaptiveAvgPool(16x16) --> flatten --> 1024-d vector

Concatenate [2048 + 1024] = 3072-d
+-- Fusion Head: Linear --> 1024, LayerNorm, GELU, Dropout(0.3)
      +-- Ordinal Head: 1024 --> 256 --> 128 --> 5 logits
```

### Key Components

| Component | Details |
|-----------|---------|
| **Backbone** | EfficientNet-B5 (pretrained, `timm`) |
| **Attention** | CBAM — channel attention (AvgPool + MaxPool + FC) + spatial attention (7x7 conv) |
| **Lesion Branch** | Lightweight U-Net, 4 output channels (MA, HE, EX, NV), no external supervision |
| **Fusion** | Linear(3072->1024) -> LayerNorm -> GELU -> Dropout(0.3) |
| **Classifier** | 3-layer ordinal head with progressive dropout (0.5 -> 0.25) |
| **Input size** | 384 x 384 |

---

## Datasets

The model is trained on a **combined multi-source dataset** with patient-level deduplication:

| Dataset | Description |
|---------|-------------|
| **EyePACS** | Large-scale Kaggle DR competition dataset |
| **Archive** | Additional annotated fundus images |
| **IDRiD** | Indian Diabetic Retinopathy Image Dataset |
| **DDR** | DDR DR grading dataset |

> **Note**: Messidor was excluded due to label mismatch risk.

**Data preparation** (`prepare_combined.py`) performs:
- MD5-based image deduplication across sources
- Patient-level stratified 5-fold cross-validation splits (no patient leakage across folds)
- Export to CSV splits for reproducible training

**Preprocessing** (`preprocess_all.py`): Ben Graham-style CLAHE + circular crop, resized to 384x384.

---

## Results

### 5-Fold Cross-Validation (Ensemble)

| Metric | Value |
|--------|-------|
| **Mean QWK** | **0.8422 +/- 0.0046** |
| **Best Fold QWK** | 0.8498 (Fold 3) |
| **Ensemble QWK** | 0.8405 |
| **Overall Accuracy** | 79.50% |
| **Balanced Accuracy** | 62.47% |
| **Macro F1** | 62.23% |
| **Weighted F1** | 80.47% |
| **Macro AUC** | 0.8953 |
| **Weighted AUC** | 0.9079 |
| **MCC** | 0.6065 |
| **Mean Absolute Error** | 0.259 |
| **Within-1 Accuracy** | **95.08%** |
| **LAOT Baseline QWK** | 0.8296 |
| **Beats LAOT** | Yes |

### Per-Class Performance (Ensemble)

| Class | F1 | AUC | Sensitivity | Specificity |
|-------|:---:|:---:|:-----------:|:-----------:|
| **No DR** | 0.9037 | 0.9184 | 90.68% | 80.61% |
| **Mild NPDR** | 0.2524 | 0.7465 | 32.16% | 91.28% |
| **Moderate NPDR** | 0.7161 | 0.9119 | 64.11% | 96.02% |
| **Severe NPDR** | 0.4593 | 0.9242 | 55.19% | 97.70% |
| **Proliferative DR** | 0.7799 | 0.9754 | 70.24% | 99.61% |

> **Note**: Mild NPDR is the hardest class to separate, a well-known challenge in DR grading due to subtle distinguishing features and class imbalance.

### Per-Fold QWK

| Fold | QWK |
|------|-----|
| Fold 1 | 0.8363 |
| Fold 2 | 0.8387 |
| Fold 3 | **0.8498** |
| Fold 4 | 0.8437 |
| Fold 5 | 0.8425 |

---

## Training Setup

### Loss Function

A **composite loss** combining four objectives to handle class imbalance and directly optimize the ordinal metric:

```
L = 0.40 x CrossEntropy
  + 0.20 x AsymmetricFocal  (gamma=4 for Mild, gamma=2 otherwise)
  + 0.10 x LabelSmoothing   (epsilon=0.1)
  + 0.30 x QWKLoss          (soft confusion matrix)
```

### Optimizer & Scheduler

| Setting | Value |
|---------|-------|
| Optimizer | AdamW |
| Learning Rate (frozen) | 1e-4 |
| Learning Rate (finetune) | 1e-5 |
| Weight Decay | 2e-2 |
| Scheduler | CosineAnnealingWarmRestarts (T0=10, T_mult=2) |
| Gradient Accumulation | 12 steps |
| Mixed Precision | AMP (fp16) |

### Class Imbalance Handling

| Strategy | Details |
|----------|---------|
| **Class weights** | Cube-root inverse frequency (cbrt mode) |
| **Manual boost** | Mild x1.2, Moderate x2.0, Severe x1.5, PDR x1.5 |
| **Weighted sampler** | Per-sample over-sampling at batch construction |
| **Mild Mixup** | In-class mixup for Mild NPDR (alpha=0.4) |

### Augmentation

Training augmentations include random flips, rotations, colour jitter, and Gaussian blur. Inference uses **5-run TTA** with horizontal/vertical flips.

### Other Settings

| Setting | Value |
|---------|-------|
| Batch Size | 8 (effective 96 with grad accum) |
| Epochs | 50 |
| Backbone frozen for | First 5 epochs |
| Early stopping patience | 16 epochs |
| Monitored metric | QWK |
| Cross-validation | 5-fold stratified (patient-level) |
| TTA Runs | 5 |

---

## Project Structure

```
Diabetic-Retinopathy/
+-- config.py               # All hyperparameters and constants
+-- train.py                # Training loop, k-fold CV, TTA, threshold optimization
+-- test.py                 # Ensemble evaluation and metric reporting
+-- dataset.py              # Dataset class and transforms
+-- loss.py                 # CompositeLoss, QWKLoss, class weight calculator
+-- metrics.py              # QWK, AUC, F1, and per-class metric helpers
+-- prepare_combined.py     # Multi-dataset merge, deduplication, fold splits
+-- preprocess_all.py       # Image preprocessing (CLAHE + crop + resize)
+-- fix.py                  # Dataset fix/patch utilities
+-- merge_archive.py        # Archive dataset merge helper
+-- models/
|   +-- hybrid_model.py     # HybridModel, CBAMAttention, LesionUNet
+-- results/                # 5-fold ensemble results (current best run)
|   +-- ensemble_results.json
|   +-- fold1_history.json
|   +-- training_summary.json
+-- results1/               # Previous experimental run results
+-- fold1_best.pth          # Best checkpoint for Fold 1
+-- data/                   # (not tracked) Raw + preprocessed images + CSV splits
```

---

## Installation

### Requirements

- Python >= 3.9
- PyTorch >= 2.0 (with CUDA recommended)
- timm
- scikit-learn
- pandas, numpy, Pillow, tqdm

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install timm scikit-learn pandas numpy Pillow tqdm
```

---

## Usage

### 1. Prepare Data

Place raw datasets under `data/raw/` following the structure expected by `prepare_combined.py`, then run:

```bash
# Preprocess images (CLAHE + crop + resize to 384x384)
python preprocess_all.py

# Merge datasets, deduplicate, create 5-fold stratified splits
python prepare_combined.py
```

### 2. Train

```bash
# Full 5-fold training
python train.py

# Quick smoke test (1 fold, limited steps)
FAST_DEV_RUN=1 python train.py
```

Checkpoints are saved to `checkpoints/fold{N}_best.pth`. Training history is logged to `results/fold{N}_history.json`.

### 3. Evaluate (Ensemble)

```bash
python test.py
```

Outputs the full ensemble metrics report to `results/ensemble_results.json`, including QWK, accuracy, AUC, F1, sensitivity/specificity per class, and the confusion matrix.

---

## Configuration

All hyperparameters are centralized in `config.py`. Key settings:

```python
IMG_SIZE    = 384          # Input resolution
NUM_CLASSES = 5            # DR grades 0-4
BATCH_SIZE  = 8            # Per-GPU batch size
EPOCHS      = 50
LR          = 1e-4         # Initial learning rate
LR_FINETUNE = 1e-5         # After backbone unfreezing
WEIGHT_DECAY = 2e-2

# Loss weights
LOSS_CE_WEIGHT    = 0.40
LOSS_FOCAL_WEIGHT = 0.20
LOSS_LS_WEIGHT    = 0.10
LOSS_QWK_WEIGHT   = 0.30

TTA_RUNS  = 5              # Test-time augmentation runs
NUM_FOLDS = 5              # K-fold cross-validation
```

---

## Acknowledgements

This project uses the following public datasets:
- [EyePACS](https://www.kaggle.com/c/diabetic-retinopathy-detection) - Kaggle DR Detection Challenge
- [IDRiD](https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid) - Indian Diabetic Retinopathy Image Dataset
- [DDR Dataset](https://github.com/nkicsl/DDR-dataset) - DDR: An Useful Clinical Asset for Automated Diabetic Retinopathy Grading

Model backbone provided by [timm](https://github.com/huggingface/pytorch-image-models) (Ross Wightman).
