import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────
# Data
# ─────────────────────────────────────────
IMG_SIZE    = 384
NUM_CLASSES = 5
NUM_WORKERS = 4

TRAIN_SPLITS   = "data/splits"
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"

# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────
BATCH_SIZE       = 8
EPOCHS           = 50
FREEZE_EPOCHS    = 5
GRAD_ACCUM_STEPS = 12

FAST_DEV_RUN    = os.getenv("FAST_DEV_RUN", "0") == "1"
MAX_TRAIN_STEPS = None
MAX_VAL_STEPS   = None

# ─────────────────────────────────────────
# Optimizer
# ─────────────────────────────────────────
LR           = 1e-4
LR_FINETUNE  = 1e-5
WEIGHT_DECAY = 2e-2    # increased from 1e-2

# ─────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────
T0      = 10
T_MULT  = 2
ETA_MIN = 1e-6

# ─────────────────────────────────────────
# Loss
# 0.4 CE + 0.2 Focal + 0.1 LS + 0.3 QWK
# ─────────────────────────────────────────
LOSS_CE_WEIGHT    = 0.40
LOSS_FOCAL_WEIGHT = 0.20
LOSS_LS_WEIGHT    = 0.10
LOSS_QWK_WEIGHT   = 0.30
FOCAL_GAMMA       = 2.0
LABEL_SMOOTHING   = 0.1

# ─────────────────────────────────────────
# Class imbalance
# ─────────────────────────────────────────
USE_CLASS_WEIGHTS    = True
USE_WEIGHTED_SAMPLER = True
CLASS_WEIGHT_TYPE    = "cbrt"
MANUAL_CLASS_BOOST   = {
    1: 1.2,   # Mild
    2: 2.0,   # Moderate
    3: 1.5,   # Severe
    4: 1.5,   # PDR
}

# ─────────────────────────────────────────
# Mild Mixup Augmentation
# mixes Mild samples with each other
# forces model to learn Mild-specific features
# ─────────────────────────────────────────
USE_MILD_MIXUP    = True
MILD_MIXUP_ALPHA  = 0.4   # Beta distribution parameter

# ─────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────
EARLY_STOP_PATIENCE = 16
MONITOR_METRIC      = "qwk"

# ─────────────────────────────────────────
# Mixed precision
# ─────────────────────────────────────────
USE_AMP = True

# ─────────────────────────────────────────
# TTA
# ─────────────────────────────────────────
TTA_RUNS = 5

# ─────────────────────────────────────────
# K-Fold
# ─────────────────────────────────────────
NUM_FOLDS = 1 if FAST_DEV_RUN else 5

# ─────────────────────────────────────────
# DR Grade labels
# ─────────────────────────────────────────
DR_GRADES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR"
}