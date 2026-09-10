import numpy as np
import torch
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
import config


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# ─────────────────────────────────────────
# QWK — primary metric
# ─────────────────────────────────────────
def quadratic_weighted_kappa(preds, targets):
    preds   = _to_numpy(preds).astype(int)
    targets = _to_numpy(targets).astype(int)
    try:
        return float(cohen_kappa_score(
            targets, preds, weights="quadratic"
        ))
    except Exception:
        return 0.0


# ─────────────────────────────────────────
# AUC — fixed to normalize probs
# ─────────────────────────────────────────
def auc_score(probs, targets):
    probs   = _to_numpy(probs)
    targets = _to_numpy(targets)
    try:
        # normalize probs to sum to 1 — fixes the AUC bug
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)

        present = list(np.unique(targets).astype(int))
        if len(present) < 2:
            return 0.0

        return float(roc_auc_score(
            targets,
            probs,
            multi_class="ovr",
            average="macro",
            labels=list(range(config.NUM_CLASSES))
        ))
    except Exception:
        return 0.0


# ─────────────────────────────────────────
# Per-class AUC
# ─────────────────────────────────────────
def per_class_auc(probs, targets):
    probs   = _to_numpy(probs)
    targets = _to_numpy(targets)

    # normalize
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)

    result = {}
    grades = [config.DR_GRADES[i] for i in range(config.NUM_CLASSES)]

    for i in range(config.NUM_CLASSES):
        binary = (targets == i).astype(int)
        try:
            if binary.sum() == 0 or binary.sum() == len(binary):
                result[grades[i]] = 0.0
            else:
                result[grades[i]] = float(
                    roc_auc_score(binary, probs[:, i])
                )
        except Exception:
            result[grades[i]] = 0.0

    return result


# ─────────────────────────────────────────
# Compute all metrics
# ─────────────────────────────────────────
def compute_all_metrics(preds, targets, probs=None):
    preds   = _to_numpy(preds).astype(int)
    targets = _to_numpy(targets).astype(int)
    grades  = [config.DR_GRADES[i] for i in range(config.NUM_CLASSES)]

    metrics = {}

    # QWK
    metrics["qwk"] = quadratic_weighted_kappa(preds, targets)

    # accuracy
    metrics["accuracy"] = float(accuracy_score(targets, preds))

    # macro F1
    metrics["macro_f1"] = float(f1_score(
        targets, preds,
        average="macro",
        zero_division=0
    ))

    # per-class F1
    metrics["per_class_f1"] = {
        grades[i]: float(f1_score(
            targets, preds,
            labels=[i],
            average="macro",
            zero_division=0
        ))
        for i in range(config.NUM_CLASSES)
    }

    # AUC
    if probs is not None:
        metrics["macro_auc"]    = auc_score(probs, targets)
        metrics["per_class_auc"] = per_class_auc(probs, targets)
    else:
        metrics["macro_auc"]    = 0.0
        metrics["per_class_auc"] = {g: 0.0 for g in grades}

    return metrics