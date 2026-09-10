import torch
import torch.nn as nn
import torch.nn.functional as F
import config


# ─────────────────────────────────────────
# QWK Loss
# directly optimizes QWK metric
# ─────────────────────────────────────────
class QWKLoss(nn.Module):

    def __init__(self, num_classes=config.NUM_CLASSES):
        super().__init__()
        self.num_classes = num_classes

        weights = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                weights[i][j] = ((i - j) ** 2) / \
                                 ((num_classes - 1) ** 2)

        self.register_buffer("weights", weights)

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        probs = torch.clamp(probs, min=1e-6, max=1.0)  # ← NaN guard
        n     = logits.size(0)

        targets_one_hot = torch.zeros_like(probs)
        targets_one_hot.scatter_(1, targets.view(-1, 1), 1.0)

        # soft confusion matrix
        conf = torch.mm(probs.t(), targets_one_hot)
        conf = conf / (conf.sum() + 1e-8)

        # expected matrix
        pred_hist   = probs.sum(0) / n
        target_hist = targets_one_hot.sum(0) / n
        expected    = torch.outer(pred_hist, target_hist)
        expected    = expected / (expected.sum() + 1e-8)

        num   = (self.weights * conf).sum()
        denom = (self.weights * expected).sum()

        loss = num / (denom + 1e-8)
        return torch.nan_to_num(loss, nan=0.0,   # ← NaN guard
                                posinf=1.0, neginf=0.0)


# ─────────────────────────────────────────
# Composite Loss
# 0.4 CE + 0.2 Focal + 0.1 LS + 0.3 QWK
# ─────────────────────────────────────────
class CompositeLoss(nn.Module):

    def __init__(
        self,
        class_weights=None,
        gamma=config.FOCAL_GAMMA,
        smoothing=config.LABEL_SMOOTHING,
        ce_weight=config.LOSS_CE_WEIGHT,
        focal_weight=config.LOSS_FOCAL_WEIGHT,
        ls_weight=config.LOSS_LS_WEIGHT,
        qwk_weight=config.LOSS_QWK_WEIGHT,
    ):
        super().__init__()

        self.gamma        = gamma
        self.smoothing    = smoothing
        self.ce_weight    = ce_weight
        self.focal_weight = focal_weight
        self.ls_weight    = ls_weight
        self.qwk_weight   = qwk_weight
        self.qwk_loss     = QWKLoss()

        self.register_buffer(
            "class_weights",
            class_weights if class_weights is not None
            else torch.ones(config.NUM_CLASSES)
        )

    def cross_entropy_loss(self, logits, targets):
        return F.cross_entropy(
            logits, targets,
            weight=self.class_weights
        )

    def focal_loss(self, logits, targets):
        ce    = F.cross_entropy(
                    logits, targets,
                    weight=self.class_weights,
                    reduction="none"
                )
        probs = F.softmax(logits, dim=1)
        pt    = probs.gather(
                    1, targets.view(-1, 1)
                ).squeeze(1)
        pt    = torch.clamp(pt, min=1e-6, max=1.0)  # ← NaN guard

        # higher gamma for Mild (class 1) specifically
        gamma_per_sample = torch.full_like(pt, self.gamma)
        gamma_per_sample[targets == 1] = 4.0  # asymmetric focal

        focal = ((1 - pt) ** gamma_per_sample) * ce
        focal = torch.nan_to_num(focal, nan=0.0)  # ← NaN guard
        return focal.mean()

    def label_smoothing_loss(self, logits, targets):
        n        = logits.size(1)
        log_prob = F.log_softmax(logits, dim=1)
        smooth   = torch.full_like(
                       log_prob, self.smoothing / (n - 1)
                   )
        smooth.scatter_(
            1, targets.view(-1, 1), 1.0 - self.smoothing
        )
        return -(smooth * log_prob).sum(dim=1).mean()

    def forward(self, logits, targets):
        ce    = self.cross_entropy_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        ls    = self.label_smoothing_loss(logits, targets)
        qwk   = self.qwk_loss(logits, targets)

        # guard all individual losses
        ce    = torch.nan_to_num(ce,    nan=0.0)
        focal = torch.nan_to_num(focal, nan=0.0)
        ls    = torch.nan_to_num(ls,    nan=0.0)
        qwk   = torch.nan_to_num(qwk,  nan=0.0)

        total = (
            self.ce_weight    * ce    +
            self.focal_weight * focal +
            self.ls_weight    * ls    +
            self.qwk_weight   * qwk
        )

        total = torch.nan_to_num(total, nan=0.0)  # ← final guard

        return total, {
            "loss_ce":    ce.item(),
            "loss_focal": focal.item(),
            "loss_ls":    ls.item(),
            "loss_qwk":   qwk.item(),
            "loss_total": total.item(),
        }


# ─────────────────────────────────────────
# Class weight calculator
# cbrt — gentle correction
# ─────────────────────────────────────────
def compute_class_weights(labels, mode=config.CLASS_WEIGHT_TYPE):
    labels  = torch.tensor(labels)
    counts  = torch.zeros(config.NUM_CLASSES)

    for c in range(config.NUM_CLASSES):
        counts[c] = (labels == c).sum().float()

    counts = torch.clamp(counts, min=1.0)

    if mode == "cbrt":
        weights = 1.0 / torch.pow(counts, 1/3)
    elif mode == "sqrt":
        weights = 1.0 / torch.sqrt(counts)
    elif mode == "inverse":
        weights = 1.0 / counts
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # apply manual boost for weak classes
    for cls, boost in config.MANUAL_CLASS_BOOST.items():
        weights[cls] = weights[cls] * boost

    weights = weights / weights.sum() * config.NUM_CLASSES

    print("  Class weights (loss):")
    for i, w in enumerate(weights):
        print(f"    Class {i} ({config.DR_GRADES[i]:<15}): {w:.4f}")

    return weights.to(config.DEVICE)