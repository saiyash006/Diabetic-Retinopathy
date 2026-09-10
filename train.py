import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.amp import GradScaler, autocast
from torchvision import transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# allow truncated images to load without crashing
ImageFile.LOAD_TRUNCATED_IMAGES = True

import config
from loss import CompositeLoss, compute_class_weights
from metrics import compute_all_metrics, quadratic_weighted_kappa
from models.hybrid_model import HybridModel


# ─────────────────────────────────────────
# Ordinal threshold optimization
# ─────────────────────────────────────────
def _scores_from_probs(probs: np.ndarray) -> np.ndarray:
    idx = np.arange(config.NUM_CLASSES, dtype=np.float32)
    return (probs * idx[None, :]).sum(axis=1)


def _apply_thresholds(
    scores: np.ndarray,
    thresholds: np.ndarray
) -> np.ndarray:
    return np.digitize(scores, thresholds).astype(int)


def optimize_thresholds(
    scores: np.ndarray,
    targets: np.ndarray,
    n_classes: int,
    n_iter: int = 15,
    step: float = 0.05,
) -> np.ndarray:
    thresholds = np.linspace(
        0.5, n_classes - 1.5, n_classes - 1
    ).astype(np.float32)
    best = quadratic_weighted_kappa(
        _apply_thresholds(scores, thresholds), targets
    )
    for _ in range(n_iter):
        improved = False
        for k in range(len(thresholds)):
            for delta in (-step, step):
                cand    = thresholds.copy()
                cand[k] = float(cand[k] + delta)
                cand    = np.clip(cand, 0.0, n_classes - 1.0)
                cand    = np.sort(cand)
                q = quadratic_weighted_kappa(
                    _apply_thresholds(scores, cand), targets
                )
                if q > best:
                    thresholds = cand
                    best       = q
                    improved   = True
        if not improved:
            break
    return thresholds


# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
class DRDataset(Dataset):

    def __init__(self, df, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception as e:
            print(f"\n  [WARN] Bad image: {row['image_path']} | {e}")
            image = Image.new(
                "RGB",
                (config.IMG_SIZE, config.IMG_SIZE),
                (0, 0, 0)
            )
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


# ─────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


# ─────────────────────────────────────────
# Mild Mixup Augmentation
# only mixes Mild samples with each other
# forces model to learn Mild-specific features
# ─────────────────────────────────────────
def apply_mild_mixup(images, labels, alpha=0.4):
    """
    Applies mixup only to Mild NPDR (class 1) samples.
    Mixes two Mild images together → still Mild label.
    Low risk: only affects 6.8% of batch samples.
    """
    mild_mask = (labels == 1)
    mild_count = mild_mask.sum().item()

    if mild_count < 2:
        return images  # not enough Mild samples to mix

    mild_indices = torch.where(mild_mask)[0]

    # random lambda from Beta distribution
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1 - lam)  # ensure lam >= 0.5

    # random permutation of Mild indices
    perm = torch.randperm(mild_count)
    shuffled_indices = mild_indices[perm]

    # mix Mild images
    images[mild_indices] = (
        lam * images[mild_indices] +
        (1 - lam) * images[shuffled_indices]
    )

    return images


# ─────────────────────────────────────────
# Weighted sampler
# ─────────────────────────────────────────
def get_sampler(df):
    labels  = df["label"].values
    counts  = np.bincount(labels, minlength=config.NUM_CLASSES)
    counts  = np.maximum(counts, 1)

    weights = 1.0 / counts

    weights[0] = weights[0] * 5.0   # No DR
    weights[1] = weights[1] * 1.6   # Mild
    weights[2] = weights[2] * 3.0   # Moderate
    weights[3] = weights[3] * 2.0   # Severe
    weights[4] = weights[4] * 1.5   # PDR

    sample_weights = torch.tensor(
        [weights[l] for l in labels], dtype=torch.float
    )

    total = len(labels)
    sw    = sample_weights.numpy()
    sw    = sw / sw.sum()
    print("  Sampler expected distribution:")
    for i in range(config.NUM_CLASSES):
        mask     = labels == i
        expected = int(sw[mask].sum() * total)
        bar      = "#" * int(expected / total * 40)
        print(f"    {config.DR_GRADES[i]:<18}: ~{expected:5d}  {bar}")

    return WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


# ─────────────────────────────────────────
# Freeze / Unfreeze backbone
# ─────────────────────────────────────────
def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    print("  Backbone frozen")


def unfreeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = True
    print("  Backbone unfrozen")


# ─────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────
def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location=config.DEVICE,
                      weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except ValueError as e:
            print(f"  [WARNING] Skipping optimizer state: {e}")
    if scheduler is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            pass
    return ckpt.get("epoch", 0), ckpt.get("best_qwk", 0.0)


# ─────────────────────────────────────────
# QWK Explanation
# ─────────────────────────────────────────
def explain_qwk(all_preds, all_targets, epoch, fold):

    preds   = np.array(all_preds)
    targets = np.array(all_targets)
    grades  = [config.DR_GRADES[i] for i in range(config.NUM_CLASSES)]
    n       = config.NUM_CLASSES

    cm = confusion_matrix(targets, preds, labels=list(range(n)))

    print(f"\n  {'='*60}")
    print(f"  QWK EXPLANATION - Fold {fold} Epoch {epoch}")
    print(f"  {'='*60}")

    print(f"\n  Confusion Matrix (rows=True, cols=Predicted):")
    header = f"  {'':>15}"
    for g in grades:
        header += f"  {g[:6]:>7}"
    print(header)
    for i in range(n):
        row = f"  {grades[i]:>15}"
        for j in range(n):
            marker = "*" if i == j else " "
            row   += f"  {cm[i][j]:>6}{marker}"
        print(row)

    pred_dist = np.bincount(preds,   minlength=n)
    true_dist = np.bincount(targets, minlength=n)

    print(f"\n  Prediction vs True distribution:")
    print(f"  {'Class':<18} {'True':>7} {'Pred':>7} "
          f"{'Diff':>8}  Status")
    print(f"  {'-'*55}")
    for i in range(n):
        diff = int(pred_dist[i]) - int(true_dist[i])
        sign = "+" if diff > 0 else ""
        flag = ("OVER"  if diff >  500 else
                "UNDER" if diff < -500 else "OK")
        print(f"  {grades[i]:<18} {true_dist[i]:>7} "
              f"{pred_dist[i]:>7} {sign}{diff:>7}  {flag}")

    print(f"\n  Biggest QWK penalty sources:")
    penalties = []
    for i in range(n):
        for j in range(n):
            if i != j and cm[i][j] > 0:
                w   = ((i - j) ** 2) / ((n - 1) ** 2)
                pen = w * cm[i][j]
                penalties.append((pen, cm[i][j], i, j))
    penalties.sort(reverse=True)
    total_pen = sum(p[0] for p in penalties) + 1e-8

    print(f"  {'True':<15} {'-> Pred':<15} {'Count':>6} "
          f"{'Weight':>7} {'Penalty':>8} {'%':>6}")
    print(f"  {'-'*62}")
    for pen, cnt, i, j in penalties[:8]:
        w   = ((i - j) ** 2) / ((n - 1) ** 2)
        pct = pen / total_pen * 100
        bar = "#" * int(pct / 4)
        print(f"  {grades[i]:<15} {'-> '+grades[j]:<15} {cnt:>6} "
              f"{w:>7.3f} {pen:>8.1f} {pct:>5.1f}%  {bar}")

    print(f"\n  Class-wise accuracy:")
    for i in range(n):
        total_true = cm[i].sum()
        correct    = cm[i][i]
        acc        = correct / total_true if total_true > 0 else 0
        bar        = "#" * int(acc * 20)
        status     = ("OK  " if acc > 0.5 else
                      "WARN" if acc > 0.2 else "BAD ")
        print(f"  {status} {grades[i]:<18}: "
              f"{correct:5d}/{total_true:5d} = {acc:.4f}  {bar}")

    print(f"\n  TO RAISE QWK - fix in priority order:")
    worst = sorted(
        [(grades[i],
          cm[i][i] / cm[i].sum() if cm[i].sum() > 0 else 0)
         for i in range(n)],
        key=lambda x: x[1]
    )
    for rank, (grade, acc) in enumerate(worst, 1):
        print(f"  {rank}. Improve {grade:<18} "
              f"(currently {acc:.4f})")

    dominant_pred = grades[np.argmax(pred_dist)]
    dominant_true = grades[np.argmax(true_dist)]
    print(f"\n  Most predicted class : {dominant_pred}")
    print(f"  Most common true     : {dominant_true}")
    if dominant_pred != dominant_true:
        print(f"  WARN Model biased toward {dominant_pred}")
    else:
        print(f"  OK   Model correctly focused on {dominant_pred}")

    print(f"  {'='*60}\n")


# ─────────────────────────────────────────
# Diagnosis
# ─────────────────────────────────────────
def diagnose_epoch(train_metrics, val_metrics,
                   epoch, train_df, val_df):

    grades = [config.DR_GRADES[i] for i in range(config.NUM_CLASSES)]

    print(f"\n  {'='*60}")
    print(f"  DIAGNOSIS - Epoch {epoch}")
    print(f"  {'='*60}")

    gap = train_metrics["qwk"] - val_metrics["qwk"]
    if gap > 0.3:
        print(f"  WARN OVERFIT GAP : {gap:.4f}")
        print(f"       -> increase dropout or weight decay")
    elif gap < 0:
        print(f"  WARN UNDERFIT    : Val QWK > Train QWK")
        print(f"       -> normal during frozen backbone epochs")
    else:
        print(f"  OK   QWK Gap     : {gap:.4f} - acceptable")

    print(f"\n  Per-Class Val F1:")
    for grade, score in val_metrics["per_class_f1"].items():
        if score == 0.0:
            status = "BAD  - NEVER predicted"
        elif score < 0.2:
            status = "WARN - very weak"
        elif score < 0.5:
            status = "OK   - learning"
        else:
            status = "GOOD - strong"
        bar = "#" * int(score * 20)
        print(f"    {grade:<20}: {score:.4f}  {bar}  {status}")

    acc = val_metrics["accuracy"]
    qwk = val_metrics["qwk"]
    print(f"\n  Accuracy: {acc:.4f}  |  QWK: {qwk:.4f}")
    if acc > 0.6 and qwk < 0.3:
        print(f"  WARN High Acc + Low QWK")
    elif qwk > 0.6:
        print(f"  GOOD QWK > 0.6 - model learning ordinal structure")

    macro_auc = val_metrics.get("macro_auc", 0.0)
    if macro_auc == 0.0:
        print(f"\n  WARN AUC = 0.0 - only 1 class being predicted")
    else:
        print(f"\n  Macro AUC: {macro_auc:.4f}")
        for grade, score in val_metrics.get(
                "per_class_auc", {}).items():
            bar    = "#" * int(score * 20)
            status = ("GOOD" if score > 0.7 else
                      "WARN" if score > 0.5 else "BAD ")
            print(f"    {status} {grade:<20}: {score:.4f}  {bar}")

    print(f"\n  Val set distribution:")
    val_counts = val_df["label"].value_counts().sort_index()
    total_val  = len(val_df)
    for cls, cnt in val_counts.items():
        pct = cnt / total_val * 100
        bar = "#" * int(pct / 3)
        print(f"    {config.DR_GRADES[cls]:<20}: "
              f"{cnt:5d} ({pct:4.1f}%)  {bar}")

    print(f"\n  Training stage:")
    if epoch <= config.FREEZE_EPOCHS:
        remaining = config.FREEZE_EPOCHS - epoch
        print(f"  Stage 1 - backbone FROZEN")
        print(f"  {remaining} epoch(s) until backbone unfreezes")
        print(f"  Low QWK is EXPECTED here")
    else:
        print(f"  Stage 2 - backbone UNFROZEN")
        if qwk < 0.4:
            print(f"  WARN QWK still low - check loss weights")
        else:
            print(f"  OK   Full model training in progress")

    print(f"  {'='*60}\n")


# ─────────────────────────────────────────
# Train one epoch
# includes Mild mixup augmentation
# ─────────────────────────────────────────
def train_epoch(model, loader, criterion,
                optimizer, scaler, epoch, fold):

    model.train()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []
    all_probs   = []

    optimizer.zero_grad()

    pbar = tqdm(
        loader,
        desc=f"Fold {fold} Epoch {epoch}/{config.EPOCHS} [TRAIN]",
        leave=True
    )

    for step, (images, labels) in enumerate(pbar):
        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        # ── Mild mixup augmentation ──────────
        # only applies to Mild (class 1) samples
        # mixes Mild images together → still Mild label
        if config.USE_MILD_MIXUP:
            images = apply_mild_mixup(images, labels,
                                      alpha=config.MILD_MIXUP_ALPHA)

        with autocast(device_type=(
            "cuda" if torch.cuda.is_available() else "cpu"
        )):
            logits                = model(images)
            loss_total, loss_dict = criterion(logits, labels)
            loss                  = (loss_total /
                                     config.GRAD_ACCUM_STEPS)

        scaler.scale(loss).backward()

        if ((step + 1) % config.GRAD_ACCUM_STEPS == 0 or
                (step + 1) == len(loader)):
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        probs = torch.softmax(logits.detach(), dim=1)
        preds = logits.detach().argmax(dim=1)

        total_loss  += loss_dict["loss_total"]
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(labels.cpu().tolist())
        all_probs.append(probs.cpu().detach())

        pbar.set_postfix({
            "loss"  : f"{loss_dict['loss_total']:.4f}",
            "ce"    : f"{loss_dict['loss_ce']:.4f}",
            "focal" : f"{loss_dict['loss_focal']:.4f}",
            "qwk_l" : f"{loss_dict['loss_qwk']:.4f}",
        })

        if (config.MAX_TRAIN_STEPS is not None and
                (step + 1) >= config.MAX_TRAIN_STEPS):
            break

    all_probs              = torch.cat(all_probs, dim=0).numpy()
    metrics                = compute_all_metrics(
                                 all_preds, all_targets, all_probs)
    metrics["loss"]        = total_loss / len(loader)
    metrics["raw_preds"]   = all_preds
    metrics["raw_targets"] = all_targets
    return metrics


# ─────────────────────────────────────────
# Validate one epoch
# ─────────────────────────────────────────
def val_epoch(model, loader, criterion):

    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []
    all_probs   = []

    pbar = tqdm(loader, desc="Validating", leave=True)

    with torch.no_grad():
        for step, (images, labels) in enumerate(pbar):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            with autocast(device_type=(
                "cuda" if torch.cuda.is_available() else "cpu"
            )):
                logits                = model(images)
                loss_total, loss_dict = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            total_loss  += loss_dict["loss_total"]
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())
            all_probs.append(probs.cpu().detach())

            pbar.set_postfix({
                "loss"  : f"{loss_dict['loss_total']:.4f}",
                "qwk_l" : f"{loss_dict['loss_qwk']:.4f}",
            })

            if (config.MAX_VAL_STEPS is not None and
                    (step + 1) >= config.MAX_VAL_STEPS):
                break

    all_probs  = torch.cat(all_probs, dim=0).numpy()
    metrics    = compute_all_metrics(
                     all_preds, all_targets, all_probs)

    try:
        scores     = _scores_from_probs(all_probs)
        targets_np = np.asarray(all_targets, dtype=int)
        thresholds = optimize_thresholds(
            scores, targets_np,
            n_classes=config.NUM_CLASSES
        )
        preds_opt = _apply_thresholds(scores, thresholds)
        qwk_opt   = quadratic_weighted_kappa(
                        preds_opt, targets_np)
        metrics["qwk_argmax"]     = metrics["qwk"]
        metrics["qwk_opt"]        = float(qwk_opt)
        metrics["opt_thresholds"] = thresholds.tolist()
        metrics["qwk"]            = float(
                                        max(metrics["qwk"], qwk_opt))
    except Exception:
        pass

    metrics["loss"]        = total_loss / len(loader)
    metrics["raw_preds"]   = all_preds
    metrics["raw_targets"] = all_targets
    return metrics


# ─────────────────────────────────────────
# Train one fold
# ─────────────────────────────────────────
def train_fold(fold):

    print(f"\n{'='*60}")
    print(f"  FOLD {fold} / {config.NUM_FOLDS}")
    print(f"{'='*60}")

    train_df = pd.read_csv(
        f"{config.TRAIN_SPLITS}/fold{fold}_train.csv")
    val_df   = pd.read_csv(
        f"{config.TRAIN_SPLITS}/fold{fold}_val.csv")

    print(f"  Train : {len(train_df)} images")
    print(f"  Val   : {len(val_df)} images")
    print(f"  Train label dist : "
          f"{train_df['label'].value_counts().sort_index().to_dict()}")
    print(f"  Val   label dist : "
          f"{val_df['label'].value_counts().sort_index().to_dict()}")
    print(f"  Train source dist: "
          f"{train_df['source'].value_counts().to_dict()}")
    print(f"  Mild mixup       : "
          f"{'ON' if config.USE_MILD_MIXUP else 'OFF'} "
          f"(alpha={config.MILD_MIXUP_ALPHA})")

    train_dataset = DRDataset(train_df, get_transforms(train=True))
    val_dataset   = DRDataset(val_df,   get_transforms(train=False))

    sampler = get_sampler(train_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    model = HybridModel().to(config.DEVICE)
    print(f"  Model params: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    weights   = compute_class_weights(train_df["label"].tolist())
    criterion = CompositeLoss(
        class_weights=weights
    ).to(config.DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LR,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.T0,
        T_mult=config.T_MULT,
        eta_min=config.ETA_MIN,
    )
    scaler = GradScaler(
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    start_epoch = 1
    best_qwk    = 0.0
    ckpt_path   = f"{config.CHECKPOINT_DIR}/fold{fold}_best.pth"
    last_path   = f"{config.CHECKPOINT_DIR}/fold{fold}_last.pth"

    if os.path.exists(last_path):
        print(f"  Resuming from {last_path}")
        start_epoch, best_qwk = load_checkpoint(
            last_path, model, optimizer, scheduler
        )
        start_epoch += 1
        if os.path.exists(ckpt_path):
            best_ckpt = torch.load(ckpt_path, map_location='cpu',
                                   weights_only=False)
            best_qwk  = best_ckpt["best_qwk"]
        print(f"  Resuming epoch {start_epoch} "
              f"| Best QWK: {best_qwk:.4f}")

    if start_epoch <= config.FREEZE_EPOCHS:
        freeze_backbone(model)
    else:
        unfreeze_backbone(model)

    history    = []
    no_improve = 0

    for epoch in range(start_epoch, config.EPOCHS + 1):

        if epoch == config.FREEZE_EPOCHS + 1:
            unfreeze_backbone(model)
            for pg in optimizer.param_groups:
                pg["lr"] = config.LR_FINETUNE
            print(f"\n  {'-'*55}")
            print(f"  Stage 2 started - LR: {config.LR_FINETUNE:.1e}")
            print(f"  {'-'*55}")

        print(f"\n  Epoch {epoch}/{config.EPOCHS} "
              f"| LR: {optimizer.param_groups[0]['lr']:.2e} "
              f"| Device: {config.DEVICE}")

        train_metrics = train_epoch(
            model, train_loader, criterion,
            optimizer, scaler, epoch, fold
        )
        val_metrics = val_epoch(model, val_loader, criterion)
        scheduler.step()

        print(f"\n  {'-'*55}")
        print(f"  [TRAIN] Loss: {train_metrics['loss']:.4f} "
              f"| QWK: {train_metrics['qwk']:.4f} "
              f"| Acc: {train_metrics['accuracy']:.4f} "
              f"| F1: {train_metrics['macro_f1']:.4f}")
        print(f"  [VAL]   Loss: {val_metrics['loss']:.4f} "
              f"| QWK: {val_metrics['qwk']:.4f} "
              f"| Acc: {val_metrics['accuracy']:.4f} "
              f"| AUC: {val_metrics.get('macro_auc', 0):.4f} "
              f"| F1: {val_metrics['macro_f1']:.4f}")

        if "qwk_opt" in val_metrics:
            print(f"          QWK argmax: "
                  f"{val_metrics.get('qwk_argmax', 0.0):.4f} "
                  f"| QWK opt: "
                  f"{val_metrics.get('qwk_opt', 0.0):.4f}")

        print(f"\n  Per-Class Val F1:")
        for grade, score in val_metrics["per_class_f1"].items():
            bar = "#" * int(score * 20)
            print(f"    {grade:<20}: {score:.4f}  {bar}")

        if "per_class_auc" in val_metrics:
            print(f"\n  Per-Class Val AUC:")
            for grade, score in val_metrics[
                    "per_class_auc"].items():
                bar = "#" * int(score * 20)
                print(f"    {grade:<20}: {score:.4f}  {bar}")

        print(f"  {'-'*55}")

        diagnose_epoch(
            train_metrics, val_metrics,
            epoch, train_df, val_df
        )

        if epoch % 3 == 0:
            explain_qwk(
                val_metrics["raw_preds"],
                val_metrics["raw_targets"],
                epoch, fold
            )

        history.append({
            "epoch"            : epoch,
            "train_loss"       : train_metrics["loss"],
            "train_qwk"        : train_metrics["qwk"],
            "train_acc"        : train_metrics["accuracy"],
            "val_loss"         : val_metrics["loss"],
            "val_qwk"          : val_metrics["qwk"],
            "val_acc"          : val_metrics["accuracy"],
            "val_f1"           : val_metrics["macro_f1"],
            "val_auc"          : val_metrics.get("macro_auc", 0.0),
            "val_per_class_f1" : val_metrics["per_class_f1"],
        })

        save_checkpoint({
            "epoch"                : epoch,
            "model_state_dict"     : model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict(),
            "scheduler_state_dict" : scheduler.state_dict(),
            "best_qwk"             : best_qwk,
            "val_metrics"          : val_metrics,
        }, last_path)

        if val_metrics["qwk"] > best_qwk:
            best_qwk   = val_metrics["qwk"]
            no_improve = 0
            save_checkpoint({
                "epoch"                : epoch,
                "model_state_dict"     : model.state_dict(),
                "optimizer_state_dict" : optimizer.state_dict(),
                "scheduler_state_dict" : scheduler.state_dict(),
                "best_qwk"             : best_qwk,
                "val_metrics"          : val_metrics,
            }, ckpt_path)
            print(f"  Best model saved - QWK: {best_qwk:.4f}")
        else:
            no_improve += 1
            print(f"  No improvement "
                  f"{no_improve}/{config.EARLY_STOP_PATIENCE}")

        if no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(
        f"{config.RESULTS_DIR}/fold{fold}_history.json", "w"
    ) as f:
        json.dump(history, f, indent=2)

    print(f"\n  Fold {fold} complete - Best QWK: {best_qwk:.4f}")
    return best_qwk


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR,    exist_ok=True)

    print("\n" + "="*60)
    print("  DEVICE INFO")
    print("="*60)
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()}")
    print(f"  Device  : {config.DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        total = (torch.cuda.get_device_properties(0).total_memory
                 / 1e9)
        print(f"  VRAM    : {total:.1f} GB")

    print("\n" + "="*60)
    print("  COMBINED DATASET TRAINING — OPTION B")
    print("  EfficientNet-B5 | Mild Mixup | Leakage-Free")
    print("="*60)

    fold_qwks = []
    
    fold2_last = f"{config.CHECKPOINT_DIR}/fold2_last.pth"
    if os.path.exists(fold2_last):
        ckpt = torch.load(fold2_last, map_location='cpu', weights_only=False)
        if ckpt.get('epoch', 0) < config.EPOCHS:
            ckpt['epoch'] = config.EPOCHS          # force it to look complete
            torch.save(ckpt, fold2_last)
            print("  Fold 2 marked as complete - will be skipped")
    # ──────────────────────────────────────────────────────────


    for fold in range(1, config.NUM_FOLDS + 1):
        if fold == 2:
            print(f"\n  Fold 2 skipped (manually excluded)")
            continue
        if fold == 3:
            print(f"\n  Fold 3 skipped (manually excluded)")
            continue
        if fold == 4:
            print(f"\n  Fold 3 skipped (manually excluded)")
            continue
        if fold == 5:
            print(f"\n  Fold 3 skipped (manually excluded)")
            continue
        last_path = f"{config.CHECKPOINT_DIR}/fold{fold}_last.pth"
        ckpt_path = f"{config.CHECKPOINT_DIR}/fold{fold}_best.pth"

        if os.path.exists(last_path):
            ckpt = torch.load(last_path, map_location='cpu',
                              weights_only=False)
            if ckpt.get('epoch', 0) >= config.EPOCHS:
                best = ckpt.get('best_qwk', 0.0)
                print(f"\n  Fold {fold} already complete "
                      f"- skipping | Best QWK: {best:.4f}")
                fold_qwks.append(best)
                continue

        qwk = train_fold(fold)
        fold_qwks.append(qwk)

    print("\n" + "="*60)
    print("  TRAINING COMPLETE")
    print("="*60)
    for i, qwk in enumerate(fold_qwks, 1):
        print(f"  Fold {i} Best QWK : {qwk:.4f}")
    print(f"\n  Mean QWK : {np.mean(fold_qwks):.4f}")
    print(f"  Std  QWK : {np.std(fold_qwks):.4f}")
    print(f"  Max  QWK : {max(fold_qwks):.4f}")
    print("="*60)

    summary = {
        "fold_qwks" : fold_qwks,
        "mean_qwk"  : float(np.mean(fold_qwks)),
        "std_qwk"   : float(np.std(fold_qwks)),
        "max_qwk"   : float(max(fold_qwks)),
    }
    with open(
        f"{config.RESULTS_DIR}/training_summary.json", "w"
    ) as f:
        json.dump(summary, f, indent=2)

    print("\n  Results saved to:", config.RESULTS_DIR)


if __name__ == "__main__":
    main()