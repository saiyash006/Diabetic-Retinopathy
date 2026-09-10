import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
    cohen_kappa_score,
    matthews_corrcoef,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import json
import warnings
warnings.filterwarnings("ignore")

ImageFile.LOAD_TRUNCATED_IMAGES = True

import config
from metrics import (
    compute_all_metrics,
    quadratic_weighted_kappa
)
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
    n_iter: int = 50,
    step: float = 0.02,
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
def get_val_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_tta_transform():
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ─────────────────────────────────────────
# Compute ALL metrics
# ─────────────────────────────────────────
def compute_full_metrics(preds, targets, probs):
    grades  = [config.DR_GRADES[i] for i in range(config.NUM_CLASSES)]
    n       = config.NUM_CLASSES
    results = {}

    # ── QWK (primary metric) ─────────────
    results["qwk"]          = quadratic_weighted_kappa(preds, targets)
    results["qwk_linear"]   = float(cohen_kappa_score(
                                  targets, preds, weights="linear"))

    # ── Accuracy ─────────────────────────
    results["accuracy"]         = float(np.mean(preds == targets))
    results["balanced_accuracy"] = float(balanced_accuracy_score(
                                       targets, preds))

    # ── F1 scores ────────────────────────
    results["macro_f1"]    = float(f1_score(
                                targets, preds, average="macro",
                                zero_division=0))
    results["weighted_f1"] = float(f1_score(
                                targets, preds, average="weighted",
                                zero_division=0))
    results["micro_f1"]    = float(f1_score(
                                targets, preds, average="micro",
                                zero_division=0))
    results["per_class_f1"] = {
        grades[i]: float(f1_score(
            targets, preds, labels=[i],
            average="macro", zero_division=0
        ))
        for i in range(n)
    }

    # ── Precision / Recall ───────────────
    results["macro_precision"] = float(precision_score(
                                     targets, preds, average="macro",
                                     zero_division=0))
    results["macro_recall"]    = float(recall_score(
                                     targets, preds, average="macro",
                                     zero_division=0))
    results["per_class_precision"] = {
        grades[i]: float(precision_score(
            targets, preds, labels=[i],
            average="macro", zero_division=0
        ))
        for i in range(n)
    }
    results["per_class_recall"] = {
        grades[i]: float(recall_score(
            targets, preds, labels=[i],
            average="macro", zero_division=0
        ))
        for i in range(n)
    }

    # ── AUC (ROC) ────────────────────────
    try:
        norm_probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)
        results["macro_auc"] = float(roc_auc_score(
            targets, norm_probs,
            multi_class="ovr",
            average="macro",
            labels=list(range(n))
        ))
        results["weighted_auc"] = float(roc_auc_score(
            targets, norm_probs,
            multi_class="ovr",
            average="weighted",
            labels=list(range(n))
        ))
        results["per_class_auc"] = {}
        for i in range(n):
            binary = (targets == i).astype(int)
            try:
                results["per_class_auc"][grades[i]] = float(
                    roc_auc_score(binary, norm_probs[:, i])
                )
            except Exception:
                results["per_class_auc"][grades[i]] = 0.0
    except Exception:
        results["macro_auc"]    = 0.0
        results["weighted_auc"] = 0.0
        results["per_class_auc"] = {g: 0.0 for g in grades}

    # ── Average Precision (PR-AUC) ───────
    try:
        results["per_class_ap"] = {}
        ap_scores = []
        for i in range(n):
            binary = (targets == i).astype(int)
            try:
                ap = float(average_precision_score(
                    binary, norm_probs[:, i]
                ))
                results["per_class_ap"][grades[i]] = ap
                ap_scores.append(ap)
            except Exception:
                results["per_class_ap"][grades[i]] = 0.0
        results["macro_ap"] = float(np.mean(ap_scores))
    except Exception:
        results["macro_ap"]    = 0.0
        results["per_class_ap"] = {g: 0.0 for g in grades}

    # ── MCC ──────────────────────────────
    try:
        results["mcc"] = float(matthews_corrcoef(targets, preds))
    except Exception:
        results["mcc"] = 0.0

    # ── Sensitivity / Specificity ────────
    cm = confusion_matrix(targets, preds, labels=list(range(n)))
    results["per_class_sensitivity"] = {}
    results["per_class_specificity"] = {}
    for i in range(n):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results["per_class_sensitivity"][grades[i]] = float(sens)
        results["per_class_specificity"][grades[i]] = float(spec)

    # ── Class-wise accuracy ──────────────
    results["per_class_accuracy"] = {}
    for i in range(n):
        total   = cm[i].sum()
        correct = cm[i][i]
        results["per_class_accuracy"][grades[i]] = float(
            correct / total if total > 0 else 0
        )

    # ── Ordinal error analysis ───────────
    results["mean_absolute_error"] = float(
        np.mean(np.abs(preds.astype(int) - targets.astype(int)))
    )
    results["within_one_accuracy"] = float(
        np.mean(np.abs(preds.astype(int) - targets.astype(int)) <= 1)
    )

    # ── Confusion matrix ─────────────────
    results["confusion_matrix"] = cm.tolist()

    return results


# ─────────────────────────────────────────
# Print detailed results
# ─────────────────────────────────────────
def print_results(preds, targets, probs, title="Results", fold=None):
    grades = [config.DR_GRADES[i] for i in range(config.NUM_CLASSES)]
    n      = config.NUM_CLASSES

    m  = compute_full_metrics(preds, targets, probs)
    cm = np.array(m["confusion_matrix"])

    print(f"\n  {'='*65}")
    print(f"  {title}")
    if fold:
        print(f"  Fold {fold}")
    print(f"  {'='*65}")

    # ── Primary metrics ──────────────────
    print(f"\n  ── PRIMARY METRICS ──────────────────────────────")
    print(f"  QWK (quadratic weighted kappa) : {m['qwk']:.4f}  ← main metric")
    print(f"  QWK (linear weighted kappa)    : {m['qwk_linear']:.4f}")
    print(f"  MCC (Matthews corr coef)       : {m['mcc']:.4f}")

    # ── Accuracy ─────────────────────────
    print(f"\n  ── ACCURACY ─────────────────────────────────────")
    print(f"  Overall accuracy    : {m['accuracy']:.4f}")
    print(f"  Balanced accuracy   : {m['balanced_accuracy']:.4f}")
    print(f"  Within-1 accuracy   : {m['within_one_accuracy']:.4f}  (pred within 1 grade)")
    print(f"  Mean absolute error : {m['mean_absolute_error']:.4f}  (ordinal distance)")

    # ── F1 scores ────────────────────────
    print(f"\n  ── F1 SCORES ────────────────────────────────────")
    print(f"  Macro F1    : {m['macro_f1']:.4f}")
    print(f"  Weighted F1 : {m['weighted_f1']:.4f}")
    print(f"  Micro F1    : {m['micro_f1']:.4f}")
    print(f"\n  Per-Class F1:")
    for grade, score in m["per_class_f1"].items():
        bar    = "#" * int(score * 25)
        status = ("GOOD" if score > 0.5 else
                  "OK  " if score > 0.3 else "WARN")
        print(f"    {status} {grade:<20}: {score:.4f}  {bar}")

    # ── Precision / Recall ───────────────
    print(f"\n  ── PRECISION / RECALL ───────────────────────────")
    print(f"  Macro Precision : {m['macro_precision']:.4f}")
    print(f"  Macro Recall    : {m['macro_recall']:.4f}")
    print(f"\n  Per-Class Precision / Recall / Sensitivity / Specificity:")
    print(f"  {'Class':<20} {'Prec':>6} {'Rec':>6} {'Sens':>6} {'Spec':>6}")
    print(f"  {'-'*50}")
    for grade in grades:
        prec = m["per_class_precision"].get(grade, 0)
        rec  = m["per_class_recall"].get(grade, 0)
        sens = m["per_class_sensitivity"].get(grade, 0)
        spec = m["per_class_specificity"].get(grade, 0)
        print(f"  {grade:<20} {prec:>6.4f} {rec:>6.4f} "
              f"{sens:>6.4f} {spec:>6.4f}")

    # ── AUC scores ───────────────────────
    print(f"\n  ── AUC SCORES ───────────────────────────────────")
    print(f"  Macro AUC    : {m['macro_auc']:.4f}")
    print(f"  Weighted AUC : {m['weighted_auc']:.4f}")
    print(f"  Macro AP     : {m['macro_ap']:.4f}  (PR-AUC)")
    print(f"\n  Per-Class ROC-AUC / PR-AUC:")
    print(f"  {'Class':<20} {'ROC-AUC':>8} {'PR-AUC':>8}")
    print(f"  {'-'*40}")
    for grade in grades:
        auc = m["per_class_auc"].get(grade, 0)
        ap  = m["per_class_ap"].get(grade, 0)
        bar = "#" * int(auc * 20)
        print(f"  {grade:<20} {auc:>8.4f} {ap:>8.4f}  {bar}")

    # ── Confusion matrix ─────────────────
    print(f"\n  ── CONFUSION MATRIX (rows=True, cols=Predicted) ─")
    header = f"  {'':>20}"
    for g in grades:
        header += f"  {g[:7]:>8}"
    print(header)
    for i in range(n):
        row = f"  {grades[i]:>20}"
        for j in range(n):
            marker = "*" if i == j else " "
            row   += f"  {cm[i][j]:>7}{marker}"
        print(row)

    # ── Class-wise accuracy ──────────────
    print(f"\n  ── CLASS-WISE ACCURACY ──────────────────────────")
    for i in range(n):
        total   = cm[i].sum()
        correct = cm[i][i]
        acc     = m["per_class_accuracy"][grades[i]]
        bar     = "#" * int(acc * 25)
        status  = ("GOOD" if acc > 0.6 else
                   "OK  " if acc > 0.4 else "WARN")
        print(f"  {status} {grades[i]:<20}: "
              f"{correct:5d}/{total:5d} = {acc:.4f}  {bar}")

    print(f"  {'='*65}\n")
    return m


# ─────────────────────────────────────────
# Single fold inference
# ─────────────────────────────────────────
def predict_fold(model, loader, tta_runs=config.TTA_RUNS,
                 device=config.DEVICE):
    model.eval()
    all_probs   = []
    all_targets = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Standard pass"):
            images = images.to(device)
            logits = model(images)
            probs  = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_targets.extend(labels.tolist())

    base_probs = torch.cat(all_probs, dim=0).numpy()

    if tta_runs <= 0:
        return base_probs, np.array(all_targets)

    tta_dataset = DRDataset(
        loader.dataset.df,
        transform=get_tta_transform()
    )
    tta_loader  = DataLoader(
        tta_dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    tta_probs_sum = base_probs.copy()

    for run in range(tta_runs):
        run_probs = []
        with torch.no_grad():
            for images, _ in tqdm(
                tta_loader,
                desc=f"  TTA run {run+1}/{tta_runs}"
            ):
                images = images.to(device)
                logits = model(images)
                probs  = F.softmax(logits, dim=1)
                run_probs.append(probs.cpu())
        tta_probs_sum += torch.cat(run_probs, dim=0).numpy()

    final_probs = tta_probs_sum / (tta_runs + 1)
    return final_probs, np.array(all_targets)


# ─────────────────────────────────────────
# Main evaluation
# ─────────────────────────────────────────
def main():

    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("\n" + "="*65)
    print("  ENSEMBLE EVALUATION — FULL METRICS")
    print(f"  TTA runs : {config.TTA_RUNS}")
    print(f"  Folds    : {config.NUM_FOLDS}")
    print(f"  Device   : {config.DEVICE}")
    print("="*65)

    all_fold_probs   = []
    all_fold_targets = []
    fold_qwks        = []
    fold_results     = []

    for fold in range(1, config.NUM_FOLDS + 1):

        ckpt_path = f"{config.CHECKPOINT_DIR}/fold{fold}_best.pth"
        val_csv   = f"{config.TRAIN_SPLITS}/fold{fold}_val.csv"

        if not os.path.exists(ckpt_path):
            print(f"\n  Fold {fold} — checkpoint not found, skipping")
            continue
        if not os.path.exists(val_csv):
            print(f"\n  Fold {fold} — val CSV not found, skipping")
            continue

        print(f"\n  {'─'*60}")
        print(f"  FOLD {fold}")
        print(f"  {'─'*60}")

        model = HybridModel().to(config.DEVICE)
        ckpt  = torch.load(ckpt_path, map_location=config.DEVICE,
                           weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint — "
              f"epoch {ckpt.get('epoch', '?')} | "
              f"best QWK {ckpt.get('best_qwk', 0):.4f}")

        val_df      = pd.read_csv(val_csv)
        val_dataset = DRDataset(val_df, get_val_transform())
        val_loader  = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"  Val samples : {len(val_df)}")
        print(f"  Label dist  : "
              f"{val_df['label'].value_counts().sort_index().to_dict()}")

        probs, targets = predict_fold(
            model, val_loader, tta_runs=config.TTA_RUNS
        )

        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)

        preds_argmax = probs.argmax(axis=1)
        scores       = _scores_from_probs(probs)
        thresholds   = optimize_thresholds(
            scores, targets, n_classes=config.NUM_CLASSES
        )
        preds_opt  = _apply_thresholds(scores, thresholds)
        qwk_opt    = quadratic_weighted_kappa(preds_opt, targets)
        qwk_argmax = quadratic_weighted_kappa(preds_argmax, targets)

        print(f"\n  QWK argmax : {qwk_argmax:.4f}")
        print(f"  QWK opt    : {qwk_opt:.4f}")
        print(f"  Thresholds : {thresholds.round(3).tolist()}")

        best_preds = (preds_opt if qwk_opt >= qwk_argmax
                      else preds_argmax)
        best_qwk   = max(qwk_opt, qwk_argmax)

        fold_qwks.append(best_qwk)

        fold_metrics = print_results(
            best_preds, targets, probs,
            title=f"FOLD {fold} RESULTS", fold=fold
        )
        fold_results.append(fold_metrics)

        all_fold_probs.append(probs)
        all_fold_targets.append(targets)

        del model
        torch.cuda.empty_cache()

    # ── ensemble ─────────────────────────
    if len(all_fold_probs) > 1:

        print("\n" + "="*65)
        print("  FULL ENSEMBLE RESULTS")
        print("="*65)

        ensemble_probs   = np.concatenate(all_fold_probs,   axis=0)
        ensemble_targets = np.concatenate(all_fold_targets, axis=0)
        ensemble_probs   = ensemble_probs / (
            ensemble_probs.sum(axis=1, keepdims=True) + 1e-8
        )

        preds_argmax = ensemble_probs.argmax(axis=1)
        qwk_argmax   = quadratic_weighted_kappa(
                           preds_argmax, ensemble_targets)

        scores     = _scores_from_probs(ensemble_probs)
        thresholds = optimize_thresholds(
            scores, ensemble_targets,
            n_classes=config.NUM_CLASSES,
            n_iter=100, step=0.01
        )
        preds_opt = _apply_thresholds(scores, thresholds)
        qwk_opt   = quadratic_weighted_kappa(
                        preds_opt, ensemble_targets)

        print(f"\n  QWK argmax  : {qwk_argmax:.4f}")
        print(f"  QWK opt     : {qwk_opt:.4f}")
        print(f"  Thresholds  : {thresholds.round(3).tolist()}")

        best_preds = (preds_opt if qwk_opt >= qwk_argmax
                      else preds_argmax)
        best_qwk   = max(qwk_opt, qwk_argmax)

        ensemble_metrics = print_results(
            best_preds, ensemble_targets, ensemble_probs,
            title="FULL ENSEMBLE RESULTS"
        )

        # ── per fold summary ─────────────
        print("\n  ── PER-FOLD SUMMARY ─────────────────────────────")
        print(f"  {'Fold':<8} {'QWK':>6} {'Acc':>6} {'MacF1':>7} "
              f"{'MacAUC':>8} {'MCC':>7} {'WI1':>6}")
        print(f"  {'-'*55}")
        for i, (qwk, res) in enumerate(
            zip(fold_qwks, fold_results), 1
        ):
            print(f"  Fold {i:<4} "
                  f"{qwk:>6.4f} "
                  f"{res.get('accuracy', 0):>6.4f} "
                  f"{res.get('macro_f1', 0):>7.4f} "
                  f"{res.get('macro_auc', 0):>8.4f} "
                  f"{res.get('mcc', 0):>7.4f} "
                  f"{res.get('within_one_accuracy', 0):>6.4f}")

        print(f"\n  Mean QWK : {np.mean(fold_qwks):.4f}")
        print(f"  Std  QWK : {np.std(fold_qwks):.4f}")
        print(f"  Max  QWK : {max(fold_qwks):.4f}")
        print(f"  Ensemble : {best_qwk:.4f}")

        if best_qwk > 0.8296:
            print(f"\n  ✅ BEATS LAOT (0.8296) by "
                  f"{best_qwk - 0.8296:.4f}!")
        else:
            print(f"\n  ❌ Gap to LAOT (0.8296) : "
                  f"{0.8296 - best_qwk:.4f}")

        # ── save full results ─────────────
        summary = {
            "fold_qwks"              : fold_qwks,
            "mean_qwk"               : float(np.mean(fold_qwks)),
            "std_qwk"                : float(np.std(fold_qwks)),
            "max_qwk"                : float(max(fold_qwks)),
            "ensemble_qwk"           : float(best_qwk),
            "ensemble_qwk_argmax"    : float(qwk_argmax),
            "ensemble_qwk_opt"       : float(qwk_opt),
            "opt_thresholds"         : thresholds.tolist(),
            "ensemble_accuracy"      : ensemble_metrics.get("accuracy", 0),
            "ensemble_balanced_acc"  : ensemble_metrics.get("balanced_accuracy", 0),
            "ensemble_macro_f1"      : ensemble_metrics.get("macro_f1", 0),
            "ensemble_weighted_f1"   : ensemble_metrics.get("weighted_f1", 0),
            "ensemble_macro_auc"     : ensemble_metrics.get("macro_auc", 0),
            "ensemble_weighted_auc"  : ensemble_metrics.get("weighted_auc", 0),
            "ensemble_macro_ap"      : ensemble_metrics.get("macro_ap", 0),
            "ensemble_mcc"           : ensemble_metrics.get("mcc", 0),
            "ensemble_mae"           : ensemble_metrics.get("mean_absolute_error", 0),
            "ensemble_within1_acc"   : ensemble_metrics.get("within_one_accuracy", 0),
            "ensemble_per_class_f1"  : ensemble_metrics.get("per_class_f1", {}),
            "ensemble_per_class_auc" : ensemble_metrics.get("per_class_auc", {}),
            "ensemble_per_class_ap"  : ensemble_metrics.get("per_class_ap", {}),
            "ensemble_per_class_sens": ensemble_metrics.get("per_class_sensitivity", {}),
            "ensemble_per_class_spec": ensemble_metrics.get("per_class_specificity", {}),
            "ensemble_confusion_matrix": ensemble_metrics.get("confusion_matrix", []),
            "fold_details"           : fold_results,
            "laot_qwk"               : 0.8296,
            "beats_laot"             : bool(best_qwk > 0.8296),
        }

        out_path = f"{config.RESULTS_DIR}/ensemble_results.json"
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n  Results saved to {out_path}")

        # ── classification report ─────────
        grades = [config.DR_GRADES[i] for i in range(config.NUM_CLASSES)]
        print("\n  ── SKLEARN CLASSIFICATION REPORT ────────────────")
        print(classification_report(
            ensemble_targets, best_preds,
            target_names=grades, digits=4
        ))

    else:
        print("\n  Only 1 fold available — no ensemble")


if __name__ == "__main__":
    main()