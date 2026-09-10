import os
import re
import glob
import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

# ─────────────────────────────────────────
# Paths
# Messidor REMOVED — label mismatch risk
# ─────────────────────────────────────────
PREPROCESSED = {
    "eyespacs" : "data/preprocessed/eyespacs/data",
    "archive"  : "data/preprocessed/archive/train_images",
    "idrid"    : "data/preprocessed/IDRID/Imagenes",
    "ddr"      : "data/preprocessed/ddr/DR_grading",
}

RAW_CSV = {
    "eyespacs" : "data/raw/eyespacs/trainLabels.csv",
    "archive"  : "data/raw/archive/archive_all.csv",
    "idrid"    : "data/raw/idrid/idrid_labels.csv",
    "ddr"      : "data/raw/ddr/DR_grading.csv",
}

SPLITS_DIR = "data/splits"
NUM_FOLDS  = 5

os.makedirs(SPLITS_DIR, exist_ok=True)


# ─────────────────────────────────────────
# Image hash for deduplication
# ─────────────────────────────────────────
def file_hash(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────
# Patient ID extractors
# ─────────────────────────────────────────
def get_patient_id(filename, source):
    stem = os.path.splitext(os.path.basename(filename))[0]

    if source == "eyespacs":
        return "eyespacs_" + stem.split("_")[0]

    elif source == "archive":
        return "archive_" + stem[:12]

    elif source == "idrid":
        cleaned = re.sub(
            r"_(left|right)$", "", stem, flags=re.IGNORECASE
        )
        return "idrid_" + cleaned

    elif source == "ddr":
        return "ddr_" + stem

    return stem


# ─────────────────────────────────────────
# Helper — find image file
# ─────────────────────────────────────────
def find_image(folder, stem):
    base = os.path.splitext(stem)[0]

    for ext in [".png", ".jpg", ".jpeg"]:
        for candidate in [base, stem]:
            path = os.path.join(folder, candidate + ext)
            if os.path.exists(path):
                return path
        path = os.path.join(folder, stem)
        if os.path.exists(path):
            return path

    for root, dirs, files in os.walk(folder):
        for f in files:
            f_base = os.path.splitext(f)[0]
            if f_base == base or f == stem:
                return os.path.join(root, f)

    return None


# ─────────────────────────────────────────
# Load eyespacs
# ─────────────────────────────────────────
def load_eyespacs():
    df   = pd.read_csv(RAW_CSV["eyespacs"])
    pre  = PREPROCESSED["eyespacs"]
    rows = []

    for _, row in df.iterrows():
        stem     = str(row["image"])
        label    = int(row["level"])
        img_path = find_image(pre, stem)
        if img_path is None:
            continue
        rows.append({
            "image_path" : img_path,
            "label"      : label,
            "source"     : "eyespacs",
            "patient_id" : get_patient_id(stem, "eyespacs"),
        })

    print(f"  eyespacs : {len(rows):>6} images loaded")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# Load archive
# ─────────────────────────────────────────
def load_archive():
    df   = pd.read_csv(RAW_CSV["archive"])
    pre  = PREPROCESSED["archive"]
    rows = []

    for _, row in df.iterrows():
        stem     = str(row["id_code"])
        label    = int(row["diagnosis"])
        img_path = find_image(pre, stem)
        if img_path is None:
            continue
        rows.append({
            "image_path" : img_path,
            "label"      : label,
            "source"     : "archive",
            "patient_id" : get_patient_id(stem, "archive"),
        })

    print(f"  archive  : {len(rows):>6} images loaded")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# Load IDRID
# ─────────────────────────────────────────
def load_idrid():
    df   = pd.read_csv(RAW_CSV["idrid"])
    pre  = PREPROCESSED["idrid"]
    rows = []

    for _, row in df.iterrows():
        stem     = str(row["id_code"])
        label    = int(row["diagnosis"])
        img_path = find_image(pre, stem)
        if img_path is None:
            continue
        rows.append({
            "image_path" : img_path,
            "label"      : label,
            "source"     : "idrid",
            "patient_id" : get_patient_id(stem, "idrid"),
        })

    print(f"  idrid    : {len(rows):>6} images loaded")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# Load DDR
# ─────────────────────────────────────────
def load_ddr():
    df   = pd.read_csv(RAW_CSV["ddr"])
    pre  = PREPROCESSED["ddr"]
    rows = []

    for _, row in df.iterrows():
        stem  = str(row["id_code"])
        label = int(row["diagnosis"])
        if label not in [0, 1, 2, 3, 4]:
            continue
        img_path = find_image(pre, stem)
        if img_path is None:
            continue
        rows.append({
            "image_path" : img_path,
            "label"      : label,
            "source"     : "ddr",
            "patient_id" : get_patient_id(stem, "ddr"),
        })

    print(f"  ddr      : {len(rows):>6} images loaded")
    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# Build master CSV
# Messidor excluded — 3-class label mismatch
# ─────────────────────────────────────────
def build_master():
    print("\n" + "="*55)
    print("  LOADING ALL DATASETS — OPTION B")
    print("  Messidor EXCLUDED (label mismatch)")
    print("="*55)

    loaders = [
        load_eyespacs,
        load_archive,
        load_idrid,
        load_ddr,
    ]

    dfs = []
    for loader in loaders:
        try:
            df = loader()
            if len(df) > 0:
                dfs.append(df)
            else:
                print(f"  WARNING: {loader.__name__} "
                      f"returned 0 images — check paths")
        except Exception as e:
            print(f"  ERROR in {loader.__name__}: {e}")

    if not dfs:
        raise ValueError("No datasets loaded — check paths")

    master = pd.concat(dfs, ignore_index=True)
    master["stratify_key"] = (
        master["label"].astype(str) + "_" + master["source"]
    )

    print(f"\n  {'─'*50}")
    print(f"  Total images (before dedup) : {len(master)}")

    # ─────────────────────────────────────────
    # Image-level hash deduplication
    # ─────────────────────────────────────────
    print(f"\n  Computing image hashes...")
    print(f"  (this takes 2-5 mins for ~50k images)")
    master["img_hash"] = master["image_path"].apply(file_hash)

    dupes = master[master.duplicated(subset="img_hash", keep=False)]
    if len(dupes) > 0:
        print(f"\n  WARNING: {len(dupes)} duplicate images found!")
        print(f"  Breakdown by source:")
        for src, cnt in dupes.groupby("source")["img_hash"].count().items():
            print(f"    {src:<12}: {cnt} duplicates")
        master = master.drop_duplicates(subset="img_hash", keep="first")
        print(f"  Removed duplicates. Clean images: {len(master)}")
    else:
        print(f"  No duplicate images found — clean!")

    print(f"\n  Source distribution:")
    for src, cnt in master["source"].value_counts().items():
        pct = cnt / len(master) * 100
        bar = "█" * int(pct / 2)
        print(f"    {src:<12}: {cnt:>6} ({pct:4.1f}%)  {bar}")

    print(f"\n  Label distribution:")
    lbls = {0:"No DR", 1:"Mild", 2:"Moderate", 3:"Severe", 4:"PDR"}
    for lbl, cnt in master["label"].value_counts().sort_index().items():
        pct = cnt / len(master) * 100
        bar = "█" * int(pct / 2)
        print(f"    {lbls[lbl]:<12}: {cnt:>6} ({pct:4.1f}%)  {bar}")

    print(f"\n  Unique patients per source:")
    for src in master["source"].unique():
        n = master[master["source"]==src]["patient_id"].nunique()
        print(f"    {src:<12}: {n:>6} unique patients")

    master.to_csv(f"{SPLITS_DIR}/master.csv", index=False)
    print(f"\n  master.csv saved — {len(master)} rows")
    return master


# ─────────────────────────────────────────
# Build leakage-free K-Fold splits
# ─────────────────────────────────────────
def build_splits(master):
    print("\n" + "="*55)
    print("  BUILDING LEAKAGE-FREE 5-FOLD SPLITS")
    print("="*55)

    # Remove old splits first
    old_files = glob.glob(f"{SPLITS_DIR}/fold*.csv")
    for f in old_files:
        os.remove(f)
    if old_files:
        print(f"  Removed {len(old_files)} old split files")

    patient_df = master.drop_duplicates(
        subset="patient_id"
    )[["patient_id", "source", "label"]].copy()

    print(f"  Total unique patients : {len(patient_df)}")
    print(f"  Splitting each source independently")

    skf = StratifiedKFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=42
    )
    kf  = KFold(
        n_splits=NUM_FOLDS,
        shuffle=True,
        random_state=42
    )

    patient_df["fold"] = -1

    for source in patient_df["source"].unique():
        src_mask = patient_df["source"] == source
        src_df   = patient_df[src_mask].copy()
        src_idx  = src_df.index.tolist()
        labels   = src_df["label"].values

        label_counts = pd.Series(labels).value_counts()
        min_count    = label_counts.min()

        if min_count >= NUM_FOLDS:
            splits = list(skf.split(
                np.arange(len(src_df)), labels
            ))
            split_type = "stratified"
        else:
            splits     = list(kf.split(np.arange(len(src_df))))
            split_type = "simple"

        for fold_idx, (_, val_idx) in enumerate(splits):
            actual_idx = [src_idx[i] for i in val_idx]
            patient_df.loc[actual_idx, "fold"] = fold_idx

        print(f"  {source:<12}: {len(src_df):>6} patients "
              f"— {split_type} split")

    unassigned = (patient_df["fold"] == -1).sum()
    assert unassigned == 0, \
        f"{unassigned} patients not assigned to any fold"

    for fold in range(NUM_FOLDS):
        val_pids   = set(
            patient_df[
                patient_df["fold"] == fold
            ]["patient_id"]
        )
        train_pids = set(
            patient_df[
                patient_df["fold"] != fold
            ]["patient_id"]
        )

        assert len(train_pids & val_pids) == 0, \
            f"Fold {fold+1}: patient leakage detected!"

        train_df = master[
            master["patient_id"].isin(train_pids)
        ].copy()
        val_df   = master[
            master["patient_id"].isin(val_pids)
        ].copy()

        train_df.to_csv(
            f"{SPLITS_DIR}/fold{fold+1}_train.csv", index=False)
        val_df.to_csv(
            f"{SPLITS_DIR}/fold{fold+1}_val.csv",   index=False)

        total = len(train_df) + len(val_df)
        print(f"\n  Fold {fold+1}:")
        print(f"    Train : {len(train_df):>6} "
              f"({len(train_df)/total*100:.1f}%) "
              f"| Val : {len(val_df):>6} "
              f"({len(val_df)/total*100:.1f}%)")
        print(f"    Train sources : "
              f"{train_df['source'].value_counts().to_dict()}")
        print(f"    Val   sources : "
              f"{val_df['source'].value_counts().to_dict()}")
        print(f"    Train labels  : "
              f"{train_df['label'].value_counts().sort_index().to_dict()}")
        print(f"    Val   labels  : "
              f"{val_df['label'].value_counts().sort_index().to_dict()}")
        print(f"    Train patients: {len(train_pids):>6} "
              f"| Val patients: {len(val_pids):>6}")
        print(f"    ✅ Zero patient leakage confirmed")

    print(f"\n  All {NUM_FOLDS} folds saved to {SPLITS_DIR}/")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
if __name__ == "__main__":
    master = build_master()
    build_splits(master)

    print("\n" + "="*55)
    print("  DONE — ready to train Option B")
    print("="*55)
    print("  Run: python train.py")