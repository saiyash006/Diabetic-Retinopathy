import pandas as pd
import hashlib
import os

def file_hash(path, chunk_size=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

master = pd.read_csv("data/splits/master.csv")
print(f"Before dedup : {len(master)} images")

# Hash (already done above — reuse if still in memory)
master["img_hash"] = master["image_path"].apply(file_hash)

# Show what gets removed
dupes = master[master.duplicated(subset="img_hash", keep=False)]
print(f"\nDuplicate pairs by source:")
print(dupes.groupby("source")["img_hash"].count())

# Keep first occurrence only
master_clean = master.drop_duplicates(subset="img_hash", keep="first")
print(f"\nAfter dedup  : {len(master_clean)} images")
print(f"Removed      : {len(master) - len(master_clean)} images")

# Save clean master
master_clean.to_csv("data/splits/master.csv", index=False)
print("master.csv updated!")

# Regenerate all 5 fold splits from clean master
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np

SPLITS_DIR = "data/splits"
NUM_FOLDS  = 5

patient_df = master_clean.drop_duplicates(
    subset="patient_id"
)[["patient_id", "source", "label"]].copy()

skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
kf  = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

patient_df["fold"] = -1

for source in patient_df["source"].unique():
    src_mask = patient_df["source"] == source
    src_df   = patient_df[src_mask].copy()
    src_idx  = src_df.index.tolist()
    labels   = src_df["label"].values
    min_count = pd.Series(labels).value_counts().min()

    if min_count >= NUM_FOLDS:
        splits = list(skf.split(np.arange(len(src_df)), labels))
    else:
        splits = list(kf.split(np.arange(len(src_df))))

    for fold_idx, (_, val_idx) in enumerate(splits):
        actual_idx = [src_idx[i] for i in val_idx]
        patient_df.loc[actual_idx, "fold"] = fold_idx

for fold in range(NUM_FOLDS):
    val_pids   = set(patient_df[patient_df["fold"] == fold]["patient_id"])
    train_pids = set(patient_df[patient_df["fold"] != fold]["patient_id"])

    assert len(train_pids & val_pids) == 0, f"Fold {fold+1}: leakage!"

    train_df = master_clean[master_clean["patient_id"].isin(train_pids)].copy()
    val_df   = master_clean[master_clean["patient_id"].isin(val_pids)].copy()

    train_df.to_csv(f"{SPLITS_DIR}/fold{fold+1}_train.csv", index=False)
    val_df.to_csv(f"{SPLITS_DIR}/fold{fold+1}_val.csv",   index=False)

    print(f"Fold {fold+1}: train={len(train_df)}, val={len(val_df)} — leakage free")

print("\nAll 5 clean folds saved!")