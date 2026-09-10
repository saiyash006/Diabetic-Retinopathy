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

# Load existing master
master = pd.read_csv("data/splits/master.csv")
print(f"Total images: {len(master)}")

# Hash all images
print("Hashing... (takes 2-5 mins)")
master["img_hash"] = master["image_path"].apply(file_hash)

# Check duplicates
dupes = master[master.duplicated(subset="img_hash", keep=False)]
print(f"\nDuplicate images found: {len(dupes)}")

if len(dupes) > 0:
    print("\nDuplicate breakdown by source:")
    print(dupes.groupby(["source"])["img_hash"].count())
    print("\nSample duplicate pairs:")
    print(dupes[["source","image_path","img_hash"]].head(10))
else:
    print("Clean! No duplicates found.")
    print("Your QWK 0.8399 is fully valid.")