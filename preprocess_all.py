import os
import cv2
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

RAW_DIR = "data/raw"
OUT_DIR = "data/preprocessed"

IMAGE_SIZE = 384

VALID_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def detect_device():
    """
    Detect GPU availability (for information only).
    Preprocessing itself runs on CPU.
    """
    if torch.cuda.is_available():
        print("GPU detected:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")


def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return img


def ben_graham(img, sigma=10):
    return cv2.addWeighted(
        img,
        4,
        cv2.GaussianBlur(img, (0, 0), sigma),
        -4,
        128
    )


def preprocess_image(args):
    img_path, input_dir, output_dir = args

    img = cv2.imread(img_path)

    if img is None:
        return

    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    img = apply_clahe(img)
    img = ben_graham(img)

    rel = os.path.relpath(img_path, input_dir)

    out_path = os.path.join(output_dir, rel)

    out_path = os.path.splitext(out_path)[0] + ".png"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    cv2.imwrite(out_path, img)


def preprocess_dataset(dataset):

    input_dir = os.path.join(RAW_DIR, dataset)
    output_dir = os.path.join(OUT_DIR, dataset)

    image_paths = []

    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                image_paths.append(os.path.join(root, f))

    print(f"\nProcessing dataset: {dataset}")
    print(f"Images found: {len(image_paths)}")

    tasks = [(img_path, input_dir, output_dir) for img_path in image_paths]

    workers = max(1, cpu_count() - 1)

    with Pool(workers) as pool:
        list(tqdm(pool.imap(preprocess_image, tasks), total=len(tasks)))


def main():

    print("Starting preprocessing...\n")

    detect_device()

    datasets = os.listdir(RAW_DIR)

    print("\nDatasets detected:")
    for d in datasets:
        print("-", d)

    for dataset in datasets:
        preprocess_dataset(dataset)

    print("\nPreprocessing finished")
    print("Output saved in:", OUT_DIR)


if __name__ == "__main__":
    main()