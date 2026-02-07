import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pandas as pd
import numpy as np
import pydicom
import tensorflow as tf
from sklearn.model_selection import train_test_split

METADATA_FILE = "MRI_metadata.csv"
DATA_DIR = "MRI"
LABEL_MAP = {'CN': 0, 'MCI': 1, 'AD': 2}

# Force float16 globally
DTYPE = np.float16


def preprocess_image(image_array):
    image_array = image_array.astype(np.float32)

    img_min = np.min(image_array)
    img_max = np.max(image_array)

    if img_max - img_min == 0:
        return None

    img = (image_array - img_min) / (img_max - img_min)

    img = tf.image.resize(img[..., np.newaxis], (224, 224))
    img = tf.image.grayscale_to_rgb(img)

    return img.numpy().astype(DTYPE)


def load_and_preprocess_all(base_path, csv_path):
    print("Loading metadata...")
    df = pd.read_csv(csv_path)
    df['Subject'] = df['Subject'].astype(str)

    data = []
    labels = []

    subject_folders = [
        f for f in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, f))
    ]

    total_files = 0

    for subject_id in subject_folders:

        match = df[df['Subject'] == subject_id]

        if match.empty:
            continue

        group = match.iloc[0]['Group']

        if group not in LABEL_MAP:
            continue

        subject_path = os.path.join(base_path, subject_id)

        for root, _, files in os.walk(subject_path):

            for file in files:

                if not file.endswith('.dcm'):
                    continue

                file_path = os.path.join(root, file)

                try:
                    dicom = pydicom.dcmread(file_path)

                    processed_img = preprocess_image(dicom.pixel_array)

                    if processed_img is None:
                        continue

                    data.append(processed_img)
                    labels.append(LABEL_MAP[group])

                    total_files += 1

                    if total_files % 500 == 0:
                        print(f"Processed {total_files} images")

                except Exception:
                    continue

    print(f"Total processed images: {total_files}")

    return np.array(data, dtype=DTYPE), np.array(labels, dtype=np.int8)


def save_compressed(name, arr):
    print(f"Saving {name} shape={arr.shape} dtype={arr.dtype}")
    np.savez_compressed(f"{name}.npz", data=arr)
    print(f"Saved {name}.npz")


if __name__ == "__main__":

    print("Starting preprocessing...")

    X, y = load_and_preprocess_all(DATA_DIR, METADATA_FILE)

    if len(X) == 0:
        print("No data found.")
        exit()

    print("Splitting dataset...")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=0.15,
        random_state=42,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=0.176,
        random_state=42,
        stratify=y_train_val
    )

    print("Saving datasets...")

    save_compressed("X_train", X_train)
    save_compressed("y_train", y_train)

    save_compressed("X_val", X_val)
    save_compressed("y_val", y_val)

    save_compressed("X_test", X_test)
    save_compressed("y_test", y_test)

    print("Done.")
