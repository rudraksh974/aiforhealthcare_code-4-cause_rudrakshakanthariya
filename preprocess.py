import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import pydicom
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tqdm import tqdm

METADATA_FILE = "MRI_metadata.csv"
DATA_DIR = "MRI"

LABEL_MAP = {'CN': 0, 'MCI': 1, 'AD': 2}

DTYPE = np.float16

def preprocess_image(image):

    image = image.astype(np.float32)

    if np.std(image) < 5:
        return None

    p1, p99 = np.percentile(image, (1, 99))

    if p99 - p1 == 0:
        return None

    image = np.clip(image, p1, p99)
    image = (image - p1) / (p99 - p1)

    image = image ** 0.8

    image = tf.image.resize(
        image[..., np.newaxis],
        (224, 224),
        method="bicubic"
    )

    image = tf.image.grayscale_to_rgb(image)


    return image.numpy().astype(DTYPE)

def load_and_preprocess_all(base_path, csv_path):

    print("Loading metadata...")

    df = pd.read_csv(csv_path)
    df['Subject'] = df['Subject'].astype(str)

    subject_to_label = dict(
        zip(df.Subject, df.Group)
    )

    data = []
    labels = []

    subjects = os.listdir(base_path)

    total = 0
    skipped = 0


    for subject in tqdm(subjects, desc="Subjects"):

        if subject not in subject_to_label:
            continue

        group = subject_to_label[subject]

        if group not in LABEL_MAP:
            continue

        label = LABEL_MAP[group]

        subject_path = os.path.join(base_path, subject)


        for root, _, files in os.walk(subject_path):

            for file in files:

                if not file.endswith(".dcm"):
                    continue

                path = os.path.join(root, file)

                try:

                    dicom = pydicom.dcmread(
                        path,
                        force=True
                    )

                    img = dicom.pixel_array

                    processed = preprocess_image(img)

                    if processed is None:
                        skipped += 1
                        continue

                    data.append(processed)
                    labels.append(label)

                    total += 1

                    if total % 1000 == 0:
                        print(f"Processed: {total}")

                except:
                    skipped += 1


    print("\nDone loading.")
    print("Total:", total)
    print("Skipped:", skipped)

    return (
        np.array(data, dtype=DTYPE),
        np.array(labels, dtype=np.int8)
    )

def save_compressed(name, arr):

    print(f"Saving {name} shape={arr.shape}")

    np.savez_compressed(
        f"{name}.npz",
        data=arr
    )

    print(f"Saved {name}.npz")

if __name__ == "__main__":

    print("Starting preprocessing pipeline...")

    X, y = load_and_preprocess_all(
        DATA_DIR,
        METADATA_FILE
    )

    if len(X) == 0:
        print("No data found.")
        exit()


    print("\nSplitting dataset...")


    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.15,
        stratify=y,
        random_state=42
    )


    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.176,
        stratify=y_train_val,
        random_state=42
    )


    print("\nSaving datasets...")


    save_compressed("X_train", X_train)
    save_compressed("y_train", y_train)

    save_compressed("X_val", X_val)
    save_compressed("y_val", y_val)

    save_compressed("X_test", X_test)
    save_compressed("y_test", y_test)


    print("\nPreprocessing COMPLETE.")
