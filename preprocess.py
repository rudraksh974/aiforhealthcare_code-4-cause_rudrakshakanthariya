import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import pandas as pd
import numpy as np
import pydicom
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split

METADATA_FILE = "MRI_metadata.csv" 
DATA_DIR = "MRI"
LABEL_MAP = {'CN': 0, 'MCI': 1, 'AD': 2}

def preprocess_image(image_array):
    img_min = np.min(image_array)
    img_max = np.max(image_array)
    img = (image_array - img_min) / (img_max - img_min + 1e-8)
    
    img = tf.image.resize(img[..., np.newaxis], (224, 224))
    img = tf.image.grayscale_to_rgb(img)
    
    return img.numpy()

def load_and_preprocess_all(base_path, csv_path):
    df = pd.read_csv(csv_path)
    df['Subject'] = df['Subject'].astype(str)
    
    data = []
    labels = []

    subject_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]

    for subject_id in subject_folders:
        match = df[df['Subject'] == subject_id]
        
        if not match.empty:
            group = match.iloc[0]['Group']
            
            if group in LABEL_MAP:
                subject_path = os.path.join(base_path, subject_id)
                for root, _, files in os.walk(subject_path):
                    for file in files:
                        if file.endswith('.dcm'):
                            file_path = os.path.join(root, file)
                            try:
                                dicom = pydicom.dcmread(file_path)
                                processed_img = preprocess_image(dicom.pixel_array)
                                data.append(processed_img)
                                labels.append(LABEL_MAP[group])
                            except Exception:
                                continue
    
    return np.array(data), np.array(labels)

if __name__ == "__main__":
    X, y = load_and_preprocess_all(DATA_DIR, METADATA_FILE)
    
    if len(X) > 0:
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
        )

        datasets = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }

        for name, arr in datasets.items():
            np.save(f'{name}.npy', arr)