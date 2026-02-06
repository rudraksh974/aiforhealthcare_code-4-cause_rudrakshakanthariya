# Brain MRI Preprocessing Pipeline for Alzheimer’s Disease Classification

## Project Overview

This project implements a complete preprocessing pipeline for brain MRI scans as part of an Alzheimer’s Disease classification hackathon.

The objective is to convert raw MRI DICOM scans into clean, structured, and model-ready datasets suitable for deep learning architectures such as VGG16, ResNet, and other CNN-based models.

This pipeline ensures that MRI data is properly matched with clinical metadata, transformed into consistent image format, labeled correctly, and split into training, validation, and testing sets.

---

## Features

The preprocessing pipeline performs the following operations:

- Reads raw MRI DICOM files  
- Matches MRI scans with clinical metadata  
- Converts MRI slices into CNN-compatible format  
- Encodes diagnosis labels  
- Performs stratified train/validation/test split  
- Saves processed datasets for reproducible model training  

---

## Dataset Structure

After cloning the repository, the dataset must be reconstructed from split parts.

```
mri-project/
│
├── preprocess.py
├── requirements.txt
├── reconstruct_dataset.sh
├── README.md
│
└── dataset/
    ├── X_train_part_aa
    ├── X_train_part_ab
    ├── ...
    ├── y_train_part_aa
    ├── ...
```

---

## Setup Instructions

### Step 1: Clone repository

```
git clone <your-repository-url>
cd mri-project
```

### Step 2: Install dependencies

```
pip install -r requirements.txt
```

### Step 3: Reconstruct dataset

```
chmod +x reconstruct_dataset.sh
./reconstruct_dataset.sh
```

This will generate:

```
dataset/X_train.npy
dataset/X_val.npy
dataset/X_test.npy
dataset/y_train.npy
dataset/y_val.npy
dataset/y_test.npy
```

---

## Metadata Description

The metadata file (`MRI_metadata.csv`) contains:

| Column | Description |
|------|-------------|
| Subject | Unique subject ID matching MRI folder name |
| Group | Diagnosis label |

Diagnosis classes:

- CN → Cognitively Normal  
- MCI → Mild Cognitive Impairment  
- AD → Alzheimer’s Disease  

---

## Preprocessing Steps

### 1. DICOM Image Loading

- Reads `.dcm` files using pydicom  
- Extracts pixel arrays from MRI slices  

### 2. Image Transformation

Each MRI slice is:

- Resized to 224 × 224  
- Converted from grayscale to RGB  
- Normalized using VGG16 preprocessing  
- Converted to NumPy array format  

This ensures compatibility with pretrained CNN architectures.

---

### 3. Metadata Matching

- MRI scans are matched with metadata entries  
- Only valid labeled subjects are included  
- Ensures accurate labeling and dataset integrity  

---

### 4. Dataset Creation

- Feature array stored in:

```
X → MRI image tensors
```

- Label array stored in:

```
y → diagnosis labels
```

---

### 5. Stratified Data Splitting

Dataset split using stratified sampling:

| Split | Percentage |
|------|------------|
| Training | 70% |
| Validation | 15% |
| Testing | 15% |

Ensures balanced class distribution.

---

### 6. Saving Processed Dataset

Processed datasets saved as:

```
dataset/X_train.npy
dataset/X_val.npy
dataset/X_test.npy

dataset/y_train.npy
dataset/y_val.npy
dataset/y_test.npy
```

These can be directly used for deep learning model training.

---

## Running the Preprocessing Pipeline

```
python preprocess.py
```

---

## Libraries Used

- Python  
- NumPy  
- Pandas  
- TensorFlow / Keras  
- pydicom  
- scikit-learn  
- tqdm  

---

## Current Status

- MRI DICOM scans successfully processed  
- Images converted to CNN-compatible format  
- Labels encoded and validated  
- Stratified train / validation / test splits created  
- Dataset saved in reusable NumPy format  
- Fully reproducible pipeline  

---

## Reproducibility

Anyone can reproduce the dataset using:

```
git clone <repo>
cd mri-project
pip install -r requirements.txt
./reconstruct_dataset.sh
python preprocess.py
```

---

## Hackathon Compliance

This repository contains:

- Full preprocessing pipeline  
- Dataset reconstruction script  
- Requirements file for environment setup  
- Reproducible dataset structure  

Fully compliant with hackathon submission requirements.
