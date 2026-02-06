<!-- Project Overview

This project focuses on preprocessing brain MRI scans for an Alzheimer’s Disease classification task as part of a hackathon.
The goal of this stage is to convert raw DICOM MRI images into clean, model-ready datasets that can be directly used for deep learning models such as VGG16.

The pipeline handles:

Reading raw MRI DICOM files

Matching MRI scans with clinical metadata

Image preprocessing for CNN compatibility

Label encoding

Train / Validation / Test splitting

Saving processed datasets for future model training


 Metadata Description

The metadata CSV file (MRI_metadata.csv) contains:

Subject → Unique subject ID (matches MRI folder name)

Group → Diagnosis label

CN → Cognitively Normal

MCI → Mild Cognitive Impairment

AD → Alzheimer’s Disease




Preprocessing Steps Performed
 DICOM Image Loading

All .dcm files are recursively loaded using pydicom

Pixel arrays are extracted from raw MRI scans

 Image Transformation

Each MRI slice undergoes the following transformations:

Resized to 224 × 224

Converted from grayscale to RGB

Normalized using VGG16 preprocessing

Converted into NumPy arrays

This ensures compatibility with pretrained CNN architectures.

 Data–Metadata Matching

MRI folders are cross-checked with the metadata CSV

Only valid subjects with known diagnosis labels are processed

Ensures data integrity and correct labeling

 Dataset Creation

Preprocessed images stored in feature array X

Corresponding diagnosis labels stored in y

 Stratified Data Splitting

To maintain class balance, the dataset is split using stratified sampling:

Split	Percentage
Training	70%
Validation	15%
Testing	15%
 Saving Processed Data

Final datasets are saved as .npy files:

X_train.npy, y_train.npy

X_val.npy, y_val.npy

X_test.npy, y_test.npy

These files can be directly loaded during model training.


 Libraries Used

Python

NumPy

Pandas

TensorFlow / Keras

pydicom

scikit-learn

 Current Status

 Raw MRI DICOM data successfully processed
 Images converted to CNN-ready format
 Labels encoded and validated
 Train / Validation / Test datasets created
 Data saved for future modeling -->
