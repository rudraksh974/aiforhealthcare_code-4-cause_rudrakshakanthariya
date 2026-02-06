#!/bin/bash

set -e

echo "Reconstructing dataset..."

cd dataset

cat X_train_part_* > X_train.npy
cat X_val_part_* > X_val.npy
cat X_test_part_* > X_test.npy

cat y_train_part_* > y_train.npy
cat y_val_part_* > y_val.npy
cat y_test_part_* > y_test.npy

echo "Dataset reconstructed successfully."
