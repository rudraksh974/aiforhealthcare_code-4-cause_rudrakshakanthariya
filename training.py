import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight  # 🔴 CHANGED: for class imbalance

tf.keras.backend.clear_session()
print("TensorFlow Version:", tf.__version__)

print("\nLoading data...")
X_train = np.load("X_train.npz")["data"].astype(np.float32)
y_train = np.load("y_train.npz")["data"]

X_test = np.load("X_test.npz")["data"].astype(np.float32)
y_test = np.load("y_test.npz")["data"]

# Filter CN vs AD
train_mask = (y_train == 0) | (y_train == 2)
test_mask = (y_test == 0) | (y_test == 2)

X_train = X_train[train_mask]
y_train = y_train[train_mask]
X_test = X_test[test_mask]
y_test = y_test[test_mask]

y_train = (y_train == 2).astype(np.float32)
y_test = (y_test == 2).astype(np.float32)

# 🔴 CHANGED: Multi-slice handling (each slice becomes one sample)
if X_train.ndim == 4:  # (subjects, slices, H, W)
    X_train = X_train.reshape(-1, 224, 224)
    y_train = np.repeat(y_train, X_train.shape[0] // len(y_train))

    X_test = X_test.reshape(-1, 224, 224)
    y_test = np.repeat(y_test, X_test.shape[0] // len(y_test))

# Expand dims if grayscale
if X_train.ndim == 3:
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

# Convert grayscale → RGB
if X_train.shape[-1] == 1:
    print("Converting 1-channel Grayscale to 3-channel RGB...")
    X_train = np.repeat(X_train, 3, axis=-1)
    X_test = np.repeat(X_test, 3, axis=-1)

print(f"Final Training Shape: {X_train.shape}")

# 🔴 CHANGED: Compute class weights (important for CN vs AD imbalance)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Class Weights:", class_weights)


def build_model():
    inputs = Input(shape=(224, 224, 3), name="input_layer")

    base_model = EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs
    )

    base_model.trainable = False  # initial freezing (correct)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="swish")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="swish")(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    # 🔴 CHANGED: Lower LR for stability
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


print("\nBuilding model...")
model = build_model()
model.summary()

train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3)
]

# ===============================
# PHASE 1: FEATURE EXTRACTION
# ===============================
print("\nStarting Phase 1 Training...")
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=16),
    validation_data=(X_test, y_test),
    epochs=1,
    class_weight=class_weights,  # CHANGED
    callbacks=callbacks
)

# ===============================
# PHASE 2: FINE-TUNING
# ===============================
print("\nUnfreezing top layers for fine-tuning...")

for layer in model.layers[-30:]:  # CHANGED: unfreeze top layers
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # 🔴 CHANGED: lower LR
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
)

history_ft = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=8),
    validation_data=(X_test, y_test),
    epochs=1,
    class_weight=class_weights,
    callbacks=callbacks
)

# ===============================
# EVALUATION
# ===============================
print("\nEvaluating...")
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob > 0.5).astype(int)

print("\nBalanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=["CN", "AD"]))

print("\nTraining Complete.")