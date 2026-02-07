import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report

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

if X_train.ndim == 3:
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

if X_train.shape[-1] == 1:
    print("Converting 1-channel Grayscale to 3-channel RGB...")
    X_train = np.repeat(X_train, 3, axis=-1)
    X_test = np.repeat(X_test, 3, axis=-1)

print(f"Final Data Shape: {X_train.shape}") 

def build_model():
    inputs = Input(shape=(224, 224, 3), name="input_layer")

    base_model = EfficientNetV2B0(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs 
    )

    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="swish")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="swish")(x)
    
    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

print("\nBuilding model...")
model = build_model()
model.summary()

train_datagen = ImageDataGenerator(
    rotation_range=10, zoom_range=0.1, horizontal_flip=True
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3)
]

print("\nStarting Training...")
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_test, y_test),
    epochs=10,
    callbacks=callbacks
)

print("\nEvaluating...")
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob > 0.5).astype(int)

print("\nBalanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_prob))

print("\nClassification Report:\n", 
      classification_report(y_test, y_pred, target_names=["CN", "AD"]))

print("\nTraining Complete.")