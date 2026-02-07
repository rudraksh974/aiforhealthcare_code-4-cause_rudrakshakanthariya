import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, balanced_accuracy_score, f1_score


# =========================
# GPU SAFE MODE
# =========================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))


# =========================
# LOAD NPZ SAFE
# =========================

def load_npz(path):

    file = np.load(path, allow_pickle=True)
    data = file[file.files[0]]

    print(path, data.shape)

    return data


print("\nLoading datasets...")

X_train = load_npz("X_train.npz")
y_train = load_npz("y_train.npz")

X_val = load_npz("X_val.npz")
y_val = load_npz("y_val.npz")

X_test = load_npz("X_test.npz")
y_test = load_npz("y_test.npz")


# convert labels
y_train = y_train.astype(np.int32)
y_val = y_val.astype(np.int32)
y_test = y_test.astype(np.int32)


# one hot
y_train_cat = tf.keras.utils.to_categorical(y_train, 3)
y_val_cat = tf.keras.utils.to_categorical(y_val, 3)
y_test_cat = tf.keras.utils.to_categorical(y_test, 3)


# =========================
# CLASS WEIGHTS
# =========================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(weights))

print("\nClass weights:", class_weights)


# =========================
# GENERATOR
# =========================

def generator(X, y_cat, y_labels, batch):

    n = len(X)

    while True:

        idx = np.random.randint(0, n, batch)

        batch_x = X[idx].astype("float32") / 255.0
        batch_y = y_cat[idx]

        sample_weights = np.array(
            [class_weights[label] for label in y_labels[idx]]
        )

        yield batch_x, batch_y, sample_weights


# =========================
# BUILD CONVNEXT MODEL
# =========================

def build_model():

    base = ConvNeXtBase(

        weights="imagenet",
        include_top=False,
        input_shape=(224,224,3)

    )

    # freeze initially
    base.trainable = False


    x = base.output

    x = GlobalAveragePooling2D()(x)

    x = BatchNormalization()(x)

    x = Dense(1024, activation="relu")(x)
    x = Dropout(0.5)(x)

    x = Dense(512, activation="relu")(x)
    x = Dropout(0.4)(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.3)(x)

    output = Dense(3, activation="softmax")(x)


    model = Model(base.input, output)


    model.compile(

        optimizer=tf.keras.optimizers.Adam(1e-3),

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )


    return model


model = build_model()

model.summary()


# =========================
# TRAIN PHASE 1
# =========================

BATCH = 16

train_gen = generator(X_train, y_train_cat, y_train, BATCH)
val_gen = generator(X_val, y_val_cat, y_val, BATCH)


callbacks = [

    tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        patience=2,
        factor=0.2
    )

]


print("\nTraining Phase 1...")

model.fit(

    train_gen,

    validation_data=val_gen,

    steps_per_epoch=len(X_train)//BATCH,

    validation_steps=len(X_val)//BATCH,

    epochs=10,

    callbacks=callbacks

)


# =========================
# FINE TUNE
# =========================

print("\nFine tuning...")

base = model.layers[1]

# unfreeze last 50% layers
for layer in base._layers[-150:]:
    layer.trainable = True


model.compile(

    optimizer=tf.keras.optimizers.Adam(1e-5),

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)


model.fit(

    train_gen,

    validation_data=val_gen,

    steps_per_epoch=len(X_train)//BATCH,

    validation_steps=len(X_val)//BATCH,

    epochs=10,

    callbacks=callbacks

)


# =========================
# EVALUATE
# =========================

print("\nEvaluating...")

test_gen = generator(X_test, y_test_cat, y_test, BATCH)

y_prob = model.predict(
    test_gen,
    steps=len(X_test)//BATCH
)

y_pred = np.argmax(y_prob, axis=1)


print("\nBalanced Accuracy:",
      balanced_accuracy_score(y_test[:len(y_pred)], y_pred))

print("\nF1 Score:",
      f1_score(y_test[:len(y_pred)], y_pred, average="macro"))

print("\nClassification Report:\n")

print(classification_report(
    y_test[:len(y_pred)],
    y_pred
))


# =========================
# SAVE
# =========================

model.save("task_3.h5")

print("\nDONE — ConvNeXt model saved.")
