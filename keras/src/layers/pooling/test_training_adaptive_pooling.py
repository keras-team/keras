import os

os.environ["KERAS_BACKEND"] = "torch"  # Force Torch backend
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time

import numpy as np
import torch

import keras
from keras.src import backend as K
from keras.src import layers
from keras.src import models

# Skip if not Torch
if K.backend() != "torch":
    print(f"⚠️ Skipping: Torch backend required, current backend={K.backend()}")
    exit(0)

print("=" * 80)
print("🚀 Torch GPU Adaptive Pooling Training Test")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Running on: {device.upper()}")
if device == "cuda":
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
print(f"🔧 Backend: {K.backend()}")
print(f"📦 Keras Version: {keras.__version__}")
print(f"🧠 Torch Version: {torch.__version__}")

# Data in channels-first format
np.random.seed(42)
x_train = np.random.randn(1000, 3, 32, 32).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)
x_val = np.random.randn(200, 3, 32, 32).astype(np.float32)
y_val = np.random.randint(0, 10, 200)


def make_model(pool_type="avg"):
    pool_layer = (
        layers.AdaptiveAveragePooling2D((4, 4), data_format="channels_first")
        if pool_type == "avg"
        else layers.AdaptiveMaxPooling2D((4, 4), data_format="channels_first")
    )
    return models.Sequential(
        [
            layers.Input(shape=(3, 32, 32)),
            layers.Conv2D(
                32,
                3,
                activation="relu",
                padding="same",
                data_format="channels_first",
            ),
            layers.BatchNormalization(axis=1),
            layers.Conv2D(
                64,
                3,
                activation="relu",
                padding="same",
                data_format="channels_first",
            ),
            pool_layer,
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )


for pool in ["avg", "max"]:
    print("\n" + "=" * 80)
    print(f"🔹 Training Model with Adaptive{pool.capitalize()}Pooling2D")
    print("=" * 80)

    model = make_model(pool)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("\n🧠 Model Summary:")
    model.summary()

    start = time.time()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=3,
        batch_size=32,
        verbose=2,
    )
    elapsed = time.time() - start

    print(f"\n✅ {pool.capitalize()}Pooling2D Training Done")
    print(f"⏱️  Training time: {elapsed:.2f}s")
    print(f"📈 Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    print(
        f"📊 Final validation accuracy: "
        f"{history.history['val_accuracy'][-1]:.4f}"
    )

    test_input = np.random.randn(1, 3, 32, 32).astype(np.float32)
    preds = model.predict(test_input, verbose=0)
    print(f"✓ Inference OK - Output shape: {preds.shape}")

print("\n" + "=" * 80)
print("🏁 All Adaptive Pooling Tests Completed Successfully on Torch GPU")
print("=" * 80)
