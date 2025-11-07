import os

os.environ["KERAS_BACKEND"] = "torch"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time

import numpy as np
import torch

import keras
from keras.src import layers
from keras.src import models

print("=" * 80)
print("🚀 Real GPU Training Test with Adaptive Pooling (Torch Backend)")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"💻 Running on: {device.upper()}")
if device == "cuda":
    print(f"🔥 GPU: {torch.cuda.get_device_name(0)}")
print(f"🔧 Backend: {keras.backend.backend()}")
print(f"📦 Keras Version: {keras.__version__}")
print(f"🧠 Torch Version: {torch.__version__}")

np.random.seed(42)
x_train = np.random.randn(1000, 32, 32, 3).astype(np.float32)
y_train = np.random.randint(0, 10, 1000)
x_val = np.random.randn(200, 32, 32, 3).astype(np.float32)
y_val = np.random.randint(0, 10, 200)


def make_model(pool_type="avg"):
    pool_layer = (
        layers.AdaptiveAveragePooling2D((4, 4))
        if pool_type == "avg"
        else layers.AdaptiveMaxPooling2D((4, 4))
    )
    return models.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(32, 3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.Conv2D(64, 3, activation="relu", padding="same"),
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
        "📊 Final validation accuracy: "
        f"{history.history['val_accuracy'][-1]:.4f}"
    )

    test_input = np.random.randn(1, 32, 32, 3).astype(np.float32)
    preds = model.predict(test_input, verbose=0)
    print(f"✓ Inference OK - Output shape: {preds.shape}")

print("\n" + "=" * 80)
print("🏁 All Adaptive Pooling Tests Completed Successfully on Torch GPU")
print("=" * 80)
