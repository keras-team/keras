import os

os.environ["KERAS_BACKEND"] = "torch"
import cProfile
import pstats

import torch

import keras
from keras import layers

cnn = keras.Sequential(
    [
        keras.Input(shape=(3, 32, 32)),
        layers.Conv2D(
            64,
            3,
            padding="same",
            activation="relu",
            data_format="channels_first",
        ),
        layers.Conv2D(
            64,
            3,
            padding="same",
            activation="relu",
            data_format="channels_first",
        ),
        layers.MaxPooling2D(data_format="channels_first"),
        layers.Conv2D(
            128,
            3,
            padding="same",
            activation="relu",
            data_format="channels_first",
        ),
        layers.GlobalAveragePooling2D(data_format="channels_first"),
        layers.Dense(10),
    ]
)

x = torch.randn(4, 3, 32, 32, device="mps")
for _ in range(5):
    cnn(x)

pr = cProfile.Profile()
pr.enable()
for _ in range(200):
    cnn(x)
pr.disable()

p = pstats.Stats(pr)
p.strip_dirs().sort_stats("cumtime").print_stats(30)
