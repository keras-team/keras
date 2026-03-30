import os

os.environ["KERAS_BACKEND"] = "torch"
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

with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU],
) as prof:
    for _ in range(500):
        cnn(x)

print(
    prof.key_averages().table(
        sort_by="cpu_time_total", row_limit=30, max_name_column_width=60
    )
)
