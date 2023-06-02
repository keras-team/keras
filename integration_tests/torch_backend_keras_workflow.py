import numpy as np

import keras_core
from keras_core import layers

model = keras_core.Sequential(
    [
        layers.Dense(1),
    ]
)
model.compile(loss="mse", optimizer="adam", metrics=["mae"])
history = model.fit(
    x=np.random.rand(100, 10),
    y=np.random.rand(100, 1),
    epochs=10,
    shuffle=False,
)
