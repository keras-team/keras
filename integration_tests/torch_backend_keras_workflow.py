import numpy as np

import keras_core
from keras_core import layers

model = keras_core.Sequential(
    [
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["mae", "accuracy"]
)
x = np.random.rand(100, 10)
y = np.random.randint(0, 2, size=(100, 1))
history = model.fit(
    x=x,
    y=y,
    epochs=10,
    shuffle=False,
    validation_data=(x, y),
)
model.predict(x)
