import numpy as np

import keras_core
from keras_core import layers
from keras_core import operations as ops

keras_core.utils.set_random_seed(1337)
x = np.random.rand(100, 32, 32, 3)
y = np.random.randint(0, 2, size=(100, 1))

# Test sequential model.
model = keras_core.Sequential(
    [
        layers.Conv2D(filters=10, kernel_size=3),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["mae", "accuracy"]
)
history = model.fit(
    x=x,
    y=y,
    epochs=10,
    validation_data=(x, y),
    verbose=0,
)
model.evaluate(x, y, verbose=0)
model.predict(x, verbose=0)

# Test on batch functions
model.train_on_batch(x, y)
model.test_on_batch(x, y)
model.predict_on_batch(x)

# Test functional model.
inputs = keras_core.Input(shape=(32, 32, 3))
outputs = layers.Conv2D(filters=10, kernel_size=3)(inputs)
outputs = layers.GlobalAveragePooling2D()(outputs)
outputs = layers.Dense(1)(outputs)
model = keras_core.Model(inputs, outputs)
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["mae", "accuracy"]
)
history = model.fit(
    x=x,
    y=y,
    epochs=10,
    validation_data=(x, y),
    verbose=0,
)
model.evaluate(x, y, verbose=0)
model.predict(x, verbose=0)


# Test custom layer
class Linear(layers.Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return ops.matmul(inputs, self.w) + self.b


inputs = keras_core.Input(shape=(32, 32, 3))
outputs = layers.Conv2D(filters=10, kernel_size=3)(inputs)
outputs = layers.GlobalAveragePooling2D()(outputs)
outputs = Linear(1)(outputs)
outputs = layers.Activation("sigmoid")(outputs)
model = keras_core.Model(inputs, outputs)
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["mae", "accuracy"]
)
history = model.fit(
    x=x,
    y=y,
    epochs=10,
    validation_data=(x, y),
    verbose=0,
)
model.evaluate(x, y, verbose=0)
model.predict(x, verbose=0)
