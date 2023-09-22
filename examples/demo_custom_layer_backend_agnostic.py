import numpy as np

import keras
from keras import Model
from keras import initializers
from keras import layers
from keras import losses
from keras import metrics
from keras import ops
from keras import optimizers


class MyDense(layers.Layer):
    def __init__(self, units, name=None):
        super().__init__(name=name)
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.w = self.add_weight(
            shape=(input_dim, self.units),
            initializer=initializers.GlorotNormal(),
            name="kernel",
            trainable=True,
        )

        self.b = self.add_weight(
            shape=(self.units,),
            initializer=initializers.Zeros(),
            name="bias",
            trainable=True,
        )

    def call(self, inputs):
        # Use Keras ops to create backend-agnostic layers/metrics/etc.
        return ops.matmul(inputs, self.w) + self.b


class MyDropout(layers.Layer):
    def __init__(self, rate, name=None):
        super().__init__(name=name)
        self.rate = rate
        # Use seed_generator for managing RNG state.
        # It is a state element and its seed variable is
        # tracked as part of `layer.variables`.
        self.seed_generator = keras.random.SeedGenerator(1337)

    def call(self, inputs):
        # Use `keras.random` for random ops.
        return keras.random.dropout(
            inputs, self.rate, seed=self.seed_generator
        )


class MyModel(Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.dense1 = MyDense(hidden_dim)
        self.dense2 = MyDense(hidden_dim)
        self.dense3 = MyDense(output_dim)
        self.dp = MyDropout(0.5)

    def call(self, x):
        x1 = self.dense1(x)
        x2 = self.dense2(x)
        # Why not use some ops here as well
        x = ops.concatenate([x1, x2], axis=-1)
        x = self.dp(x)
        return self.dense3(x)


model = MyModel(hidden_dim=256, output_dim=16)

x = np.random.random((50000, 128))
y = np.random.random((50000, 16))
batch_size = 32
epochs = 5

model.compile(
    optimizer=optimizers.SGD(learning_rate=0.001),
    loss=losses.MeanSquaredError(),
    metrics=[metrics.MeanSquaredError()],
)
history = model.fit(x, y, batch_size=batch_size, epochs=epochs)

model.summary()

print("History:")
print(history.history)
