import numpy as np

import keras_core
from keras_core import Model
from keras_core import backend
from keras_core import initializers
from keras_core import layers
from keras_core import losses
from keras_core import metrics
from keras_core import operations as ops
from keras_core import optimizers


class MyDense(layers.Layer):
    def __init__(self, units, name=None):
        super().__init__(name=name)
        self.units = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        w_shape = (input_dim, self.units)
        w_value = initializers.GlorotUniform()(w_shape)
        # State must be stored in backend.Variable objects.
        self.w = backend.Variable(w_value, name="kernel", trainable=True)

        # You can also use add_weight
        self.b = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
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
        self.seed_generator = keras_core.random.SeedGenerator(1337)

    def call(self, inputs):
        # Use `keras_core.random` for random ops.
        return keras_core.random.dropout(
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
