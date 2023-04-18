import numpy as np

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
        self.w = backend.Variable(w_value, name="kernel")

        b_shape = (self.units,)
        b_value = initializers.Zeros()(b_shape)
        self.b = backend.Variable(b_value, name="bias")

    def call(self, inputs):
        # Use Keras ops to create backend-agnostic layers/metrics/etc.
        return ops.matmul(inputs, self.w) + self.b


class MyModel(Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.dense1 = MyDense(hidden_dim)
        self.dense2 = MyDense(hidden_dim)
        self.dense3 = MyDense(output_dim)

    def call(self, x):
        x1 = self.dense1(x)
        x2 = self.dense2(x)
        # Why not use some ops here as well
        x = ops.concatenate([x1, x2], axis=-1)
        return self.dense3(x)


model = MyModel(hidden_dim=256, output_dim=16)
model.summary()

x = np.random.random((50000, 128))
y = np.random.random((50000, 16))
batch_size = 32
epochs = 10

model.compile(
    optimizer=optimizers.SGD(learning_rate=0.001),
    loss=losses.MeanSquaredError(),
    metrics=[metrics.MeanSquaredError()],
)
history = model.fit(x, y, batch_size=batch_size, epochs=epochs)

print("History:")
print(history.history)
