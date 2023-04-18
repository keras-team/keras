import numpy as np

from keras_core import Model
from keras_core import layers
from keras_core import losses
from keras_core import metrics
from keras_core import optimizers


class MyModel(Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim)
        self.dense2 = layers.Dense(hidden_dim)
        self.dense3 = layers.Dense(output_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
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
