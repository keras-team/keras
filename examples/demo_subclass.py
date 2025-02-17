import numpy as np

from keras import Model
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers


class MyModel(Model):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.dense1 = layers.Dense(hidden_dim, activation="relu")
        self.dense2 = layers.Dense(hidden_dim, activation="relu")
        self.dense3 = layers.Dense(output_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


model = MyModel(hidden_dim=256, output_dim=16)

x = np.random.random((50000, 128))
y = np.random.random((50000, 16))
batch_size = 32
epochs = 6

model.compile(
    optimizer=optimizers.SGD(learning_rate=0.001),
    loss=losses.MeanSquaredError(),
    metrics=[metrics.MeanSquaredError()],
)
history = model.fit(
    x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2
)

print("History:")
print(history.history)

model.summary()
