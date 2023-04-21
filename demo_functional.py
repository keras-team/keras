import numpy as np

from keras_core import Model
from keras_core import layers
from keras_core import losses
from keras_core import metrics
from keras_core import optimizers

inputs = layers.Input((128,), batch_size=32)
x = layers.Dense(256, activation="relu")(inputs)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dense(256, activation="relu")(x)
outputs = layers.Dense(16)(x)
model = Model(inputs, outputs)

model.summary()

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
