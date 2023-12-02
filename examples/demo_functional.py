import numpy as np

from keras import Model
from keras import layers
from keras import losses
from keras import metrics
from keras import optimizers
import keras

keras.config.disable_traceback_filtering()

inputs = layers.Input((100,))
x = layers.Dense(512, activation="relu")(inputs)
residual = x
x = layers.Dense(512, activation="relu")(x)
x = layers.Dense(512, activation="relu")(x)
x += residual
x = layers.Dense(512, activation="relu")(x)
residual = x
x = layers.Dense(512, activation="relu")(x)
x = layers.Dense(512, activation="relu")(x)
x += residual
residual = x
x = layers.Dense(512, activation="relu")(x)
x = layers.Dense(512, activation="relu")(x)
x += residual
outputs = layers.Dense(16)(x)
model = Model(inputs, outputs)

model.summary()

x = np.random.random((50000, 100))
y = np.random.random((50000, 16))
batch_size = 32
epochs = 5

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss=losses.MeanSquaredError(),
    metrics=[
        metrics.CategoricalAccuracy(name="acc"),
        metrics.MeanSquaredError(name="mse"),
    ],
)

print("\nTrain model")
history = model.fit(
    x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2
)
print("\nHistory:")
print(history.history)

print("\nEvaluate model")
scores = model.evaluate(x, y, return_dict=True)
print(scores)

print("\nRun inference")
pred = model.predict(x)
print(f"Inferred output shape {pred.shape}")
