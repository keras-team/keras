import numpy as np

from keras_core import Model
from keras_core import layers
from keras_core import losses
from keras_core import metrics
from keras_core import optimizers

inputs = layers.Input((100,))
x = layers.Dense(128, activation="relu")(inputs)
residual = x
x = layers.Dense(128, activation="relu")(x)
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
