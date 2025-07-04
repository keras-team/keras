import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.layers import Dense
from keras.models import Sequential

# Assume you have your data loaded into x_train (input features) and y_train (target labels)
# X should have a shape like (number_of_samples, 44)
# y should have a shape like (number_of_samples,) with values 0 or 1 for binary classification

# Example of dummy data generation (replace with your actual data)
num_serialized_unts = 10  # serialized units being manufactured
pv = 44
num_process_variables_per_unit = (
    pv  # process variables being measured during mfg process
)
num_of_defect_outcomes = 2

X = np.random.random((num_serialized_unts, num_process_variables_per_unit))
X = X.reshape(-1, 28 * 28).astype("float32") / 255.0

y = np.random.randint(2, size=num_serialized_unts)
print(y)
# Using the Sequential API
model = keras.Sequential(
    [
        layers.Input(X),
        # Input layer
        layers.Dense(64, activation="relu"),
        # Another hidden layer with 64 units
        layers.Dense(num_of_defect_outcomes, activation="softmax"),
        # Output layer for 'number of defect outcomes' classes with softmax
    ]
)
# Output layer: For binary classification, use a single neuron with sigmoid activation
# model.add(Dense(1, activation="sigmoid"))

# Compile the model
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

# Train the model
model.fit(
    X,
    y,
    epochs=10,
)  # batch_size=32 # Adjust epochs and batch_size as needed

# You can evaluate the model on test data if available
# loss, accuracy = model.evaluate(X, y)
# print('Test loss:', loss)
# print('Test accuracy:', accuracy)
