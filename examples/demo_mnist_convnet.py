"""
MNIST Handwritten Digit Classification using a Convolutional Neural Network (CNN)

This example demonstrates how to build, train, and evaluate a simple CNN using Keras
on the MNIST dataset, which contains 70,000 grayscale images of handwritten digits (0-9).

The model architecture:
- Input Layer (28x28x1 images)
- Conv2D + MaxPooling
- Conv2D + MaxPooling
- Flatten
- Dropout
- Dense output with Softmax activation
"""

import numpy as np
import keras
from keras import layers
from keras.utils import to_categorical

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the MNIST dataset (pre-shuffled into train and test sets)
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Expand dimensions to match the expected input shape (batch_size, 28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Print dataset shapes for confirmation
print("x_train shape:", x_train.shape)
print(f"{x_train.shape[0]} training samples")
print(f"{x_test.shape[0]} test samples")

# One-hot encode the labels (e.g., 3 -> [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

# Set training parameters
batch_size = 128
epochs = 3

# Build the CNN model using the Sequential API
model = keras.Sequential([
    layers.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax"),
])

# Display model architecture
model.summary()

# Compile the model with loss function, optimizer, and metric
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train the model using 90% of the data and validate on 10%
model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1
)

# Evaluate model performance on test set
score = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest loss: {score[0]:.4f}")
print(f"Test accuracy: {score[1]*100:.2f}%")
