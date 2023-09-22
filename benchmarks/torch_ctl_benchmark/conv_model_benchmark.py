"""Benchmark Keras performance with torch custom training loop.

In this file we use a convolution model. Training loop is written in the
vanilla torch way, and we compare the performance between building model with
Keras and torch.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import keras
from benchmarks.torch_ctl_benchmark.benchmark_utils import train_loop
from keras import layers

num_classes = 2
input_shape = (3, 256, 256)
batch_size = 128
num_batches = 20
num_epochs = 1

x_train = np.random.normal(
    size=(num_batches * batch_size, *input_shape)
).astype(np.float32)
y_train = np.random.randint(0, num_classes, size=(num_batches * batch_size,))

# Create a TensorDataset
dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_train), torch.from_numpy(y_train)
)
# Create a DataLoader
train_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=False
)


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(3, 32, kernel_size=(3, 3))
        self.activation = torch.nn.ReLU()
        self.max_pool = torch.nn.MaxPool2d((2, 2))
        self.flatten = torch.nn.Flatten()
        self.dense = torch.nn.LazyLinear(num_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.softmax(x)
        return x


def run_keras_custom_training_loop():
    keras_model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(num_classes),
            layers.Softmax(),
        ]
    )
    optimizer = optim.Adam(keras_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    train_loop(
        keras_model,
        train_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        framework="keras",
    )


def run_torch_custom_training_loop():
    torch_model = TorchModel()
    optimizer = optim.Adam(torch_model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    train_loop(
        torch_model,
        train_loader,
        num_epochs=num_epochs,
        optimizer=optimizer,
        loss_fn=loss_fn,
        framework="torch",
    )


if __name__ == "__main__":
    run_keras_custom_training_loop()
    run_torch_custom_training_loop()
