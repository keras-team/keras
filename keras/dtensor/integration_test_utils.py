# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""E2E test for DTensor with Mnist model.

Note that this is used as prototype and verification of current functionality,
and will be changed rapidly. Please don't reply on any of these methods as a
public API/contract.
"""


import numpy as np
import tensorflow.compat.v2 as tf
from absl import logging

from keras import layers
from keras import losses
from keras import models
from keras.datasets import mnist
from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import layout_map as layout_map_lib
from keras.utils import np_utils

NUM_CLASS = 10  # MNIST has 10 digits


def get_model_with_layout_map(layout_map):
    """Builds a Sequential CNN model to recognize MNIST digits.

    Args:
      layout_map: dict of string name -> Layout, for weights creation.

    Returns:
      a CNN Keras model used for MNIST
    """

    with layout_map_lib.layout_map_scope(layout_map):
        # Define a CNN model to recognize MNIST digits.
        model = models.Sequential()
        model.add(
            layers.Conv2D(
                32,
                name="conv2d_1",
                kernel_size=(3, 3),
                activation="relu",
                input_shape=(28, 28, 1),  # channel last gray scale input
            )
        )
        model.add(
            layers.Conv2D(
                64,
                name="conv2d_2",
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Dropout(0.25))
        model.add(layers.Flatten())
        model.add(
            layers.Dense(
                128,
                name="dense_1",
                activation="relu",
            )
        )
        model.add(layers.Dropout(0.5))
        model.add(
            layers.Dense(
                NUM_CLASS,
                name="dense_2",
                activation="softmax",
            )
        )
        return model


def get_all_replicated_layout_map(mesh):
    layout_map = layout_map_lib.LayoutMap(mesh=mesh)

    layout_4d = dtensor.Layout.replicated(mesh, rank=4)
    layout_2d = dtensor.Layout.replicated(mesh, rank=2)
    layout_1d = dtensor.Layout.replicated(mesh, rank=1)

    layout_map["conv2d.*kernel"] = layout_4d
    layout_map["conv2d.*bias"] = layout_1d
    layout_map["dense.*kernel"] = layout_2d
    layout_map["dense.*bias"] = layout_1d

    return layout_map


def get_mnist_datasets(num_class, batch_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.expand_dims(x_train, axis=-1).astype("float32")
    x_test = np.expand_dims(x_test, axis=-1).astype("float32")
    x_train /= 255  # normalize to 0~1
    x_test /= 255

    y_train = np_utils.to_categorical(y_train, num_class)
    y_test = np_utils.to_categorical(y_test, num_class)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .repeat()
        .batch(batch_size, drop_remainder=True)
    )
    eval_ds = (
        tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .repeat()
        .batch(batch_size, drop_remainder=True)
    )

    return train_ds, eval_ds


def train_mnist_model_batch_sharded(
    model, optimizer, mesh, num_epochs, steps_per_epoch, global_batch_size
):

    dataset, _ = get_mnist_datasets(NUM_CLASS, global_batch_size)

    input_image_layout = dtensor.Layout.batch_sharded(mesh, "batch", rank=4)
    input_label_layout = dtensor.Layout.batch_sharded(mesh, "batch", rank=2)
    loss_obj = losses.CategoricalCrossentropy()

    num_local_devices = mesh.num_local_devices()
    iterator = iter(dataset)
    train_losses = []
    for epoch in range(num_epochs):
        total_loss = 0.00
        for _ in range(steps_per_epoch):
            images, labels = next(iterator)
            images = tf.split(images, num_local_devices)
            labels = tf.split(labels, num_local_devices)
            d_images = dtensor.pack(images, input_image_layout)
            d_labels = dtensor.pack(labels, input_label_layout)
            total_loss += train_step(
                model, d_images, d_labels, loss_obj, optimizer
            )

        train_loss = tf.reduce_mean(total_loss / steps_per_epoch)

        logging.info("Epoch %d, Loss: %f", epoch, train_loss)
        train_losses.append(train_loss)
    return train_losses


# Change to use model.fit when dataset has the correct layout info populated
# in the iterator, which is the long term solution
@tf.function
def train_step(model, feature, label, loss_obj, optimizer):

    with tf.GradientTape() as tape:
        predict = model(feature, training=True)
        loss = loss_obj(label, predict)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss
