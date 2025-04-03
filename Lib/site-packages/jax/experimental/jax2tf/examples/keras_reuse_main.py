# Copyright 2020 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Demonstrates reuse of a jax2tf model in Keras.

Includes the flags from saved_model_main.py.

See README.md.
"""
import logging
import warnings
from absl import app
from absl import flags
from jax.experimental.jax2tf.examples import mnist_lib
from jax.experimental.jax2tf.examples import saved_model_main
import tensorflow as tf
import tensorflow_datasets as tfds  # type: ignore
with warnings.catch_warnings():
  warnings.simplefilter("ignore")
  import tensorflow_hub as hub  # type: ignore


FLAGS = flags.FLAGS


def main(_):
  FLAGS.model_classifier_layer = False  # We only need the features
  # Train the model and save the feature extractor
  saved_model_main.train_and_save()

  tf_accelerator, _ = saved_model_main.tf_accelerator_and_tolerances()
  feature_model_dir = saved_model_main.savedmodel_dir()

  # With Keras, we use the tf.distribute.OneDeviceStrategy as the high-level
  # analogue of the tf.device(...) placement seen above.
  # It works on CPU, GPU and TPU.
  # Actual high-performance training would use the appropriately replicated
  # TF Distribution Strategy.
  strategy = tf.distribute.OneDeviceStrategy(tf_accelerator)
  with strategy.scope():
    images = tf.keras.layers.Input(
        mnist_lib.input_shape, batch_size=mnist_lib.train_batch_size)
    keras_feature_extractor = hub.KerasLayer(feature_model_dir, trainable=True)
    features = keras_feature_extractor(images)
    predictor = tf.keras.layers.Dense(10, activation="softmax")
    predictions = predictor(features)
    keras_model = tf.keras.Model(images, predictions)

  keras_model.compile(
      loss=tf.keras.losses.categorical_crossentropy,
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
      metrics=["accuracy"])
  logging.info(keras_model.summary())

  train_ds = mnist_lib.load_mnist(
      tfds.Split.TRAIN, batch_size=mnist_lib.train_batch_size)
  test_ds = mnist_lib.load_mnist(
      tfds.Split.TEST, batch_size=mnist_lib.test_batch_size)
  keras_model.fit(train_ds, epochs=FLAGS.num_epochs, validation_data=test_ds)

  if saved_model_main.SHOW_IMAGES.value:
    mnist_lib.plot_images(
        test_ds,
        1,
        5,
        f"Keras inference with reuse of {saved_model_main.model_description()}",
        inference_fn=lambda images: keras_model(tf.convert_to_tensor(images)))


if __name__ == "__main__":
  app.run(main)
