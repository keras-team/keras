# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Distribution tests for keras.layers.preprocessing.image_preprocessing."""

import tensorflow.compat.v2 as tf

import numpy as np

import keras
from keras import keras_parameterized
from keras.distribute import strategy_combinations
from keras.layers.preprocessing import image_preprocessing
from keras.layers.preprocessing import preprocessing_test_utils


@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.combine(
        strategy=strategy_combinations.all_strategies +
        strategy_combinations.multi_worker_mirrored_strategies,
        mode=["eager", "graph"]))
class ImagePreprocessingDistributionTest(
    keras_parameterized.TestCase,
    preprocessing_test_utils.PreprocessingLayerTest):

  def test_distribution(self, strategy):
    if "CentralStorage" in type(strategy).__name__:
      self.skipTest("Does not work with CentralStorageStrategy yet.")
    # TODO(b/159738418): large image input causes OOM in ubuntu multi gpu.
    np_images = np.random.random((32, 32, 32, 3)).astype(np.float32)
    image_dataset = tf.data.Dataset.from_tensor_slices(np_images).batch(
        16, drop_remainder=True)

    with strategy.scope():
      input_data = keras.Input(shape=(32, 32, 3), dtype=tf.float32)
      image_preprocessor = keras.Sequential([
          image_preprocessing.Resizing(height=256, width=256),
          image_preprocessing.RandomCrop(height=224, width=224),
          image_preprocessing.RandomTranslation(.1, .1),
          image_preprocessing.RandomRotation(.2),
          image_preprocessing.RandomFlip(),
          image_preprocessing.RandomZoom(.2, .2)])
      preprocessed_image = image_preprocessor(input_data)
      flatten_layer = keras.layers.Flatten(data_format="channels_last")
      output = flatten_layer(preprocessed_image)
      cls_layer = keras.layers.Dense(units=1, activation="sigmoid")
      output = cls_layer(output)
      model = keras.Model(inputs=input_data, outputs=output)
    model.compile(loss="binary_crossentropy")
    _ = model.predict(image_dataset)


if __name__ == "__main__":
  tf.compat.v1.enable_v2_behavior()
  tf.__internal__.distribute.multi_process_runner.test_main()
