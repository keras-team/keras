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
"""Tests of `worker_training_state.py` utilities."""

import tensorflow.compat.v2 as tf

import os
import sys

from absl.testing import parameterized
from keras import callbacks
from keras.distribute import multi_worker_testing_utils


class ModelCheckpointTest(tf.test.TestCase, parameterized.TestCase):

  @tf.__internal__.distribute.combinations.generate(
      tf.__internal__.test.combinations.combine(
          mode=['eager'],
          file_format=['h5', 'tf'],
          save_weights_only=[True, False]))
  def testCheckpointExists(self, file_format, save_weights_only):
    train_ds, _ = multi_worker_testing_utils.mnist_synthetic_dataset(64, 2)
    model = multi_worker_testing_utils.get_mnist_model((28, 28, 1))
    saving_dir = self.get_temp_dir()
    saving_filepath = os.path.join(saving_dir, 'checkpoint.' + file_format)
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath=saving_filepath, save_weights_only=save_weights_only)
    ]
    self.assertFalse(tf.io.gfile.exists(saving_filepath))
    model.fit(x=train_ds, epochs=2, steps_per_epoch=2, callbacks=callbacks_list)
    tf_saved_model_exists = tf.io.gfile.exists(saving_filepath)
    tf_weights_only_checkpoint_exists = tf.io.gfile.exists(saving_filepath +
                                                               '.index')
    self.assertTrue(tf_saved_model_exists or tf_weights_only_checkpoint_exists)


if __name__ == '__main__':
  with tf.compat.v1.test.mock.patch.object(sys, 'exit', os._exit):
    tf.test.main()
