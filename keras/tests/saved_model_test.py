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
"""Tests for trackable object SavedModel save."""

import tensorflow.compat.v2 as tf

import os
import subprocess
from tensorflow.python.framework import test_util as tf_test_utils
from keras import Model, models
from keras.layers import core, Layer
from keras.optimizers.optimizer_v2 import adam


class _Embedding(Layer):
  def __init__(self, input_dim, output_dim, trainable=True):
    super(_Embedding, self).__init__(dtype=tf.float32)
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.embedding_table = None
    self.trainable = trainable

  def build(self, input_shape):
    self.embedding_table = self.add_weight(
        "embedding_table", shape=[self.input_dim, self.output_dim],
        dtype=tf.float32, trainable=self.trainable)

  def call(self, indices):
    return tf.gather(params=self.embedding_table, indices=indices)

class _TestModel(Model):
  def __init__(self, rows, cols):
    super().__init__()
    self.embedding = _Embedding(input_dim=rows, output_dim=cols)

  def call(self, x):
    with tf.device('/cpu:0'):
      x = self.embedding(x)
    with tf.device('/gpu:0'):
      x = tf.math.reduce_sum(x, axis=1)
    return x


class _ModelWithOptimizerUsingDefun(tf.train.Checkpoint):

  def __init__(self):
    self.dense = core.Dense(1)
    self.optimizer = adam.Adam(0.01)

  @tf.function(
      input_signature=(tf.TensorSpec([None, 2], tf.float32),
                       tf.TensorSpec([None], tf.float32)),
  )
  def call(self, x, y):
    with tf.GradientTape() as tape:
      loss = tf.reduce_mean((self.dense(x) - y) ** 2.)
    trainable_variables = self.dense.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))
    return {"loss": loss}


class MemoryTests(tf.test.TestCase):

  def setUp(self):
    super(MemoryTests, self).setUp()
    self._model = _ModelWithOptimizerUsingDefun()

  @tf_test_utils.assert_no_garbage_created
  def DISABLED_test_no_reference_cycles(self):
    x = tf.constant([[3., 4.]])
    y = tf.constant([2.])
    self._model.call(x, y)
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    tf.saved_model.save(self._model, save_dir, self._model.call)

class SavedModelLoadMemoryTests(tf.test.TestCase):

  def get_gpu_memory(self):
    command = "nvidia-smi --query-gpu=memory.total --format=csv"
    memory_free_info = subprocess.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    if isinstance(memory_free_values, list):
        return int(memory_free_values[0] / 1024)
    return int(memory_free_values)

  @tf_test_utils.run_gpu_only
  def test_no_oom_loading_large_tenor(self):
    if not tf.config.get_soft_device_placement():
      self.skipTest("This test only works for soft device placement is on")
    max_gpu_memory_in_Gb = self.get_gpu_memory()
    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    ncols = 128
    nrows = (max_gpu_memory_in_Gb + 1) * 2**30 // ncols // 4
    model = _TestModel(rows=nrows, cols=ncols)
    x = tf.zeros(shape=(65536, 1), dtype=tf.int32)
    y = model(x)
    models.save_model(
        model=model, filepath=save_dir, overwrite=True,
        options=tf.saved_model.SaveOptions(
            experimental_variable_policy=tf.saved_model.experimental.VariablePolicy.SAVE_VARIABLE_DEVICES))
    loaded = models.load_model(
        save_dir,
        options=tf.saved_model.SaveOptions(
            experimental_variable_policy=tf.saved_model.experimental.VariablePolicy.SAVE_VARIABLE_DEVICES))


if __name__ == "__main__":
  tf.test.main()