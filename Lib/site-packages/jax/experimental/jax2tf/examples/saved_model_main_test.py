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
"""Tests for mnist_lib, saved_model_lib, saved_model_main."""

import os
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized

from jax._src import config
from jax._src import test_util as jtu

from jax.experimental.jax2tf.examples import saved_model_main
from jax.experimental.jax2tf.tests import tf_test_util

config.parse_flags_with_absl()
FLAGS = flags.FLAGS


class SavedModelMainTest(tf_test_util.JaxToTfTestCase):

  def setUp(self):
    super().setUp()
    FLAGS.model_path = os.path.join(absltest.get_default_test_tmpdir(),
                                    "saved_models")
    FLAGS.num_epochs = 1
    FLAGS.test_savedmodel = True
    FLAGS.mock_data = True

  @parameterized.named_parameters(
      dict(
          testcase_name=f"_{model}_batch={serving_batch_size}",
          model=model,
          serving_batch_size=serving_batch_size)
      for model in ["mnist_pure_jax", "mnist_flax"]
      for serving_batch_size in [1, -1])
  def test_train_and_save_full(self,
                               model="mnist_flax",
                               serving_batch_size=-1):
    if (serving_batch_size == -1 and
        config.jax2tf_default_native_serialization.value and
        not config.dynamic_shapes.value):
      self.skipTest("shape polymorphism but --jax_dynamic_shapes is not set.")
    FLAGS.model = model
    FLAGS.model_classifier_layer = True
    FLAGS.serving_batch_size = serving_batch_size
    saved_model_main.train_and_save()

  @parameterized.named_parameters(
      dict(testcase_name=f"_{model}", model=model)
      for model in ["mnist_pure_jax", "mnist_flax"])
  def test_train_and_save_features(self, model="mnist_flax"):
    FLAGS.model = model
    FLAGS.model_classifier_layer = False
    saved_model_main.train_and_save()


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
