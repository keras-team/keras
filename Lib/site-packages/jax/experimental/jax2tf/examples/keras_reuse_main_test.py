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

import os
from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax._src import test_util as jtu

from jax.experimental.jax2tf.examples import keras_reuse_main
from jax.experimental.jax2tf.tests import tf_test_util

jax.config.parse_flags_with_absl()
FLAGS = flags.FLAGS


class KerasReuseMainTest(tf_test_util.JaxToTfTestCase):

  def setUp(self):
    super().setUp()
    FLAGS.model_path = os.path.join(absltest.get_default_test_tmpdir(),
                                    "saved_models")
    FLAGS.num_epochs = 1
    FLAGS.test_savedmodel = True
    FLAGS.mock_data = True
    FLAGS.show_images = False
    FLAGS.serving_batch_size = 1

  @parameterized.named_parameters(
      dict(testcase_name=f"_{model}", model=model)
      for model in ["mnist_pure_jax", "mnist_flax"])
  @jtu.ignore_warning(message="the imp module is deprecated")
  def test_keras_reuse(self, model="mnist_pure_jax"):
    FLAGS.model = model
    keras_reuse_main.main(None)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
