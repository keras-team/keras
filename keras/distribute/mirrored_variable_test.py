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
"""Test MirroredVariable in MirroredStrategy and MultiWorkerMirroredStrategy."""

import tensorflow.compat.v2 as tf
from keras.distribute import distributed_training_utils
from keras.layers import core


def _mimic_two_cpus():
  try:
    cpus = tf.config.list_physical_devices("CPU")
  except tf.errors.NotFoundError:
    # Testing device not available. Skip the test.
    return False

  tf.config.set_logical_device_configuration(cpus[0], [
      tf.config.LogicalDeviceConfiguration(),
      tf.config.LogicalDeviceConfiguration(),
  ])
  return True


def get_strategy_with_mimicing_cpus():
  if not _mimic_two_cpus():
    return None
  return (tf.distribute.MultiWorkerMirroredStrategy
          ._from_local_devices(("/device:CPU:0", "/device:CPU:1")))


@tf.__internal__.distribute.combinations.generate(
    tf.__internal__.test.combinations.combine(
        distribution=list(
            filter(None.__ne__, [
                tf.__internal__.distribute.combinations.mirrored_strategy_with_gpu_and_cpu,
                get_strategy_with_mimicing_cpus()
            ])),
        mode=["graph", "eager"]))
class MirroredVariableCreationTest(tf.test.TestCase):
  """Base class that tests mirrored variable creator.

  Currently it assumes all strategy objects have two replicas.
  """

  @classmethod
  def setUpClass(cls):
    _mimic_two_cpus()

  def assertAllDifferent(self, objs):
    for i in range(len(objs)):
      for j in range(len(objs)):
        if i == j:
          continue
        self.assertIsNot(objs[i], objs[j])

  def _is_mirrored(self, val):
    if distributed_training_utils.is_distributed_variable(val):
      if val._policy:  # pylint: disable=protected-access
        return val._policy._is_mirrored()  # pylint: disable=protected-access
    # Since `Mirrored` is a private symbol in tf.distribute, we're checking
    # with `DistributedValues` as an approximation.
    return isinstance(val, tf.distribute.DistributedValues)

  def testWithLayers(self, distribution):

    def model_fn(features):

      layer1 = core.Dense(1)
      layer1(features)
      layer2 = core.Dense(1)
      layer2(features)
      # We rely on names and orders to make sure replica references the same
      # MirroredVariable. Uniquifying names may involve global states,
      # merge_call switches threads so we need to test things work after
      # merge_call.
      tf.distribute.get_replica_context().merge_call(lambda _: _)
      layer3 = core.Dense(1)
      layer3(features)
      return [(layer1.kernel, layer1.bias), (layer2.kernel, layer2.bias),
              (layer3.kernel, layer3.bias)]

    iterator = distribution.make_input_fn_iterator(
        lambda _: tf.data.Dataset.from_tensors([[1.]]).repeat(10))
    self.evaluate(iterator.initializer)
    features = iterator.get_next()

    with distribution.scope():
      result = distribution.extended.call_for_each_replica(
          model_fn, args=(features,))
      for kernel, bias in result:
        self.assertTrue(self._is_mirrored(kernel))
        self.assertAllDifferent(distribution.experimental_local_results(kernel))
        self.assertTrue(self._is_mirrored(bias))
        self.assertAllDifferent(distribution.experimental_local_results(kernel))


if __name__ == "__main__":
  tf.test.main()
