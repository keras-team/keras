# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for pickling / deepcopying of Keras Models."""
# pylint: disable=g-bad-import-order
import tensorflow.compat.v2 as tf

import copy
import pickle
import numpy as np

from keras import keras_parameterized
from keras import testing_utils


class TestPickleProtocol(keras_parameterized.TestCase):
  """Tests pickle protoocol support."""

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.parameterized.named_parameters(
      ('copy', copy.copy), ('deepcopy', copy.deepcopy),
      *((f'pickle_protocol_level_{protocol}',
         lambda model: pickle.loads(pickle.dumps(model, protocol=protocol)))  # pylint: disable=cell-var-from-loop
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1)))
  def test_built_models(self, serializer):
    """Built models should be copyable and picklable for all model types."""
    if not tf.__internal__.tf2.enabled():
      self.skipTest('pickle model only available in v2 when tf format is used.')
    model = testing_utils.get_small_mlp(
        num_hidden=1, num_classes=2, input_dim=3)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

    # train
    x = np.random.random(size=(1000, 3))
    y = np.random.randint(low=0, high=2, size=(1000,))
    model.fit(x, y)  # builds model
    y1 = model.predict(x)
    # roundtrip with training
    model = serializer(model)
    y2 = model.predict(x)
    # check that the predictions are the same
    self.assertAllClose(y1, y2)
    # and that we can continue training
    model.fit(x, y)
    y3 = model.predict(x)
    # check that the predictions are the same
    self.assertNotAllClose(y2, y3)

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.parameterized.named_parameters(
      ('copy', copy.copy),
      ('deepcopy', copy.deepcopy),
  )
  def test_unbuilt_models(self, serializer):
    """Unbuilt models should be copyable & deepcopyable for all model types."""
    if not tf.__internal__.tf2.enabled():
      self.skipTest('pickle model only available in v2 when tf format is used.')
    original_model = testing_utils.get_small_mlp(
        num_hidden=1, num_classes=2, input_dim=3)
    # roundtrip without compiling or training
    model = serializer(original_model)
    # compile
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')
    # roundtrip compiled but not trained
    model = serializer(model)


if __name__ == '__main__':
  tf.test.main()
