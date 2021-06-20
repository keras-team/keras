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

import tensorflow.compat.v2 as tf

import copy
import pickle

import numpy as np

from keras import keras_parameterized
from keras import testing_utils


class TestPickleProtocol(keras_parameterized.TestCase):
  """Tests pickle protoocol support.
  """

  @keras_parameterized.run_with_all_model_types
  def test_pickle_model(self):
    """Test copy.copy, copy.deepcopy and pickle on Functional Model."""

    def roundtrip(model):
      model = copy.copy(model)
      model = copy.deepcopy(model)
      for protocol in range(pickle.HIGHEST_PROTOCOL+1):
        model = pickle.loads(pickle.dumps(model, protocol=protocol))
      return model

    # create model
    original_model = testing_utils.get_small_mlp(
      num_hidden=1, num_classes=2, input_dim=3
    )
    original_weights = original_model.get_weights()

    # roundtrip without compiling
    model = roundtrip(original_model)
    # compile
    model.compile(optimizer='sgd', loss='mse')
    # roundtrip compiled but not trained
    model = roundtrip(model)
    # train
    x = np.random.random((1000, 3))
    y = np.random.random((1000, 2))
    model.fit(x, y)
    y1 = model.predict(x)
    # roundtrip with training
    model = roundtrip(model)
    y2 = model.predict(x)
    self.assertAllClose(y1, y2)
    # check that the original model has not been changed
    final_weights = original_model.get_weights()
    self.assertAllClose(original_weights, final_weights)
    # but that the weights on the trained model are different
    self.assertNotAllClose(original_weights, model.get_weights())

if __name__ == '__main__':
  tf.test.main()
