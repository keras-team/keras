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


SERIALIZERS = (
  ("copy", copy.copy),
  ("deepcopy", copy.deepcopy),
  *(
    (f"pickle-{protocol}", lambda model: pickle.loads(pickle.dumps(model, protocol=protocol)))
    for protocol in range(pickle.HIGHEST_PROTOCOL+1)
  )
)

class TestPickleProtocol(keras_parameterized.TestCase):
  """Tests pickle protoocol support.
  """

  @keras_parameterized.run_with_all_model_types
  @keras_parameterized.parameterized.named_parameters(*SERIALIZERS)
  def test_pickle_model_fitted(self, serializer):
    """Fitted models should be copyable/picklable."""

    # create model
    original_model = testing_utils.get_small_mlp(
      num_hidden=1, num_classes=2, input_dim=3
    )
    original_model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy')

    # train
    x = np.random.random((1000, 3))
    y = np.random.randint(low=0, high=2, size=(1000, ))
    original_model.fit(x, y)
    y1 = original_model.predict(x)
    original_weights = original_model.get_weights()

    # roundtrip and check that the predictions are the same
    new_model = serializer(original_model)

    y2 = new_model.predict(x)
    self.assertAllClose(y1, y2)

    # make sure we can keep training
    new_model.fit(x, y)
    new_weights = new_model.get_weights()
    # the weights on the new model should have changed
    self.assertNotAllClose(new_weights, original_weights)
    # but the weights on the original model have not been touched
    self.assertAllClose(original_weights, original_model.get_weights())

if __name__ == '__main__':
  tf.test.main()
