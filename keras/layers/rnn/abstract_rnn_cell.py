# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Base class for RNN cells."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras.engine import base_layer
from keras.layers.rnn import rnn_utils

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.AbstractRNNCell')
class AbstractRNNCell(base_layer.Layer):
  """Abstract object representing an RNN cell.

  See [the Keras RNN API guide](https://www.tensorflow.org/guide/keras/rnn)
  for details about the usage of RNN API.

  This is the base class for implementing RNN cells with custom behavior.

  Every `RNNCell` must have the properties below and implement `call` with
  the signature `(output, next_state) = call(input, state)`.

  Examples:

  ```python
    class MinimalRNNCell(AbstractRNNCell):

      def __init__(self, units, **kwargs):
        self.units = units
        super(MinimalRNNCell, self).__init__(**kwargs)

      @property
      def state_size(self):
        return self.units

      def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

      def call(self, inputs, states):
        prev_output = states[0]
        h = backend.dot(inputs, self.kernel)
        output = h + backend.dot(prev_output, self.recurrent_kernel)
        return output, output
  ```

  This definition of cell differs from the definition used in the literature.
  In the literature, 'cell' refers to an object with a single scalar output.
  This definition refers to a horizontal array of such units.

  An RNN cell, in the most abstract setting, is anything that has
  a state and performs some operation that takes a matrix of inputs.
  This operation results in an output matrix with `self.output_size` columns.
  If `self.state_size` is an integer, this operation also results in a new
  state matrix with `self.state_size` columns.  If `self.state_size` is a
  (possibly nested tuple of) TensorShape object(s), then it should return a
  matching structure of Tensors having shape `[batch_size].concatenate(s)`
  for each `s` in `self.batch_size`.
  """

  def call(self, inputs, states):
    """The function that contains the logic for one RNN step calculation.

    Args:
      inputs: the input tensor, which is a slide from the overall RNN input by
        the time dimension (usually the second dimension).
      states: the state tensor from previous step, which has the same shape
        as `(batch, state_size)`. In the case of timestep 0, it will be the
        initial state user specified, or zero filled tensor otherwise.

    Returns:
      A tuple of two tensors:
        1. output tensor for the current timestep, with size `output_size`.
        2. state tensor for next step, which has the shape of `state_size`.
    """
    raise NotImplementedError

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.

    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return rnn_utils.generate_zero_filled_state_for_cell(
        self, inputs, batch_size, dtype)
