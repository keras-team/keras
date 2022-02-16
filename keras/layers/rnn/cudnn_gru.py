# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Fast GRU layer backed by cuDNN."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

import collections

from keras import constraints
from keras import initializers
from keras import regularizers
from keras.layers.rnn import gru_lstm_utils
from keras.layers.rnn.base_cudnn_rnn import _CuDNNRNN
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export(v1=['keras.layers.CuDNNGRU'])
class CuDNNGRU(_CuDNNRNN):
  """Fast GRU implementation backed by cuDNN.

  More information about cuDNN can be found on the [NVIDIA
  developer website](https://developer.nvidia.com/cudnn).
  Can only be run on GPU.

  Args:
      units: Positive integer, dimensionality of the output space.
      kernel_initializer: Initializer for the `kernel` weights matrix, used for
        the linear transformation of the inputs.
      recurrent_initializer: Initializer for the `recurrent_kernel` weights
        matrix, used for the linear transformation of the recurrent state.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation").
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix.
      bias_constraint: Constraint function applied to the bias vector.
      return_sequences: Boolean. Whether to return the last output in the output
        sequence, or the full sequence.
      return_state: Boolean. Whether to return the last state in addition to the
        output.
      go_backwards: Boolean (default False). If True, process the input sequence
        backwards and return the reversed sequence.
      stateful: Boolean (default False). If True, the last state for each sample
        at index i in a batch will be used as initial state for the sample of
        index i in the following batch.
  """

  def __init__(self,
               units,
               kernel_initializer='glorot_uniform',
               recurrent_initializer='orthogonal',
               bias_initializer='zeros',
               kernel_regularizer=None,
               recurrent_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               recurrent_constraint=None,
               bias_constraint=None,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               **kwargs):
    self.units = units
    cell_spec = collections.namedtuple('cell', 'state_size')
    self._cell = cell_spec(state_size=self.units)
    super(CuDNNGRU, self).__init__(
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        **kwargs)

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.recurrent_initializer = initializers.get(recurrent_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.recurrent_constraint = constraints.get(recurrent_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

  @property
  def cell(self):
    return self._cell

  def build(self, input_shape):
    super(CuDNNGRU, self).build(input_shape)
    if isinstance(input_shape, list):
      input_shape = input_shape[0]
    input_dim = int(input_shape[-1])

    self.kernel = self.add_weight(
        shape=(input_dim, self.units * 3),
        name='kernel',
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint)

    self.recurrent_kernel = self.add_weight(
        shape=(self.units, self.units * 3),
        name='recurrent_kernel',
        initializer=self.recurrent_initializer,
        regularizer=self.recurrent_regularizer,
        constraint=self.recurrent_constraint)

    self.bias = self.add_weight(
        shape=(self.units * 6,),
        name='bias',
        initializer=self.bias_initializer,
        regularizer=self.bias_regularizer,
        constraint=self.bias_constraint)

    self.built = True

  def _process_batch(self, inputs, initial_state):
    if not self.time_major:
      inputs = tf.transpose(inputs, perm=(1, 0, 2))
    input_h = initial_state[0]
    input_h = tf.expand_dims(input_h, axis=0)

    params = gru_lstm_utils.canonical_to_params(
        weights=[
            self.kernel[:, self.units:self.units * 2],
            self.kernel[:, :self.units],
            self.kernel[:, self.units * 2:],
            self.recurrent_kernel[:, self.units:self.units * 2],
            self.recurrent_kernel[:, :self.units],
            self.recurrent_kernel[:, self.units * 2:],
        ],
        biases=[
            self.bias[self.units:self.units * 2],
            self.bias[:self.units],
            self.bias[self.units * 2:self.units * 3],
            self.bias[self.units * 4:self.units * 5],
            self.bias[self.units * 3:self.units * 4],
            self.bias[self.units * 5:],
        ],
        shape=self._vector_shape)

    args = {
        'input': inputs,
        'input_h': input_h,
        'input_c': 0,
        'params': params,
        'is_training': True,
        'rnn_mode': 'gru',
    }

    outputs, h, _, _, _ = tf.raw_ops.CudnnRNNV2(**args)

    if self.stateful or self.return_state:
      h = h[0]
    if self.return_sequences:
      if self.time_major:
        output = outputs
      else:
        output = tf.transpose(outputs, perm=(1, 0, 2))
    else:
      output = outputs[-1]
    return output, [h]

  def get_config(self):
    config = {
        'units': self.units,
        'kernel_initializer': initializers.serialize(self.kernel_initializer),
        'recurrent_initializer':
            initializers.serialize(self.recurrent_initializer),
        'bias_initializer': initializers.serialize(self.bias_initializer),
        'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
        'recurrent_regularizer':
            regularizers.serialize(self.recurrent_regularizer),
        'bias_regularizer': regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint': constraints.serialize(self.kernel_constraint),
        'recurrent_constraint':
            constraints.serialize(self.recurrent_constraint),
        'bias_constraint': constraints.serialize(self.bias_constraint)
    }
    base_config = super(CuDNNGRU, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
