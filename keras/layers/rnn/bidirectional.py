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
"""Bidirectional wrapper for RNNs."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

import copy

from keras import backend
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.layers.rnn import rnn_utils
from keras.layers.rnn.base_wrapper import Wrapper
from keras.utils import generic_utils
from keras.utils import tf_inspect
from keras.utils import tf_utils
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.Bidirectional')
class Bidirectional(Wrapper):
  """Bidirectional wrapper for RNNs.

  Args:
    layer: `keras.layers.RNN` instance, such as `keras.layers.LSTM` or
      `keras.layers.GRU`. It could also be a `keras.layers.Layer` instance
      that meets the following criteria:
      1. Be a sequence-processing layer (accepts 3D+ inputs).
      2. Have a `go_backwards`, `return_sequences` and `return_state`
        attribute (with the same semantics as for the `RNN` class).
      3. Have an `input_spec` attribute.
      4. Implement serialization via `get_config()` and `from_config()`.
      Note that the recommended way to create new RNN layers is to write a
      custom RNN cell and use it with `keras.layers.RNN`, instead of
      subclassing `keras.layers.Layer` directly.
      - When the `returns_sequences` is true, the output of the masked timestep
      will be zero regardless of the layer's original `zero_output_for_mask`
      value.
    merge_mode: Mode by which outputs of the forward and backward RNNs will be
      combined. One of {'sum', 'mul', 'concat', 'ave', None}. If None, the
      outputs will not be combined, they will be returned as a list. Default
      value is 'concat'.
    backward_layer: Optional `keras.layers.RNN`, or `keras.layers.Layer`
      instance to be used to handle backwards input processing.
      If `backward_layer` is not provided, the layer instance passed as the
      `layer` argument will be used to generate the backward layer
      automatically.
      Note that the provided `backward_layer` layer should have properties
      matching those of the `layer` argument, in particular it should have the
      same values for `stateful`, `return_states`, `return_sequences`, etc.
      In addition, `backward_layer` and `layer` should have different
      `go_backwards` argument values.
      A `ValueError` will be raised if these requirements are not met.

  Call arguments:
    The call arguments for this layer are the same as those of the wrapped RNN
      layer.
    Beware that when passing the `initial_state` argument during the call of
    this layer, the first half in the list of elements in the `initial_state`
    list will be passed to the forward RNN call and the last half in the list
    of elements will be passed to the backward RNN call.

  Raises:
    ValueError:
      1. If `layer` or `backward_layer` is not a `Layer` instance.
      2. In case of invalid `merge_mode` argument.
      3. If `backward_layer` has mismatched properties compared to `layer`.

  Examples:

  ```python
  model = Sequential()
  model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
  model.add(Bidirectional(LSTM(10)))
  model.add(Dense(5))
  model.add(Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

   # With custom backward layer
   model = Sequential()
   forward_layer = LSTM(10, return_sequences=True)
   backward_layer = LSTM(10, activation='relu', return_sequences=True,
                         go_backwards=True)
   model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
                           input_shape=(5, 10)))
   model.add(Dense(5))
   model.add(Activation('softmax'))
   model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
  ```
  """

  def __init__(self,
               layer,
               merge_mode='concat',
               weights=None,
               backward_layer=None,
               **kwargs):
    if not isinstance(layer, Layer):
      raise ValueError(
          'Please initialize `Bidirectional` layer with a '
          f'`tf.keras.layers.Layer` instance. Received: {layer}')
    if backward_layer is not None and not isinstance(backward_layer, Layer):
      raise ValueError(
          '`backward_layer` need to be a `tf.keras.layers.Layer` instance. '
          f'Received: {backward_layer}')
    if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
      raise ValueError(f'Invalid merge mode. Received: {merge_mode}. '
                       'Merge mode should be one of '
                       '{"sum", "mul", "ave", "concat", None}')
    # We don't want to track `layer` since we're already tracking the two copies
    # of it we actually run.
    self._setattr_tracking = False
    super(Bidirectional, self).__init__(layer, **kwargs)
    self._setattr_tracking = True

    # Recreate the forward layer from the original layer config, so that it will
    # not carry over any state from the layer.
    self.forward_layer = self._recreate_layer_from_config(layer)

    if backward_layer is None:
      self.backward_layer = self._recreate_layer_from_config(
          layer, go_backwards=True)
    else:
      self.backward_layer = backward_layer
      # Keep the custom backward layer config, so that we can save it later. The
      # layer's name might be updated below with prefix 'backward_', and we want
      # to preserve the original config.
      self._backward_layer_config = generic_utils.serialize_keras_object(
          backward_layer)

    self.forward_layer._name = 'forward_' + self.forward_layer.name
    self.backward_layer._name = 'backward_' + self.backward_layer.name

    self._verify_layer_config()

    def force_zero_output_for_mask(layer):
      # Force the zero_output_for_mask to be True if returning sequences.
      if getattr(layer, 'zero_output_for_mask', None) is not None:
        layer.zero_output_for_mask = layer.return_sequences

    force_zero_output_for_mask(self.forward_layer)
    force_zero_output_for_mask(self.backward_layer)

    self.merge_mode = merge_mode
    if weights:
      nw = len(weights)
      self.forward_layer.initial_weights = weights[:nw // 2]
      self.backward_layer.initial_weights = weights[nw // 2:]
    self.stateful = layer.stateful
    self.return_sequences = layer.return_sequences
    self.return_state = layer.return_state
    self.supports_masking = True
    self._trainable = True
    self._num_constants = 0
    self.input_spec = layer.input_spec

  @property
  def _use_input_spec_as_call_signature(self):
    return self.layer._use_input_spec_as_call_signature  # pylint: disable=protected-access

  def _verify_layer_config(self):
    """Ensure the forward and backward layers have valid common property."""
    if self.forward_layer.go_backwards == self.backward_layer.go_backwards:
      raise ValueError(
          'Forward layer and backward layer should have different '
          '`go_backwards` value.'
          f'forward_layer.go_backwards = {self.forward_layer.go_backwards},'
          f'backward_layer.go_backwards = {self.backward_layer.go_backwards}')

    common_attributes = ('stateful', 'return_sequences', 'return_state')
    for a in common_attributes:
      forward_value = getattr(self.forward_layer, a)
      backward_value = getattr(self.backward_layer, a)
      if forward_value != backward_value:
        raise ValueError(
            'Forward layer and backward layer are expected to have the same '
            f'value for attribute "{a}", got "{forward_value}" for forward '
            f'layer and "{backward_value}" for backward layer')

  def _recreate_layer_from_config(self, layer, go_backwards=False):
    # When recreating the layer from its config, it is possible that the layer
    # is a RNN layer that contains custom cells. In this case we inspect the
    # layer and pass the custom cell class as part of the `custom_objects`
    # argument when calling `from_config`.
    # See https://github.com/tensorflow/tensorflow/issues/26581 for more detail.
    config = layer.get_config()
    if go_backwards:
      config['go_backwards'] = not config['go_backwards']
    if 'custom_objects' in tf_inspect.getfullargspec(
        layer.__class__.from_config).args:
      custom_objects = {}
      cell = getattr(layer, 'cell', None)
      if cell is not None:
        custom_objects[cell.__class__.__name__] = cell.__class__
        # For StackedRNNCells
        stacked_cells = getattr(cell, 'cells', [])
        for c in stacked_cells:
          custom_objects[c.__class__.__name__] = c.__class__
      return layer.__class__.from_config(config, custom_objects=custom_objects)
    else:
      return layer.__class__.from_config(config)

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    output_shape = self.forward_layer.compute_output_shape(input_shape)
    if self.return_state:
      state_shape = tf_utils.convert_shapes(output_shape[1:], to_tuples=False)
      output_shape = tf_utils.convert_shapes(output_shape[0], to_tuples=False)
    else:
      output_shape = tf_utils.convert_shapes(output_shape, to_tuples=False)

    if self.merge_mode == 'concat':
      output_shape = output_shape.as_list()
      output_shape[-1] *= 2
      output_shape = tf.TensorShape(output_shape)
    elif self.merge_mode is None:
      output_shape = [output_shape, copy.copy(output_shape)]

    if self.return_state:
      if self.merge_mode is None:
        return output_shape + state_shape + copy.copy(state_shape)
      return [output_shape] + state_shape + copy.copy(state_shape)
    return output_shape

  def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
    """`Bidirectional.__call__` implements the same API as the wrapped `RNN`."""
    inputs, initial_state, constants = rnn_utils.standardize_args(
        inputs, initial_state, constants, self._num_constants)

    if isinstance(inputs, list):
      if len(inputs) > 1:
        initial_state = inputs[1:]
      inputs = inputs[0]

    if initial_state is None and constants is None:
      return super(Bidirectional, self).__call__(inputs, **kwargs)

    # Applies the same workaround as in `RNN.__call__`
    additional_inputs = []
    additional_specs = []
    if initial_state is not None:
      # Check if `initial_state` can be split into half
      num_states = len(initial_state)
      if num_states % 2 > 0:
        raise ValueError(
            'When passing `initial_state` to a Bidirectional RNN, '
            'the state should be a list containing the states of '
            'the underlying RNNs. '
            f'Received: {initial_state}')

      kwargs['initial_state'] = initial_state
      additional_inputs += initial_state
      state_specs = tf.nest.map_structure(
          lambda state: InputSpec(shape=backend.int_shape(state)),
          initial_state)
      self.forward_layer.state_spec = state_specs[:num_states // 2]
      self.backward_layer.state_spec = state_specs[num_states // 2:]
      additional_specs += state_specs
    if constants is not None:
      kwargs['constants'] = constants
      additional_inputs += constants
      constants_spec = [InputSpec(shape=backend.int_shape(constant))
                        for constant in constants]
      self.forward_layer.constants_spec = constants_spec
      self.backward_layer.constants_spec = constants_spec
      additional_specs += constants_spec

      self._num_constants = len(constants)
      self.forward_layer._num_constants = self._num_constants
      self.backward_layer._num_constants = self._num_constants

    is_keras_tensor = backend.is_keras_tensor(
        tf.nest.flatten(additional_inputs)[0])
    for tensor in tf.nest.flatten(additional_inputs):
      if backend.is_keras_tensor(tensor) != is_keras_tensor:
        raise ValueError('The initial state of a Bidirectional'
                         ' layer cannot be specified with a mix of'
                         ' Keras tensors and non-Keras tensors'
                         ' (a "Keras tensor" is a tensor that was'
                         ' returned by a Keras layer, or by `Input`)')

    if is_keras_tensor:
      # Compute the full input spec, including state
      full_input = [inputs] + additional_inputs
      # The original input_spec is None since there could be a nested tensor
      # input. Update the input_spec to match the inputs.
      full_input_spec = [None for _ in range(len(tf.nest.flatten(inputs)))
                        ] + additional_specs
      # Removing kwargs since the value are passed with input list.
      kwargs['initial_state'] = None
      kwargs['constants'] = None

      # Perform the call with temporarily replaced input_spec
      original_input_spec = self.input_spec
      self.input_spec = full_input_spec
      output = super(Bidirectional, self).__call__(full_input, **kwargs)
      self.input_spec = original_input_spec
      return output
    else:
      return super(Bidirectional, self).__call__(inputs, **kwargs)

  def call(self,
           inputs,
           training=None,
           mask=None,
           initial_state=None,
           constants=None):
    """`Bidirectional.call` implements the same API as the wrapped `RNN`."""
    kwargs = {}
    if generic_utils.has_arg(self.layer.call, 'training'):
      kwargs['training'] = training
    if generic_utils.has_arg(self.layer.call, 'mask'):
      kwargs['mask'] = mask
    if generic_utils.has_arg(self.layer.call, 'constants'):
      kwargs['constants'] = constants

    if generic_utils.has_arg(self.layer.call, 'initial_state'):
      if isinstance(inputs, list) and len(inputs) > 1:
        # initial_states are keras tensors, which means they are passed in
        # together with inputs as list. The initial_states need to be split into
        # forward and backward section, and be feed to layers accordingly.
        forward_inputs = [inputs[0]]
        backward_inputs = [inputs[0]]
        pivot = (len(inputs) - self._num_constants) // 2 + 1
        # add forward initial state
        forward_inputs += inputs[1:pivot]
        if not self._num_constants:
          # add backward initial state
          backward_inputs += inputs[pivot:]
        else:
          # add backward initial state
          backward_inputs += inputs[pivot:-self._num_constants]
          # add constants for forward and backward layers
          forward_inputs += inputs[-self._num_constants:]
          backward_inputs += inputs[-self._num_constants:]
        forward_state, backward_state = None, None
        if 'constants' in kwargs:
          kwargs['constants'] = None
      elif initial_state is not None:
        # initial_states are not keras tensors, eg eager tensor from np array.
        # They are only passed in from kwarg initial_state, and should be passed
        # to forward/backward layer via kwarg initial_state as well.
        forward_inputs, backward_inputs = inputs, inputs
        half = len(initial_state) // 2
        forward_state = initial_state[:half]
        backward_state = initial_state[half:]
      else:
        forward_inputs, backward_inputs = inputs, inputs
        forward_state, backward_state = None, None

      y = self.forward_layer(forward_inputs,
                             initial_state=forward_state, **kwargs)
      y_rev = self.backward_layer(backward_inputs,
                                  initial_state=backward_state, **kwargs)
    else:
      y = self.forward_layer(inputs, **kwargs)
      y_rev = self.backward_layer(inputs, **kwargs)

    if self.return_state:
      states = y[1:] + y_rev[1:]
      y = y[0]
      y_rev = y_rev[0]

    if self.return_sequences:
      time_dim = 0 if getattr(self.forward_layer, 'time_major', False) else 1
      y_rev = backend.reverse(y_rev, time_dim)
    if self.merge_mode == 'concat':
      output = backend.concatenate([y, y_rev])
    elif self.merge_mode == 'sum':
      output = y + y_rev
    elif self.merge_mode == 'ave':
      output = (y + y_rev) / 2
    elif self.merge_mode == 'mul':
      output = y * y_rev
    elif self.merge_mode is None:
      output = [y, y_rev]
    else:
      raise ValueError(
          f'Unrecognized value for `merge_mode`. Received: {self.merge_mode}'
          'Expected values are ["concat", "sum", "ave", "mul"]')

    if self.return_state:
      if self.merge_mode is None:
        return output + states
      return [output] + states
    return output

  def reset_states(self):
    self.forward_layer.reset_states()
    self.backward_layer.reset_states()

  def build(self, input_shape):
    with backend.name_scope(self.forward_layer.name):
      self.forward_layer.build(input_shape)
    with backend.name_scope(self.backward_layer.name):
      self.backward_layer.build(input_shape)
    self.built = True

  def compute_mask(self, inputs, mask):
    if isinstance(mask, list):
      mask = mask[0]
    if self.return_sequences:
      if not self.merge_mode:
        output_mask = [mask, mask]
      else:
        output_mask = mask
    else:
      output_mask = [None, None] if not self.merge_mode else None

    if self.return_state:
      states = self.forward_layer.states
      state_mask = [None for _ in states]
      if isinstance(output_mask, list):
        return output_mask + state_mask * 2
      return [output_mask] + state_mask * 2
    return output_mask

  @property
  def constraints(self):
    constraints = {}
    if hasattr(self.forward_layer, 'constraints'):
      constraints.update(self.forward_layer.constraints)
      constraints.update(self.backward_layer.constraints)
    return constraints

  def get_config(self):
    config = {'merge_mode': self.merge_mode}
    if self._num_constants:
      config['num_constants'] = self._num_constants

    if hasattr(self, '_backward_layer_config'):
      config['backward_layer'] = self._backward_layer_config
    base_config = super(Bidirectional, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # Instead of updating the input, create a copy and use that.
    config = copy.deepcopy(config)
    num_constants = config.pop('num_constants', 0)
    # Handle forward layer instantiation (as would parent class).
    from keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top
    config['layer'] = deserialize_layer(
        config['layer'], custom_objects=custom_objects)
    # Handle (optional) backward layer instantiation.
    backward_layer_config = config.pop('backward_layer', None)
    if backward_layer_config is not None:
      backward_layer = deserialize_layer(
          backward_layer_config, custom_objects=custom_objects)
      config['backward_layer'] = backward_layer
    # Instantiate the wrapper, adjust it and return it.
    layer = cls(**config)
    layer._num_constants = num_constants  # pylint: disable=protected-access
    return layer
