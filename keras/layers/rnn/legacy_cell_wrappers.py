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
"""Module implementing the V1 version of RNN cell wrappers."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import numbers

from keras.layers.rnn.cell_wrappers import _enumerated_map_structure_up_to
from keras.layers.rnn.cell_wrappers import _parse_config_to_function
from keras.layers.rnn.cell_wrappers import _serialize_function_to_config
from keras.layers.rnn.legacy_cells import RNNCell
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.util.tf_export import tf_export


# This can be used with self.assertRaisesRegexp for assert_like_rnncell.
ASSERT_LIKE_RNNCELL_ERROR_REGEXP = "is not an RNNCell"


def _hasattr(obj, attr_name):
  try:
    getattr(obj, attr_name)
  except AttributeError:
    return False
  else:
    return True


def assert_like_rnncell(cell_name, cell):
  """Raises a TypeError if cell is not like an RNNCell.

  NOTE: Do not rely on the error message (in particular in tests) which can be
  subject to change to increase readability. Use
  ASSERT_LIKE_RNNCELL_ERROR_REGEXP.

  Args:
    cell_name: A string to give a meaningful error referencing to the name of
      the functionargument.
    cell: The object which should behave like an RNNCell.

  Raises:
    TypeError: A human-friendly exception.
  """
  conditions = [
      _hasattr(cell, "output_size"),
      _hasattr(cell, "state_size"),
      _hasattr(cell, "get_initial_state") or _hasattr(cell, "zero_state"),
      callable(cell),
  ]
  errors = [
      "'output_size' property is missing", "'state_size' property is missing",
      "either 'zero_state' or 'get_initial_state' method is required",
      "is not callable"
  ]

  if not all(conditions):

    errors = [error for error, cond in zip(errors, conditions) if not cond]
    raise TypeError("The argument {!r} ({}) is not an RNNCell: {}.".format(
        cell_name, cell, ", ".join(errors)))


class _RNNCellWrapperV1(RNNCell):
  """Base class for cells wrappers V1 compatibility.

  This class along with `_RNNCellWrapperV2` allows to define cells wrappers that
  are compatible with V1 and V2, and defines helper methods for this purpose.
  """

  def __init__(self, cell, *args, **kwargs):
    super(_RNNCellWrapperV1, self).__init__(*args, **kwargs)
    assert_like_rnncell("cell", cell)
    self.cell = cell
    if isinstance(cell, tf.__internal__.tracking.Trackable):
      self._track_trackable(self.cell, name="cell")

  def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
    """Calls the wrapped cell and performs the wrapping logic.

    This method is called from the wrapper's `call` or `__call__` methods.

    Args:
      inputs: A tensor with wrapped cell's input.
      state: A tensor or tuple of tensors with wrapped cell's state.
      cell_call_fn: Wrapped cell's method to use for step computation (cell's
        `__call__` or 'call' method).
      **kwargs: Additional arguments.

    Returns:
      A pair containing:
      - Output: A tensor with cell's output.
      - New state: A tensor or tuple of tensors with new wrapped cell's state.
    """
    raise NotImplementedError

  def __call__(self, inputs, state, scope=None):
    """Runs the RNN cell step computation.

    We assume that the wrapped RNNCell is being built within its `__call__`
    method. We directly use the wrapped cell's `__call__` in the overridden
    wrapper `__call__` method.

    This allows to use the wrapped cell and the non-wrapped cell equivalently
    when using `__call__`.

    Args:
      inputs: A tensor with wrapped cell's input.
      state: A tensor or tuple of tensors with wrapped cell's state.
      scope: VariableScope for the subgraph created in the wrapped cells'
        `__call__`.

    Returns:
      A pair containing:

      - Output: A tensor with cell's output.
      - New state: A tensor or tuple of tensors with new wrapped cell's state.
    """
    return self._call_wrapped_cell(
        inputs, state, cell_call_fn=self.cell.__call__, scope=scope)

  @property
  def state_size(self):
    return self.cell.state_size

  @property
  def output_size(self):
    return self.cell.output_size

  def zero_state(self, batch_size, dtype):
    with tf.name_scope(type(self).__name__ + "ZeroState"):
      return self.cell.zero_state(batch_size, dtype)

  def get_config(self):
    config = {
        "cell": {
            "class_name": self.cell.__class__.__name__,
            "config": self.cell.get_config()
        },
    }
    base_config = super(_RNNCellWrapperV1, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    config = config.copy()
    cell = config.pop("cell")
    try:
      assert_like_rnncell("cell", cell)
      return cls(cell, **config)
    except TypeError:
      raise ValueError("RNNCellWrapper cannot reconstruct the wrapped cell. "
                       "Please overwrite the cell in the config with a RNNCell "
                       "instance.")


@keras_export(v1=["keras.__internal__.legacy.rnn_cell.DropoutWrapper"])
@tf_export(v1=["nn.rnn_cell.DropoutWrapper"])
class DropoutWrapper(_RNNCellWrapperV1):
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self,
               cell,
               input_keep_prob=1.0,
               output_keep_prob=1.0,
               state_keep_prob=1.0,
               variational_recurrent=False,
               input_size=None,
               dtype=None,
               seed=None,
               dropout_state_filter_visitor=None,
               **kwargs):
    """Create a cell with added input, state, and/or output dropout.

    If `variational_recurrent` is set to `True` (**NOT** the default behavior),
    then the same dropout mask is applied at every step, as described in:
    [A Theoretically Grounded Application of Dropout in Recurrent
    Neural Networks. Y. Gal, Z. Ghahramani](https://arxiv.org/abs/1512.05287).

    Otherwise a different dropout mask is applied at every time step.

    Note, by default (unless a custom `dropout_state_filter` is provided),
    the memory state (`c` component of any `LSTMStateTuple`) passing through
    a `DropoutWrapper` is never modified.  This behavior is described in the
    above article.

    Args:
      cell: an RNNCell, a projection to output_size is added to it.
      input_keep_prob: unit Tensor or float between 0 and 1, input keep
        probability; if it is constant and 1, no input dropout will be added.
      output_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is constant and 1, no output dropout will be added.
      state_keep_prob: unit Tensor or float between 0 and 1, output keep
        probability; if it is constant and 1, no output dropout will be added.
        State dropout is performed on the outgoing states of the cell. **Note**
        the state components to which dropout is applied when `state_keep_prob`
        is in `(0, 1)` are also determined by the argument
        `dropout_state_filter_visitor` (e.g. by default dropout is never applied
        to the `c` component of an `LSTMStateTuple`).
      variational_recurrent: Python bool.  If `True`, then the same dropout
        pattern is applied across all time steps per run call. If this parameter
        is set, `input_size` **must** be provided.
      input_size: (optional) (possibly nested tuple of) `TensorShape` objects
        containing the depth(s) of the input tensors expected to be passed in to
        the `DropoutWrapper`.  Required and used **iff** `variational_recurrent
        = True` and `input_keep_prob < 1`.
      dtype: (optional) The `dtype` of the input, state, and output tensors.
        Required and used **iff** `variational_recurrent = True`.
      seed: (optional) integer, the randomness seed.
      dropout_state_filter_visitor: (optional), default: (see below).  Function
        that takes any hierarchical level of the state and returns a scalar or
        depth=1 structure of Python booleans describing which terms in the state
        should be dropped out.  In addition, if the function returns `True`,
        dropout is applied across this sublevel.  If the function returns
        `False`, dropout is not applied across this entire sublevel.
        Default behavior: perform dropout on all terms except the memory (`c`)
          state of `LSTMCellState` objects, and don't try to apply dropout to
        `TensorArray` objects: ```
        def dropout_state_filter_visitor(s):
          if isinstance(s, LSTMCellState): # Never perform dropout on the c
            state. return LSTMCellState(c=False, h=True)
          elif isinstance(s, TensorArray): return False return True ```
      **kwargs: dict of keyword arguments for base layer.

    Raises:
      TypeError: if `cell` is not an `RNNCell`, or `keep_state_fn` is provided
        but not `callable`.
      ValueError: if any of the keep_probs are not between 0 and 1.
    """
    super(DropoutWrapper, self).__init__(cell, dtype=dtype, **kwargs)

    if (dropout_state_filter_visitor is not None and
        not callable(dropout_state_filter_visitor)):
      raise TypeError("dropout_state_filter_visitor must be callable. "
                      f"Received: {dropout_state_filter_visitor}")
    self._dropout_state_filter = (
        dropout_state_filter_visitor or _default_dropout_state_filter_visitor)
    with tf.name_scope("DropoutWrapperInit"):

      def tensor_and_const_value(v):
        tensor_value = tf.convert_to_tensor(v)
        const_value = tf.get_static_value(tensor_value)
        return (tensor_value, const_value)

      for prob, attr in [(input_keep_prob, "input_keep_prob"),
                         (state_keep_prob, "state_keep_prob"),
                         (output_keep_prob, "output_keep_prob")]:
        tensor_prob, const_prob = tensor_and_const_value(prob)
        if const_prob is not None:
          if const_prob < 0 or const_prob > 1:
            raise ValueError(
                f"Parameter {attr} must be between 0 and 1. "
                "Received {const_prob}")
          setattr(self, "_%s" % attr, float(const_prob))
        else:
          setattr(self, "_%s" % attr, tensor_prob)

    # Set variational_recurrent, seed before running the code below
    self._variational_recurrent = variational_recurrent
    self._input_size = input_size
    self._seed = seed

    self._recurrent_input_noise = None
    self._recurrent_state_noise = None
    self._recurrent_output_noise = None

    if variational_recurrent:
      if dtype is None:
        raise ValueError(
            "When variational_recurrent=True, dtype must be provided")

      def convert_to_batch_shape(s):
        # Prepend a 1 for the batch dimension; for recurrent
        # variational dropout we use the same dropout mask for all
        # batch elements.
        return tf.concat(([1], tf.TensorShape(s).as_list()), 0)

      def batch_noise(s, inner_seed):
        shape = convert_to_batch_shape(s)
        return tf.random.uniform(shape, seed=inner_seed, dtype=dtype)

      if (not isinstance(self._input_keep_prob, numbers.Real) or
          self._input_keep_prob < 1.0):
        if input_size is None:
          raise ValueError(
              "When variational_recurrent=True and input_keep_prob < 1.0 or "
              "is unknown, input_size must be provided")
        self._recurrent_input_noise = _enumerated_map_structure_up_to(
            input_size,
            lambda i, s: batch_noise(s, inner_seed=self._gen_seed("input", i)),
            input_size)
      self._recurrent_state_noise = _enumerated_map_structure_up_to(
          cell.state_size,
          lambda i, s: batch_noise(s, inner_seed=self._gen_seed("state", i)),
          cell.state_size)
      self._recurrent_output_noise = _enumerated_map_structure_up_to(
          cell.output_size,
          lambda i, s: batch_noise(s, inner_seed=self._gen_seed("output", i)),
          cell.output_size)

  def _gen_seed(self, salt_prefix, index):
    if self._seed is None:
      return None
    salt = "%s_%d" % (salt_prefix, index)
    string = (str(self._seed) + salt).encode("utf-8")
    return int(hashlib.md5(string).hexdigest()[:8], 16) & 0x7FFFFFFF

  @property
  def wrapped_cell(self):
    return self.cell

  def build(self, inputs_shape):
    self.cell.build(inputs_shape)
    self.built = True

  def _variational_recurrent_dropout_value(
      self, unused_index, value, noise, keep_prob):
    """Performs dropout given the pre-calculated noise tensor."""
    # uniform [keep_prob, 1.0 + keep_prob)
    random_tensor = keep_prob + noise

    # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
    binary_tensor = tf.floor(random_tensor)
    ret = tf.divide(value, keep_prob) * binary_tensor
    ret.set_shape(value.get_shape())
    return ret

  def _dropout(self,
               values,
               salt_prefix,
               recurrent_noise,
               keep_prob,
               shallow_filtered_substructure=None):
    """Decides whether to perform standard dropout or recurrent dropout."""

    if shallow_filtered_substructure is None:
      # Put something so we traverse the entire structure; inside the
      # dropout function we check to see if leafs of this are bool or not.
      shallow_filtered_substructure = values

    if not self._variational_recurrent:

      def dropout(i, do_dropout, v):
        if not isinstance(do_dropout, bool) or do_dropout:
          return tf.nn.dropout(
              v, rate=1. - keep_prob, seed=self._gen_seed(salt_prefix, i))
        else:
          return v

      return _enumerated_map_structure_up_to(
          shallow_filtered_substructure, dropout,
          *[shallow_filtered_substructure, values])
    else:

      def dropout(i, do_dropout, v, n):
        if not isinstance(do_dropout, bool) or do_dropout:
          return self._variational_recurrent_dropout_value(i, v, n, keep_prob)
        else:
          return v

      return _enumerated_map_structure_up_to(
          shallow_filtered_substructure, dropout,
          *[shallow_filtered_substructure, values, recurrent_noise])

  def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
    """Runs the wrapped cell and applies dropout.

    Args:
      inputs: A tensor with wrapped cell's input.
      state: A tensor or tuple of tensors with wrapped cell's state.
      cell_call_fn: Wrapped cell's method to use for step computation (cell's
        `__call__` or 'call' method).
      **kwargs: Additional arguments.

    Returns:
      A pair containing:

      - Output: A tensor with cell's output.
      - New state: A tensor or tuple of tensors with new wrapped cell's state.
    """

    def _should_dropout(p):
      return (not isinstance(p, float)) or p < 1

    if _should_dropout(self._input_keep_prob):
      inputs = self._dropout(inputs, "input", self._recurrent_input_noise,
                             self._input_keep_prob)
    output, new_state = cell_call_fn(inputs, state, **kwargs)
    if _should_dropout(self._state_keep_prob):
      # Identify which subsets of the state to perform dropout on and
      # which ones to keep.
      shallow_filtered_substructure = tf.__internal__.nest.get_traverse_shallow_structure(
          self._dropout_state_filter, new_state)
      new_state = self._dropout(new_state, "state", self._recurrent_state_noise,
                                self._state_keep_prob,
                                shallow_filtered_substructure)
    if _should_dropout(self._output_keep_prob):
      output = self._dropout(output, "output", self._recurrent_output_noise,
                             self._output_keep_prob)
    return output, new_state

  def get_config(self):
    """Returns the config of the dropout wrapper."""
    config = {
        "input_keep_prob": self._input_keep_prob,
        "output_keep_prob": self._output_keep_prob,
        "state_keep_prob": self._state_keep_prob,
        "variational_recurrent": self._variational_recurrent,
        "input_size": self._input_size,
        "seed": self._seed,
    }
    if self._dropout_state_filter != _default_dropout_state_filter_visitor:  # pylint: disable=comparison-with-callable
      function, function_type, function_module = _serialize_function_to_config(
          self._dropout_state_filter)
      config.update({"dropout_fn": function,
                     "dropout_fn_type": function_type,
                     "dropout_fn_module": function_module})
    base_config = super(DropoutWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if "dropout_fn" in config:
      config = config.copy()
      dropout_state_filter = _parse_config_to_function(
          config, custom_objects, "dropout_fn", "dropout_fn_type",
          "dropout_fn_module")
      config.pop("dropout_fn")
      config["dropout_state_filter_visitor"] = dropout_state_filter
    return super(DropoutWrapper, cls).from_config(
        config, custom_objects=custom_objects)


@keras_export(v1=["keras.__internal__.legacy.rnn_cell.ResidualWrapper"])
@tf_export(v1=["nn.rnn_cell.ResidualWrapper"])
class ResidualWrapper(_RNNCellWrapperV1):
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, cell, residual_fn=None, **kwargs):
    """Constructs a `ResidualWrapper` for `cell`.

    Args:
      cell: An instance of `RNNCell`.
      residual_fn: (Optional) The function to map raw cell inputs and raw cell
        outputs to the actual cell outputs of the residual network.
        Defaults to calling nest.map_structure on (lambda i, o: i + o), inputs
          and outputs.
      **kwargs: dict of keyword arguments for base layer.
    """
    super(ResidualWrapper, self).__init__(cell, **kwargs)
    self._residual_fn = residual_fn

  def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
    """Run the cell and then apply the residual_fn on its inputs to its outputs.

    Args:
      inputs: cell inputs.
      state: cell state.
      cell_call_fn: Wrapped cell's method to use for step computation (cell's
        `__call__` or 'call' method).
      **kwargs: Additional arguments passed to the wrapped cell's `call`.

    Returns:
      Tuple of cell outputs and new state.

    Raises:
      TypeError: If cell inputs and outputs have different structure (type).
      ValueError: If cell inputs and outputs have different structure (value).
    """
    outputs, new_state = cell_call_fn(inputs, state, **kwargs)

    # Ensure shapes match
    def assert_shape_match(inp, out):
      inp.get_shape().assert_is_compatible_with(out.get_shape())

    def default_residual_fn(inputs, outputs):
      tf.nest.assert_same_structure(inputs, outputs)
      tf.nest.map_structure(assert_shape_match, inputs, outputs)
      return tf.nest.map_structure(lambda inp, out: inp + out, inputs, outputs)

    res_outputs = (self._residual_fn or default_residual_fn)(inputs, outputs)
    return (res_outputs, new_state)

  def get_config(self):
    """Returns the config of the residual wrapper."""
    if self._residual_fn is not None:
      function, function_type, function_module = _serialize_function_to_config(
          self._residual_fn)
      config = {
          "residual_fn": function,
          "residual_fn_type": function_type,
          "residual_fn_module": function_module
      }
    else:
      config = {}
    base_config = super(ResidualWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    if "residual_fn" in config:
      config = config.copy()
      residual_function = _parse_config_to_function(config, custom_objects,
                                                    "residual_fn",
                                                    "residual_fn_type",
                                                    "residual_fn_module")
      config["residual_fn"] = residual_function
    return super(ResidualWrapper, cls).from_config(
        config, custom_objects=custom_objects)


@keras_export(v1=["keras.__internal__.legacy.rnn_cell.DeviceWrapper"])
@tf_export(v1=["nn.rnn_cell.DeviceWrapper"])
class DeviceWrapper(_RNNCellWrapperV1):
  """Operator that ensures an RNNCell runs on a particular device."""

  def __init__(self, cell, device, **kwargs):
    """Construct a `DeviceWrapper` for `cell` with device `device`.

    Ensures the wrapped `cell` is called with `tf.device(device)`.

    Args:
      cell: An instance of `RNNCell`.
      device: A device string or function, for passing to `tf.device`.
      **kwargs: dict of keyword arguments for base layer.
    """
    super(DeviceWrapper, self).__init__(cell, **kwargs)
    self._device = device

  def zero_state(self, batch_size, dtype):
    with tf.name_scope(type(self).__name__ + "ZeroState"):
      with tf.compat.v1.device(self._device):
        return self.cell.zero_state(batch_size, dtype)

  def _call_wrapped_cell(self, inputs, state, cell_call_fn, **kwargs):
    """Run the cell on specified device."""
    with tf.compat.v1.device(self._device):
      return cell_call_fn(inputs, state, **kwargs)

  def get_config(self):
    config = {"device": self._device}
    base_config = super(DeviceWrapper, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _default_dropout_state_filter_visitor(substate):
  from keras.layers.rnn.legacy_cells import LSTMStateTuple  # pylint: disable=g-import-not-at-top
  if isinstance(substate, LSTMStateTuple):
    # Do not perform dropout on the memory state.
    return LSTMStateTuple(c=False, h=True)
  elif isinstance(substate, tf.TensorArray):
    return False
  return True
