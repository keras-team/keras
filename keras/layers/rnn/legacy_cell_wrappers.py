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

from keras.layers.rnn import base_cell_wrappers
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
class DropoutWrapper(base_cell_wrappers.DropoutWrapperBase,
                     _RNNCellWrapperV1):
  """Operator adding dropout to inputs and outputs of the given cell."""

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    super(DropoutWrapper, self).__init__(*args, **kwargs)

  __init__.__doc__ = base_cell_wrappers.DropoutWrapperBase.__init__.__doc__


@keras_export(v1=["keras.__internal__.legacy.rnn_cell.ResidualWrapper"])
@tf_export(v1=["nn.rnn_cell.ResidualWrapper"])
class ResidualWrapper(base_cell_wrappers.ResidualWrapperBase,
                      _RNNCellWrapperV1):
  """RNNCell wrapper that ensures cell inputs are added to the outputs."""

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    super(ResidualWrapper, self).__init__(*args, **kwargs)

  __init__.__doc__ = base_cell_wrappers.ResidualWrapperBase.__init__.__doc__


@keras_export(v1=["keras.__internal__.legacy.rnn_cell.DeviceWrapper"])
@tf_export(v1=["nn.rnn_cell.DeviceWrapper"])
class DeviceWrapper(base_cell_wrappers.DeviceWrapperBase,
                    _RNNCellWrapperV1):

  def __init__(self, *args, **kwargs):  # pylint: disable=useless-super-delegation
    super(DeviceWrapper, self).__init__(*args, **kwargs)

  __init__.__doc__ = base_cell_wrappers.DeviceWrapperBase.__init__.__doc__
