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
"""Utilities for RNN cells and layers."""


import tensorflow.compat.v2 as tf

from keras.utils import control_flow_util

# isort: off
from tensorflow.python.platform import tf_logging as logging


def standardize_args(inputs, initial_state, constants, num_constants):
    """Standardizes `__call__` to a single list of tensor inputs.

    When running a model loaded from a file, the input tensors
    `initial_state` and `constants` can be passed to `RNN.__call__()` as part
    of `inputs` instead of by the dedicated keyword arguments. This method
    makes sure the arguments are separated and that `initial_state` and
    `constants` are lists of tensors (or None).

    Args:
      inputs: Tensor or list/tuple of tensors. which may include constants
        and initial states. In that case `num_constant` must be specified.
      initial_state: Tensor or list of tensors or None, initial states.
      constants: Tensor or list of tensors or None, constant tensors.
      num_constants: Expected number of constants (if constants are passed as
        part of the `inputs` list.

    Returns:
      inputs: Single tensor or tuple of tensors.
      initial_state: List of tensors or None.
      constants: List of tensors or None.
    """
    if isinstance(inputs, list):
        # There are several situations here:
        # In the graph mode, __call__ will be only called once. The
        # initial_state and constants could be in inputs (from file loading).
        # In the eager mode, __call__ will be called twice, once during
        # rnn_layer(inputs=input_t, constants=c_t, ...), and second time will be
        # model.fit/train_on_batch/predict with real np data. In the second
        # case, the inputs will contain initial_state and constants as eager
        # tensor.
        #
        # For either case, the real input is the first item in the list, which
        # could be a nested structure itself. Then followed by initial_states,
        # which could be a list of items, or list of list if the initial_state
        # is complex structure, and finally followed by constants which is a
        # flat list.
        assert initial_state is None and constants is None
        if num_constants:
            constants = inputs[-num_constants:]
            inputs = inputs[:-num_constants]
        if len(inputs) > 1:
            initial_state = inputs[1:]
            inputs = inputs[:1]

        if len(inputs) > 1:
            inputs = tuple(inputs)
        else:
            inputs = inputs[0]

    def to_list_or_none(x):
        if x is None or isinstance(x, list):
            return x
        if isinstance(x, tuple):
            return list(x)
        return [x]

    initial_state = to_list_or_none(initial_state)
    constants = to_list_or_none(constants)

    return inputs, initial_state, constants


def is_multiple_state(state_size):
    """Check whether the state_size contains multiple states."""
    return hasattr(state_size, "__len__") and not isinstance(
        state_size, tf.TensorShape
    )


def generate_zero_filled_state_for_cell(cell, inputs, batch_size, dtype):
    if inputs is not None:
        batch_size = tf.shape(inputs)[0]
        dtype = inputs.dtype
    return generate_zero_filled_state(batch_size, cell.state_size, dtype)


def generate_zero_filled_state(batch_size_tensor, state_size, dtype):
    """Generate a zero filled tensor with shape [batch_size, state_size]."""
    if batch_size_tensor is None or dtype is None:
        raise ValueError(
            "batch_size and dtype cannot be None while constructing initial "
            f"state. Received: batch_size={batch_size_tensor}, dtype={dtype}"
        )

    def create_zeros(unnested_state_size):
        flat_dims = tf.TensorShape(unnested_state_size).as_list()
        init_state_size = [batch_size_tensor] + flat_dims
        return tf.zeros(init_state_size, dtype=dtype)

    if tf.nest.is_nested(state_size):
        return tf.nest.map_structure(create_zeros, state_size)
    else:
        return create_zeros(state_size)


def caching_device(rnn_cell):
    """Returns the caching device for the RNN variable.

    This is useful for distributed training, when variable is not located as
    same device as the training worker. By enabling the device cache, this
    allows worker to read the variable once and cache locally, rather than read
    it every time step from remote when it is needed.

    Note that this is assuming the variable that cell needs for each time step
    is having the same value in the forward path, and only gets updated in the
    backprop. It is true for all the default cells (SimpleRNN, GRU, LSTM). If
    the cell body relies on any variable that gets updated every time step, then
    caching device will cause it to read the stall value.

    Args:
      rnn_cell: the rnn cell instance.
    """
    if tf.executing_eagerly():
        # caching_device is not supported in eager mode.
        return None
    if not getattr(rnn_cell, "_enable_caching_device", False):
        return None
    # Don't set a caching device when running in a loop, since it is possible
    # that train steps could be wrapped in a tf.while_loop. In that scenario
    # caching prevents forward computations in loop iterations from re-reading
    # the updated weights.
    if control_flow_util.IsInWhileLoop(tf.compat.v1.get_default_graph()):
        logging.warning(
            "Variable read device caching has been disabled because the "
            "RNN is in tf.while_loop loop context, which will cause "
            "reading stalled value in forward path. This could slow down "
            "the training due to duplicated variable reads. Please "
            "consider updating your code to remove tf.while_loop if possible."
        )
        return None
    if (
        rnn_cell._dtype_policy.compute_dtype
        != rnn_cell._dtype_policy.variable_dtype
    ):
        logging.warning(
            "Variable read device caching has been disabled since it "
            "doesn't work with the mixed precision API. This is "
            "likely to cause a slowdown for RNN training due to "
            "duplicated read of variable for each timestep, which "
            "will be significant in a multi remote worker setting. "
            "Please consider disabling mixed precision API if "
            "the performance has been affected."
        )
        return None
    # Cache the value on the device that access the variable.
    return lambda op: op.device


def config_for_enable_caching_device(rnn_cell):
    """Return the dict config for RNN cell wrt to enable_caching_device field.

    Since enable_caching_device is a internal implementation detail for speed up
    the RNN variable read when running on the multi remote worker setting, we
    don't want this config to be serialized constantly in the JSON. We will only
    serialize this field when a none default value is used to create the cell.
    Args:
      rnn_cell: the RNN cell for serialize.

    Returns:
      A dict which contains the JSON config for enable_caching_device value or
      empty dict if the enable_caching_device value is same as the default
      value.
    """
    default_enable_caching_device = (
        tf.compat.v1.executing_eagerly_outside_functions()
    )
    if rnn_cell._enable_caching_device != default_enable_caching_device:
        return {"enable_caching_device": rnn_cell._enable_caching_device}
    return {}
