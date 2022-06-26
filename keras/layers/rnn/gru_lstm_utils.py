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
"""Utilities used by both the GRU and LSTM classes."""


import uuid

import tensorflow.compat.v2 as tf

# isort: off
from tensorflow.python.eager.context import get_device_name

# The following string constants are used by Defun approach for unified backend
# of LSTM and GRU.
_FUNCTION_API_NAME_ATTRIBUTE = "api_implements"
_FUNCTION_DEVICE_ATTRIBUTE = "api_preferred_device"
CPU_DEVICE_NAME = "CPU"
GPU_DEVICE_NAME = "GPU"

# The following number constants are used to represent the runtime of the defun
# backend function. Since the CPU/GPU implementation are mathematically same, we
# need some signal for the function to indicate which function is executed. This
# is for testing purpose to verify the correctness of swapping backend function.
RUNTIME_UNKNOWN = 0
RUNTIME_CPU = 1
RUNTIME_GPU = 2

CUDNN_AVAILABLE_MSG = "Layer %s will use cuDNN kernels when running on GPU."
CUDNN_NOT_AVAILABLE_MSG = (
    "Layer %s will not use cuDNN kernels since it "
    "doesn't meet the criteria. It will "
    "use a generic GPU kernel as fallback when running "
    "on GPU."
)


def use_new_gru_lstm_impl():
    return False


# TODO(b/169707691): The wrapper can be removed if TFLite doesn't need to rely
# on supportive attributes from LSTM/GRU.
class DefunWrapper:
    """A wrapper with no deep copy of the Defun in LSTM/GRU layer."""

    def __init__(self, time_major, go_backwards, layer_name):
        self.time_major = time_major
        self.go_backwards = go_backwards
        self.layer_name = layer_name
        if self.layer_name not in ["lstm", "gru"]:
            raise ValueError(
                "Defun wrapper only applies to LSTM and GRU layer, "
                "but given {}".format(self.layer_name)
            )
        # The first two attributes are added to support TFLite use case.
        supportive_attributes = {
            "time_major": self.time_major,
            "go_backwards": self.go_backwards,
            _FUNCTION_API_NAME_ATTRIBUTE: self.layer_name
            + "_"
            + str(uuid.uuid4()),
        }
        if self.layer_name == "lstm":
            from keras.layers.rnn import (
                lstm,
            )

            layer_func = lstm.lstm_with_backend_selection
        else:
            from keras.layers.rnn import (
                gru,
            )

            layer_func = gru.gru_with_backend_selection

        self.defun_layer = tf.__internal__.function.defun_with_attributes(
            layer_func, attributes=supportive_attributes, autograph=False
        )

    def __deepcopy__(self, memo):
        new_wrapper = type(self)(
            self.time_major, self.go_backwards, self.layer_name
        )
        memo[id(self)] = new_wrapper
        return new_wrapper


def canonical_to_params(weights, biases, shape, transpose_weights=False):
    """Utility function convert variable to cuDNN compatible parameter.

    Note that Keras weights for kernels are different from the cuDNN format.
    Eg.:

    ```
      Keras                 cuDNN
      [[0, 1, 2],  <--->  [[0, 2, 4],
       [3, 4, 5]]          [1, 3, 5]]
    ```

    If the input weights need to be in a unified format, then set
    `transpose_weights=True` to convert the weights.

    Args:
      weights: list of weights for the individual kernels and recurrent kernels.
      biases: list of biases for individual gate.
      shape: the shape for the converted variables that will be feed to cuDNN.
      transpose_weights: boolean, whether to transpose the weights.

    Returns:
      The converted weights that can be feed to cuDNN ops as param.
    """

    def convert(w):
        return tf.transpose(w) if transpose_weights else w

    weights = [tf.reshape(convert(x), shape) for x in weights]
    biases = [tf.reshape(x, shape) for x in biases]
    return tf.concat(weights + biases, axis=0)


def is_sequence_right_padded(mask):
    """Check the mask tensor and see if it right padded.

    For cuDNN kernel, it uses the sequence length param to skip the tailing
    timestep. If the data is left padded, or not a strict right padding (has
    masked value in the middle of the sequence), then cuDNN kernel won't be work
    properly in those cases.

    Left padded data: [[False, False, True, True, True]].
    Right padded data: [[True, True, True, False, False]].
    Mixture of mask/unmasked data: [[True, False, True, False, False]].

    Note that for the mixed data example above, the actually data RNN should see
    are those 2 Trues (index 0 and 2), the index 1 False should be ignored and
    not pollute the internal states.

    Args:
      mask: the Boolean tensor with shape [batch, timestep]

    Returns:
      boolean scalar tensor, whether the mask is strictly right padded.
    """
    max_seq_length = tf.shape(mask)[1]
    count_of_true = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    right_padded_mask = tf.sequence_mask(count_of_true, maxlen=max_seq_length)
    return tf.reduce_all(tf.equal(mask, right_padded_mask))


def has_fully_masked_sequence(mask):
    # See https://github.com/tensorflow/tensorflow/issues/33148 for more
    # details.  Cudnn kernel will error out if the input sequence contains any
    # fully masked data. We walk around this issue by rerouting the computation
    # to standard kernel, until the issue on cudnn side has been fixed.  For a
    # fully masked sequence, it will contain all Falses. To make it easy to
    # check, we inverse the boolean, check if any of the sequence has all True.
    return tf.reduce_any(tf.reduce_all(tf.logical_not(mask), axis=1))


def is_cudnn_supported_inputs(mask, time_major):
    if time_major:
        mask = tf.transpose(mask)

    return tf.logical_and(
        is_sequence_right_padded(mask),
        tf.logical_not(has_fully_masked_sequence(mask)),
    )


def calculate_sequence_by_mask(mask, time_major):
    """Calculate the sequence length tensor (1-D) based on the masking tensor.

    The masking tensor is a 2D boolean tensor with shape [batch, timestep]. For
    any timestep that should be masked, the corresponding field will be False.
    Consider the following example:
      a = [[True, True, False, False],
           [True, True, True, False]]
    It is a (2, 4) tensor, and the corresponding sequence length result should
    be 1D tensor with value [2, 3]. Note that the masking tensor must be right
    padded that could be checked by, e.g., `is_sequence_right_padded()`.

    Args:
      mask: Boolean tensor with shape [batch, timestep] or [timestep, batch] if
        time_major=True.
      time_major: Boolean, which indicates whether the mask is time major or
        batch major.
    Returns:
      sequence_length: 1D int32 tensor.
    """
    timestep_index = 0 if time_major else 1
    return tf.reduce_sum(tf.cast(mask, tf.int32), axis=timestep_index)


def generate_defun_backend(
    unique_api_name, preferred_device, func, supportive_attributes
):
    function_attributes = {
        _FUNCTION_API_NAME_ATTRIBUTE: unique_api_name,
        _FUNCTION_DEVICE_ATTRIBUTE: preferred_device,
    }
    function_attributes.update(supportive_attributes)
    return tf.__internal__.function.defun_with_attributes(
        func=func, attributes=function_attributes, autograph=False
    )


def get_context_device_type():
    """Parse the current context and return the device type, eg CPU/GPU."""
    current_device = get_device_name()
    if current_device is None:
        return None
    return tf.compat.v1.DeviceSpec.from_string(current_device).device_type


def runtime(runtime_name):
    with tf.device("/cpu:0"):
        return tf.constant(runtime_name, dtype=tf.float32, name="runtime")


def read_variable_value(v):
    """Read the value of a variable if it is variable."""
    if isinstance(v, tf.Variable):
        return v.read_value()
    return v


def function_register(func, *args, **kwargs):
    """Register a specialization of a `Function` into the graph.

    This won't actually call the function with the inputs, and only put the
    function definition into graph. Register function with different input param
    will result into multiple version of functions registered in graph.

    Args:
      func: the `Function` instance that generated by a @defun
      *args: input arguments for the Python function.
      **kwargs: input keyword arguments for the Python function.

    Returns:
      a `ConcreteFunction` object specialized to inputs and execution context.

    Raises:
      ValueError: When the input function is not a defun wrapped python
        function.
    """
    concrete_func = func.get_concrete_function(*args, **kwargs)
    concrete_func.add_to_graph()
    concrete_func.add_gradient_functions_to_graph()
    return concrete_func
