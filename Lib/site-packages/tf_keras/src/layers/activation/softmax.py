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
"""Softmax activation layer."""


import tensorflow.compat.v2 as tf

from tf_keras.src import backend
from tf_keras.src.engine.base_layer import Layer
from tf_keras.src.utils import tf_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


def _large_compatible_negative(tensor_type):
    """Large negative number as Tensor.

    This function is necessary because the standard value for epsilon
    in this module (-1e9) cannot be represented using tf.float16

    Args:
        tensor_type: a dtype to determine the type.

    Returns:
        a large negative number.
    """
    # In case of dtype=float16 (e.g., for mixed-precision), the largest
    # negative number (dtypes.float16.min) is divided by 2, in order to
    # avoid overflows when summing negative inputs.
    if tensor_type == tf.float16:
        return tf.float16.min / 2.0
    return -1e9


@keras_export("keras.layers.Softmax")
class Softmax(Layer):
    """Softmax activation function.

    Example without mask:

    >>> inp = np.asarray([[1., 2., 1.]])
    >>> layer = tf.keras.layers.Softmax()
    >>> layer(inp).numpy()
    array([[0.21194157, 0.5761169 , 0.21194157]], dtype=float32)
    >>> mask = np.asarray([[True, False, True]], dtype=bool)
    >>> layer(inp, mask).numpy()
    array([[0.5, 0. , 0.5]], dtype=float32)

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        axis: Integer, or list of Integers, axis along which the softmax
            normalization is applied.
    Call arguments:
        inputs: The inputs, or logits to the softmax layer.
        mask: A boolean mask of the same shape as `inputs`. The mask
            specifies 1 to keep and 0 to mask. Defaults to `None`.


    Returns:
        Softmaxed output with the same shape as `inputs`.
    """

    def __init__(self, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis

    def call(self, inputs, mask=None):
        if mask is not None:
            # Since mask is 1.0 for positions we want to keep and 0.0 for masked
            # positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -1e.9 for masked positions.
            adder = (1.0 - tf.cast(mask, inputs.dtype)) * (
                _large_compatible_negative(inputs.dtype)
            )

            # Since we are adding it to the raw scores before the softmax, this
            # is effectively the same as removing these entirely.
            inputs += adder
        if isinstance(self.axis, (tuple, list)):
            if len(self.axis) > 1:
                return tf.exp(
                    inputs
                    - tf.reduce_logsumexp(inputs, axis=self.axis, keepdims=True)
                )
            else:
                return backend.softmax(inputs, axis=self.axis[0])
        return backend.softmax(inputs, axis=self.axis)

    def get_config(self):
        config = {"axis": self.axis}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape

