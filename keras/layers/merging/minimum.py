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
"""Layer that computes the minimum (element-wise) of several inputs."""


import tensorflow.compat.v2 as tf

from keras.layers.merging.base_merge import _Merge

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Minimum")
class Minimum(_Merge):
    """Layer that computes the minimum (element-wise) a list of inputs.

    It takes as input a list of tensors, all of the same shape, and returns
    a single tensor (also of the same shape).

    >>> tf.keras.layers.Minimum()([np.arange(5).reshape(5, 1),
    ...                            np.arange(5, 10).reshape(5, 1)])
    <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
    array([[0],
         [1],
         [2],
         [3],
         [4]])>

    >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
    >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
    >>> minned = tf.keras.layers.Minimum()([x1, x2])
    >>> minned.shape
    TensorShape([5, 8])
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = tf.minimum(output, inputs[i])
        return output


@keras_export("keras.layers.minimum")
def minimum(inputs, **kwargs):
    """Functional interface to the `Minimum` layer.

    Args:
        inputs: A list of input tensors.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor, the element-wise minimum of the inputs.
    """
    return Minimum(**kwargs)(inputs)
