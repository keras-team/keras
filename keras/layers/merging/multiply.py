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
"""Layer that multiplies (element-wise) several inputs."""


from keras.layers.merging.base_merge import _Merge

from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Multiply")
class Multiply(_Merge):
    """Layer that multiplies (element-wise) a list of inputs.

    It takes as input a list of tensors, all of the same shape, and returns
    a single tensor (also of the same shape).

    >>> tf.keras.layers.Multiply()([np.arange(5).reshape(5, 1),
    ...                             np.arange(5, 10).reshape(5, 1)])
    <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
    array([[ 0],
         [ 6],
         [14],
         [24],
         [36]])>

    >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
    >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
    >>> multiplied = tf.keras.layers.Multiply()([x1, x2])
    >>> multiplied.shape
    TensorShape([5, 8])
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output = output * inputs[i]
        return output


@keras_export("keras.layers.multiply")
def multiply(inputs, **kwargs):
    """Functional interface to the `Multiply` layer.

    Example:

    >>> x1 = np.arange(3.0)
    >>> x2 = np.arange(3.0)
    >>> tf.keras.layers.multiply([x1, x2])
    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([0., 1., 4.], ...)>

    Usage in a functional model:

    >>> input1 = tf.keras.layers.Input(shape=(16,))
    >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1) #shape=(None, 8)
    >>> input2 = tf.keras.layers.Input(shape=(32,))
    >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2) #shape=(None, 8)
    >>> out = tf.keras.layers.multiply([x1,x2]) #shape=(None, 8)
    >>> out = tf.keras.layers.Dense(4)(out)
    >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

    Args:
        inputs: A list of input tensors.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor, the element-wise product of the inputs.
    """
    return Multiply(**kwargs)(inputs)
