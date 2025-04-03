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
"""Layer that adds several inputs."""


from tf_keras.src.layers.merging.base_merge import _Merge

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Add")
class Add(_Merge):
    """Layer that adds a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x1 = tf.random.normal(input_shape)
    >>> x2 = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Add()([x1, x2])
    >>> print(y.shape)
    (2, 3, 4)

    Used in a functional model:

    >>> input1 = tf.keras.layers.Input(shape=(16,))
    >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = tf.keras.layers.Input(shape=(32,))
    >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
    >>> # equivalent to `added = tf.keras.layers.add([x1, x2])`
    >>> added = tf.keras.layers.Add()([x1, x2])
    >>> out = tf.keras.layers.Dense(4)(added)
    >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output


@keras_export("keras.layers.add")
def add(inputs, **kwargs):
    """Functional interface to the `tf.keras.layers.Add` layer.

    Args:
        inputs: A list of input tensors with the same shape.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor as the sum of the inputs. It has the same shape as the inputs.

    Examples:

    >>> input_shape = (2, 3, 4)
    >>> x1 = tf.random.normal(input_shape)
    >>> x2 = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.add([x1, x2])
    >>> print(y.shape)
    (2, 3, 4)

    Used in a functional model:

    >>> input1 = tf.keras.layers.Input(shape=(16,))
    >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = tf.keras.layers.Input(shape=(32,))
    >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
    >>> added = tf.keras.layers.add([x1, x2])
    >>> out = tf.keras.layers.Dense(4)(added)
    >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

    """
    return Add(**kwargs)(inputs)

