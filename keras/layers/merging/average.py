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
"""Layer that averages several inputs."""


from keras.layers.merging.base_merge import _Merge

from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Average")
class Average(_Merge):
    """Layer that averages a list of inputs element-wise.

    It takes as input a list of tensors, all of the same shape, and returns
    a single tensor (also of the same shape).

    Example:

    >>> x1 = np.ones((2, 2))
    >>> x2 = np.zeros((2, 2))
    >>> y = tf.keras.layers.Average()([x1, x2])
    >>> y.numpy().tolist()
    [[0.5, 0.5], [0.5, 0.5]]

    Usage in a functional model:

    >>> input1 = tf.keras.layers.Input(shape=(16,))
    >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = tf.keras.layers.Input(shape=(32,))
    >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
    >>> avg = tf.keras.layers.Average()([x1, x2])
    >>> out = tf.keras.layers.Dense(4)(avg)
    >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

    Raises:
      ValueError: If there is a shape mismatch between the inputs and the shapes
        cannot be broadcasted to match.
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output += inputs[i]
        return output / len(inputs)


@keras_export("keras.layers.average")
def average(inputs, **kwargs):
    """Functional interface to the `tf.keras.layers.Average` layer.

    Example:

    >>> x1 = np.ones((2, 2))
    >>> x2 = np.zeros((2, 2))
    >>> y = tf.keras.layers.Average()([x1, x2])
    >>> y.numpy().tolist()
    [[0.5, 0.5], [0.5, 0.5]]

    Usage in a functional model:

    >>> input1 = tf.keras.layers.Input(shape=(16,))
    >>> x1 = tf.keras.layers.Dense(8, activation='relu')(input1)
    >>> input2 = tf.keras.layers.Input(shape=(32,))
    >>> x2 = tf.keras.layers.Dense(8, activation='relu')(input2)
    >>> avg = tf.keras.layers.Average()([x1, x2])
    >>> out = tf.keras.layers.Dense(4)(avg)
    >>> model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)

    Args:
        inputs: A list of input tensors.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor, the average of the inputs.

    Raises:
      ValueError: If there is a shape mismatch between the inputs and the shapes
        cannot be broadcasted to match.
    """
    return Average(**kwargs)(inputs)
