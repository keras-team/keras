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
"""Contains the Reshape layer."""


import numpy as np
import tensorflow.compat.v2 as tf

from keras.engine.base_layer import Layer

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Reshape")
class Reshape(Layer):
    """Layer that reshapes inputs into the given shape.

    Input shape:
      Arbitrary, although all dimensions in the input shape must be known/fixed.
      Use the keyword argument `input_shape` (tuple of integers, does not
      include the samples/batch size axis) when using this layer as the first
      layer in a model.

    Output shape:
      `(batch_size,) + target_shape`

    Example:

    >>> # as first layer in a Sequential model
    >>> model = tf.keras.Sequential()
    >>> model.add(tf.keras.layers.Reshape((3, 4), input_shape=(12,)))
    >>> # model.output_shape == (None, 3, 4), `None` is the batch size.
    >>> model.output_shape
    (None, 3, 4)

    >>> # as intermediate layer in a Sequential model
    >>> model.add(tf.keras.layers.Reshape((6, 2)))
    >>> model.output_shape
    (None, 6, 2)

    >>> # also supports shape inference using `-1` as dimension
    >>> model.add(tf.keras.layers.Reshape((-1, 2, 2)))
    >>> model.output_shape
    (None, 3, 2, 2)
    """

    def __init__(self, target_shape, **kwargs):
        """Creates a `tf.keras.layers.Reshape`  layer instance.

        Args:
          target_shape: Target shape. Tuple of integers, does not include the
            samples dimension (batch size).
          **kwargs: Any additional layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.target_shape = tuple(target_shape)

    def _fix_unknown_dimension(self, input_shape, output_shape):
        """Find and replace a missing dimension in an output shape.

        This is a near direct port of the internal Numpy function
        `_fix_unknown_dimension` in `numpy/core/src/multiarray/shape.c`

        Args:
          input_shape: Shape of array being reshaped
          output_shape: Desired shape of the array with at most a single -1
            which indicates a dimension that should be derived from the input
            shape.

        Returns:
          The new output shape with a -1 replaced with its computed value.

        Raises:
          ValueError: If the total array size of the output_shape is
          different than the input_shape, or more than one unknown dimension
          is specified.
        """
        output_shape = list(output_shape)
        msg = (
            "total size of new array must be unchanged, "
            "input_shape = {}, output_shape = {}".format(
                input_shape, output_shape
            )
        )

        known, unknown = 1, None
        for index, dim in enumerate(output_shape):
            if dim < 0:
                if unknown is None:
                    unknown = index
                else:
                    raise ValueError(
                        "There must be at most one unknown dimension in "
                        f"output_shape. Received: output_shape={output_shape}."
                    )
            else:
                known *= dim

        original = np.prod(input_shape, dtype=int)
        if unknown is not None:
            if known == 0 or original % known != 0:
                raise ValueError(msg)
            output_shape[unknown] = original // known
        elif original != known:
            raise ValueError(msg)
        return output_shape

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if None in input_shape[1:]:
            output_shape = [input_shape[0]]
            # input shape (partially) unknown? replace -1's with None's
            output_shape += tuple(
                s if s != -1 else None for s in self.target_shape
            )
        else:
            output_shape = [input_shape[0]]
            output_shape += self._fix_unknown_dimension(
                input_shape[1:], self.target_shape
            )
        return tf.TensorShape(output_shape)

    def call(self, inputs):
        result = tf.reshape(inputs, (tf.shape(inputs)[0],) + self.target_shape)
        if not tf.executing_eagerly():
            # Set the static shape for the result since it might lost during
            # array_ops reshape, eg, some `None` dim in the result could be
            # inferred.
            result.set_shape(self.compute_output_shape(inputs.shape))
        return result

    def get_config(self):
        config = {"target_shape": self.target_shape}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
