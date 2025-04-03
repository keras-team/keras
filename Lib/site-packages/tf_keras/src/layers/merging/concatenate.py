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
"""Layer that concatenates several inputs."""


import tensorflow.compat.v2 as tf

from tf_keras.src import backend
from tf_keras.src.layers.merging.base_merge import _Merge
from tf_keras.src.utils import tf_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Concatenate")
class Concatenate(_Merge):
    """Layer that concatenates a list of inputs.

    It takes as input a list of tensors, all of the same shape except
    for the concatenation axis, and returns a single tensor that is the
    concatenation of all inputs.

    >>> x = np.arange(20).reshape(2, 2, 5)
    >>> print(x)
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
     [[10 11 12 13 14]
      [15 16 17 18 19]]]
    >>> y = np.arange(20, 30).reshape(2, 1, 5)
    >>> print(y)
    [[[20 21 22 23 24]]
     [[25 26 27 28 29]]]
    >>> tf.keras.layers.Concatenate(axis=1)([x, y])
    <tf.Tensor: shape=(2, 3, 5), dtype=int64, numpy=
    array([[[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [20, 21, 22, 23, 24]],
           [[10, 11, 12, 13, 14],
            [15, 16, 17, 18, 19],
            [25, 26, 27, 28, 29]]])>

    >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
    >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
    >>> concatted = tf.keras.layers.Concatenate()([x1, x2])
    >>> concatted.shape
    TensorShape([5, 16])

    """

    def __init__(self, axis=-1, **kwargs):
        """Instantiates a Concatenate layer.

        >>> x = np.arange(20).reshape(2, 2, 5)
        >>> print(x)
        [[[ 0  1  2  3  4]
          [ 5  6  7  8  9]]
         [[10 11 12 13 14]
          [15 16 17 18 19]]]
        >>> y = np.arange(20, 30).reshape(2, 1, 5)
        >>> print(y)
        [[[20 21 22 23 24]]
         [[25 26 27 28 29]]]
        >>> tf.keras.layers.Concatenate(axis=1)([x, y])
        <tf.Tensor: shape=(2, 3, 5), dtype=int64, numpy=
        array([[[ 0,  1,  2,  3,  4],
                [ 5,  6,  7,  8,  9],
                [20, 21, 22, 23, 24]],
               [[10, 11, 12, 13, 14],
                [15, 16, 17, 18, 19],
                [25, 26, 27, 28, 29]]])>

        Args:
          axis: Axis along which to concatenate.
          **kwargs: standard layer keyword arguments.
        """
        super().__init__(**kwargs)
        self.axis = axis
        self.supports_masking = True
        self._reshape_required = False

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # Used purely for shape validation.
        if len(input_shape) < 1 or not isinstance(input_shape[0], tuple):
            raise ValueError(
                "A `Concatenate` layer should be called on a list of "
                f"at least 1 input. Received: input_shape={input_shape}"
            )
        if all(shape is None for shape in input_shape):
            return
        reduced_inputs_shapes = [list(shape) for shape in input_shape]
        shape_set = set()
        for i in range(len(reduced_inputs_shapes)):
            del reduced_inputs_shapes[i][self.axis]
            shape_set.add(tuple(reduced_inputs_shapes[i]))

        if len(shape_set) != 1:
            err_msg = (
                "A `Concatenate` layer requires inputs with matching shapes "
                "except for the concatenation axis. "
                f"Received: input_shape={input_shape}"
            )
            # Make sure all the shapes have same ranks.
            ranks = set(len(shape) for shape in shape_set)
            if len(ranks) != 1:
                raise ValueError(err_msg)
            # Get the only rank for the set.
            (rank,) = ranks
            for axis in range(rank):
                # Skip the Nones in the shape since they are dynamic, also the
                # axis for concat has been removed above.
                unique_dims = set(
                    shape[axis]
                    for shape in shape_set
                    if shape[axis] is not None
                )
                if len(unique_dims) > 1:
                    raise ValueError(err_msg)

    def _merge_function(self, inputs):
        return backend.concatenate(inputs, axis=self.axis)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if (not isinstance(input_shape, (tuple, list))) or (
            not isinstance(input_shape[0], (tuple, list))
        ):
            # The tf_utils.shape_type_conversion decorator turns tensorshapes
            # into tuples, so we need to verify that `input_shape` is a
            # list/tuple, *and* that the individual elements are themselves
            # shape tuples.
            raise ValueError(
                "A `Concatenate` layer should be called on a list of inputs. "
                f"Received: input_shape={input_shape}"
            )
        input_shapes = input_shape
        output_shape = list(input_shapes[0])
        for shape in input_shapes[1:]:
            if output_shape[self.axis] is None or shape[self.axis] is None:
                output_shape[self.axis] = None
                break
            output_shape[self.axis] += shape[self.axis]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        if not isinstance(mask, (tuple, list)):
            raise ValueError(f"`mask` should be a list. Received mask={mask}")
        if not isinstance(inputs, (tuple, list)):
            raise ValueError(
                f"`inputs` should be a list. Received: inputs={inputs}"
            )
        if len(mask) != len(inputs):
            raise ValueError(
                "The lists `inputs` and `mask` should have the same length. "
                f"Received: inputs={inputs} of length {len(inputs)}, and "
                f"mask={mask} of length {len(mask)}"
            )
        if all(m is None for m in mask):
            return None
        # Make a list of masks while making sure
        # the dimensionality of each mask
        # is the same as the corresponding input.
        masks = []
        for input_i, mask_i in zip(inputs, mask):
            if mask_i is None:
                # Input is unmasked. Append all 1s to masks,
                masks.append(tf.ones_like(input_i, dtype="bool"))
            elif backend.ndim(mask_i) < backend.ndim(input_i):
                # Mask is smaller than the input, expand it
                masks.append(tf.expand_dims(mask_i, axis=-1))
            else:
                masks.append(mask_i)
        concatenated = backend.concatenate(masks, axis=self.axis)
        return backend.all(concatenated, axis=-1, keepdims=False)

    def get_config(self):
        config = {
            "axis": self.axis,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.layers.concatenate")
def concatenate(inputs, axis=-1, **kwargs):
    """Functional interface to the `Concatenate` layer.

    >>> x = np.arange(20).reshape(2, 2, 5)
    >>> print(x)
    [[[ 0  1  2  3  4]
      [ 5  6  7  8  9]]
     [[10 11 12 13 14]
      [15 16 17 18 19]]]
    >>> y = np.arange(20, 30).reshape(2, 1, 5)
    >>> print(y)
    [[[20 21 22 23 24]]
     [[25 26 27 28 29]]]
    >>> tf.keras.layers.concatenate([x, y],
    ...                             axis=1)
    <tf.Tensor: shape=(2, 3, 5), dtype=int64, numpy=
    array([[[ 0,  1,  2,  3,  4],
          [ 5,  6,  7,  8,  9],
          [20, 21, 22, 23, 24]],
         [[10, 11, 12, 13, 14],
          [15, 16, 17, 18, 19],
          [25, 26, 27, 28, 29]]])>

    Args:
        inputs: A list of input tensors.
        axis: Concatenation axis.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor, the concatenation of the inputs alongside axis `axis`.
    """
    return Concatenate(axis=axis, **kwargs)(inputs)

