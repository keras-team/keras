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
"""Layer that computes the dot product between two inputs."""


from keras import backend
from keras.engine import base_layer_utils
from keras.layers.merging.base_merge import _Merge
from keras.utils import tf_utils
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Dot")
class Dot(_Merge):
    """Layer that computes a dot product between samples in two tensors.

    E.g. if applied to a list of two tensors `a` and `b` of shape
    `(batch_size, n)`, the output will be a tensor of shape `(batch_size, 1)`
    where each entry `i` will be the dot product between
    `a[i]` and `b[i]`.

    >>> x = np.arange(10).reshape(1, 5, 2)
    >>> print(x)
    [[[0 1]
      [2 3]
      [4 5]
      [6 7]
      [8 9]]]
    >>> y = np.arange(10, 20).reshape(1, 2, 5)
    >>> print(y)
    [[[10 11 12 13 14]
      [15 16 17 18 19]]]
    >>> tf.keras.layers.Dot(axes=(1, 2))([x, y])
    <tf.Tensor: shape=(1, 2, 2), dtype=int64, numpy=
    array([[[260, 360],
            [320, 445]]])>

    >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
    >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
    >>> dotted = tf.keras.layers.Dot(axes=1)([x1, x2])
    >>> dotted.shape
    TensorShape([5, 1])


    """

    def __init__(self, axes, normalize=False, **kwargs):
        """Initializes a layer that computes the element-wise dot product.

          >>> x = np.arange(10).reshape(1, 5, 2)
          >>> print(x)
          [[[0 1]
            [2 3]
            [4 5]
            [6 7]
            [8 9]]]
          >>> y = np.arange(10, 20).reshape(1, 2, 5)
          >>> print(y)
          [[[10 11 12 13 14]
            [15 16 17 18 19]]]
          >>> tf.keras.layers.Dot(axes=(1, 2))([x, y])
          <tf.Tensor: shape=(1, 2, 2), dtype=int64, numpy=
          array([[[260, 360],
                  [320, 445]]])>

        Args:
          axes: Integer or tuple of integers,
            axis or axes along which to take the dot product. If a tuple, should
            be two integers corresponding to the desired axis from the first input
            and the desired axis from the second input, respectively. Note that the
            size of the two selected axes must match.
          normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
          **kwargs: Standard layer keyword arguments.
        """
        super().__init__(**kwargs)
        if not isinstance(axes, int):
            if not isinstance(axes, (list, tuple)):
                raise TypeError(
                    "Invalid type for argument `axes`: it should be "
                    f"a list or an int. Received: axes={axes}"
                )
            if len(axes) != 2:
                raise ValueError(
                    "Invalid format for argument `axes`: it should contain two "
                    f"elements. Received: axes={axes}"
                )
            if not isinstance(axes[0], int) or not isinstance(axes[1], int):
                raise ValueError(
                    "Invalid format for argument `axes`: list elements should be "
                    f"integers. Received: axes={axes}"
                )
        self.axes = axes
        self.normalize = normalize
        self.supports_masking = True
        self._reshape_required = False

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape[0], tuple) or len(input_shape) != 2:
            raise ValueError(
                "A `Dot` layer should be called on a list of 2 inputs. "
                f"Received: input_shape={input_shape}"
            )
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1 is None or shape2 is None:
            return
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        if shape1[axes[0]] != shape2[axes[1]]:
            raise ValueError(
                "Incompatible input shapes: "
                f"axis values {shape1[axes[0]]} (at axis {axes[0]}) != "
                f"{shape2[axes[1]]} (at axis {axes[1]}). "
                f"Full input shapes: {shape1}, {shape2}"
            )

    def _merge_function(self, inputs):
        base_layer_utils.no_ragged_support(inputs, self.name)
        if len(inputs) != 2:
            raise ValueError(
                "A `Dot` layer should be called on exactly 2 inputs. "
                f"Received: inputs={inputs}"
            )
        x1 = inputs[0]
        x2 = inputs[1]
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [
                    self.axes % backend.ndim(x1),
                    self.axes % backend.ndim(x2),
                ]
            else:
                axes = [self.axes] * 2
        else:
            axes = []
            for i in range(len(self.axes)):
                if self.axes[i] < 0:
                    axes.append(self.axes[i] % backend.ndim(inputs[i]))
                else:
                    axes.append(self.axes[i])
        if self.normalize:
            x1 = tf.linalg.l2_normalize(x1, axis=axes[0])
            x2 = tf.linalg.l2_normalize(x2, axis=axes[1])
        output = backend.batch_dot(x1, x2, axes)
        return output

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2:
            raise ValueError(
                "A `Dot` layer should be called on a list of 2 inputs. "
                f"Received: input_shape={input_shape}"
            )
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        shape1.pop(axes[0])
        shape2.pop(axes[1])
        shape2.pop(0)
        output_shape = shape1 + shape2
        if len(output_shape) == 1:
            output_shape += [1]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            "axes": self.axes,
            "normalize": self.normalize,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


@keras_export("keras.layers.dot")
def dot(inputs, axes, normalize=False, **kwargs):
    """Functional interface to the `Dot` layer.

    Args:
        inputs: A list of input tensors (at least 2).
        axes: Integer or tuple of integers,
            axis or axes along which to take the dot product.
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.

    Returns:
        A tensor, the dot product of the samples from the inputs.
    """
    return Dot(axes=axes, normalize=normalize, **kwargs)(inputs)
