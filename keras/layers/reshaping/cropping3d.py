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
"""Keras cropping layer for 3D input."""


import tensorflow.compat.v2 as tf

from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Cropping3D")
class Cropping3D(Layer):
    """Cropping layer for 3D data (e.g. spatial or spatio-temporal).

      Examples:

    >>> input_shape = (2, 28, 28, 10, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> y = tf.keras.layers.Cropping3D(cropping=(2, 4, 2))(x)
    >>> print(y.shape)
    (2, 24, 20, 6, 3)

    Args:
      cropping: Int, or tuple of 3 ints, or tuple of 3 tuples of 2 ints.
        - If int: the same symmetric cropping
          is applied to depth, height, and width.
        - If tuple of 3 ints: interpreted as two different
          symmetric cropping values for depth, height, and width:
          `(symmetric_dim1_crop, symmetric_dim2_crop, symmetric_dim3_crop)`.
        - If tuple of 3 tuples of 2 ints: interpreted as
          `((left_dim1_crop, right_dim1_crop), (left_dim2_crop,
            right_dim2_crop), (left_dim3_crop, right_dim3_crop))`
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
        while `channels_first` corresponds to inputs with shape
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      5D tensor with shape:
      - If `data_format` is `"channels_last"`:
        `(batch_size, first_axis_to_crop, second_axis_to_crop,
        third_axis_to_crop, depth)`
      - If `data_format` is `"channels_first"`:
        `(batch_size, depth, first_axis_to_crop, second_axis_to_crop,
          third_axis_to_crop)`

    Output shape:
      5D tensor with shape:
      - If `data_format` is `"channels_last"`:
        `(batch_size, first_cropped_axis, second_cropped_axis,
        third_cropped_axis, depth)`
      - If `data_format` is `"channels_first"`:
        `(batch_size, depth, first_cropped_axis, second_cropped_axis,
          third_cropped_axis)`
    """

    def __init__(
        self, cropping=((1, 1), (1, 1), (1, 1)), data_format=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(cropping, int):
            self.cropping = (
                (cropping, cropping),
                (cropping, cropping),
                (cropping, cropping),
            )
        elif hasattr(cropping, "__len__"):
            if len(cropping) != 3:
                raise ValueError(
                    "`cropping` should have 3 elements. "
                    f"Received: {cropping}."
                )
            dim1_cropping = conv_utils.normalize_tuple(
                cropping[0], 2, "1st entry of cropping", allow_zero=True
            )
            dim2_cropping = conv_utils.normalize_tuple(
                cropping[1], 2, "2nd entry of cropping", allow_zero=True
            )
            dim3_cropping = conv_utils.normalize_tuple(
                cropping[2], 2, "3rd entry of cropping", allow_zero=True
            )
            self.cropping = (dim1_cropping, dim2_cropping, dim3_cropping)
        else:
            raise ValueError(
                "`cropping` should be either an int, "
                "a tuple of 3 ints "
                "(symmetric_dim1_crop, symmetric_dim2_crop, "
                "symmetric_dim3_crop), "
                "or a tuple of 3 tuples of 2 ints "
                "((left_dim1_crop, right_dim1_crop),"
                " (left_dim2_crop, right_dim2_crop),"
                " (left_dim3_crop, right_dim2_crop)). "
                f"Received: {cropping}."
            )
        self.input_spec = InputSpec(ndim=5)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()

        if self.data_format == "channels_first":
            if input_shape[2] is not None:
                dim1 = (
                    input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
                )
            else:
                dim1 = None
            if input_shape[3] is not None:
                dim2 = (
                    input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
                )
            else:
                dim2 = None
            if input_shape[4] is not None:
                dim3 = (
                    input_shape[4] - self.cropping[2][0] - self.cropping[2][1]
                )
            else:
                dim3 = None
            return tf.TensorShape(
                [input_shape[0], input_shape[1], dim1, dim2, dim3]
            )
        elif self.data_format == "channels_last":
            if input_shape[1] is not None:
                dim1 = (
                    input_shape[1] - self.cropping[0][0] - self.cropping[0][1]
                )
            else:
                dim1 = None
            if input_shape[2] is not None:
                dim2 = (
                    input_shape[2] - self.cropping[1][0] - self.cropping[1][1]
                )
            else:
                dim2 = None
            if input_shape[3] is not None:
                dim3 = (
                    input_shape[3] - self.cropping[2][0] - self.cropping[2][1]
                )
            else:
                dim3 = None
            return tf.TensorShape(
                [input_shape[0], dim1, dim2, dim3, input_shape[4]]
            )

    def call(self, inputs):

        if self.data_format == "channels_first":
            if (
                self.cropping[0][1]
                == self.cropping[1][1]
                == self.cropping[2][1]
                == 0
            ):
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] :,
                    self.cropping[2][0] :,
                ]
            elif self.cropping[0][1] == self.cropping[1][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] :,
                    self.cropping[2][0] : -self.cropping[2][1],
                ]
            elif self.cropping[1][1] == self.cropping[2][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] :,
                    self.cropping[2][0] :,
                ]
            elif self.cropping[0][1] == self.cropping[2][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] : -self.cropping[1][1],
                    self.cropping[2][0] :,
                ]
            elif self.cropping[0][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] : -self.cropping[1][1],
                    self.cropping[2][0] : -self.cropping[2][1],
                ]
            elif self.cropping[1][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] :,
                    self.cropping[2][0] : -self.cropping[2][1],
                ]
            elif self.cropping[2][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] : -self.cropping[1][1],
                    self.cropping[2][0] :,
                ]
            return inputs[
                :,
                :,
                self.cropping[0][0] : -self.cropping[0][1],
                self.cropping[1][0] : -self.cropping[1][1],
                self.cropping[2][0] : -self.cropping[2][1],
            ]
        else:
            if (
                self.cropping[0][1]
                == self.cropping[1][1]
                == self.cropping[2][1]
                == 0
            ):
                return inputs[
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] :,
                    self.cropping[2][0] :,
                    :,
                ]
            elif self.cropping[0][1] == self.cropping[1][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] :,
                    self.cropping[2][0] : -self.cropping[2][1],
                    :,
                ]
            elif self.cropping[1][1] == self.cropping[2][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] :,
                    self.cropping[2][0] :,
                    :,
                ]
            elif self.cropping[0][1] == self.cropping[2][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] : -self.cropping[1][1],
                    self.cropping[2][0] :,
                    :,
                ]
            elif self.cropping[0][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] : -self.cropping[1][1],
                    self.cropping[2][0] : -self.cropping[2][1],
                    :,
                ]
            elif self.cropping[1][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] :,
                    self.cropping[2][0] : -self.cropping[2][1],
                    :,
                ]
            elif self.cropping[2][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] : -self.cropping[1][1],
                    self.cropping[2][0] :,
                    :,
                ]
            return inputs[
                :,
                self.cropping[0][0] : -self.cropping[0][1],
                self.cropping[1][0] : -self.cropping[1][1],
                self.cropping[2][0] : -self.cropping[2][1],
                :,
            ]

    def get_config(self):
        config = {"cropping": self.cropping, "data_format": self.data_format}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
