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
"""Keras cropping layer for 2D input."""


import tensorflow.compat.v2 as tf

from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Cropping2D")
class Cropping2D(Layer):
    """Cropping layer for 2D input (e.g. picture).

    It crops along spatial dimensions, i.e. height and width.

    Examples:

    >>> input_shape = (2, 28, 28, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> y = tf.keras.layers.Cropping2D(cropping=((2, 2), (4, 4)))(x)
    >>> print(y.shape)
    (2, 24, 20, 3)

    Args:
      cropping: Int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        - If int: the same symmetric cropping
          is applied to height and width.
        - If tuple of 2 ints:
          interpreted as two different
          symmetric cropping values for height and width:
          `(symmetric_height_crop, symmetric_width_crop)`.
        - If tuple of 2 tuples of 2 ints:
          interpreted as
          `((top_crop, bottom_crop), (left_crop, right_crop))`
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch_size, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
        `(batch_size, rows, cols, channels)`
      - If `data_format` is `"channels_first"`:
        `(batch_size, channels, rows, cols)`

    Output shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
        `(batch_size, cropped_rows, cropped_cols, channels)`
      - If `data_format` is `"channels_first"`:
        `(batch_size, channels, cropped_rows, cropped_cols)`
    """

    def __init__(self, cropping=((0, 0), (0, 0)), data_format=None, **kwargs):
        super().__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        if isinstance(cropping, int):
            self.cropping = ((cropping, cropping), (cropping, cropping))
        elif hasattr(cropping, "__len__"):
            if len(cropping) != 2:
                raise ValueError(
                    "`cropping` should have two elements. "
                    f"Received: {cropping}."
                )
            height_cropping = conv_utils.normalize_tuple(
                cropping[0], 2, "1st entry of cropping", allow_zero=True
            )
            width_cropping = conv_utils.normalize_tuple(
                cropping[1], 2, "2nd entry of cropping", allow_zero=True
            )
            self.cropping = (height_cropping, width_cropping)
        else:
            raise ValueError(
                "`cropping` should be either an int, "
                "a tuple of 2 ints "
                "(symmetric_height_crop, symmetric_width_crop), "
                "or a tuple of 2 tuples of 2 ints "
                "((top_crop, bottom_crop), (left_crop, right_crop)). "
                f"Received: {cropping}."
            )
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()

        if self.data_format == "channels_first":
            return tf.TensorShape(
                [
                    input_shape[0],
                    input_shape[1],
                    input_shape[2] - self.cropping[0][0] - self.cropping[0][1]
                    if input_shape[2]
                    else None,
                    input_shape[3] - self.cropping[1][0] - self.cropping[1][1]
                    if input_shape[3]
                    else None,
                ]
            )
        else:
            return tf.TensorShape(
                [
                    input_shape[0],
                    input_shape[1] - self.cropping[0][0] - self.cropping[0][1]
                    if input_shape[1]
                    else None,
                    input_shape[2] - self.cropping[1][0] - self.cropping[1][1]
                    if input_shape[2]
                    else None,
                    input_shape[3],
                ]
            )

    def call(self, inputs):

        if self.data_format == "channels_first":
            if (
                inputs.shape[2] is not None
                and sum(self.cropping[0]) >= inputs.shape[2]
            ) or (
                inputs.shape[3] is not None
                and sum(self.cropping[1]) >= inputs.shape[3]
            ):
                raise ValueError(
                    "Argument `cropping` must be "
                    "greater than the input shape. Received: inputs.shape="
                    f"{inputs.shape}, and cropping={self.cropping}"
                )
            if self.cropping[0][1] == self.cropping[1][1] == 0:
                return inputs[
                    :, :, self.cropping[0][0] :, self.cropping[1][0] :
                ]
            elif self.cropping[0][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] : -self.cropping[1][1],
                ]
            elif self.cropping[1][1] == 0:
                return inputs[
                    :,
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] :,
                ]
            return inputs[
                :,
                :,
                self.cropping[0][0] : -self.cropping[0][1],
                self.cropping[1][0] : -self.cropping[1][1],
            ]
        else:
            if (
                inputs.shape[1] is not None
                and sum(self.cropping[0]) >= inputs.shape[1]
            ) or (
                inputs.shape[2] is not None
                and sum(self.cropping[1]) >= inputs.shape[2]
            ):
                raise ValueError(
                    "Argument `cropping` must be "
                    "greater than the input shape. Received: inputs.shape="
                    f"{inputs.shape}, and cropping={self.cropping}"
                )
            if self.cropping[0][1] == self.cropping[1][1] == 0:
                return inputs[
                    :, self.cropping[0][0] :, self.cropping[1][0] :, :
                ]
            elif self.cropping[0][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] :,
                    self.cropping[1][0] : -self.cropping[1][1],
                    :,
                ]
            elif self.cropping[1][1] == 0:
                return inputs[
                    :,
                    self.cropping[0][0] : -self.cropping[0][1],
                    self.cropping[1][0] :,
                    :,
                ]
            return inputs[
                :,
                self.cropping[0][0] : -self.cropping[0][1],
                self.cropping[1][0] : -self.cropping[1][1],
                :,
            ]

    def get_config(self):
        config = {"cropping": self.cropping, "data_format": self.data_format}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
