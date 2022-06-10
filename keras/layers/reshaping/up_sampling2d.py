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
"""Keras upsampling layer for 2D inputs."""


import tensorflow.compat.v2 as tf

from keras import backend
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.UpSampling2D")
class UpSampling2D(Layer):
    """Upsampling layer for 2D inputs.

    Repeats the rows and columns of the data
    by `size[0]` and `size[1]` respectively.

    Examples:

    >>> input_shape = (2, 2, 1, 3)
    >>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
    >>> print(x)
    [[[[ 0  1  2]]
      [[ 3  4  5]]]
     [[[ 6  7  8]]
      [[ 9 10 11]]]]
    >>> y = tf.keras.layers.UpSampling2D(size=(1, 2))(x)
    >>> print(y)
    tf.Tensor(
      [[[[ 0  1  2]
         [ 0  1  2]]
        [[ 3  4  5]
         [ 3  4  5]]]
       [[[ 6  7  8]
         [ 6  7  8]]
        [[ 9 10 11]
         [ 9 10 11]]]], shape=(2, 2, 2, 3), dtype=int64)

    Args:
      size: Int, or tuple of 2 integers.
        The upsampling factors for rows and columns.
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
      interpolation: A string, one of `"area"`, `"bicubic"`, `"bilinear"`,
        `"gaussian"`, `"lanczos3"`, `"lanczos5"`, `"mitchellcubic"`,
        `"nearest"`.

    Input shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch_size, rows, cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch_size, channels, rows, cols)`

    Output shape:
      4D tensor with shape:
      - If `data_format` is `"channels_last"`:
          `(batch_size, upsampled_rows, upsampled_cols, channels)`
      - If `data_format` is `"channels_first"`:
          `(batch_size, channels, upsampled_rows, upsampled_cols)`
    """

    def __init__(
        self, size=(2, 2), data_format=None, interpolation="nearest", **kwargs
    ):
        super().__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.size = conv_utils.normalize_tuple(size, 2, "size")
        interpolations = {
            "area": tf.image.ResizeMethod.AREA,
            "bicubic": tf.image.ResizeMethod.BICUBIC,
            "bilinear": tf.image.ResizeMethod.BILINEAR,
            "gaussian": tf.image.ResizeMethod.GAUSSIAN,
            "lanczos3": tf.image.ResizeMethod.LANCZOS3,
            "lanczos5": tf.image.ResizeMethod.LANCZOS5,
            "mitchellcubic": tf.image.ResizeMethod.MITCHELLCUBIC,
            "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        }
        interploations_list = '"' + '", "'.join(interpolations.keys()) + '"'
        if interpolation not in interpolations:
            raise ValueError(
                "`interpolation` argument should be one of: "
                f'{interploations_list}. Received: "{interpolation}".'
            )
        self.interpolation = interpolation
        self.input_spec = InputSpec(ndim=4)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        if self.data_format == "channels_first":
            height = (
                self.size[0] * input_shape[2]
                if input_shape[2] is not None
                else None
            )
            width = (
                self.size[1] * input_shape[3]
                if input_shape[3] is not None
                else None
            )
            return tf.TensorShape(
                [input_shape[0], input_shape[1], height, width]
            )
        else:
            height = (
                self.size[0] * input_shape[1]
                if input_shape[1] is not None
                else None
            )
            width = (
                self.size[1] * input_shape[2]
                if input_shape[2] is not None
                else None
            )
            return tf.TensorShape(
                [input_shape[0], height, width, input_shape[3]]
            )

    def call(self, inputs):
        return backend.resize_images(
            inputs,
            self.size[0],
            self.size[1],
            self.data_format,
            interpolation=self.interpolation,
        )

    def get_config(self):
        config = {
            "size": self.size,
            "data_format": self.data_format,
            "interpolation": self.interpolation,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
