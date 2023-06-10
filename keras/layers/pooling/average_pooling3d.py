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
"""Average pooling 3D layer."""


import tensorflow.compat.v2 as tf

from keras.layers.pooling.base_pooling3d import Pooling3D

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.AveragePooling3D", "keras.layers.AvgPool3D")
class AveragePooling3D(Pooling3D):
    """Average pooling operation for 3D data (spatial or spatio-temporal).

    Downsamples the input along its spatial dimensions (depth, height, and
    width) by taking the average value over an input window
    (of size defined by `pool_size`) for each channel of the input.
    The window is shifted by `strides` along each dimension.

    Args:
      pool_size: tuple of 3 integers,
        factors by which to downscale (dim1, dim2, dim3).
        `(2, 2, 2)` will halve the size of the 3D input in each dimension.
      strides: tuple of 3 integers, or None. Strides values.
      padding: One of `"valid"` or `"same"` (case-insensitive).
        `"valid"` means no padding. `"same"` results in padding evenly to
        the left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
        while `channels_first` corresponds to inputs with shape
        `(batch, channels, spatial_dim1, spatial_dim2, spatial_dim3)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`

    Example:

    ```python
    depth = 30
    height = 30
    width = 30
    input_channels = 3

    inputs = tf.keras.Input(shape=(depth, height, width, input_channels))
    layer = tf.keras.layers.AveragePooling3D(pool_size=3)
    outputs = layer(inputs)  # Shape: (batch_size, 10, 10, 10, 3)
    ```
    """

    def __init__(
        self,
        pool_size=(2, 2, 2),
        strides=None,
        padding="valid",
        data_format=None,
        **kwargs
    ):
        super().__init__(
            tf.nn.avg_pool3d,
            pool_size=pool_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            **kwargs
        )


# Alias

AvgPool3D = AveragePooling3D
