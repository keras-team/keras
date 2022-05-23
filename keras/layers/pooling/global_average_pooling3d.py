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
"""Global average pooling 3D layer."""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

from keras import backend
from keras.layers.pooling.base_global_pooling3d import GlobalPooling3D

from tensorflow.python.util.tf_export import keras_export


@keras_export(
    "keras.layers.GlobalAveragePooling3D", "keras.layers.GlobalAvgPool3D"
)
class GlobalAveragePooling3D(GlobalPooling3D):
    """Global Average pooling operation for 3D data.

    Args:
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
      keepdims: A boolean, whether to keep the spatial dimensions or not.
        If `keepdims` is `False` (default), the rank of the tensor is reduced
        for spatial dimensions.
        If `keepdims` is `True`, the spatial dimensions are retained with
        length 1.
        The behavior is the same as for `tf.reduce_mean` or `np.mean`.

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:
      - If `keepdims`=False:
        2D tensor with shape `(batch_size, channels)`.
      - If `keepdims`=True:
        - If `data_format='channels_last'`:
          5D tensor with shape `(batch_size, 1, 1, 1, channels)`
        - If `data_format='channels_first'`:
          5D tensor with shape `(batch_size, channels, 1, 1, 1)`
    """

    def call(self, inputs):
        if self.data_format == "channels_last":
            return backend.mean(inputs, axis=[1, 2, 3], keepdims=self.keepdims)
        else:
            return backend.mean(inputs, axis=[2, 3, 4], keepdims=self.keepdims)


# Alias

GlobalAvgPool3D = GlobalAveragePooling3D
