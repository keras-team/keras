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
"""Global max pooling 1D layer."""


from keras import backend
from keras.layers.pooling.base_global_pooling1d import GlobalPooling1D

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.GlobalMaxPool1D", "keras.layers.GlobalMaxPooling1D")
class GlobalMaxPooling1D(GlobalPooling1D):
    """Global max pooling operation for 1D temporal data.

    Downsamples the input representation by taking the maximum value over
    the time dimension.

    For example:

    >>> x = tf.constant([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> x = tf.reshape(x, [3, 3, 1])
    >>> x
    <tf.Tensor: shape=(3, 3, 1), dtype=float32, numpy=
    array([[[1.], [2.], [3.]],
           [[4.], [5.], [6.]],
           [[7.], [8.], [9.]]], dtype=float32)>
    >>> max_pool_1d = tf.keras.layers.GlobalMaxPooling1D()
    >>> max_pool_1d(x)
    <tf.Tensor: shape=(3, 1), dtype=float32, numpy=
    array([[3.],
           [6.],
           [9.], dtype=float32)>

    Args:
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, steps, features)` while `channels_first`
        corresponds to inputs with shape
        `(batch, features, steps)`.
      keepdims: A boolean, whether to keep the temporal dimension or not.
        If `keepdims` is `False` (default), the rank of the tensor is reduced
        for spatial dimensions.
        If `keepdims` is `True`, the temporal dimension are retained with
        length 1.
        The behavior is the same as for `tf.reduce_max` or `np.max`.

    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape:
        `(batch_size, steps, features)`
      - If `data_format='channels_first'`:
        3D tensor with shape:
        `(batch_size, features, steps)`

    Output shape:
      - If `keepdims`=False:
        2D tensor with shape `(batch_size, features)`.
      - If `keepdims`=True:
        - If `data_format='channels_last'`:
          3D tensor with shape `(batch_size, 1, features)`
        - If `data_format='channels_first'`:
          3D tensor with shape `(batch_size, features, 1)`
    """

    def call(self, inputs):
        steps_axis = 1 if self.data_format == "channels_last" else 2
        return backend.max(inputs, axis=steps_axis, keepdims=self.keepdims)


# Alias

GlobalMaxPool1D = GlobalMaxPooling1D
