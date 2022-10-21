import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export

from keras.layers.pooling.base_generalized_pooling import BaseGeneralizedPooling


@keras_export("keras.layers.GeneralizedMeanPooling1D")
class GeneralizedMeanPooling1D(BaseGeneralizedPooling):
    """Generalized mean pooling operation for temporal data.

    Generalized Mean Pooling (GeM) computes the generalized mean of each
    channel in a tensor. It provides a parameter `power` that sets an
    exponent enabling the pooling to increase or decrease the contrast
    between salient features in the feature map.

    The GeM layer is an generalization of the average pooling layer and
    spatial max pooling layer. When `power` = 1`, it will act as a average
    pooling layer and when `power = inf`, it will act as a spatial
    max-pooling layer.

    Examples:

    1. When pool_size=2, strides=1, padding='valid'

    >>> input_shape = (2, 3, 4)
    >>> x = tf.random.normal(input_shape)
    >>> gem_pool_1d = tf.keras.layers.GeneralizedMeanPooling1D(power=3,
    ...    pool_size=2, strides=1, padding='valid',
    ...    data_format='channels_last')
    >>> gem_pool_1d(x)
    <tf.Tensor: shape=(1, 4, 1), dtype=float32, numpy=
    array([[[1.6509637],
            [2.596247 ],
            [3.5700185],
            [4.5548835]]], dtype=float32)>

    2. When pool_size=2, strides=1, padding='valid'

    >>> input_shape = (2, 3, 4)
    >>> x = tf.random.normal(input_shape)
    >>> gem_pool_1d = tf.keras.layers.GeneralizedMeanPooling1D(power=3,
    ...    pool_size=2, strides=2, padding='valid',
    ...    data_format='channels_last')
    >>> gem_pool_1d(x)
    <tf.Tensor: shape=(1, 2, 1), dtype=float32, numpy=
    array([[[1.6509637],
            [3.5700185]]], dtype=float32)>

    3. When pool_size=2, strides=1, padding='same'

    >>> input_shape = (2, 3, 4)
    >>> x = tf.random.normal(input_shape)
    >>> gem_pool_1d = tf.keras.layers.GeneralizedMeanPooling1D(power=3,
    ...    pool_size=2, strides=1, padding='same',
    ...    data_format='channels_last')
    >>> gem_pool_1d(x)
    <tf.Tensor: shape=(1, 5, 1), dtype=float32, numpy=
    array([[[1.6509637],
            [2.596247 ],
            [3.5700185],
            [4.5548835],
            [5.0000005]]], dtype=float32)>

    Args:
      power: Float power > 0 is an inverse exponent parameter, used during
        the generalized mean pooling computation. Setting this exponent as
        power > 1 increases the contrast of the pooled feature map and focuses
        on the salient features of the image. GeM is a generalization of the
        average pooling when `power` = 1 and of spatial max-pooling layer when
        `power` = inf or a large number.
      pool_size: Integer, size of the average pooling windows.
      strides: Integer, or None. Factor by which to downscale.
        E.g. 2 will halve the input.
        If None, it will default to `pool_size`.
      padding: A string. The padding method, either 'valid' or 'same'.
        `'valid'` means no padding. `'same'` results in padding evenly to
        the left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, steps, features)` while `channels_first` corresponds
        to inputs with shape `(batch, features, steps)`.
      name: A string, the name of the layer.

    Input shape:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch_size, steps, features)`.
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch_size, features, steps)`.

    Output shape:
      - If `data_format='channels_last'`:
        3D tensor with shape `(batch_size, downsampled_steps, features)`.
      - If `data_format='channels_first'`:
        3D tensor with shape `(batch_size, features, downsampled_steps)`.

     References:
        - [Filip Radenović, et al.](https://arxiv.org/abs/1711.02512)
    """

    def __init__(
        self,
        power=3.0,
        pool_size=2,
        strides=None,
        padding="valid",
        data_format="channels_last",
        name="GeneralizedMeanPooling1D",
        **kwargs
    ):
        self.power = power
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.data_format = data_format
        super().__init__(name=name, **kwargs)

    def call(self, inputs):
        x = tf.pow(inputs, self.power)
        x = tf.nn.avg_pool1d(
            x, self.pool_size, self.strides, self.padding, self.data_format
        )
        x = tf.pow(x, (1.0 / self.power))
        return x
