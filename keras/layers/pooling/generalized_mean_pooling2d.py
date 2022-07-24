import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export

from keras.layers.pooling.base_generalized_pooling import BaseGeneralizedPooling


@keras_export("keras.layers.GeneralizedMeanPooling2D")
class GeneralizedMeanPooling2D(BaseGeneralizedPooling):
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

    >>> x = tf.constant([[1., 2., 3.],
    ...                  [4., 5., 6.],
    ...                  [7., 8., 9.]])
    >>> x = tf.reshape(x, [1, 3, 3, 1])
    >>> gem_pool_2d = tf.keras.layers.GeneralizedMeanPooling2D(power=3,
    ...    pool_size=2, strides=1, padding='valid',
    ...    data_format='channels_last')
    >>> gem_pool_2d(x)
    <tf.Tensor: shape=(1, 2, 2, 1), dtype=float32, numpy=
    array([[[[3.6717105],
            [4.546836 ]],

            [[6.390677 ],
            [7.3403287]]]], dtype=float32)>

    2. When pool_size=2, strides=2, padding='valid'

    >>> x = tf.constant([[1., 2., 3., 4.],
    ...                  [5., 6., 7., 8.],
    ...                  [9., 10., 11., 12.]])
    >>> x = tf.reshape(x, [1, 3, 4, 1])
    >>> gem_pool_2d = tf.keras.layers.GeneralizedMeanPooling2D(power=3,
    ...    pool_size=2, strides=2, padding='valid',
    ...    data_format='channels_last')
    >>> gem_pool_2d(x)
    <tf.Tensor: shape=(1, 1, 2, 1), dtype=float32, numpy=
    array([[[[4.4395204],
            [6.184108 ]]]], dtype=float32)>

    3. When pool_size=2, strides=1, padding='same'

    >>> x = tf.constant([[1., 2., 3.],
    ...                  [4., 5., 6.],
    ...                  [7., 8., 9.]])
    >>> x = tf.reshape(x, [1, 3, 3, 1])
    >>> gem_pool_2d = tf.keras.layers.GeneralizedMeanPooling1D(power=3,
    ...    pool_size=2, strides=1, padding='same',
    ...    data_format='channels_last')
    >>> gem_pool_2d(x)
    <tf.Tensor: shape=(1, 3, 3, 1), dtype=float32, numpy=
    array([[[[3.6717105],
            [4.546836 ],
            [4.952891 ]],

            [[6.390677 ],
            [7.3403287],
            [7.7887416]],

            [[7.533188 ],
            [8.529311 ],
            [9.000002 ]]]], dtype=float32)>

    Args:
      power: Float power > 0 is an inverse exponent parameter, used during
        the generalized mean pooling computation. Setting this exponent as
        power > 1 increases the contrast of the pooled feature map and focuses
        on the salient features of the image. GeM is a generalization of the
        average pooling when `power` = 1 and of spatial max-pooling layer when
        `power` = inf or a large number.
      pool_size: An integer or tuple/list of 2 integers:
        (pool_height, pool_width) specifying the size of the pooling window.
        Can be a single integer to specify the same value for all spatial
        dimensions.
      strides: An integer or tuple/list of 2 integers, specifying the strides
        of the pooling operation. Can be a single integer to specify the same
        value for all spatial dimensions.
      padding: A string. The padding method, either 'valid' or 'same'.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`.
      name: A string, the name of the layer.

    Input shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, rows, cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, rows, cols)`.

    Output shape:
      - If `data_format='channels_last'`:
        4D tensor with shape `(batch_size, pooled_rows, pooled_cols, channels)`.
      - If `data_format='channels_first'`:
        4D tensor with shape `(batch_size, channels, pooled_rows, pooled_cols)`.

     References:
        - https://arxiv.org/pdf/1711.02512.pdf
    """

    def __init__(
        self,
        power=3.0,
        pool_size=2,
        strides=None,
        padding="valid",
        data_format=None,
        name=None,
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
        x = tf.nn.avg_pool2d(
            x, self.pool_size, self.strides, self.padding, self.data_format
        )
        x = tf.pow(x, (1.0 / self.power))
        return x

    def get_config(self):
        config = {
            "power": self.power,
            "strides": self.strides,
            "pool_size": self.pool_size,
            "padding": self.padding,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
