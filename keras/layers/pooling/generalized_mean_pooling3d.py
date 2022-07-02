import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export

from keras.layers.pooling.base_generalized_pooling3d import GeneralizedPooling3D


@keras_export("keras.layers.GeneralizedMeanPooling3D")
class GeneralizedMeanPooling3D(GeneralizedPooling3D):
    """Generalized mean pooling operation for temporal data.

    Generalized Mean Pooling (GeM) computes the generalized mean of each
    channel in a tensor. It provides a parameter `p` that sets an exponent
    enabling the pooling to increase or decrease the contrast between salient
    features in the feature map.

    The GeM layer is an generalization of the average pooling layer and spatial
    max pooling layer. When `p` = 1`, it will act as a average pooling layer and
    when `p = inf`, it will act as a spatial max pooling layer.

    Examples:

    1. When pool_size=2, strides=2, padding='valid'

    >>> x = tf.constant(
    ...     [[[
    ...          [[3.], [2.]],
    ...          [[2.], [3.]]],
    ...       [
    ...          [[1.], [2.]],
    ...          [[4.], [2.]]]
    ...     ]])
    >>> x = tf.reshape(x, [1, 2, 2, 2, 1])
    >>> gem_pool_3d = tf.keras.layers.GeneralizedMeanPooling3D(power=3,
    ...    pool_size=(2, 2, 2), strides=(2, 2, 2),
    ...    padding='valid', data_format='channels_last')
    >>> gem_pool_3d(x)
    <tf.Tensor: shape=(1, 2, 2, 2, 1), dtype=float32, numpy=
    array([[[[[3.],
            [2.]],

            [[2.],
            [3.]]],

            [[[1.],
            [2.]],

            [[4.],
            [2.]]]]], dtype=float32)>

    2.
    ```python
    depth  = 30
    height = 30
    width  = 30
    input_channels = 3

    inputs = tf.keras.Input(shape=(depth, height, width, input_channels))
    layer = tf.keras.layers.GeneralizedMeanPooling3D(pool_size=3)
    outputs = layer(inputs) # Shape: (batch_size, 10, 10, 10, 3)
    ```


    Args:
      power: Float power > 0 is an inverse exponent parameter, used during
        the generalized mean pooling computation. Setting this exponent as
        power > 1 increases the contrast of the pooled feature map and
        focuses on the salient features of the image. GeM is a generalization
        of the average pooling commonly used in classification networks
        (power = 1) and of spatial max-pooling layer (power = inf).
      pool_size: An integer or tuple/list of 3 integers:
        (pool_depth, pool_height, pool_width) specifying the size of the
        pooling window. Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 3 integers, specifying the strides
        of the pooling operation. Can be a single integer to specify the same
        value for all spatial dimensions.
      padding: A string. The padding method, either 'valid' or 'same'.
        Case-insensitive.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`. The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, depth, height, width, channels)` while
        `channels_first` corresponds to inputs with shape
        `(batch, channels, depth, height, width)`.
      name: A string, the name of the layer.

    Input shape:
      - If `data_format='channels_last'`:
        5D tensor with shape:
        `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`
      - If `data_format='channels_first'`:
        5D tensor with shape:
        `(batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)`

    Output shape:
      - If `data_format='channels_last'`:
        5D tensor with shape
        `(batch_size, pooled_dim1, pooled_dim2, pooled_dim3, channels)`.
      - If `data_format='channels_first'`:
        5D tensor with shape
        `(batch_size, channels, pooled_dim1, pooled_dim2, pooled_dim3)`.

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
        x = tf.nn.avg_pool3d(
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
