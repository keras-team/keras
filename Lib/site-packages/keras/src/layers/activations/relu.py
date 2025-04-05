from keras.src import activations
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.ReLU")
class ReLU(Layer):
    """Rectified Linear Unit activation function layer.

    Formula:
    ``` python
    f(x) = max(x,0)
    f(x) = max_value if x >= max_value
    f(x) = x if threshold <= x < max_value
    f(x) = negative_slope * (x - threshold) otherwise
    ```

    Example:
    ``` python
    relu_layer = keras.layers.ReLU(
        max_value=10,
        negative_slope=0.5,
        threshold=0,
    )
    input = np.array([-10, -5, 0.0, 5, 10])
    result = relu_layer(input)
    # result = [-5. , -2.5,  0. ,  5. , 10.]
    ```

    Args:
        max_value: Float >= 0. Maximum activation value. None means unlimited.
            Defaults to `None`.
        negative_slope: Float >= 0. Negative slope coefficient.
            Defaults to `0.0`.
        threshold: Float >= 0. Threshold value for thresholded activation.
            Defaults to `0.0`.
        **kwargs: Base layer keyword arguments, such as `name` and `dtype`.
    """

    def __init__(
        self, max_value=None, negative_slope=0.0, threshold=0.0, **kwargs
    ):
        super().__init__(**kwargs)
        if max_value is not None and max_value < 0.0:
            raise ValueError(
                "max_value of a ReLU layer cannot be a negative "
                f"value. Received: max_value={max_value}"
            )
        if negative_slope is None or negative_slope < 0.0:
            raise ValueError(
                "negative_slope of a ReLU layer cannot be a negative "
                f"value. Received: negative_slope={negative_slope}"
            )
        if threshold is None or threshold < 0.0:
            raise ValueError(
                "threshold of a ReLU layer cannot be a negative "
                f"value. Received: threshold={threshold}"
            )

        self.max_value = max_value
        self.negative_slope = negative_slope
        self.threshold = threshold
        self.supports_masking = True
        self.built = True

    def call(self, inputs):
        return activations.relu(
            inputs,
            negative_slope=self.negative_slope,
            max_value=self.max_value,
            threshold=self.threshold,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "max_value": self.max_value,
                "negative_slope": self.negative_slope,
                "threshold": self.threshold,
            }
        )
        return config

    def compute_output_shape(self, input_shape):
        return input_shape
