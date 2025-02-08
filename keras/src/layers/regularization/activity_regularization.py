from keras.src import regularizers
from keras.src import utils
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer


@keras_export("keras.layers.ActivityRegularization")
class ActivityRegularization(Layer):
    """Layer that applies an update to the cost function based input activity.

    Args:
        l1: L1 regularization factor (positive float).
        l2: L2 regularization factor (positive float).

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.
    """

    def __init__(self, l1=0.0, l2=0.0, **kwargs):
        super().__init__(
            activity_regularizer=regularizers.L1L2(l1=l1, l2=l2), **kwargs
        )
        self.supports_masking = True
        self.l1 = l1
        self.l2 = l2

        # We can only safely mark the layer as built when build is not
        # overridden.
        if utils.is_default(self.build):
            self.built = True
            self._post_build()
            self._lock_state()

    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        base_config = super().get_config()
        base_config.pop("activity_regularizer", None)
        config = {"l1": self.l1, "l2": self.l2}
        return {**base_config, **config}
