from keras.api_export import keras_export
from keras.layers.layer import Layer


@keras_export("keras.layers.Identity")
class Identity(Layer):
    """Identity layer.

    This layer should be used as a placeholder when no operation is to be
    performed. The layer just returns its `inputs` argument as output.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs
