from keras_core.api_export import keras_core_export
from keras_core.layers.layer import Layer


@keras_core_export("keras_core.layers.Identity")
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
