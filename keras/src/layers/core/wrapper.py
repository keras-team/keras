from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.saving import serialization_lib


@keras_export("keras.layers.Wrapper")
class Wrapper(Layer):
    """Abstract wrapper base class.

    Wrappers take another layer and augment it in various ways.
    Do not use this class as a layer, it is only an abstract base class.
    Two usable wrappers are the `TimeDistributed` and `Bidirectional` layers.

    Args:
        layer: The layer to be wrapped.
    """

    def __init__(self, layer, **kwargs):
        try:
            assert isinstance(layer, Layer)
        except Exception:
            raise ValueError(
                f"Layer {layer} supplied to Wrapper isn't "
                "a supported layer type. Please "
                "ensure wrapped layer is a valid Keras layer."
            )
        super().__init__(**kwargs)
        self.layer = layer

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True

    def get_config(self):
        config = {"layer": serialization_lib.serialize_keras_object(self.layer)}
        base_config = super().get_config()
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        layer = serialization_lib.deserialize_keras_object(
            config.pop("layer"),
            custom_objects=custom_objects,
        )
        return cls(layer, **config)
