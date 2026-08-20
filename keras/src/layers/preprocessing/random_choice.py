from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.preprocessing.data_layer import DataLayer
from keras.src.random.seed_generator import SeedGenerator
from keras.src.saving import serialization_lib


@keras_export("keras.layers.RandomChoice")
class RandomChoice(DataLayer):
    """Apply one randomly-picked layer from a list to the input.

    During training, on each call this layer picks one of the wrapped layers
    uniformly at random and applies it. The choice is batch-wide — the same
    layer is applied to every sample in the batch.

    During inference (`training=False`) the layer is always a no-op.

    Note that every wrapped layer is evaluated on each training call and all
    but the selected result are discarded, so the per-call compute cost is
    `len(layers)` evaluations rather than one.

    Args:
        layers: List of Keras `Layer` instances. Each must accept the same
            input shape and emit a same-shape output.
        seed: Optional integer. Random seed used to pick a layer.

    Example:

    ```python
    augmenter = keras.layers.RandomChoice([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ])
    ```
    """

    def __init__(self, layers, seed=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(layers, (list, tuple)) or len(layers) == 0:
            raise ValueError(
                "`layers` must be a non-empty list of Keras `Layer` "
                f"instances. Received: layers={layers}"
            )
        for i, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise TypeError(
                    "Each entry in `layers` must be a Keras `Layer` "
                    f"instance. Received layers[{i}]={layer} of type "
                    f"{type(layer)}"
                )
        self._wrapped_layers = list(layers)
        self.seed = seed
        self.generator = SeedGenerator(seed)

    @property
    def layers(self):
        return self._wrapped_layers

    def call(self, inputs, training=True):
        if not training:
            return inputs

        n = len(self._wrapped_layers)
        candidates = [
            layer(inputs, training=training) for layer in self._wrapped_layers
        ]
        seed = self._get_seed_generator(self.backend._backend)
        choice = self.backend.random.randint(
            shape=(1,), minval=0, maxval=n, seed=seed
        )
        # Stack all candidate outputs along a new leading axis and gather the
        # one selected by `choice`.
        stacked = self.backend.numpy.stack(candidates, axis=0)
        return self.backend.numpy.take(stacked, choice, axis=0)[0]

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layers": [
                    serialization_lib.serialize_keras_object(layer)
                    for layer in self._wrapped_layers
                ],
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = {**config}
        config["layers"] = [
            serialization_lib.deserialize_keras_object(
                x, custom_objects=custom_objects
            )
            for x in config["layers"]
        ]
        return cls(**config)
