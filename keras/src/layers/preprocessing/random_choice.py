from keras.src import ops
from keras.src import random
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.random.seed_generator import SeedGenerator
from keras.src.saving import serialization_lib


@keras_export("keras.layers.RandomChoice")
class RandomChoice(Layer):
    """Apply one randomly-picked layer from a list to the input.

    During training, on each call this layer picks one of the wrapped `layers`
    uniformly at random and applies it. All other layers are evaluated and
    discarded, so the per-call compute cost is `len(layers)` evaluations; this
    keeps the layer compatible with `tf.data` / `grain` pipelines where
    Python-side branching is not available. The choice is batch-wide — the
    same layer is applied to every sample in the batch.

    During inference (`training=False`) the layer is always a no-op.

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
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    @property
    def layers(self):
        return self._wrapped_layers

    def call(self, inputs, training=True):
        if not training:
            return inputs

        n = len(self._wrapped_layers)
        # Stack all candidate outputs along a new leading axis and gather the
        # one selected by `choice`. This avoids relying on `ops.where` with a
        # scalar mask broadcasting against an N-D tensor — which is uneven
        # across backends.
        candidates = [
            layer(inputs, training=training) for layer in self._wrapped_layers
        ]
        stacked = ops.stack(candidates, axis=0)
        choice = random.randint(
            shape=(1,), minval=0, maxval=n, seed=self.generator
        )
        return ops.take(stacked, choice, axis=0)[0]

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
