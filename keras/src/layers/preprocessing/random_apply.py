from keras.src import ops
from keras.src import random
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.random.seed_generator import SeedGenerator
from keras.src.saving import serialization_lib


@keras_export("keras.layers.RandomApply")
class RandomApply(Layer):
    """Apply a wrapped layer to the input with a given probability.

    During training, on each call this layer flips a single Bernoulli coin: with
    probability `rate` it applies the wrapped `layer`, otherwise it passes the
    input through unchanged. The decision is batch-wide — the same coin flip is
    used for every sample in the batch — which keeps the layer compatible with
    a `tf.data` or `grain` pipeline.

    During inference (`training=False`) the layer is always a no-op.

    Args:
        layer: A Keras `Layer` to apply with probability `rate`. Typically a
            preprocessing layer, but any layer that accepts the same input
            shape and emits a same-shape output will work.
        rate: Float in `[0, 1]`. Probability of applying the wrapped layer.
            Defaults to `0.5`.
        seed: Optional integer. Random seed used for the Bernoulli draw.

    Example:

    ```python
    augmenter = keras.layers.RandomApply(
        keras.layers.RandomFlip("horizontal"), rate=0.3
    )
    ```
    """

    def __init__(self, layer, rate=0.5, seed=None, **kwargs):
        super().__init__(**kwargs)
        if not isinstance(layer, Layer):
            raise TypeError(
                "`layer` must be a Keras `Layer` instance. "
                f"Received: layer={layer} of type {type(layer)}"
            )
        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError(
                f"`rate` must be a float in `[0, 1]`. Received: rate={rate}"
            )
        self.layer = layer
        self.rate = float(rate)
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def call(self, inputs, training=True):
        if not training:
            return inputs

        transformed = self.layer(inputs, training=training)
        u = random.uniform(
            shape=(), minval=0.0, maxval=1.0, seed=self.generator
        )
        should_apply = ops.less(u, self.rate)
        return ops.where(should_apply, transformed, inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer": serialization_lib.serialize_keras_object(self.layer),
                "rate": self.rate,
                "seed": self.seed,
            }
        )
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        config = {**config}
        config["layer"] = serialization_lib.deserialize_keras_object(
            config["layer"], custom_objects=custom_objects
        )
        return cls(**config)
