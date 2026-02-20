from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.random import SeedGenerator
from keras.src.saving import serialization_lib


@keras_export("keras.layers.RandomApply")
class RandomApply(Layer):
    """Randomly apply a layer to inputs with a given probability.

    At training time, the layer wrapped by `RandomApply` will be
    applied with probability `rate`. During inference, the input is
    returned unchanged.

    **Note:** This layer is safe to use inside a `tf.data` or `grain`
    pipeline (independently of which backend you're using).

    Args:
        layer: A `keras.Layer`. The layer to apply randomly.
        rate: Float between 0 and 1. The probability of applying the
            layer during training. Default is `0.5`.
        seed: Integer. Used to create a random seed.
        name: String. The name of the layer.

    Example:

    ```python
    from keras import layers

    # Apply random flip with 50% probability
    layer = layers.RandomApply(
        layer=layers.RandomFlip("horizontal"),
        rate=0.5,
    )
    images = np.random.uniform(0, 255, (4, 224, 224, 3)).astype("float32")
    output = layer(images, training=True)
    ```
    """

    def __init__(self, layer, rate=0.5, seed=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        if not isinstance(layer, Layer):
            raise ValueError(
                "Expected `layer` to be a `keras.Layer` instance. "
                f"Received: layer={layer} (of type {type(layer)})"
            )
        if not 0.0 <= rate <= 1.0:
            raise ValueError(
                "Expected `rate` to be a float between 0 and 1. "
                f"Received: rate={rate}"
            )
        self._layer = layer
        self.rate = rate
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    @property
    def layer(self):
        return self._layer

    def build(self, input_shape):
        self._layer.build(input_shape)
        self.built = True

    def call(self, inputs, training=True):
        if not training:
            return inputs

        seed_generator = self.generator
        random_value = backend.random.uniform(shape=(), seed=seed_generator)
        if random_value < self.rate:
            kwargs = {}
            if self._layer._call_has_training_arg:
                kwargs["training"] = training
            return self._layer(inputs, **kwargs)
        return inputs

    @classmethod
    def from_config(cls, config):
        config["layer"] = serialization_lib.deserialize_keras_object(
            config["layer"]
        )
        return cls(**config)

    def get_config(self):
        config = {
            "layer": serialization_lib.serialize_keras_object(self._layer),
            "rate": self.rate,
            "seed": self.seed,
            "name": self.name,
        }
        return config
