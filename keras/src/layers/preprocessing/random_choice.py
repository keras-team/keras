from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.random import SeedGenerator
from keras.src.saving import serialization_lib


@keras_export("keras.layers.RandomChoice")
class RandomChoice(Layer):
    """Randomly pick one layer from a list and apply it to the input.

    At training time, one layer is chosen uniformly at random and applied.
    During inference, the input is returned unchanged.

    **Note:** This layer is safe to use inside a `tf.data` or `grain`
    pipeline (independently of which backend you're using).

    Args:
        layers: A list of `keras.Layer` instances. The candidate layers
            to choose from.
        seed: Integer. Used to create a random seed.
        name: String. The name of the layer.

    Example:

    ```python
    from keras import layers

    # Randomly pick one augmentation to apply
    layer = layers.RandomChoice(
        layers=[
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ],
    )
    images = np.random.uniform(0, 255, (4, 224, 224, 3)).astype("float32")
    output = layer(images, training=True)
    ```
    """

    def __init__(self, layers, seed=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        if not isinstance(layers, (list, tuple)):
            raise ValueError(
                "Expected `layers` to be a list or tuple of "
                "`keras.Layer` instances. "
                f"Received: layers={layers}"
            )
        if len(layers) == 0:
            raise ValueError(
                "Expected `layers` to contain at least one layer. "
                "Received: empty list."
            )
        for i, layer in enumerate(layers):
            if not isinstance(layer, Layer):
                raise ValueError(
                    "Expected all elements of `layers` to be "
                    "`keras.Layer` instances. "
                    f"Received: layers[{i}]={layer} "
                    f"(of type {type(layer)})"
                )
        self._layers_list = list(layers)
        self.seed = seed
        self.generator = SeedGenerator(seed)
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    @property
    def layers(self):
        return self._layers_list

    def build(self, input_shape):
        for layer in self._layers_list:
            layer.build(input_shape)
        self.built = True

    def call(self, inputs, training=True):
        if not training:
            return inputs

        seed_generator = self.generator
        idx = backend.random.randint(
            shape=(),
            minval=0,
            maxval=len(self._layers_list),
            seed=seed_generator,
        )
        idx = int(idx)
        layer = self._layers_list[idx]
        kwargs = {}
        if layer._call_has_training_arg:
            kwargs["training"] = training
        return layer(inputs, **kwargs)

    @classmethod
    def from_config(cls, config):
        config["layers"] = [
            serialization_lib.deserialize_keras_object(x)
            for x in config["layers"]
        ]
        return cls(**config)

    def get_config(self):
        config = {
            "layers": serialization_lib.serialize_keras_object(
                self._layers_list
            ),
            "seed": self.seed,
            "name": self.name,
        }
        return config
