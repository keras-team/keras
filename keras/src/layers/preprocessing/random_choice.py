from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.preprocessing.data_layer import DataLayer
from keras.src.layers.preprocessing.random_apply import _check_same_shape
from keras.src.random.seed_generator import SeedGenerator
from keras.src.saving import serialization_lib


@keras_export("keras.layers.RandomChoice")
class RandomChoice(DataLayer):
    """Apply one randomly-picked layer from a list to the input.

    During training, this layer picks one of the wrapped layers uniformly at
    random and applies it. By default (`batchwise=False`) the choice is made
    independently for each sample, so different samples in a batch can be
    transformed by different layers. Set `batchwise=True` for a single
    batch-wide choice.

    Per-sample choices require a batched input, detected with the image
    convention: a rank-4 tensor `(batch, ...)`, or a dict input whose
    `"images"` entry is rank-4. Any other input (e.g. a single unbatched image)
    receives one shared choice regardless of `batchwise`.

    During inference (`training=False`) the layer is always a no-op.

    Note that every wrapped layer is evaluated on each training call and all
    but the selected result are discarded, so the per-call compute cost is
    `len(layers)` evaluations rather than one.

    Args:
        layers: List of Keras `Layer` instances. Each must accept the same
            input shape and emit a same-shape output.
        batchwise: Boolean. If `True`, a single layer is chosen for the whole
            batch. If `False` (default), the choice is made per sample.
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

    def __init__(self, layers, batchwise=False, seed=None, **kwargs):
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
        self.batchwise = batchwise
        self.seed = seed
        self.generator = SeedGenerator(seed)

    @property
    def wrapped_layers(self):
        return self._wrapped_layers

    def build(self, input_shape):
        # Build the wrapped layers so the whole stack reports as built. Without
        # this, a functional model emits an "unbuilt state" warning on the
        # openvino backend. Mirrors `RandAugment.build`.
        for layer in self._wrapped_layers:
            layer.build(input_shape)

    def call(self, inputs, training=True):
        if not training:
            return inputs

        n = len(self._wrapped_layers)
        # The image preprocessing layers rebind keys on the structure they are
        # given and return that same object, so every wrapped layer gets its
        # own copy. Sharing one would chain the augmentations together instead
        # of producing independent candidates, and would leave the caller's
        # structure mutated.
        original = tree.map_structure(lambda x: x, inputs)
        candidates = [
            layer(tree.map_structure(lambda x: x, inputs), training=training)
            for layer in self._wrapped_layers
        ]
        seed = self._get_seed_generator(self.backend._backend)
        # One index per sample (`batchwise=False`) or a single index shared by
        # the whole batch (`batchwise=True`).
        num_draws = 1 if self.batchwise else self._sample_count(original)
        choice = self.backend.random.randint(
            shape=(num_draws,), minval=0, maxval=n, seed=seed
        )

        def _select(original_leaf, *candidate_leaves):
            reference_shape = getattr(original_leaf, "shape", None)
            for i, candidate_leaf in enumerate(candidate_leaves):
                _check_same_shape(
                    getattr(candidate_leaf, "shape", None),
                    reference_shape,
                    f"`layers[{i}]`, a "
                    f"`{self._wrapped_layers[i].__class__.__name__}`,",
                )
            # Select each candidate where `choice` equals its index. The same
            # `choice` is used for every leaf, so a structured input is
            # transformed by exactly one of the wrapped layers per sample.
            rank = len(self.backend.ops.shape(original_leaf))
            result = candidate_leaves[0]
            for i in range(1, len(candidate_leaves)):
                mask = self.backend.ops.numpy.reshape(
                    self.backend.ops.numpy.equal(choice, i),
                    [-1] + [1] * (rank - 1),
                )
                result = self.backend.ops.numpy.where(
                    mask, candidate_leaves[i], result
                )
            return result

        return tree.map_structure(_select, original, *candidates)

    def _sample_count(self, inputs):
        """Per-sample draw count, or 1 when the input is not a batch.

        Follows the image convention used by the preprocessing layers: the
        batch is read from the `"images"` entry of a dict input (otherwise the
        first leaf), and a rank-4 tensor `(batch, ...)` is a batch while lower
        ranks are a single unbatched sample.
        """
        if isinstance(inputs, dict) and "images" in inputs:
            sample = inputs["images"]
        else:
            sample = tree.flatten(inputs)[0]
        shape = self.backend.ops.shape(sample)
        if len(shape) == 4:
            return shape[0]
        return 1

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
                "batchwise": self.batchwise,
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
