from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.layers.layer import Layer
from keras.src.layers.preprocessing.data_layer import DataLayer
from keras.src.random.seed_generator import SeedGenerator
from keras.src.saving import serialization_lib


def _check_same_shape(shape, reference_shape, description):
    """Raise if a wrapped layer changed the shape of a leaf.

    Only statically-known dimensions are compared, so a dynamic batch axis
    under `tf.data` or a symbolic build does not trip the check.
    """
    if shape is None or reference_shape is None:
        return
    shape = tuple(shape)
    reference_shape = tuple(reference_shape)
    mismatch = len(shape) != len(reference_shape) or any(
        d is not None and r is not None and d != r
        for d, r in zip(shape, reference_shape)
    )
    if mismatch:
        raise ValueError(
            f"{description} must emit an output with the same shape as its "
            f"input, because the output is selected element-wise against the "
            f"unmodified input. Received an input of shape {reference_shape} "
            f"and an output of shape {shape}. Layers that change the shape of "
            f"their input, such as `Resizing` or `Cropping2D`, cannot be "
            f"wrapped."
        )


@keras_export("keras.layers.RandomApply")
class RandomApply(DataLayer):
    """Apply a wrapped layer to the input with a given probability.

    During training, this layer flips a Bernoulli coin with probability `rate`.
    On heads it applies the wrapped `layer`, on tails it passes the input
    through unchanged. By default (`batchwise=False`) the coin is flipped
    independently for each sample, so a batch can contain a mix of augmented and
    untouched samples. Set `batchwise=True` for a single batch-wide decision
    (required by layers that mix samples together, such as `MixUp` or `CutMix`).

    During inference (`training=False`) the layer is always a no-op.

    Note that the wrapped layer is evaluated on every training call, including
    the calls where its output is discarded. The `rate` argument controls which
    result is returned, not whether the work is done.

    Args:
        layer: A Keras `Layer` to apply with probability `rate`. Typically a
            preprocessing layer, but any layer that accepts the same input
            shape and emits a same-shape output will work.
        rate: Float in `[0, 1]`. Probability of applying the wrapped layer.
            Defaults to `0.5`.
        batchwise: Boolean. If `True`, a single decision is made for the whole
            batch. If `False` (default), the decision is made per sample.
        seed: Optional integer. Random seed used for the Bernoulli draw.

    Example:

    ```python
    augmenter = keras.layers.RandomApply(
        keras.layers.RandomFlip("horizontal"), rate=0.3
    )
    ```
    """

    def __init__(self, layer, rate=0.5, batchwise=False, seed=None, **kwargs):
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
        self.batchwise = batchwise
        self.seed = seed
        self.generator = SeedGenerator(seed)

    def call(self, inputs, training=True):
        if not training:
            return inputs

        # Snapshot the input structure before invoking the wrapped layer. The
        # image preprocessing layers rebind keys on the structure they are
        # given and return that same object, so without a snapshot both
        # branches of the selection below would observe augmented values and
        # `rate=0.0` would not be a strict pass-through. The wrapped layer
        # gets its own copy so the caller's structure is left untouched.
        original = tree.map_structure(lambda x: x, inputs)
        transformed = self.layer(
            tree.map_structure(lambda x: x, inputs), training=training
        )

        seed = self._get_seed_generator(self.backend._backend)
        # One coin flip per sample (`batchwise=False`) or a single flip shared
        # by the whole batch (`batchwise=True`).
        num_draws = 1 if self.batchwise else self._sample_count(original)
        apply = self.backend.ops.numpy.less(
            self.backend.random.uniform(
                shape=(num_draws,), minval=0.0, maxval=1.0, seed=seed
            ),
            self.rate,
        )

        def _select(transformed_leaf, original_leaf):
            _check_same_shape(
                getattr(transformed_leaf, "shape", None),
                getattr(original_leaf, "shape", None),
                f"The wrapped layer `{self.layer.__class__.__name__}`",
            )
            # Broadcast the coin over the leaf's trailing axes. The same draw
            # is used for every leaf, so a structured input's image and its
            # bounding boxes stay aligned per sample.
            rank = len(self.backend.ops.shape(original_leaf))
            mask = self.backend.ops.numpy.reshape(
                apply, [-1] + [1] * (rank - 1)
            )
            return self.backend.ops.numpy.where(
                mask, transformed_leaf, original_leaf
            )

        return tree.map_structure(_select, transformed, original)

    def _sample_count(self, inputs):
        """Per-sample draw count, or 1 for unbatched input.

        Follows the image convention: a rank-4 tensor `(batch, ...)` is
        batched; lower ranks are treated as a single unbatched sample.
        """
        sample = tree.flatten(inputs)[0]
        shape = self.backend.ops.shape(sample)
        if len(shape) >= 4:
            return shape[0]
        return 1

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "layer": serialization_lib.serialize_keras_object(self.layer),
                "rate": self.rate,
                "batchwise": self.batchwise,
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
