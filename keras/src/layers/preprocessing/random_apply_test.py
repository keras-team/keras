import numpy as np

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.layers.preprocessing.random_apply import RandomApply
from keras.src.saving import serialization_lib


class _AddOne(layers.Layer):
    """Deterministic helper: adds 1.0 to its input."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def call(self, inputs, training=True):
        return inputs + 1.0


class RandomApplyTest(testing.TestCase):
    def test_rejects_non_layer(self):
        with self.assertRaisesRegex(TypeError, "Keras `Layer`"):
            RandomApply(lambda x: x)

    def test_rejects_invalid_rate(self):
        with self.assertRaisesRegex(ValueError, "rate"):
            RandomApply(_AddOne(), rate=1.5)
        with self.assertRaisesRegex(ValueError, "rate"):
            RandomApply(_AddOne(), rate=-0.1)

    def test_inference_is_noop(self):
        # training=False should always pass through regardless of `rate`.
        layer = RandomApply(_AddOne(), rate=1.0, seed=0)
        x = np.ones((4, 3, 3, 1), dtype="float32")
        out = backend.convert_to_numpy(layer(x, training=False))
        self.assertAllClose(out, x)

    def test_rate_one_always_applies(self):
        layer = RandomApply(_AddOne(), rate=1.0, seed=42)
        x = np.zeros((4, 3, 3, 1), dtype="float32")
        out = backend.convert_to_numpy(layer(x, training=True))
        self.assertAllClose(out, np.ones_like(x))

    def test_rate_zero_never_applies(self):
        layer = RandomApply(_AddOne(), rate=0.0, seed=42)
        x = np.zeros((4, 3, 3, 1), dtype="float32")
        out = backend.convert_to_numpy(layer(x, training=True))
        self.assertAllClose(out, x)

    def test_rate_half_mixes(self):
        # With many independent calls and rate=0.5, both branches should fire.
        layer = RandomApply(_AddOne(), rate=0.5, seed=0)
        x = np.zeros((1, 1, 1, 1), dtype="float32")
        outs = [
            backend.convert_to_numpy(layer(x, training=True)).item()
            for _ in range(100)
        ]
        seen_apply = any(v == 1.0 for v in outs)
        seen_skip = any(v == 0.0 for v in outs)
        self.assertTrue(seen_apply, "rate=0.5 should sometimes apply")
        self.assertTrue(seen_skip, "rate=0.5 should sometimes skip")

    def test_serialization_roundtrip(self):
        layer = RandomApply(_AddOne(), rate=0.3, seed=7)
        config = serialization_lib.serialize_keras_object(layer)
        revived = serialization_lib.deserialize_keras_object(
            config, custom_objects={"_AddOne": _AddOne}
        )
        self.assertEqual(revived.rate, 0.3)
        self.assertEqual(revived.seed, 7)
        self.assertIsInstance(revived.layer, _AddOne)

    def test_wraps_a_preprocessing_layer(self):
        # End-to-end: wrap RandomFlip and confirm it still produces shape-
        # preserving outputs.
        layer = RandomApply(layers.RandomFlip("horizontal"), rate=0.5, seed=0)
        x = np.random.uniform(size=(2, 4, 4, 3)).astype("float32")
        out = backend.convert_to_numpy(layer(x, training=True))
        self.assertEqual(out.shape, x.shape)
