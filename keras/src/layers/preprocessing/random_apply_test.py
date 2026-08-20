import numpy as np

import keras
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

    def _bbox_data(self):
        return {
            "images": np.random.uniform(size=(2, 8, 8, 3)).astype("float32"),
            "bounding_boxes": {
                "boxes": np.array(
                    [[[0.0, 0.0, 4.0, 4.0]], [[1.0, 1.0, 5.0, 5.0]]],
                    dtype="float32",
                ),
                "labels": np.array([[0], [1]], dtype="float32"),
            },
        }

    def test_dict_input_with_bounding_boxes(self):
        # A wrapped image preprocessing layer accepts a dict of images and
        # bounding boxes; the wrapper must pass the structure through rather
        # than assuming a single tensor.
        layer = RandomApply(
            layers.RandomFlip("horizontal", bounding_box_format="xyxy"),
            rate=1.0,
            seed=0,
        )
        out = layer(self._bbox_data(), training=True)
        self.assertIsInstance(out, dict)
        self.assertEqual(sorted(out.keys()), ["bounding_boxes", "images"])
        self.assertEqual(
            backend.convert_to_numpy(out["images"]).shape, (2, 8, 8, 3)
        )
        self.assertEqual(
            backend.convert_to_numpy(out["bounding_boxes"]["boxes"]).shape,
            (2, 1, 4),
        )

    def test_rate_zero_is_strict_passthrough_for_dict_input(self):
        # `BaseImagePreprocessingLayer` rebinds keys on the dict it is given
        # and returns that same object, so a naive select would compare the
        # augmented values against themselves and let augmentation leak
        # through at rate=0.0.
        for seed in range(8):
            data = self._bbox_data()
            expected = data["images"].copy()
            layer = RandomApply(
                layers.RandomFlip("horizontal", bounding_box_format="xyxy"),
                rate=0.0,
                seed=seed,
            )
            out = layer(data, training=True)
            self.assertAllClose(
                backend.convert_to_numpy(out["images"]), expected
            )

    def test_does_not_mutate_caller_structure(self):
        data = self._bbox_data()
        expected = data["images"].copy()
        layer = RandomApply(
            layers.RandomFlip("horizontal", bounding_box_format="xyxy"),
            rate=1.0,
            seed=3,
        )
        layer(data, training=True)
        self.assertAllClose(data["images"], expected)

    def test_rejects_shape_changing_layer(self):
        # The wrapped output is selected against the unmodified input, so a
        # layer that resizes its input cannot be wrapped. Without an explicit
        # check this surfaces as a raw backend error from the stack op.
        layer = RandomApply(layers.Resizing(2, 2), rate=0.5, seed=0)
        x = np.random.uniform(size=(2, 4, 4, 3)).astype("float32")
        with self.assertRaisesRegex(ValueError, "same shape as its input"):
            layer(x, training=True)

    def test_dynamic_batch_dim_is_not_a_shape_mismatch(self):
        # The check must compare only statically-known dimensions, or a
        # symbolic build with an unknown batch axis would raise.
        inputs = keras.Input((8, 8, 3))
        outputs = RandomApply(
            layers.RandomFlip("horizontal"), rate=0.5, seed=0
        )(inputs)
        model = keras.Model(inputs, outputs)
        x = np.random.uniform(size=(4, 8, 8, 3)).astype("float32")
        self.assertEqual(model.predict(x, verbose=0).shape, (4, 8, 8, 3))
