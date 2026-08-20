import numpy as np

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.layers.preprocessing.random_choice import RandomChoice
from keras.src.saving import serialization_lib


class _AddConst(layers.Layer):
    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = float(value)
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def call(self, inputs, training=True):
        return inputs + self.value

    def get_config(self):
        config = super().get_config()
        config["value"] = self.value
        return config


class _AddConstInPlace(layers.Layer):
    """Rebinds a key on the input dict, as the image preprocessing layers do."""

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = float(value)
        self._convert_input_args = False
        self._allow_non_tensor_positional_args = True

    def call(self, inputs, training=True):
        inputs["images"] = inputs["images"] + self.value
        return inputs


class RandomChoiceTest(testing.TestCase):
    def test_rejects_empty_layers(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            RandomChoice([])

    def test_rejects_non_layer_entry(self):
        with self.assertRaisesRegex(TypeError, "Keras `Layer`"):
            RandomChoice([_AddConst(1.0), lambda x: x])

    def test_inference_is_noop(self):
        layer = RandomChoice([_AddConst(1.0), _AddConst(2.0)], seed=0)
        x = np.zeros((4, 3, 3, 1), dtype="float32")
        out = backend.convert_to_numpy(layer(x, training=False))
        self.assertAllClose(out, x)

    def test_single_layer_always_chosen(self):
        layer = RandomChoice([_AddConst(5.0)], seed=0)
        x = np.zeros((4, 3, 3, 1), dtype="float32")
        out = backend.convert_to_numpy(layer(x, training=True))
        self.assertAllClose(out, np.full_like(x, 5.0))

    def test_choice_covers_all_layers(self):
        # With three layers and a seed, repeated calls should produce all
        # three distinct add-values eventually.
        layer = RandomChoice(
            [_AddConst(1.0), _AddConst(2.0), _AddConst(3.0)], seed=0
        )
        x = np.zeros((1, 1, 1, 1), dtype="float32")
        outs = {
            backend.convert_to_numpy(layer(x, training=True)).item()
            for _ in range(200)
        }
        self.assertEqual(outs, {1.0, 2.0, 3.0})

    def test_serialization_roundtrip(self):
        layer = RandomChoice(
            [_AddConst(1.0), _AddConst(2.0)], seed=11, name="rc"
        )
        config = serialization_lib.serialize_keras_object(layer)
        revived = serialization_lib.deserialize_keras_object(
            config, custom_objects={"_AddConst": _AddConst}
        )
        self.assertEqual(revived.seed, 11)
        self.assertEqual(len(revived.layers), 2)
        self.assertEqual(revived.layers[0].value, 1.0)
        self.assertEqual(revived.layers[1].value, 2.0)

    def test_wraps_preprocessing_layers(self):
        layer = RandomChoice(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.1),
            ],
            seed=0,
        )
        x = np.random.uniform(size=(2, 8, 8, 3)).astype("float32")
        out = backend.convert_to_numpy(layer(x, training=True))
        self.assertEqual(out.shape, x.shape)

    def test_dict_input_with_bounding_boxes(self):
        layer = RandomChoice(
            [
                layers.RandomFlip("horizontal", bounding_box_format="xyxy"),
                layers.RandomRotation(0.1, bounding_box_format="xyxy"),
            ],
            seed=0,
        )
        data = {
            "images": np.random.uniform(size=(2, 8, 8, 3)).astype("float32"),
            "bounding_boxes": {
                "boxes": np.array(
                    [[[0.0, 0.0, 4.0, 4.0]], [[1.0, 1.0, 5.0, 5.0]]],
                    dtype="float32",
                ),
                "labels": np.array([[0], [1]], dtype="float32"),
            },
        }
        out = layer(data, training=True)
        self.assertIsInstance(out, dict)
        self.assertEqual(sorted(out.keys()), ["bounding_boxes", "images"])
        self.assertEqual(
            backend.convert_to_numpy(out["images"]).shape, (2, 8, 8, 3)
        )

    def test_candidates_are_independent(self):
        # Each wrapped layer must see the original input. The image
        # preprocessing layers rebind keys on the structure they are given, so
        # sharing one dict across candidates would chain the augmentations:
        # +1 applied on top of +10 would yield 11.0 rather than one of
        # {1.0, 10.0}.
        x = np.zeros((4, 1, 1, 1), dtype="float32")
        seen = set()
        for seed in range(12):
            layer = RandomChoice(
                [_AddConstInPlace(1.0), _AddConstInPlace(10.0)], seed=seed
            )
            out = layer({"images": x}, training=True)
            value = float(backend.convert_to_numpy(out["images"]).ravel()[0])
            seen.add(value)
        self.assertTrue(
            seen.issubset({1.0, 10.0}),
            f"unexpected outcomes {sorted(seen)}; chaining suspected",
        )
