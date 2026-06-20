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
