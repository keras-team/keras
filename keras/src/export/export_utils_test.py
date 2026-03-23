"""Tests for keras.src.export.export_utils."""

import numpy as np

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.export.export_utils import convert_spec_to_tensor
from keras.src.export.export_utils import get_input_signature
from keras.src.export.export_utils import make_input_spec


class MakeInputSpecTest(testing.TestCase):
    def test_from_keras_tensor(self):
        inp = layers.Input(shape=(10,), dtype="float32", name="my_input")
        spec = make_input_spec(inp)
        self.assertIsInstance(spec, layers.InputSpec)
        self.assertEqual(spec.shape[1:], (10,))
        self.assertEqual(spec.dtype, "float32")

    def test_from_input_spec(self):
        original = layers.InputSpec(shape=(None, 5), dtype="float32")
        spec = make_input_spec(original)
        self.assertIs(spec, original)

    def test_from_input_spec_missing_shape_raises(self):
        original = layers.InputSpec(dtype="float32")
        with self.assertRaisesRegex(ValueError, "shape.*dtype.*must be"):
            make_input_spec(original)

    def test_from_input_spec_missing_dtype_raises(self):
        original = layers.InputSpec(shape=(None, 5))
        with self.assertRaisesRegex(ValueError, "shape.*dtype.*must be"):
            make_input_spec(original)

    def test_from_backend_tensor(self):
        t = backend.convert_to_tensor(np.zeros((2, 3, 4), dtype="float32"))
        spec = make_input_spec(t)
        self.assertIsInstance(spec, layers.InputSpec)
        # Batch dimension should be dynamic (None)
        self.assertIsNone(spec.shape[0])
        self.assertEqual(spec.shape[1:], (3, 4))

    def test_unsupported_type_raises(self):
        with self.assertRaisesRegex(TypeError, "Unsupported"):
            make_input_spec("not a tensor")

    def test_unsupported_type_list_raises(self):
        with self.assertRaisesRegex(TypeError, "Unsupported"):
            make_input_spec([1, 2, 3])


class GetInputSignatureTest(testing.TestCase):
    def test_functional_model(self):
        inp = layers.Input(shape=(10,), name="x")
        out = layers.Dense(5)(inp)
        model = models.Model(inputs=inp, outputs=out)
        sig = get_input_signature(model)
        self.assertIsInstance(sig, list)
        self.assertEqual(len(sig), 1)

    def test_sequential_model(self):
        model = models.Sequential([layers.Input(shape=(8,)), layers.Dense(4)])
        sig = get_input_signature(model)
        # Sequential returns a list of InputSpecs
        self.assertIsNotNone(sig)

    def test_not_a_model_raises(self):
        with self.assertRaisesRegex(TypeError, "must be a.*Model"):
            get_input_signature("not a model")

    def test_unbuilt_model_raises(self):
        model = models.Sequential([layers.Dense(4)])
        with self.assertRaisesRegex(ValueError, "not yet been built"):
            get_input_signature(model)

    def test_functional_multi_input(self):
        inp1 = layers.Input(shape=(5,), name="a")
        inp2 = layers.Input(shape=(3,), name="b")
        out = layers.Concatenate()([inp1, inp2])
        model = models.Model(inputs=[inp1, inp2], outputs=out)
        sig = get_input_signature(model)
        self.assertIsInstance(sig, list)


class ConvertSpecToTensorTest(testing.TestCase):
    def test_basic_conversion(self):
        spec = layers.InputSpec(shape=(2, 3), dtype="float32")
        tensor = convert_spec_to_tensor(spec)
        self.assertEqual(backend.standardize_shape(tensor.shape), (2, 3))

    def test_replace_none_number(self):
        spec = layers.InputSpec(shape=(None, 5), dtype="float32")
        tensor = convert_spec_to_tensor(spec, replace_none_number=4)
        self.assertEqual(backend.standardize_shape(tensor.shape), (4, 5))

    def test_without_replace_none_raises(self):
        """If shape has None and no replacement, should raise."""
        spec = layers.InputSpec(shape=(None, 5), dtype="float32")
        with self.assertRaises(Exception):
            convert_spec_to_tensor(spec)


if __name__ == "__main__":
    testing.run_tests()
