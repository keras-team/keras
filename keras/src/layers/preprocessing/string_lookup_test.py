import os

import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import saving
from keras.src import testing
from keras.src.ops import convert_to_tensor


class StringLookupTest(testing.TestCase):
    # TODO: increase coverage. Most features aren't being tested.

    def test_config(self):
        layer = layers.StringLookup(
            output_mode="int",
            vocabulary=["a", "b", "c"],
            oov_token="[OOV]",
            mask_token="[MASK]",
        )
        self.run_class_serialization_test(layer)
        self.assertEqual(layer.get_config()["vocabulary"], ["a", "b", "c"])

    def test_vocabulary_file(self):
        temp_dir = self.get_temp_dir()
        vocab_path = os.path.join(temp_dir, "vocab.txt")
        with open(vocab_path, "w") as file:
            file.write("a\nb\nc\n")

        layer = layers.StringLookup(
            output_mode="int",
            vocabulary=vocab_path,
            oov_token="[OOV]",
            mask_token="[MASK]",
            name="index",
        )
        self.assertEqual(
            [str(v) for v in layer.get_vocabulary()],
            ["[MASK]", "[OOV]", "a", "b", "c"],
        )
        self.assertIsNone(layer.get_config().get("vocabulary", None))

        # Make sure vocabulary comes from the archive, not the original file.
        os.remove(vocab_path)

        model = models.Sequential([layer])
        model_path = os.path.join(temp_dir, "test_model.keras")
        model.save(model_path)

        reloaded_model = saving.load_model(model_path)
        reloaded_layer = reloaded_model.get_layer("index")
        self.assertEqual(
            [str(v) for v in reloaded_layer.get_vocabulary()],
            ["[MASK]", "[OOV]", "a", "b", "c"],
        )

    def test_adapt_flow(self):
        layer = layers.StringLookup(
            output_mode="int",
        )
        layer.adapt(["a", "a", "a", "b", "b", "c"])
        input_data = ["b", "c", "d"]
        output = layer(input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([2, 3, 0]))

    def test_fixed_vocabulary(self):
        layer = layers.StringLookup(
            output_mode="int",
            vocabulary=["a", "b", "c"],
        )
        input_data = ["b", "c", "d"]
        output = layer(input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([2, 3, 0]))

    @pytest.mark.skipif(
        not backend.backend() == "tensorflow", reason="Requires tf.SparseTensor"
    )
    def test_sparse_inputs(self):
        import tensorflow as tf

        layer = layers.StringLookup(
            output_mode="int",
            vocabulary=["a", "b", "c"],
        )
        input_data = tf.SparseTensor(
            indices=[[0, 0], [1, 1], [2, 2]],
            values=["b", "c", "d"],
            dense_shape=(3, 3),
        )
        output = layer(input_data)
        self.assertIsInstance(output, tf.SparseTensor)
        self.assertAllClose(output, np.array([[2, 0, 0], [0, 3, 0], [0, 0, 0]]))
        self.assertAllClose(output.values, np.array([2, 3, 0]))

    def test_set_vocabulary(self):
        layer = layers.StringLookup(
            output_mode="int",
        )
        layer.set_vocabulary(["a", "b", "c"])
        input_data = ["b", "c", "d"]
        output = layer(input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([2, 3, 0]))

    def test_tf_data_compatibility(self):
        layer = layers.StringLookup(
            output_mode="int",
            vocabulary=["a", "b", "c"],
        )
        input_data = ["b", "c", "d"]
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(3).map(layer)
        output = next(iter(ds)).numpy()
        self.assertAllClose(output, np.array([2, 3, 0]))

    @pytest.mark.skipif(not backend.backend() == "tensorflow", reason="tf only")
    def test_tensor_as_vocab(self):
        vocab = convert_to_tensor(["a", "b", "c", "d"])
        data = [["a", "c", "d"], ["d", "z", "b"]]
        layer = layers.StringLookup(
            vocabulary=vocab,
        )
        output = layer(data)
        self.assertAllClose(output, np.array([[1, 3, 4], [4, 0, 2]]))

    @pytest.mark.skipif(backend.backend() != "torch", reason="Only torch")
    def test_torch_backend_compatibility(self):
        import torch

        # Forward lookup: String -> number
        forward_lookup = layers.StringLookup(
            vocabulary=["a", "b", "c"], oov_token="[OOV]"
        )
        input_data_str = ["a", "b", "[OOV]", "d"]
        output_numeric = forward_lookup(input_data_str)

        # assert instance of output is torch.Tensor
        self.assertIsInstance(output_numeric, torch.Tensor)
        expected_numeric = torch.tensor([1, 2, 0, 0])
        self.assertAllClose(output_numeric.cpu(), expected_numeric)

        oov = "[OOV]"
        # Inverse lookup: Number -> string
        inverse_lookup = layers.StringLookup(
            vocabulary=["a", "b", "c"], oov_token=oov, invert=True
        )
        input_data_int = torch.tensor([1, 2, 0], dtype=torch.int64)
        output_string = inverse_lookup(input_data_int)
        # Assert that the output is a list
        # See : https://docs.pytorch.org/text/stable/_modules/torchtext/vocab/vocab.html#Vocab.lookup_tokens
        # The torch equivalent implementation of this returns a list of strings
        self.assertIsInstance(output_string, list)
        expected_string = ["a", "b", "[OOV]"]
        self.assertEqual(output_string, expected_string)
