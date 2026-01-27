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

    def test_one_hot_output_with_higher_rank_input(self):
        input_data = np.array([["a", "b"], ["c", "unknown"]])
        vocabulary = ["a", "b", "c"]
        layer = layers.StringLookup(
            vocabulary=vocabulary, output_mode="one_hot"
        )
        output_data = layer(input_data)
        self.assertEqual(output_data.shape, (2, 2, 4))
        expected_output = np.array(
            [
                [[0, 1, 0, 0], [0, 0, 1, 0]],
                [[0, 0, 0, 1], [1, 0, 0, 0]],
            ]
        )
        self.assertAllClose(output_data, expected_output)
        output_data_3d = layer(np.expand_dims(input_data, axis=0))
        self.assertEqual(output_data_3d.shape, (1, 2, 2, 4))
        self.assertAllClose(
            output_data_3d, np.expand_dims(expected_output, axis=0)
        )

    def test_multi_hot_output_shape(self):
        input_data = np.array([["a", "b"], ["c", "unknown"]])
        vocabulary = ["a", "b", "c"]
        layer = layers.StringLookup(
            vocabulary=vocabulary, output_mode="multi_hot"
        )
        output_data = layer(input_data)
        self.assertEqual(output_data.shape, (2, 4))

    def test_count_output_shape(self):
        input_data = np.array([["a", "b"], ["c", "unknown"]])
        vocabulary = ["a", "b", "c"]
        layer = layers.StringLookup(vocabulary=vocabulary, output_mode="count")
        output_data = layer(input_data)
        self.assertEqual(output_data.shape, (2, 4))

    def test_tf_idf_output_shape(self):
        input_data = np.array([["a", "b"], ["c", "unknown"]])
        vocabulary = ["a", "b", "c"]
        idf_weights = [1.0, 1.0, 1.0]
        layer = layers.StringLookup(
            vocabulary=vocabulary,
            idf_weights=idf_weights,
            output_mode="tf_idf",
        )
        output_data = layer(input_data)
        self.assertEqual(output_data.shape, (2, 4))

    def test_max_tokens(self):
        layer = layers.StringLookup(output_mode="int", max_tokens=4)
        layer.adapt(["a", "a", "a", "b", "b", "c", "d", "e"])
        vocab = layer.get_vocabulary()
        self.assertEqual(len(vocab), 4)

    def test_mask_token(self):
        layer = layers.StringLookup(
            output_mode="int",
            vocabulary=["a", "b", "c"],
            mask_token="[MASK]",
        )
        output = layer(["[MASK]", "a", "b", "c"])
        self.assertAllClose(output, np.array([0, 2, 3, 4]))

    def test_invert(self):
        layer = layers.StringLookup(
            vocabulary=["a", "b", "c"],
            invert=True,
        )
        output = layer([1, 2, 3, 0])
        output_list = [
            x.numpy().decode("utf-8") if hasattr(x, "numpy") else str(x)
            for x in output
        ]
        self.assertEqual(output_list, ["a", "b", "c", "[UNK]"])

    def test_pad_to_max_tokens(self):
        layer = layers.StringLookup(
            vocabulary=["a", "b"],
            output_mode="multi_hot",
            max_tokens=5,
            pad_to_max_tokens=True,
        )
        output = layer(["a", "b"])
        self.assertEqual(output.shape[-1], 5)

    def test_num_oov_indices(self):
        layer = layers.StringLookup(
            vocabulary=["a", "b", "c"],
            num_oov_indices=2,
            output_mode="int",
        )
        output = layer(["a", "b", "c", "x", "y"])
        self.assertAllClose(output[:3], np.array([2, 3, 4]))
        self.assertTrue(
            all(o in [0, 1] for o in backend.convert_to_numpy(output[3:]))
        )

    def test_get_vocabulary(self):
        layer = layers.StringLookup(output_mode="int")
        layer.adapt(["a", "a", "a", "b", "b", "c"])
        vocab = layer.get_vocabulary()
        self.assertEqual(vocab[0], "[UNK]")
        self.assertEqual(vocab[1], "a")

    def test_invalid_max_tokens(self):
        with self.assertRaises(ValueError):
            layers.StringLookup(max_tokens=1)

    def test_invalid_num_oov_indices(self):
        with self.assertRaises(ValueError):
            layers.StringLookup(num_oov_indices=-1)

    def test_sparse_output(self):
        if backend.backend() != "tensorflow":
            self.skipTest("sparse=True only supported on TensorFlow")
        layer = layers.StringLookup(
            vocabulary=["a", "b", "c"],
            output_mode="multi_hot",
            sparse=True,
        )
        output = layer(["a", "b"])
        self.assertTrue(hasattr(output, "indices"))  # SparseTensor check

    def test_invalid_vocabulary_dtype(self):
        with self.assertRaises(TypeError):
            layers.StringLookup(vocabulary_dtype="int64")

    def test_num_oov_indices_zero(self):
        layer = layers.StringLookup(
            vocabulary=["a", "b", "c"],
            num_oov_indices=0,
            output_mode="int",
        )
        output = layer(["a", "b", "c"])
        self.assertAllClose(output, np.array([0, 1, 2]))

    def test_adapt_with_steps(self):
        layer = layers.StringLookup(output_mode="int")
        ds = tf_data.Dataset.from_tensor_slices(
            ["a", "b", "c", "a", "a"]
        ).batch(2)
        layer.adapt(ds, steps=2)
        vocab = layer.get_vocabulary()
        self.assertIn("a", vocab)

    def test_vocabulary_from_file(self):
        tmp_dir = self.get_temp_dir()
        vocab_file = os.path.join(tmp_dir, "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("a\nb\nc\n")
        layer = layers.StringLookup(
            vocabulary=vocab_file,
            output_mode="int",
        )
        output = layer(["a", "b", "c", "unknown"])
        self.assertAllClose(output, np.array([1, 2, 3, 0]))

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
