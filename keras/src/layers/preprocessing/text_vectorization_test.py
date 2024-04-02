import os

import numpy as np
import pytest
import tensorflow as tf
from tensorflow import data as tf_data

from keras.src import Sequential
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import saving
from keras.src import testing


class TextVectorizationTest(testing.TestCase):
    # TODO: increase coverage. Most features aren't being tested.

    def test_config(self):
        layer = layers.TextVectorization(
            output_mode="int",
            vocabulary=["one", "two"],
            output_sequence_length=5,
        )
        self.run_class_serialization_test(layer)

    def test_adapt_flow(self):
        max_tokens = 5000
        max_len = 4
        layer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=max_len,
        )
        layer.adapt(["foo bar", "bar baz", "baz bada boom"])
        input_data = [["foo qux bar"], ["qux baz"]]
        output = layer(input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([[4, 1, 3, 0], [1, 2, 0, 0]]))

    def test_fixed_vocabulary(self):
        max_tokens = 5000
        max_len = 4
        layer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=max_len,
            vocabulary=["baz", "bar", "foo"],
        )
        input_data = [["foo qux bar"], ["qux baz"]]
        output = layer(input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([[4, 1, 3, 0], [1, 2, 0, 0]]))

    def test_set_vocabulary(self):
        max_tokens = 5000
        max_len = 4
        layer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=max_len,
        )
        layer.set_vocabulary(["baz", "bar", "foo"])
        input_data = [["foo qux bar"], ["qux baz"]]
        output = layer(input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([[4, 1, 3, 0], [1, 2, 0, 0]]))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Requires string input dtype"
    )
    def test_save_load_with_ngrams_flow(self):
        input_data = np.array(["foo bar", "bar baz", "baz bada boom"])
        model = Sequential(
            [
                layers.Input(dtype="string", shape=(1,)),
                layers.TextVectorization(ngrams=(1, 2)),
            ]
        )
        model.layers[0].adapt(input_data)
        output = model(input_data)
        temp_filepath = os.path.join(self.get_temp_dir(), "model.keras")
        model.save(temp_filepath)
        model = saving.load_model(temp_filepath)
        self.assertAllClose(output, model(input_data))

    def test_tf_data_compatibility(self):
        max_tokens = 5000
        max_len = 4
        layer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=max_len,
            vocabulary=["baz", "bar", "foo"],
        )
        input_data = [["foo qux bar"], ["qux baz"]]
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(output, np.array([[4, 1, 3, 0], [1, 2, 0, 0]]))

        # Test adapt flow
        layer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=max_len,
        )
        layer.adapt(input_data)
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        for output in ds.take(1):
            output.numpy()

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Requires string tensors."
    )
    def test_tf_as_first_sequential_layer(self):
        layer = layers.TextVectorization(
            max_tokens=10,
            output_mode="int",
            output_sequence_length=3,
        )
        layer.set_vocabulary(["baz", "bar", "foo"])
        model = models.Sequential(
            [
                layer,
                layers.Embedding(5, 4),
            ]
        )
        model(backend.convert_to_tensor([["foo qux bar"], ["qux baz"]]))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Requires ragged tensors."
    )
    def test_ragged_tensor(self):
        layer = layers.TextVectorization(
            output_mode="int",
            vocabulary=["baz", "bar", "foo"],
            ragged=True,
        )
        input_data = [["foo qux bar"], ["qux baz"], ["foo"]]
        output = layer(input_data)
        self.assertIsInstance(output, tf.RaggedTensor)
        self.assertEqual(output.shape, (3, None))
        self.assertEqual(output.to_list(), [[4, 1, 3], [1, 2], [4]])

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Requires ragged tensors."
    )
    def test_ragged_tensor_output_length(self):
        layer = layers.TextVectorization(
            output_mode="int",
            vocabulary=["baz", "bar", "foo"],
            ragged=True,
            output_sequence_length=2,
        )
        input_data = [["foo qux bar"], ["qux baz"], ["foo"]]
        output = layer(input_data)
        self.assertIsInstance(output, tf.RaggedTensor)
        self.assertEqual(output.shape, (3, None))
        self.assertEqual(output.to_list(), [[4, 1], [1, 2], [4]])

    @pytest.mark.skipif(
        backend.backend() == "tensorflow",
        reason="Verify raises exception for non-TF backends",
    )
    def test_raises_exception_ragged_tensor(self):
        with self.assertRaises(ValueError):
            _ = layers.TextVectorization(
                output_mode="int",
                vocabulary=["baz", "bar", "foo"],
                ragged=True,
            )
