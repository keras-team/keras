import numpy as np
import pytest
from tensorflow import data as tf_data

from keras import backend
from keras import layers
from keras import models
from keras import testing


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
