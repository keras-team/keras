import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized
from tensorflow import data as tf_data

from keras.src import Sequential
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import saving
from keras.src import testing


class TextVectorizationTest(testing.TestCase, parameterized.TestCase):
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
        output = next(iter(ds)).numpy()
        self.assertAllClose(output, np.array([[4, 1, 3, 0], [1, 2, 0, 0]]))

        # Test adapt flow
        layer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=max_len,
        )
        layer.adapt(input_data)
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(2).map(layer)
        next(iter(ds)).numpy()

    @parameterized.named_parameters(
        [
            ("from_ragged", "whitespace"),  # intermediate tensor is ragged
            ("from_dense", None),  # intermediate tensor is dense
        ]
    )
    def test_static_output_sequence_length(self, split):
        max_tokens = 5000
        max_len = 4
        layer = layers.TextVectorization(
            max_tokens=max_tokens,
            output_mode="int",
            output_sequence_length=max_len,
            split=split,
            vocabulary=["baz", "bar", "foo"],
        )
        if split:
            input_data = [["foo qux bar"], ["qux baz"]]
        else:
            input_data = [["foo"], ["baz"]]

        def call_layer(x):
            result = layer(x)
            self.assertEqual(result.shape, (None, 4))
            return result

        ds = (
            tf_data.Dataset.from_tensor_slices(input_data)
            .batch(2)
            .map(call_layer)
        )
        next(iter(ds))

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

    def test_multi_hot_output(self):
        layer = layers.TextVectorization(
            output_mode="multi_hot", vocabulary=["foo", "bar", "baz"]
        )
        input_data = [["foo bar"], ["baz foo foo"]]
        output = layer(input_data)

        """
        First batch
        Tokens present: ["foo", "bar"]
            For each token in vocabulary:
            foo (index 1): present -> 1
            bar (index 2): present -> 1
            baz (index 3): absent -> 0
            Result: [0, 1, 1, 0]
        
        Second batch
            Tokens: ["baz", "foo", "foo"]
            For each token in vocabulary:
            foo (index 1): present -> 1
            bar (index 2): absent -> 0
            baz (index 3): present -> 1
            Result: [0, 1, 0, 1]
        """
        self.assertAllClose(output, [[0, 1, 1, 0], [0, 1, 0, 1]])

    def test_output_mode_count_output(self):
        layer = layers.TextVectorization(
            output_mode="count", vocabulary=["foo", "bar", "baz"]
        )
        output = layer(["foo bar", "baz foo foo"])
        self.assertAllClose(output, [[0, 1, 1, 0], [0, 2, 0, 1]])

    def test_output_mode_tf_idf_output(self):
        layer = layers.TextVectorization(
            output_mode="tf_idf",
            vocabulary=["foo", "bar", "baz"],
            idf_weights=[0.3, 0.5, 0.2],
        )
        output = layer(["foo bar", "baz foo foo"])
        self.assertAllClose(
            output, [[0.0, 0.3, 0.5, 0.0], [0.0, 0.6, 0.0, 0.2]]
        )

    def test_lower_and_strip_punctuation_standardization(self):
        layer = layers.TextVectorization(
            standardize="lower_and_strip_punctuation",
            vocabulary=["hello", "world", "this", "is", "nice", "test"],
        )
        output = layer(["Hello, World!. This is just a nice test!"])
        self.assertTrue(backend.is_tensor(output))

        # test output sequence length, taking first batch.
        self.assertEqual(len(output[0]), 8)

        self.assertAllEqual(output, [[2, 3, 4, 5, 1, 1, 6, 7]])

    def test_lower_standardization(self):
        layer = layers.TextVectorization(
            standardize="lower",
            vocabulary=[
                "hello,",
                "hello",
                "world",
                "this",
                "is",
                "nice",
                "test",
            ],
        )
        output = layer(["Hello, World!. This is just a nice test!"])
        self.assertTrue(backend.is_tensor(output))
        self.assertEqual(len(output[0]), 8)
        """
        The input is lowercased and tokenized into words. The vocab is:
        {0: '',
        1: '[UNK]',
        2: 'hello,',
        3: 'hello',
        4: 'world',
        5: 'this',
        6: 'is',
        7: 'nice',
        8: 'test'}
        """
        self.assertAllEqual(output, [[2, 1, 5, 6, 1, 1, 7, 1]])

    def test_char_splitting(self):
        layer = layers.TextVectorization(
            split="character", vocabulary=list("abcde"), output_mode="int"
        )
        output = layer(["abcf"])
        self.assertTrue(backend.is_tensor(output))
        self.assertEqual(len(output[0]), 4)
        self.assertAllEqual(output, [[2, 3, 4, 1]])

    def test_custom_splitting(self):
        def custom_split(text):
            return tf.strings.split(text, sep="|")

        layer = layers.TextVectorization(
            split=custom_split,
            vocabulary=["foo", "bar", "foobar"],
            output_mode="int",
        )
        output = layer(["foo|bar"])
        self.assertTrue(backend.is_tensor(output))

        # after custom split, the outputted index should be the last
        # token in the vocab.
        self.assertAllEqual(output, [[4]])
