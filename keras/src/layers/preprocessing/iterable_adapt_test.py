"""Tests for arbitrary iterable support in preprocessing layer adapt() methods."""

import numpy as np
import pytest
from absl.testing import parameterized

from keras import utils
from keras.src import layers
from keras.src import testing


class IterableAdaptTest(testing.TestCase, parameterized.TestCase):

    @parameterized.parameters([("list",), ("generator",)])
    def test_feature_space_adapt_with_iterables(self, input_type):
        def create_data():
            return [
                {
                    "float_values": float(i),
                    "string_values": f"item_{i % 10}",
                    "int_values": i % 5,
                }
                for i in range(100)
            ]
        
        if input_type == "list":
            data = create_data()
        elif input_type == "generator":
            def data_generator():
                for item in create_data():
                    yield item
            data = data_generator()
        
        feature_space = utils.FeatureSpace(
            features={
                "float_values": "float_normalized",
                "string_values": "string_categorical",
                "int_values": "integer_categorical",
            },
            output_mode="concat",
        )
        
        feature_space.adapt(data)
        
        test_data = {
            "float_values": [1.0, 2.0],
            "string_values": ["item_1", "item_2"],
            "int_values": [1, 2],
        }
        output = feature_space(test_data)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape[0], 2)

    @parameterized.parameters([("list",), ("generator",)])
    def test_string_lookup_adapt_with_iterables(self, input_type):
        vocab_data = ["apple", "banana", "cherry", "apple", "banana", "apple"]
        
        if input_type == "list":
            data = vocab_data
        elif input_type == "generator":
            def vocab_generator():
                for word in vocab_data:
                    yield word
            data = vocab_generator()
        
        layer = layers.StringLookup()
        layer.adapt(data)
        
        vocab = layer.get_vocabulary()
        self.assertIn("apple", vocab)
        self.assertIn("banana", vocab)
        self.assertIn("cherry", vocab)

    @parameterized.parameters([("list",), ("generator",)])
    def test_integer_lookup_adapt_with_iterables(self, input_type):
        vocab_data = [1, 2, 3, 1, 2, 1, 4, 5]
        
        if input_type == "list":
            data = vocab_data
        elif input_type == "generator":
            def vocab_generator():
                for value in vocab_data:
                    yield value
            data = vocab_generator()
        
        layer = layers.IntegerLookup()
        layer.adapt(data)
        
        vocab = layer.get_vocabulary()
        self.assertIn(1, vocab)
        self.assertIn(2, vocab)
        self.assertIn(3, vocab)

    @parameterized.parameters([("list",), ("generator",)])
    def test_text_vectorization_adapt_with_iterables(self, input_type):
        texts = [
            "the quick brown fox",
            "jumps over the lazy dog",
            "the quick dog",
        ]
        
        if input_type == "list":
            data = texts
        elif input_type == "generator":
            def text_generator():
                for text in texts:
                    yield text
            data = text_generator()
        
        layer = layers.TextVectorization()
        layer.adapt(data)
        
        vocab = layer.get_vocabulary()
        self.assertIn("the", vocab)
        self.assertIn("quick", vocab)
        self.assertIn("dog", vocab)

    @parameterized.parameters([("list",), ("generator",)])
    def test_discretization_adapt_with_iterables(self, input_type):
        if input_type == "list":
            data = [float(i) for i in range(100)]
        elif input_type == "generator":
            def data_generator():
                for i in range(100):
                    yield [float(i)]
            data = data_generator()
        
        layer = layers.Discretization(num_bins=5)
        layer.adapt(data)
        
        test_input = np.array([10.0, 50.0, 90.0])
        output = layer(test_input)
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (3,))

    def test_feature_space_empty_iterable_raises_error(self):
        feature_space = utils.FeatureSpace(
            features={"value": "float_normalized"},
            output_mode="concat",
        )
        
        with self.assertRaisesRegex(ValueError, "empty iterable"):
            feature_space.adapt([])

    def test_string_lookup_invalid_iterable_raises_error(self):
        layer = layers.StringLookup()
        
        with self.assertRaisesRegex(ValueError, "adapt"):
            layer.adapt(12345)

    def test_text_vectorization_generator_with_batch_size(self):
        def text_generator():
            for i in range(50):
                yield f"document number {i}"
        
        layer = layers.TextVectorization()
        layer.adapt(text_generator(), batch_size=10)
        
        vocab = layer.get_vocabulary()
        self.assertIn("number", vocab)
        self.assertIn("document", vocab)

    def test_normalization_already_supports_iterables(self):
        layer = layers.Normalization()
        data_array = np.array([[float(i)] for i in range(100)])
        layer.adapt(data_array)
        
        output = layer(np.array([[10.0], [50.0], [90.0]]))
        self.assertIsNotNone(output)
        self.assertEqual(output.shape, (3, 1))
