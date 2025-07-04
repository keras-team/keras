import numpy as np
import pytest
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
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
