import os

import numpy as np
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class IntegerLookupTest(testing.TestCase):
    def test_config(self):
        layer = layers.IntegerLookup(
            output_mode="int",
            vocabulary=[1, 2, 3],
            oov_token=1,
            mask_token=0,
        )
        self.run_class_serialization_test(layer)

    def test_adapt_flow(self):
        adapt_data = [1, 1, 1, 2, 2, 3]
        single_sample_input_data = [1, 2, 4]
        batch_input_data = [[1, 2, 4], [2, 3, 5]]

        # int mode
        layer = layers.IntegerLookup(
            output_mode="int",
        )
        layer.adapt(adapt_data)
        output = layer(single_sample_input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([1, 2, 0]))
        output = layer(batch_input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([[1, 2, 0], [2, 3, 0]]))

        # one_hot mode
        layer = layers.IntegerLookup(
            output_mode="one_hot",
        )
        layer.adapt(adapt_data)
        output = layer(single_sample_input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(
            output, np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])
        )

        # multi_hot mode
        layer = layers.IntegerLookup(
            output_mode="multi_hot",
        )
        layer.adapt(adapt_data)
        output = layer(single_sample_input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([1, 1, 1, 0]))

        # tf_idf mode
        layer = layers.IntegerLookup(
            output_mode="tf_idf",
        )
        layer.adapt(adapt_data)
        output = layer(single_sample_input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(
            output, np.array([1.133732, 0.916291, 1.098612, 0.0])
        )

        # count mode
        layer = layers.IntegerLookup(
            output_mode="count",
        )
        layer.adapt(adapt_data)
        output = layer([1, 2, 3, 4, 1, 2, 1])
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([1, 3, 2, 1]))

    def test_fixed_vocabulary(self):
        layer = layers.IntegerLookup(
            output_mode="int",
            vocabulary=[1, 2, 3, 4],
        )
        input_data = [2, 3, 4, 5]
        output = layer(input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([2, 3, 4, 0]))

    def test_set_vocabulary(self):
        layer = layers.IntegerLookup(
            output_mode="int",
        )
        layer.set_vocabulary([1, 2, 3, 4])
        input_data = [2, 3, 4, 5]
        output = layer(input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([2, 3, 4, 0]))

    def test_tf_data_compatibility(self):
        layer = layers.IntegerLookup(
            output_mode="int",
            vocabulary=[1, 2, 3, 4],
        )
        input_data = [2, 3, 4, 5]
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(4).map(layer)
        output = next(iter(ds)).numpy()
        self.assertAllClose(output, np.array([2, 3, 4, 0]))

    def test_one_hot_output_with_higher_rank_input(self):
        input_data = np.array([[1, 2], [3, 0]])
        vocabulary = [1, 2, 3]
        layer = layers.IntegerLookup(
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
        input_data = np.array([[1, 2], [3, 0]])
        vocabulary = [1, 2, 3]
        layer = layers.IntegerLookup(
            vocabulary=vocabulary, output_mode="multi_hot"
        )
        output_data = layer(input_data)
        self.assertEqual(output_data.shape, (2, 4))

    def test_count_output_shape(self):
        input_data = np.array([[1, 2], [3, 0]])
        vocabulary = [1, 2, 3]
        layer = layers.IntegerLookup(vocabulary=vocabulary, output_mode="count")
        output_data = layer(input_data)
        self.assertEqual(output_data.shape, (2, 4))

    def test_tf_idf_output_shape(self):
        input_data = np.array([[1, 2], [3, 0]])
        vocabulary = [1, 2, 3]
        idf_weights = [1.0, 1.0, 1.0]
        layer = layers.IntegerLookup(
            vocabulary=vocabulary,
            idf_weights=idf_weights,
            output_mode="tf_idf",
        )
        output_data = layer(input_data)
        self.assertEqual(output_data.shape, (2, 4))

    def test_max_tokens(self):
        layer = layers.IntegerLookup(output_mode="int", max_tokens=4)
        layer.adapt([1, 2, 3, 4, 5, 6, 1, 1, 2, 2])
        vocab = layer.get_vocabulary()
        self.assertEqual(len(vocab), 4)

    def test_mask_token(self):
        layer = layers.IntegerLookup(
            output_mode="int",
            vocabulary=[1, 2, 3],
            mask_token=0,
        )
        output = layer([0, 1, 2, 3])
        self.assertAllClose(output, np.array([0, 2, 3, 4]))

    def test_invert(self):
        layer = layers.IntegerLookup(
            vocabulary=[10, 20, 30],
            invert=True,
        )
        output = layer([1, 2, 3, 0])
        self.assertAllClose(output, np.array([10, 20, 30, -1]))

    def test_pad_to_max_tokens(self):
        layer = layers.IntegerLookup(
            vocabulary=[1, 2],
            output_mode="multi_hot",
            max_tokens=5,
            pad_to_max_tokens=True,
        )
        output = layer([1, 2])
        self.assertEqual(output.shape[-1], 5)

    def test_num_oov_indices(self):
        layer = layers.IntegerLookup(
            vocabulary=[1, 2, 3],
            num_oov_indices=2,
            output_mode="int",
        )
        output = layer([1, 2, 3, 999, 1000])
        self.assertAllClose(output[:3], np.array([2, 3, 4]))
        self.assertTrue(
            all(o in [0, 1] for o in backend.convert_to_numpy(output[3:]))
        )

    def test_get_vocabulary(self):
        layer = layers.IntegerLookup(output_mode="int")
        layer.adapt([5, 5, 5, 10, 10, 15])
        vocab = layer.get_vocabulary()
        self.assertEqual(vocab[0], -1)
        self.assertEqual(vocab[1], 5)

    def test_invalid_max_tokens(self):
        with self.assertRaises(ValueError):
            layers.IntegerLookup(max_tokens=1)

    def test_invalid_num_oov_indices(self):
        with self.assertRaises(ValueError):
            layers.IntegerLookup(num_oov_indices=-1)

    def test_sparse_output(self):
        if backend.backend() != "tensorflow":
            self.skipTest("sparse=True only supported on TensorFlow")
        layer = layers.IntegerLookup(
            vocabulary=[1, 2, 3],
            output_mode="multi_hot",
            sparse=True,
        )
        output = layer([1, 2])
        self.assertTrue(hasattr(output, "indices"))  # SparseTensor check

    def test_invalid_vocabulary_dtype(self):
        with self.assertRaises(ValueError):
            layers.IntegerLookup(vocabulary_dtype="int32")

    def test_num_oov_indices_zero(self):
        layer = layers.IntegerLookup(
            vocabulary=[1, 2, 3],
            num_oov_indices=0,
            output_mode="int",
        )
        output = layer([1, 2, 3])
        self.assertAllClose(output, np.array([0, 1, 2]))

    def test_adapt_with_steps(self):
        layer = layers.IntegerLookup(output_mode="int")
        ds = tf_data.Dataset.from_tensor_slices([1, 2, 3, 1, 1]).batch(2)
        layer.adapt(ds, steps=2)
        vocab = layer.get_vocabulary()
        self.assertIn(1, vocab)

    def test_vocabulary_from_file(self):
        tmp_dir = self.get_temp_dir()
        vocab_file = os.path.join(tmp_dir, "vocab.txt")
        with open(vocab_file, "w") as f:
            f.write("10\n20\n30\n")
        layer = layers.IntegerLookup(
            vocabulary=vocab_file,
            output_mode="int",
        )
        output = layer([10, 20, 30, 999])
        self.assertAllClose(output, np.array([1, 2, 3, 0]))
