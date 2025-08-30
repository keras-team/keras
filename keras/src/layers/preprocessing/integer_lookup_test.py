import numpy as np
from tensorflow import data as tf_data

from keras.src import backend
from keras.src import layers
from keras.src import testing


class IntegerLookupTest(testing.TestCase):
    # TODO: increase coverage. Most features aren't being tested.

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
