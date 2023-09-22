import numpy as np
from tensorflow import data as tf_data

from keras import backend
from keras import layers
from keras import testing


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
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(output, np.array([2, 3, 4, 0]))
