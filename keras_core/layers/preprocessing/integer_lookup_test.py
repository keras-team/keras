import numpy as np
import tensorflow as tf

from keras_core import backend
from keras_core import layers
from keras_core import testing


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
        layer = layers.IntegerLookup(
            output_mode="int",
        )
        layer.adapt([1, 1, 1, 1, 2, 2, 2, 3, 3, 4])
        input_data = [2, 3, 4, 5]
        output = layer(input_data)
        self.assertTrue(backend.is_tensor(output))
        self.assertAllClose(output, np.array([2, 3, 4, 0]))

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
        ds = tf.data.Dataset.from_tensor_slices(input_data).batch(4).map(layer)
        for output in ds.take(1):
            output = output.numpy()
        self.assertAllClose(output, np.array([2, 3, 4, 0]))
