import math

import numpy as np
import scipy
import tensorflow as tf
from absl.testing import parameterized

from keras import testing
from keras.trainers.data_adapters import generator_data_adapter


def example_generator(x, y, sample_weight=None, batch_size=32):
    def make():
        for i in range(math.ceil(len(x) / batch_size)):
            low = i * batch_size
            high = min(low + batch_size, len(x))
            batch_x = x[low:high]
            batch_y = y[low:high]
            if sample_weight is not None:
                yield batch_x, batch_y, sample_weight[low:high]
            else:
                yield batch_x, batch_y

    return make


class GeneratorDataAdapterTest(testing.TestCase, parameterized.TestCase):
    @parameterized.parameters(
        [
            (True,),
            (False,),
        ]
    )
    def test_basic_flow(self, use_sample_weight):
        x = np.random.random((64, 4))
        y = np.array([[i, i] for i in range(64)], dtype="float64")
        if use_sample_weight:
            sw = np.random.random((64,))
        else:
            sw = None
        make_generator = example_generator(
            x,
            y,
            sample_weight=sw,
            batch_size=16,
        )
        adapter = generator_data_adapter.GeneratorDataAdapter(make_generator())

        gen = adapter.get_numpy_iterator()
        sample_order = []
        for batch in gen:
            if use_sample_weight:
                self.assertEqual(len(batch), 3)
                bx, by, bsw = batch
            else:
                self.assertEqual(len(batch), 2)
                bx, by = batch

            self.assertIsInstance(bx, np.ndarray)
            self.assertIsInstance(by, np.ndarray)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.shape, (16, 4))
            self.assertEqual(by.shape, (16, 2))
            if use_sample_weight:
                self.assertIsInstance(bsw, np.ndarray)
            for i in range(by.shape[0]):
                sample_order.append(by[i, 0])
        self.assertAllClose(sample_order, list(range(64)))

        adapter = generator_data_adapter.GeneratorDataAdapter(
            make_generator(),
        )
        ds = adapter.get_tf_dataset()
        sample_order = []
        for batch in ds:
            if use_sample_weight:
                self.assertEqual(len(batch), 3)
                bx, by, bsw = batch
            else:
                self.assertEqual(len(batch), 2)
                bx, by = batch
            self.assertIsInstance(bx, tf.Tensor)
            self.assertIsInstance(by, tf.Tensor)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(tuple(bx.shape), (16, 4))
            self.assertEqual(tuple(by.shape), (16, 2))
            if use_sample_weight:
                self.assertIsInstance(bsw, tf.Tensor)
            for i in range(by.shape[0]):
                sample_order.append(by[i, 0])
        self.assertAllClose(sample_order, list(range(64)))

    def test_tf_sparse_tensors(self):
        def generate_tf():
            for i in range(4):
                x = tf.SparseTensor(
                    indices=[[0, 0], [1, 2]],
                    values=[1.0, 2.0],
                    dense_shape=(2, 4),
                )
                y = tf.SparseTensor(
                    indices=[[0, 0], [1, 1]],
                    values=[3.0, 4.0],
                    dense_shape=(2, 2),
                )
                yield x, y

        adapter = generator_data_adapter.GeneratorDataAdapter(generate_tf())
        ds = adapter.get_tf_dataset()
        for batch in ds:
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, tf.SparseTensor)
            self.assertIsInstance(by, tf.SparseTensor)
            self.assertEqual(bx.shape, (2, 4))
            self.assertEqual(by.shape, (2, 2))

    def test_scipy_sparse_tensors(self):
        def generate_scipy():
            for i in range(4):
                x = scipy.sparse.coo_matrix(
                    ([1.0, 2.0], ([0, 1], [0, 2])), shape=[2, 4]
                )
                y = scipy.sparse.coo_matrix(
                    ([3.0, 4.0], ([0, 1], [0, 1])), shape=[2, 2]
                )
                yield x, y

        adapter = generator_data_adapter.GeneratorDataAdapter(generate_scipy())
        ds = adapter.get_tf_dataset()
        for batch in ds:
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, tf.SparseTensor)
            self.assertIsInstance(by, tf.SparseTensor)
            self.assertEqual(bx.shape, (2, 4))
            self.assertEqual(by.shape, (2, 2))
