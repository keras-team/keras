import math

import jax
import numpy as np
import scipy
import tensorflow as tf
import torch
from absl.testing import parameterized
from jax import numpy as jnp

from keras import testing
from keras.testing.test_utils import named_product
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
    @parameterized.named_parameters(
        named_product(
            [
                {"testcase_name": "use_weight", "use_sample_weight": True},
                {"testcase_name": "no_weight", "use_sample_weight": False},
            ],
            generator_type=["np", "tf", "jax", "torch"],
            iterator_type=["np", "tf", "jax", "torch"],
        )
    )
    def test_basic_flow(self, use_sample_weight, generator_type, iterator_type):
        x = np.random.random((34, 4)).astype("float32")
        y = np.array([[i, i] for i in range(34)], dtype="float32")
        sw = np.random.random((34,)).astype("float32")
        if generator_type == "tf":
            x, y, sw = tf.constant(x), tf.constant(y), tf.constant(sw)
        elif generator_type == "jax":
            x, y, sw = jnp.array(x), jnp.array(y), jnp.array(sw)
        elif generator_type == "torch":
            x, y, sw = (
                torch.as_tensor(x),
                torch.as_tensor(y),
                torch.as_tensor(sw),
            )
        if not use_sample_weight:
            sw = None
        make_generator = example_generator(
            x,
            y,
            sample_weight=sw,
            batch_size=16,
        )

        adapter = generator_data_adapter.GeneratorDataAdapter(make_generator())
        if iterator_type == "np":
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray
        elif iterator_type == "tf":
            it = adapter.get_tf_dataset()
            expected_class = tf.Tensor
        elif iterator_type == "jax":
            it = adapter.get_jax_iterator()
            expected_class = jax.Array
        elif iterator_type == "torch":
            it = adapter.get_torch_dataloader()
            expected_class = torch.Tensor

        sample_order = []
        for i, batch in enumerate(it):
            if use_sample_weight:
                self.assertEqual(len(batch), 3)
                bx, by, bsw = batch
            else:
                self.assertEqual(len(batch), 2)
                bx, by = batch
            self.assertIsInstance(bx, expected_class)
            self.assertIsInstance(by, expected_class)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertContainsExactSubsequence(str(bx.dtype), "float32")
            if i < 2:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 4))
                self.assertEqual(by.shape, (2, 2))
            if use_sample_weight:
                self.assertIsInstance(bsw, expected_class)
            for i in range(by.shape[0]):
                sample_order.append(by[i, 0])
        self.assertAllClose(sample_order, list(range(34)))

    def test_tf_sparse_tensors_with_tf_dataset(self):
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

    def test_scipy_sparse_tensors_with_tf_dataset(self):
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
