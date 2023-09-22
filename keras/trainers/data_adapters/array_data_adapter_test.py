import numpy as np
import pandas
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras import backend
from keras import testing
from keras.trainers.data_adapters import array_data_adapter


class TestArrayDataAdapter(testing.TestCase, parameterized.TestCase):
    def make_array(self, array_type, shape, dtype="float32"):
        if array_type == "np":
            return np.ones(shape, dtype=dtype)
        elif array_type == "tf":
            return tf.ones(shape, dtype=dtype)
        elif array_type == "backend":
            if backend.backend() == "jax":
                import jax

                return jax.numpy.ones(shape, dtype=dtype)
            elif backend.backend() == "torch":
                import torch

                return torch.tensor(np.ones(shape, dtype=dtype))
            else:
                return tf.ones(shape, dtype=dtype)
        elif array_type == "pandas":
            return pandas.DataFrame(np.ones(shape, dtype=dtype))

    @parameterized.parameters([("np",), ("tf",), ("backend",), ("pandas",)])
    def test_basic_flow(self, array_type):
        x = self.make_array(array_type, (34, 4))
        y = self.make_array(array_type, (34, 2))

        adapter = array_data_adapter.ArrayDataAdapter(
            x,
            y=y,
            sample_weight=None,
            batch_size=16,
            steps=None,
            shuffle=False,
        )
        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 2)

        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, np.ndarray)
            self.assertIsInstance(by, np.ndarray)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, backend.floatx())
            if i < 2:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 4))
                self.assertEqual(by.shape, (2, 2))
        ds = adapter.get_tf_dataset()
        for i, batch in enumerate(ds):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, tf.Tensor)
            self.assertIsInstance(by, tf.Tensor)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, backend.floatx())
            if i < 2:
                self.assertEqual(tuple(bx.shape), (16, 4))
                self.assertEqual(tuple(by.shape), (16, 2))
            else:
                self.assertEqual(tuple(bx.shape), (2, 4))
                self.assertEqual(tuple(by.shape), (2, 2))

    def test_multi_inputs_and_outputs(self):
        x1 = np.random.random((34, 1))
        x2 = np.random.random((34, 2))
        y1 = np.random.random((34, 3))
        y2 = np.random.random((34, 4))
        sw = np.random.random((34,))
        adapter = array_data_adapter.ArrayDataAdapter(
            x={"x1": x1, "x2": x2},
            y=[y1, y2],
            sample_weight=sw,
            batch_size=16,
            steps=None,
            shuffle=False,
        )
        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 3)
            bx, by, bw = batch
            self.assertIsInstance(bx, dict)
            # NOTE: the y list was converted to a tuple for tf.data
            # compatibility.
            self.assertIsInstance(by, tuple)
            self.assertIsInstance(bw, tuple)

            self.assertIsInstance(bx["x1"], np.ndarray)
            self.assertIsInstance(bx["x2"], np.ndarray)
            self.assertIsInstance(by[0], np.ndarray)
            self.assertIsInstance(by[1], np.ndarray)
            self.assertIsInstance(bw[0], np.ndarray)
            self.assertIsInstance(bw[1], np.ndarray)

            self.assertEqual(bx["x1"].dtype, by[0].dtype)
            self.assertEqual(bx["x1"].dtype, backend.floatx())
            if i < 2:
                self.assertEqual(bx["x1"].shape, (16, 1))
                self.assertEqual(bx["x2"].shape, (16, 2))
                self.assertEqual(by[0].shape, (16, 3))
                self.assertEqual(by[1].shape, (16, 4))
                self.assertEqual(bw[0].shape, (16,))
                self.assertEqual(bw[1].shape, (16,))
            else:
                self.assertEqual(bx["x1"].shape, (2, 1))
                self.assertEqual(by[0].shape, (2, 3))
                self.assertEqual(bw[0].shape, (2,))
                self.assertEqual(bw[1].shape, (2,))
        ds = adapter.get_tf_dataset()
        for i, batch in enumerate(ds):
            self.assertEqual(len(batch), 3)
            bx, by, bw = batch
            self.assertIsInstance(bx, dict)
            # NOTE: the y list was converted to a tuple for tf.data
            # compatibility.
            self.assertIsInstance(by, tuple)
            self.assertIsInstance(bw, tuple)

            self.assertIsInstance(bx["x1"], tf.Tensor)
            self.assertIsInstance(bx["x2"], tf.Tensor)
            self.assertIsInstance(by[0], tf.Tensor)
            self.assertIsInstance(by[1], tf.Tensor)
            self.assertIsInstance(bw[0], tf.Tensor)
            self.assertIsInstance(bw[1], tf.Tensor)

            self.assertEqual(bx["x1"].dtype, by[0].dtype)
            self.assertEqual(bx["x1"].dtype, backend.floatx())
            if i < 2:
                self.assertEqual(tuple(bx["x1"].shape), (16, 1))
                self.assertEqual(tuple(bx["x2"].shape), (16, 2))
                self.assertEqual(tuple(by[0].shape), (16, 3))
                self.assertEqual(tuple(by[1].shape), (16, 4))
                self.assertEqual(tuple(bw[0].shape), (16,))
                self.assertEqual(tuple(bw[1].shape), (16,))
            else:
                self.assertEqual(tuple(bx["x1"].shape), (2, 1))
                self.assertEqual(tuple(by[0].shape), (2, 3))
                self.assertEqual(tuple(bw[0].shape), (2,))
                self.assertEqual(tuple(bw[1].shape), (2,))

    @parameterized.parameters([("int",), ("categorical",)])
    def test_class_weights(self, target_encoding):
        x = np.random.random((4, 2))
        if target_encoding == "int":
            y = np.array([[0], [1], [2], [3]], dtype="int32")
        else:
            y = np.array(
                [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                dtype="float32",
            )

        class_weight = {
            0: 0.1,
            1: 0.2,
            2: 0.3,
            3: 0.4,
        }
        adapter = array_data_adapter.ArrayDataAdapter(
            x,
            y=y,
            class_weight=class_weight,
            batch_size=16,
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            self.assertEqual(len(batch), 3)
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 0.2, 0.3, 0.4])

    def test_errors(self):
        # TODO
        pass

    @parameterized.parameters([("np",), ("tf",), ("backend",), ("pandas",)])
    def test_integer_inputs(self, array_type):
        x1 = self.make_array(array_type, (4, 4), dtype="float64")
        x2 = self.make_array(array_type, (4, 4), dtype="int32")
        y = self.make_array(array_type, (4, 2))

        adapter = array_data_adapter.ArrayDataAdapter(
            (x1, x2),
            y=y,
            sample_weight=None,
            batch_size=4,
            steps=None,
            shuffle=False,
        )

        (x1, x2), y = next(adapter.get_numpy_iterator())
        self.assertEqual(x1.dtype, backend.floatx())
        self.assertEqual(x2.dtype, "int32")

    def test_pandas_series(self):
        x = pandas.Series(np.ones((10,)))
        y = np.ones((10,))

        adapter = array_data_adapter.ArrayDataAdapter(
            x,
            y=y,
            sample_weight=None,
            batch_size=4,
            steps=None,
            shuffle=False,
        )

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 4)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 2)

        x, y = next(adapter.get_numpy_iterator())
        self.assertEqual(x.dtype, backend.floatx())
        self.assertIsInstance(x, np.ndarray)
        self.assertEqual(x.shape, (4, 1))

    @pytest.mark.skipif(
        backend.backend() != "tensorflow",
        reason="Only tensorflow supports raggeds",
    )
    def test_tf_ragged(self):
        x = tf.ragged.constant([[1, 2], [1, 2, 3], [1, 2], [1], []], "float64")
        y = np.ones((5,))

        adapter = array_data_adapter.ArrayDataAdapter(
            x,
            y=y,
            sample_weight=None,
            batch_size=2,
            steps=None,
            shuffle=False,
        )

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 2)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 1)

        x, y = next(adapter.get_numpy_iterator())
        self.assertEqual(x.dtype, backend.floatx())
        self.assertIsInstance(x, tf.RaggedTensor)
        self.assertEqual(x.shape, (2, None))
