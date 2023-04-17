import numpy as np
import tensorflow as tf

from keras_core import backend
from keras_core import testing
from keras_core.trainers.data_adapters import tf_dataset_adapter


class TestTFDatasetAdapter(testing.TestCase):
    def test_basic_flow(self):
        x = tf.random.normal((34, 4))
        y = tf.random.normal((34, 2))
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        adapter = tf_dataset_adapter.TFDatasetAdapter(base_ds)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, None)
        self.assertEqual(adapter.has_partial_batch, None)
        self.assertEqual(adapter.partial_batch_size, None)

        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertTrue(isinstance(bx, np.ndarray))
            self.assertTrue(isinstance(by, np.ndarray))
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, "float32")
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
            self.assertTrue(isinstance(bx, tf.Tensor))
            self.assertTrue(isinstance(by, tf.Tensor))
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, "float32")
            if i < 2:
                self.assertEqual(tuple(bx.shape), (16, 4))
                self.assertEqual(tuple(by.shape), (16, 2))
            else:
                self.assertEqual(tuple(bx.shape), (2, 4))
                self.assertEqual(tuple(by.shape), (2, 2))
