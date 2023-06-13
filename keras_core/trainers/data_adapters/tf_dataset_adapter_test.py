import numpy as np
import tensorflow as tf

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

    def _test_class_weights(self, target_encoding="int"):
        x = np.random.random((4, 2))
        if target_encoding == "int":
            y = np.array([[0], [1], [2], [3]], dtype="int64")
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
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            base_ds, class_weight=class_weight
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            self.assertEqual(len(batch), 3)
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 0.2, 0.3, 0.4])

    def test_class_weights_int_targets(self):
        self._test_class_weights(target_encoding="int")

    def test_class_weights_categorical_targets(self):
        self._test_class_weights(target_encoding="categorical")

    def test_num_batches(self):
        dataset = tf.data.Dataset.range(42)
        cardinality = int(dataset.cardinality())
        self.assertEqual(cardinality, 42)
        adapter = tf_dataset_adapter.TFDatasetAdapter(dataset)
        self.assertEqual(adapter.num_batches, 42)

        # Test for Infiniate Cardinality
        dataset = tf.data.Dataset.range(42)
        dataset = dataset.repeat()
        cardinality = int(dataset.cardinality())
        self.assertEqual(cardinality, tf.data.INFINITE_CARDINALITY)
        adapter = tf_dataset_adapter.TFDatasetAdapter(dataset)
        self.assertIsNone(adapter.num_batches)

        # Test for Unknown Cardinality
        dataset = dataset.filter(lambda x: True)
        cardinality = int(dataset.cardinality())
        self.assertEqual(cardinality, tf.data.UNKNOWN_CARDINALITY)
        adapter = tf_dataset_adapter.TFDatasetAdapter(dataset)
        self.assertIsNone(adapter.num_batches)
