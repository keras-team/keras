from unittest import mock

import jax
import numpy as np
import pytest
import tensorflow as tf
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.trainers.data_adapters import tf_dataset_adapter


class TestTFDatasetAdapter(testing.TestCase, parameterized.TestCase):
    def test_basic_flow(self):
        x = tf.random.normal((34, 4))
        y = tf.random.normal((34, 2))
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        adapter = tf_dataset_adapter.TFDatasetAdapter(base_ds)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, None)
        self.assertEqual(adapter.has_partial_batch, None)
        self.assertEqual(adapter.partial_batch_size, None)

        if backend.backend() == "numpy":
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            expected_class = tf.Tensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            expected_class = jax.Array
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
            expected_class = torch.Tensor

        for i, batch in enumerate(it):
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

    def test_invalid_dataset_type(self):
        with self.assertRaisesRegex(
            ValueError, "Expected argument `dataset` to be a tf.data.Dataset"
        ):
            invalid_data = "This is not a tf.data.Dataset"
            tf_dataset_adapter.TFDatasetAdapter(invalid_data)

    def test_class_weight_and_sample_weight_together(self):
        x = np.random.random((4, 2))
        y = np.array([[0], [1], [2], [3]], dtype="int64")
        sw = np.array([0.5, 0.5, 0.5, 0.5])
        base_ds = tf.data.Dataset.from_tensor_slices((x, y, sw)).batch(16)
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}

        with self.assertRaisesRegex(
            ValueError,
            "You cannot `class_weight` and `sample_weight` at the same time.",
        ):
            tf_dataset_adapter.TFDatasetAdapter(
                base_ds, class_weight=class_weight
            )

    def test_different_y_shapes_with_class_weight(self):
        x = np.random.random((4, 2))
        y = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype="float32",
        )
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            base_ds, class_weight=class_weight
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 0.2, 0.3, 0.4])

        y_sparse = np.array([0, 1, 2, 3], dtype="int64")
        base_ds = tf.data.Dataset.from_tensor_slices((x, y_sparse)).batch(16)
        adapter = tf_dataset_adapter.TFDatasetAdapter(
            base_ds, class_weight=class_weight
        )
        gen = adapter.get_numpy_iterator()
        for batch in gen:
            _, _, bw = batch
            self.assertAllClose(bw, [0.1, 0.2, 0.3, 0.4])

    def test_nested_y_with_class_weight(self):
        x = np.random.random((4, 2))

        # Define two target outputs, y1 and y2, for the dataset
        y1 = np.array([0, 1, 2, 3], dtype="int64")
        y2 = np.array([0, 1, 2, 3], dtype="int64")

        # Create a tf.data Dataset from the input data and two target outputs
        base_ds = tf.data.Dataset.from_tensor_slices((x, (y1, y2))).batch(16)

        # Define class weights for potential classes in the output
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}

        with self.assertRaisesRegex(
            ValueError,
            "`class_weight` is only supported for Models with a single output.",
        ):
            tf_dataset_adapter.TFDatasetAdapter(
                base_ds, class_weight=class_weight
            )

    def test_class_weights_map_fn_with_sample_weight(self):
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        class_weights_map_fn = tf_dataset_adapter.make_class_weight_map_fn(
            class_weight
        )

        x = np.array([[0.5, 0.5], [0.5, 0.5]])
        y = np.array([[1, 0], [0, 1]])
        sw = np.array([1.0, 1.0])

        with self.assertRaisesRegex(
            ValueError,
            "You cannot `class_weight` and `sample_weight` at the same time.",
        ):
            class_weights_map_fn(x, y, sw)

    def test_class_weights_map_fn_nested_y(self):
        class_weight = {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4}
        class_weights_map_fn = tf_dataset_adapter.make_class_weight_map_fn(
            class_weight
        )

        x = np.array([[0.5, 0.5]])
        y1 = np.array([1])
        y2 = np.array([0])

        with self.assertRaisesRegex(
            ValueError,
            "`class_weight` is only supported for Models with a single output.",
        ):
            class_weights_map_fn(x, (y1, y2))

    def test_distribute_dataset(self):
        x = tf.random.normal((34, 4))
        y = tf.random.normal((34, 2))
        base_ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(16)

        data_distribution = mock.Mock()
        # Mimic that there are 2 worker, and each of the worker will get batch
        # size of 8
        data_distribution.distribute_dataset = mock.MagicMock(
            return_value=base_ds.rebatch(8).shard(2, index=0)
        )

        adapter = tf_dataset_adapter.TFDatasetAdapter(
            base_ds, distribution=data_distribution
        )

        self.assertEqual(adapter.num_batches, None)
        self.assertEqual(adapter.batch_size, None)
        self.assertEqual(adapter.has_partial_batch, None)
        self.assertEqual(adapter.partial_batch_size, None)

        gen = adapter.get_numpy_iterator()
        for i, batch in enumerate(gen):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, np.ndarray)
            self.assertIsInstance(by, np.ndarray)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.dtype, "float32")
            if i < 2:
                self.assertEqual(bx.shape, (8, 4))
                self.assertEqual(by.shape, (8, 2))
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
            self.assertEqual(bx.dtype, "float32")
            if i < 2:
                self.assertEqual(tuple(bx.shape), (8, 4))
                self.assertEqual(tuple(by.shape), (8, 2))
            else:
                self.assertEqual(tuple(bx.shape), (2, 4))
                self.assertEqual(tuple(by.shape), (2, 2))

    @pytest.mark.skipif(
        not backend.SUPPORTS_SPARSE_TENSORS and backend.backend() != "numpy",
        reason="Backend does not support sparse tensors",
    )
    def test_tf_sparse_tensors(self):
        x = tf.SparseTensor(
            indices=[[0, 0], [1, 2]], values=[1.0, 2.0], dense_shape=(2, 4)
        )
        y = tf.SparseTensor(
            indices=[[0, 0], [1, 1]], values=[3.0, 4.0], dense_shape=(2, 2)
        )
        base_ds = tf.data.Dataset.from_tensors((x, y))
        adapter = tf_dataset_adapter.TFDatasetAdapter(base_ds)

        if backend.backend() == "numpy":
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            expected_class = tf.SparseTensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            expected_class = jax.experimental.sparse.BCOO

        for batch in it:
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, expected_class)
            self.assertIsInstance(by, expected_class)
            self.assertEqual(bx.shape, (2, 4))
            self.assertEqual(by.shape, (2, 2))
