import grain
import numpy as np
import pytest
import tensorflow as tf
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src.testing.test_utils import named_product
from keras.src.trainers.data_adapters import data_adapter_test
from keras.src.trainers.data_adapters import grain_dataset_adapter


class Range2DSource(grain.sources.RandomAccessDataSource):
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __getitem__(self, idx):
        return np.expand_dims(np.array([self.start + idx]), axis=0)

    def __len__(self):
        return self.stop - self.start


class GrainDatasetAdapterTest(data_adapter_test.DataAdapterTest):
    def _get_dataset(
        self, dataset_type, num_samples=34, worker_count=0, num_threads=0
    ):
        x = np.random.normal(size=(num_samples, 4)).astype("float32")
        y = np.random.normal(size=(num_samples, 2)).astype("float32")

        class MySource(grain.sources.RandomAccessDataSource):
            def __init__(self, x, y):
                self.x = x
                self.y = y

            def __getitem__(self, idx):
                return self.x[idx], self.y[idx]

            def __len__(self):
                return len(self.x)

        if dataset_type == "map_dataset":
            dataset = grain.MapDataset.source(MySource(x, y)).batch(
                batch_size=16
            )
        elif dataset_type == "iter_dataset":
            dataset = (
                grain.MapDataset.source(MySource(x, y))
                .to_iter_dataset()
                .batch(batch_size=16)
            )
        else:
            source = MySource(x, y)
            dataset = grain.DataLoader(
                data_source=source,
                operations=[grain.transforms.Batch(batch_size=16)],
                shard_options=grain.sharding.NoSharding(),
                sampler=grain.samplers.IndexSampler(
                    num_records=len(source), num_epochs=1
                ),
                worker_count=worker_count,
                read_options=grain.ReadOptions(num_threads=num_threads),
            )
        return dataset

    @parameterized.named_parameters(
        named_product(
            dataset_type=["map_dataset", "iter_dataset", "data_loader"]
        )
    )
    def test_basic_flow(self, dataset_type):
        dataset = self._get_dataset(dataset_type)
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)

        self.assertEqual(adapter.num_batches, None)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, None)
        self.assertEqual(adapter.partial_batch_size, None)

        if backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            expected_class = tf.Tensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
            expected_class = torch.Tensor
        else:
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray

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

    @parameterized.named_parameters(
        named_product(data_type=["list", "dict", "nested_list", "nested_dict"])
    )
    def test_nested_data(self, data_type):
        if data_type not in ("list", "dict", "nested_list", "nested_dict"):
            raise ValueError(
                "data_type must be one of 'list', 'dict', 'nested_list' or "
                f"'nested_dict'. Received: {data_type}"
            )

        class NestedSource(grain.sources.RandomAccessDataSource):
            def __init__(self, data_type):
                self.x = np.random.random((40, 4)).astype("float32")
                self.y = np.random.random((40, 2)).astype("float32")
                self.data_type = data_type

            def __len__(self):
                return len(self.x)

            def __getitem__(self, idx):
                x = self.x[idx]
                y = self.y[idx]
                if self.data_type == "list":
                    return x, y
                elif self.data_type == "dict":
                    return {"x": x, "y": y}
                elif self.data_type == "nested_list":
                    return x, (x, y)
                elif self.data_type == "nested_dict":
                    return {"data": {"x": x, "y": y}}

        dataset = grain.MapDataset.source(NestedSource(data_type)).batch(
            batch_size=4
        )
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)

        if backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
            expected_class = tf.Tensor
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
            expected_class = np.ndarray
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
            expected_class = torch.Tensor
        else:
            it = adapter.get_numpy_iterator()
            expected_class = np.ndarray

        for batch in it:
            if data_type == "list":
                self.assertEqual(len(batch), 2)
                bx, by = batch
            elif data_type == "dict":
                self.assertEqual(len(batch), 2)
                bx, by = batch["x"], batch["y"]
            elif data_type == "nested_list":
                self.assertEqual(len(batch), 2)
                bx, (_, by) = batch
            elif data_type == "nested_dict":
                self.assertEqual(len(batch["data"]), 2)
                bx, by = batch["data"]["x"], batch["data"]["y"]
            self.assertIsInstance(bx, expected_class)
            self.assertIsInstance(by, expected_class)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertEqual(bx.shape, (4, 4))
            self.assertEqual(by.shape, (4, 2))

    def test_multiple_calling_on_iterators(self):
        dataset = self._get_dataset("iter_dataset")
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)

        numpy_it = adapter.get_numpy_iterator()
        jax_it = adapter.get_jax_iterator()
        tf_it = adapter.get_tf_dataset()
        torch_it = adapter.get_torch_dataloader()
        for it in (numpy_it, jax_it, tf_it, torch_it):
            for batch in it:
                self.assertEqual(len(batch), 2)
                bx, by = batch
                self.assertEqual(bx.dtype, by.dtype)

    def test_num_batches(self):
        dataset = grain.MapDataset.source(Range2DSource(0, 42))
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)
        self.assertEqual(adapter.num_batches, None)

        # Test for Infinite Cardinality
        dataset = grain.MapDataset.source(Range2DSource(0, 42))
        dataset = dataset.repeat()
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)
        self.assertIsNone(adapter.num_batches)

        # Test for Unknown Cardinality
        dataset = dataset.filter(lambda x: True)
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)
        self.assertIsNone(adapter.num_batches)

    def test_invalid_dataset_type(self):
        with self.assertRaisesRegex(
            ValueError,
            (
                r"Expected `dataset` to be a grain.MapDataset, "
                r"grain.IterDataset or grain.DataLoader. "
            ),
        ):
            grain_dataset_adapter.GrainDatasetAdapter(
                "This is not a grain.Dataset"
            )

    @pytest.mark.skipif(backend.backend() != "jax", reason="JAX only")
    def test_get_jax_iterator_with_super_batch(self):
        # Even batches: 4 batches with super_batch=2 -> 2 super-batches
        ds_even = self._get_dataset("map_dataset", num_samples=64)
        adapter_even = grain_dataset_adapter.GrainDatasetAdapter(ds_even)
        self.verify_super_batched_iterator(
            adapter_even.get_jax_iterator(super_batch=2),
            expected_super_batches=2,
            has_partial_batch=False,
        )

        # Uneven batches: 5 batches with super_batch=2 -> 2 super-batches + 1
        # partial batch list
        ds_uneven = self._get_dataset("map_dataset", num_samples=80)
        adapter_uneven = grain_dataset_adapter.GrainDatasetAdapter(ds_uneven)
        self.verify_super_batched_iterator(
            adapter_uneven.get_jax_iterator(super_batch=2),
            expected_super_batches=2,
            has_partial_batch=True,
        )
