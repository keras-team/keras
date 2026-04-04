import grain
import numpy as np
import tensorflow as tf
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.testing.test_utils import named_product
from keras.src.trainers.data_adapters import grain_dataset_adapter


class Range2DSource(grain.sources.RandomAccessDataSource):
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __getitem__(self, idx):
        return np.expand_dims(np.array([self.start + idx]), axis=0)

    def __len__(self):
        return self.stop - self.start


class GrainDatasetAdapterTest(testing.TestCase):
    def _get_dataset(self, dataset_type, worker_count=0, num_threads=0):
        x = np.random.normal(size=(34, 4)).astype("float32")
        y = np.random.normal(size=(34, 2)).astype("float32")

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

        if dataset_type == "map_dataset":
            self.assertEqual(adapter.num_batches, 3)
        else:
            self.assertIsNone(adapter.num_batches)
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
        # Finite MapDataset: num_batches should be known.
        dataset = grain.MapDataset.source(Range2DSource(0, 42))
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)
        self.assertEqual(adapter.num_batches, 42)

        # Batched MapDataset
        dataset = grain.MapDataset.source(Range2DSource(0, 42)).batch(10)
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)
        self.assertEqual(adapter.num_batches, 5)

        # Infinite cardinality (repeat)
        dataset = grain.MapDataset.source(Range2DSource(0, 42))
        dataset = dataset.repeat()
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)
        self.assertIsNone(adapter.num_batches)

        # IterDataset: unknown cardinality
        dataset = grain.MapDataset.source(
            Range2DSource(0, 42)
        ).to_iter_dataset()
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

    # ------------------------------------------------------------------
    # Iterator checkpoint / resume tests
    # ------------------------------------------------------------------

    def test_get_iterator_state_map_dataset(self):
        """get_iterator_state returns a dict after iterating a MapDataset."""
        dataset = grain.MapDataset.source(Range2DSource(0, 10)).batch(2)
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)

        it = adapter.get_numpy_iterator()
        iterator = iter(it)
        next(iterator)  # consume one batch

        state = adapter.get_iterator_state()
        self.assertIsNotNone(state)
        self.assertIsInstance(state, dict)

    def test_get_iterator_state_before_iteration(self):
        """get_iterator_state returns None when no iterator is live."""
        dataset = grain.MapDataset.source(Range2DSource(0, 10)).batch(2)
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)

        self.assertIsNone(adapter.get_iterator_state())

    def test_set_and_restore_iterator_state(self):
        """set_iterator_state resumes from the saved position."""
        data = list(range(20))
        dataset = grain.MapDataset.source(data).batch(4)
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)

        # Iterate through 2 batches and save state.
        it = adapter.get_numpy_iterator()
        iterator = iter(it)
        next(iterator)  # [0,1,2,3]
        next(iterator)  # [4,5,6,7]
        state = adapter.get_iterator_state()
        self.assertIsNotNone(state)

        # Read the next batch before restoring.
        batch_2 = next(iterator)  # [8,9,10,11]

        # Schedule state restoration.
        adapter.set_iterator_state(state)

        # Create a new iterator -- state should be applied on iter().
        it2 = adapter.get_numpy_iterator()
        iterator2 = iter(it2)
        restored_batch = next(iterator2)

        # The restored iterator should yield the same data as batch_2.
        np.testing.assert_array_equal(restored_batch, batch_2)

    def test_get_iterator_state_data_loader(self):
        """DataLoader does not support checkpoint -- returns None."""
        dataset = self._get_dataset("data_loader")
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)

        it = adapter.get_numpy_iterator()
        for _ in it:
            break  # consume one batch

        # DataLoader iterators don't have get_state.
        self.assertIsNone(adapter.get_iterator_state())

    def test_iterator_state_json_serializable(self):
        """The state dict must be JSON-serializable for BackupAndRestore."""
        import json

        dataset = grain.MapDataset.source(Range2DSource(0, 10)).batch(2)
        adapter = grain_dataset_adapter.GrainDatasetAdapter(dataset)

        it = adapter.get_numpy_iterator()
        iterator = iter(it)
        next(iterator)

        state = adapter.get_iterator_state()
        roundtripped = json.loads(json.dumps(state))
        self.assertEqual(state, roundtripped)
