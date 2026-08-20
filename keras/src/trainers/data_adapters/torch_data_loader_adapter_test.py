import math

import numpy as np
import tensorflow as tf
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.distribution import distribution_lib as dist_lib
from keras.src.testing.test_utils import named_product
from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
    TorchDataLoaderAdapter,
)


class TestIterableDataset(torch.utils.data.IterableDataset):
    def __iter__(self):
        for i in range(100):
            yield torch.tensor([float(i)]), torch.tensor([float(i)])


class TestTorchDataLoaderAdapter(testing.TestCase):
    def test_basic_dataloader(self):
        x = torch.normal(2, 3, size=(34, 4))
        y = torch.normal(1, 3, size=(34, 2))
        ds = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=16)
        adapter = TorchDataLoaderAdapter(dataloader)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 2)

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

    def test_dict_batch_preserves_structure(self):
        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 2

            def __getitem__(self, idx):
                return {
                    "x": torch.tensor([idx, idx + 1], dtype=torch.float32),
                    "y": torch.tensor(idx, dtype=torch.float32),
                }

        dataloader = torch.utils.data.DataLoader(DictDataset(), batch_size=2)
        adapter = TorchDataLoaderAdapter(dataloader)

        batch = next(adapter.get_numpy_iterator())
        self.assertIsInstance(batch, dict)
        self.assertEqual(set(batch), {"x", "y"})
        self.assertIsInstance(batch["x"], np.ndarray)
        self.assertIsInstance(batch["y"], np.ndarray)
        self.assertEqual(batch["x"].shape, (2, 2))
        self.assertEqual(batch["y"].shape, (2,))

        if backend.backend() == "tensorflow":
            ds = TorchDataLoaderAdapter(dataloader).get_tf_dataset()
            self.assertIsInstance(ds.element_spec, dict)
            self.assertEqual(set(ds.element_spec), {"x", "y"})
            self.assertIsInstance(ds.element_spec["x"], tf.TensorSpec)
            self.assertIsInstance(ds.element_spec["y"], tf.TensorSpec)
            self.assertEqual(ds.element_spec["x"].shape, (None, 2))
            self.assertEqual(ds.element_spec["y"].shape, (None,))

    def test_single_tensor_batch_preserves_structure(self):
        class TensorOnlyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 4

            def __getitem__(self, idx):
                return torch.tensor(
                    [idx, idx + 1, idx + 2], dtype=torch.float32
                )

        dataloader = torch.utils.data.DataLoader(
            TensorOnlyDataset(), batch_size=2
        )
        adapter = TorchDataLoaderAdapter(dataloader)

        batch = next(adapter.get_numpy_iterator())
        self.assertIsInstance(batch, np.ndarray)
        self.assertEqual(batch.shape, (2, 3))

        if backend.backend() == "tensorflow":
            ds = TorchDataLoaderAdapter(dataloader).get_tf_dataset()
            self.assertIsInstance(ds.element_spec, tf.TensorSpec)
            self.assertEqual(ds.element_spec.shape, (None, 3))

    @parameterized.named_parameters(
        named_product(batch_size=[None, 3], implements_len=[True, False])
    )
    def test_dataloader_iterable_dataset(self, batch_size, implements_len):
        class TestIterableDataset(torch.utils.data.IterableDataset):
            def __init__(self):
                self.x = torch.normal(2, 3, size=(16, 4))
                self.y = torch.normal(1, 3, size=(16, 2))

            def __iter__(self):
                for _ in range(10):
                    yield (self.x, self.y)

        class TestIterableDatasetWithLen(TestIterableDataset):
            def __len__(self):
                return 10

        ds = (
            TestIterableDatasetWithLen()
            if implements_len
            else TestIterableDataset()
        )
        dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
        adapter = TorchDataLoaderAdapter(dataloader)

        if implements_len and batch_size:
            self.assertEqual(adapter.num_batches, math.ceil(10 / batch_size))
            self.assertEqual(adapter.batch_size, batch_size)
            self.assertEqual(adapter.has_partial_batch, True)
            self.assertEqual(adapter.partial_batch_size, 10 % batch_size)
        elif implements_len:
            self.assertEqual(adapter.num_batches, 10)
            self.assertEqual(adapter.batch_size, None)
            self.assertEqual(adapter.has_partial_batch, None)
            self.assertEqual(adapter.partial_batch_size, None)
        else:
            self.assertIsNone(adapter.num_batches)
            self.assertEqual(adapter.batch_size, batch_size)
            self.assertIsNone(adapter.has_partial_batch)
            self.assertIsNone(adapter.partial_batch_size)

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

        batch_count = 0
        for i, batch in enumerate(it):
            batch_count += 1
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertIsInstance(bx, expected_class)
            self.assertIsInstance(by, expected_class)
            self.assertEqual(bx.dtype, by.dtype)
            self.assertContainsExactSubsequence(str(bx.dtype), "float32")
            if batch_size:
                if i < 3:
                    self.assertEqual(bx.shape, (batch_size, 16, 4))
                    self.assertEqual(by.shape, (batch_size, 16, 2))
                else:
                    self.assertEqual(bx.shape, (10 % batch_size, 16, 4))
                    self.assertEqual(by.shape, (10 % batch_size, 16, 2))
            else:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))

        if batch_size:
            self.assertEqual(batch_count, math.ceil(10 / batch_size))
        else:
            self.assertEqual(batch_count, 10)

    def test_with_different_shapes(self):
        x = (
            [np.ones([4], "float32")] * 16
            + [np.ones([5], "float32")] * 16
            + [np.ones([6], "float32")] * 2
        )
        y = np.ones((34, 2), "float32")
        ds = torch.utils.data.StackDataset(x, y)
        dataloader = torch.utils.data.DataLoader(ds, batch_size=16)
        adapter = TorchDataLoaderAdapter(dataloader)

        self.assertEqual(adapter.num_batches, 3)
        self.assertEqual(adapter.batch_size, 16)
        self.assertEqual(adapter.has_partial_batch, True)
        self.assertEqual(adapter.partial_batch_size, 2)

        if backend.backend() == "tensorflow":
            it = adapter.get_tf_dataset()
        elif backend.backend() == "jax":
            it = adapter.get_jax_iterator()
        elif backend.backend() == "torch":
            it = adapter.get_torch_dataloader()
        else:
            it = adapter.get_numpy_iterator()

        for i, batch in enumerate(it):
            self.assertEqual(len(batch), 2)
            bx, by = batch
            self.assertEqual(bx.dtype, by.dtype)
            self.assertContainsExactSubsequence(str(bx.dtype), "float32")
            if i == 0:
                self.assertEqual(bx.shape, (16, 4))
                self.assertEqual(by.shape, (16, 2))
            elif i == 1:
                self.assertEqual(bx.shape, (16, 5))
                self.assertEqual(by.shape, (16, 2))
            else:
                self.assertEqual(bx.shape, (2, 6))
                self.assertEqual(by.shape, (2, 2))

    @parameterized.named_parameters(
        named_product(
            [
                {
                    "testcase_name": "dataparallel",
                    "dist_type": "dp",
                    "num_processes": 4,
                    "process_id": 1,
                    "mesh_shape": (4,),
                },
                {
                    "testcase_name": "modelparallel",
                    "dist_type": "mp",
                    "num_processes": 8,
                    "process_id": 5,
                    "mesh_shape": (2, 4),
                },
                {
                    "testcase_name": "modelparallel_large_mesh",
                    "dist_type": "mp",
                    "num_processes": 4,
                    "process_id": 2,
                    "mesh_shape": (2, 2),
                },
            ],
            [
                {
                    "testcase_name": "map_no_shuffle",
                    "dataset_type": "map",
                    "shuffle": False,
                },
                {
                    "testcase_name": "map_shuffle",
                    "dataset_type": "map",
                    "shuffle": True,
                },
                {
                    "testcase_name": "iterable",
                    "dataset_type": "iterable",
                    "shuffle": False,
                },
                {
                    "testcase_name": "iterable_no_len",
                    "dataset_type": "iterable_no_len",
                    "shuffle": False,
                },
            ],
        )
    )
    def test_sharding(
        self,
        dist_type,
        num_processes,
        process_id,
        mesh_shape,
        dataset_type,
        shuffle,
    ):
        if dist_type == "dp":
            dist = dist_lib.DataParallel(devices=["cpu:0"] * num_processes)
        else:
            device_mesh = dist_lib.DeviceMesh(
                shape=mesh_shape,
                axis_names=("data", "model"),
                devices=["cpu:0"] * np.prod(mesh_shape),
            )
            dist = dist_lib.ModelParallel(
                device_mesh=device_mesh,
                layout_map=dist_lib.LayoutMap(device_mesh),
                batch_dim_name="data",
            )
        dist._num_processes = num_processes
        dist._process_id = process_id
        dist._is_multi_process = num_processes > 1
        dist.auto_shard_dataset = True

        expected_num_replicas = dist.num_model_replicas
        expected_shard_id = dist.data_shard_id

        if dataset_type == "map":
            x = torch.arange(100).float().reshape((100, 1))
            dataset = torch.utils.data.TensorDataset(x, x)
        elif dataset_type == "iterable":
            dataset = type(
                "DS", (TestIterableDataset,), {"__len__": lambda s: 100}
            )()
        else:
            dataset = TestIterableDataset()

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=10, shuffle=shuffle
        )

        with dist.scope():
            adapter = TorchDataLoaderAdapter(dataloader)

            it_methods = ["get_numpy_iterator"]
            backend_it_method = {
                "tensorflow": "get_tf_dataset",
                "jax": "get_jax_iterator",
                "torch": "get_torch_dataloader",
            }.get(backend.backend())
            if backend_it_method:
                it_methods.append(backend_it_method)

        if dataset_type == "map":
            self.assertEqual(
                adapter._dataloader.sampler.num_replicas, expected_num_replicas
            )
            self.assertEqual(
                adapter._dataloader.sampler.rank, expected_shard_id
            )
        else:
            self.assertEqual(
                adapter._dataloader.dataset.num_data_shards,
                expected_num_replicas,
            )
            self.assertEqual(
                adapter._dataloader.dataset.data_shard_id, expected_shard_id
            )

        def get_order(it_fn):
            order = []
            for batch in it_fn():
                by = batch[1]
                by = backend.convert_to_numpy(by)
                order.extend(by[:, 0].tolist())
            return order

        for it_method in it_methods:
            it_fn = getattr(adapter, it_method)
            if not shuffle:
                batches = list(it_fn())
                # For map dataset, DistributedSampler behavior might pad.
                # But for our simple formula, it should match.
                expected_num_batches = (
                    10 - expected_shard_id + expected_num_replicas - 1
                ) // expected_num_replicas
                self.assertEqual(len(batches), expected_num_batches)

                for i, batch in enumerate(batches):
                    bx, by = batch
                    bx = backend.convert_to_numpy(bx)
                    by = backend.convert_to_numpy(by)
                    # DistributedSampler and ShardedIterableDataset both use
                    # interleaved sharding.
                    # Each replica gets samples: [rank, rank + num_replicas,
                    # rank + 2*num_replicas, ...]
                    # So batch i on this replica starts at index:
                    # (rank + i * num_replicas * batch_size)
                    expected_first_sample_value = (
                        expected_shard_id + i * expected_num_replicas * 10
                    )
                    self.assertAllClose(
                        bx[0, 0],
                        expected_first_sample_value,
                    )
                    self.assertAllClose(
                        by[0, 0],
                        expected_first_sample_value,
                    )
            else:
                # Same epoch should have same shuffle
                # DistributedSampler uses its own internal epoch
                adapter._dataloader.sampler.set_epoch(1)
                order1 = get_order(it_fn)
                adapter._dataloader.sampler.set_epoch(1)
                order2 = get_order(it_fn)
                self.assertAllClose(order1, order2)

                # Different epochs should have different shuffle
                adapter._dataloader.sampler.set_epoch(2)
                order3 = get_order(it_fn)
                self.assertNotAllClose(order1, order3)
