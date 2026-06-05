import math
import os
from unittest.mock import patch

import numpy as np
import pytest
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
        ("dataparallel", "dp", 4, 1, (4,), 4, 1),
        ("modelparallel", "mp", 8, 5, (2, 4), 2, 1),
        ("modelparallel_large_mesh", "mp", 4, 2, (8, 2), 4, 2),
    )
    @patch("torch.distributed.is_available")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("keras.src.distribution.distribution_lib.distribution_lib")
    @patch("keras.src.distribution.distribution_lib.distribution")
    def test_sharding(
        self,
        dist_type,
        world_size,
        rank,
        mesh_shape,
        expected_num_replicas,
        expected_rank,
        mock_distribution,
        mock_backend_dist_lib,
        mock_get_rank,
        mock_get_world_size,
        mock_is_available,
    ):
        mock_is_available.return_value = True
        mock_get_world_size.return_value = world_size
        mock_get_rank.return_value = rank
        mock_backend_dist_lib.num_processes.return_value = world_size
        mock_backend_dist_lib.process_id.return_value = rank

        if dist_type == "dp":
            dist = dist_lib.DataParallel(devices=["cpu:0"] * world_size)
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
        dist.auto_shard_dataset = True
        mock_distribution.return_value = dist

        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = torch.utils.data.TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        adapter = TorchDataLoaderAdapter(dataloader)
        new_dataloader = adapter.get_torch_dataloader()

        self.assertIsInstance(
            new_dataloader.sampler,
            torch.utils.data.distributed.DistributedSampler,
        )
        self.assertEqual(
            new_dataloader.sampler.num_replicas, expected_num_replicas
        )
        self.assertEqual(new_dataloader.sampler.rank, expected_rank)

    @parameterized.named_parameters(
        ("dataparallel", "dp", 4, 1, (4,), 4, 1),
        ("modelparallel", "mp", 8, 5, (2, 4), 2, 1),
    )
    @patch("torch.distributed.is_available")
    @patch("torch.distributed.get_world_size")
    @patch("torch.distributed.get_rank")
    @patch("keras.src.distribution.distribution_lib.distribution_lib")
    @patch("keras.src.distribution.distribution_lib.distribution")
    def test_sharding_iterable_dataset(
        self,
        dist_type,
        world_size,
        rank,
        mesh_shape,
        expected_num_replicas,
        expected_rank,
        mock_distribution,
        mock_backend_dist_lib,
        mock_get_rank,
        mock_get_world_size,
        mock_is_available,
    ):
        mock_is_available.return_value = True
        mock_get_world_size.return_value = world_size
        mock_get_rank.return_value = rank
        mock_backend_dist_lib.num_processes.return_value = world_size
        mock_backend_dist_lib.process_id.return_value = rank

        if dist_type == "dp":
            dist = dist_lib.DataParallel(devices=["cpu:0"] * world_size)
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
        dist.auto_shard_dataset = True
        mock_distribution.return_value = dist

        class TestIterableDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                for i in range(100):
                    yield torch.tensor([i])

            def __len__(self):
                return 100

        dataset = TestIterableDataset()
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

        adapter = TorchDataLoaderAdapter(dataloader)
        new_dataloader = adapter.get_torch_dataloader()

        self.assertTrue(
            "ShardedIterableDataset" in str(type(new_dataloader.dataset))
        )
        self.assertEqual(
            new_dataloader.dataset.num_data_shards, expected_num_replicas
        )
        self.assertEqual(new_dataloader.dataset.data_shard_id, expected_rank)

        items = list(new_dataloader)
        # 100 items total, 10 items per batch.
        # Each replica should get 100 / expected_num_replicas items.
        expected_items = 100 // expected_num_replicas
        self.assertEqual(sum(len(b) for b in items), expected_items)

        # Check that we are getting the correct items (interleaved sharding)
        first_item = next(iter(new_dataloader))[0]
        self.assertEqual(first_item.item(), expected_rank)

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="Only for torch backend",
    )
    def test_torch_dataloader_distribute_integration(self):
        # Initialize torch distributed
        if not torch.distributed.is_initialized():
            os.environ["WORLD_SIZE"] = "1"
            os.environ["RANK"] = "0"
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12356"
            torch.distributed.init_process_group(
                backend="gloo",
                rank=0,
                world_size=1,
            )

        from keras.src.distribution import distribution_lib

        dp = distribution_lib.DataParallel(devices=["cpu:0"])
        # Force multi-process to test sampler wiring even with 1 process
        dp._is_multi_process = True
        with dp.scope():
            x = torch.randn(10, 2)
            y = torch.randn(10, 1)
            dataset = torch.utils.data.TensorDataset(x, y)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

            adapter = TorchDataLoaderAdapter(dataloader)
            new_dataloader = adapter.get_torch_dataloader()

            self.assertIsInstance(
                new_dataloader.sampler,
                torch.utils.data.distributed.DistributedSampler,
            )
            self.assertEqual(new_dataloader.sampler.num_replicas, 1)
            self.assertEqual(new_dataloader.sampler.rank, 0)
