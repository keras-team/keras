from unittest import mock

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.trainers.data_adapters.data_adapter_utils import (
    _add_torch_distributed_sampler,
)
from keras.src.trainers.data_adapters.data_adapter_utils import (
    class_weight_to_sample_weights,
)


class TestDataAdapterUtils(testing.TestCase):
    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_add_torch_distributed_sampler_iterable_dataset(self):
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data import IterableDataset

        class MyIterableDataset(IterableDataset):
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

            def __len__(self):
                return len(self.data)

        data = list(range(10))
        dataset = MyIterableDataset(data)
        dataloader = DataLoader(dataset, batch_size=2)

        # Test rank 0 of 2
        distributed_dataloader = _add_torch_distributed_sampler(
            dataloader, num_data_shards=2, data_shard_id=0
        )
        self.assertIsInstance(
            distributed_dataloader.dataset, torch.utils.data.IterableDataset
        )
        self.assertEqual(len(distributed_dataloader.dataset), 5)
        output = [
            x.tolist() if torch.is_tensor(x) else x
            for x in distributed_dataloader
        ]
        self.assertEqual(output, [[0, 2], [4, 6], [8]])

        # Test rank 1 of 2
        distributed_dataloader = _add_torch_distributed_sampler(
            dataloader, num_data_shards=2, data_shard_id=1
        )
        self.assertEqual(len(distributed_dataloader.dataset), 5)
        output = [
            x.tolist() if torch.is_tensor(x) else x
            for x in distributed_dataloader
        ]
        self.assertEqual(output, [[1, 3], [5, 7], [9]])

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_add_torch_distributed_sampler_iterable_dataset_no_len(self):
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data import IterableDataset

        class MyIterableDatasetNoLen(IterableDataset):
            def __init__(self, data):
                self.data = data

            def __iter__(self):
                return iter(self.data)

        data = list(range(10))
        dataset = MyIterableDatasetNoLen(data)
        dataloader = DataLoader(dataset, batch_size=2)

        distributed_dataloader = _add_torch_distributed_sampler(
            dataloader, num_data_shards=2, data_shard_id=0
        )
        self.assertFalse(hasattr(distributed_dataloader.dataset, "__len__"))
        output = [
            x.tolist() if torch.is_tensor(x) else x
            for x in distributed_dataloader
        ]
        self.assertEqual(output, [[0, 2], [4, 6], [8]])

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_add_torch_distributed_sampler_map_dataset(self):
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset

        class MyMapDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)

        data = list(range(10))
        dataset = MyMapDataset(data)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

        distributed_dataloader = _add_torch_distributed_sampler(
            dataloader, num_data_shards=2, data_shard_id=0
        )
        self.assertIsInstance(
            distributed_dataloader.sampler,
            torch.utils.data.distributed.DistributedSampler,
        )
        self.assertTrue(distributed_dataloader.sampler.shuffle)
        self.assertEqual(distributed_dataloader.batch_size, 2)

        # Test without shuffle
        dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
        distributed_dataloader = _add_torch_distributed_sampler(
            dataloader, num_data_shards=2, data_shard_id=1
        )
        self.assertFalse(distributed_dataloader.sampler.shuffle)
        self.assertEqual(distributed_dataloader.batch_size, 3)

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_add_torch_distributed_sampler_with_batch_sampler(self):
        import torch
        from torch.utils.data import BatchSampler
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset
        from torch.utils.data import SequentialSampler

        class MyMapDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, index):
                return self.data[index]

            def __len__(self):
                return len(self.data)

        data = list(range(10))
        dataset = MyMapDataset(data)
        batch_sampler = BatchSampler(
            SequentialSampler(dataset), batch_size=4, drop_last=False
        )
        dataloader = DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=2
        )

        distributed_dataloader = _add_torch_distributed_sampler(
            dataloader, num_data_shards=2, data_shard_id=0
        )
        self.assertIsInstance(
            distributed_dataloader.sampler,
            torch.utils.data.distributed.DistributedSampler,
        )
        self.assertEqual(distributed_dataloader.batch_size, 4)
        self.assertEqual(distributed_dataloader.num_workers, 2)

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_add_torch_distributed_sampler_custom_batch_sampler(self):
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset

        from keras.src.trainers.data_adapters.data_adapter_utils import (
            DistributedBatchSampler,
        )

        class MyMapDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, i):
                return self.data[i]

            def __len__(self):
                return len(self.data)

        class MyBatchSampler:
            def __init__(self, data_source):
                self.data_source = data_source

            def __iter__(self):
                for i in range(0, len(self.data_source), 2):
                    yield list(range(i, min(i + 2, len(self.data_source))))

            def __len__(self):
                return (len(self.data_source) + 1) // 2

        dataset = MyMapDataset(list(range(10)))
        batch_sampler = MyBatchSampler(dataset)
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler)

        distributed_dataloader = _add_torch_distributed_sampler(
            dataloader, num_data_shards=2, data_shard_id=0
        )
        self.assertIsInstance(
            distributed_dataloader.batch_sampler, DistributedBatchSampler
        )
        self.assertEqual(
            distributed_dataloader.batch_sampler.num_data_shards, 2
        )
        self.assertEqual(distributed_dataloader.batch_sampler.data_shard_id, 0)
        # Verify it still yields batches correctly (sharded)
        batches = list(distributed_dataloader.batch_sampler)
        self.assertEqual(batches, [[0, 1], [4, 5], [8, 9]])

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_add_torch_distributed_sampler_user_state(self):
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset

        class MyMapDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __getitem__(self, i):
                return self.data[i]

            def __len__(self):
                return len(self.data)

        def my_collate_fn(batch):
            return batch

        def my_worker_init_fn(worker_id):
            pass

        generator = torch.Generator()

        dataset = MyMapDataset(list(range(10)))
        dataloader = DataLoader(
            dataset,
            batch_size=2,
            collate_fn=my_collate_fn,
            worker_init_fn=my_worker_init_fn,
            generator=generator,
            num_workers=1,
            prefetch_factor=2,
            persistent_workers=True,
            timeout=10,
        )

        distributed_dataloader = _add_torch_distributed_sampler(
            dataloader, num_data_shards=2, data_shard_id=0
        )
        self.assertEqual(distributed_dataloader.collate_fn, my_collate_fn)
        self.assertEqual(
            distributed_dataloader.worker_init_fn, my_worker_init_fn
        )
        self.assertEqual(distributed_dataloader.generator, generator)
        self.assertEqual(distributed_dataloader.num_workers, 1)
        self.assertEqual(distributed_dataloader.prefetch_factor, 2)
        self.assertEqual(distributed_dataloader.persistent_workers, True)
        self.assertEqual(distributed_dataloader.timeout, 10)


class TestClassWeightToSampleWeights(testing.TestCase):
    @parameterized.named_parameters(
        [
            # Simple case, where y is flat
            (
                "simple_class_labels",
                np.array([0, 1, 0, 2]),
                {0: 1.0, 1: 2.0, 2: 3.0},
                np.array([1.0, 2.0, 1.0, 3.0]),
            ),
            # Testing with one-hot encoded labels,
            # so basically the argmax statement
            (
                "one_hot_encoded_labels",
                np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]]),
                {0: 1.0, 1: 2.0, 2: 3.0},
                np.array([1.0, 2.0, 1.0, 3.0]),
            ),
            # 3 is not mapped, so it's assigned the default weight (1)
            (
                "unmapped_class",
                np.array([0, 3, 0, 2]),
                {0: 1.0, 1: 2.0, 2: 3.0},
                np.array([1.0, 1.0, 1.0, 3.0]),
            ),
            (
                "multi_dimensional_input",
                np.array([[0], [1], [0], [2]]),
                {0: 1.0, 1: 2.0, 2: 3.0},
                np.array([1.0, 2.0, 1.0, 3.0]),
            ),
            (
                "all_unmapped",
                np.array([0, 1, 0, 2]),
                {},
                np.array([1.0, 1.0, 1.0, 1.0]),
            ),
        ]
    )
    def test_class_weight_to_sample_weights(self, y, class_weight, expected):
        self.assertAllClose(
            class_weight_to_sample_weights(y, class_weight), expected
        )

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_class_weight_to_sample_weights_torch_specific(self):
        import torch

        y = torch.from_numpy(np.array([0, 1, 0, 2]))
        self.assertAllClose(
            class_weight_to_sample_weights(y, {0: 1.0, 1: 2.0, 2: 3.0}),
            np.array([1.0, 2.0, 1.0, 3.0]),
        )
        y_one_hot = torch.from_numpy(
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
        )
        self.assertAllClose(
            class_weight_to_sample_weights(y_one_hot, {0: 1.0, 1: 2.0, 2: 3.0}),
            np.array([1.0, 2.0, 1.0, 3.0]),
        )

    @pytest.mark.skipif(backend.backend() != "jax", reason="jax only")
    def test_class_weight_to_sample_weights_jax_specific(self):
        import jax

        y = jax.numpy.asarray(np.array([0, 1, 0, 2]))
        self.assertAllClose(
            class_weight_to_sample_weights(y, {0: 1.0, 1: 2.0, 2: 3.0}),
            np.array([1.0, 2.0, 1.0, 3.0]),
        )
        y_one_hot = jax.numpy.asarray(
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
        )
        self.assertAllClose(
            class_weight_to_sample_weights(y_one_hot, {0: 1.0, 1: 2.0, 2: 3.0}),
            np.array([1.0, 2.0, 1.0, 3.0]),
        )

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="tensorflow only"
    )
    def test_class_weight_to_sample_weights_tf_specific(self):
        import tensorflow as tf

        y = tf.convert_to_tensor(np.array([0, 1, 0, 2]))
        self.assertAllClose(
            class_weight_to_sample_weights(y, {0: 1.0, 1: 2.0, 2: 3.0}),
            np.array([1.0, 2.0, 1.0, 3.0]),
        )
        y_one_hot = tf.convert_to_tensor(
            np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
        )
        self.assertAllClose(
            class_weight_to_sample_weights(y_one_hot, {0: 1.0, 1: 2.0, 2: 3.0}),
            np.array([1.0, 2.0, 1.0, 3.0]),
        )


class TestMultiWorkerValidation(testing.TestCase):
    @mock.patch("keras.src.distribution.distribution_lib.distribution")
    def test_unsupported_type_multi_worker(self, mock_distribution):
        from keras.src.distribution import distribution_lib
        from keras.src.trainers.data_adapters import get_data_adapter

        # Mock a multi-worker distribution
        dist = distribution_lib.DataParallel(devices=["cpu:0", "cpu:1"])
        dist._num_processes = 2
        dist._is_multi_process = True
        dist.auto_shard_dataset = True
        mock_distribution.return_value = dist

        # Raw generator is not supported for auto-sharding
        def generator():
            yield np.ones((10, 2))

        with self.assertRaisesRegex(
            ValueError,
            "When using a multi-worker distribution with auto-sharding enabled",
        ):
            get_data_adapter(generator())


class TestDataAdapterFactory(testing.TestCase):
    def test_generator_with_y_error(self):
        from keras.src.trainers.data_adapters import get_data_adapter

        def generator():
            yield np.ones((10, 2))

        # Passing y along with a generator should raise ValueError mentioning
        # "generator"
        with self.assertRaisesRegex(ValueError, "providing `x` as a generator"):
            get_data_adapter(generator(), y=np.ones((10, 1)))
