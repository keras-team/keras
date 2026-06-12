from unittest import mock

import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.trainers.data_adapters.data_adapter_utils import (
    DistributedBatchSampler,
)
from keras.src.trainers.data_adapters.data_adapter_utils import (
    class_weight_to_sample_weights,
)


class TestDistributedBatchSampler(testing.TestCase):
    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_basic_sharding(self):
        # 10 batches total
        batch_sampler = list(range(10))

        # Test with 2 replicas
        # Rank 0 should get [0, 2, 4, 6, 8]
        sampler0 = DistributedBatchSampler(
            batch_sampler, num_data_shards=2, data_shard_id=0
        )
        self.assertEqual(list(sampler0), [0, 2, 4, 6, 8])
        self.assertEqual(len(sampler0), 5)

        # Rank 1 should get [1, 3, 5, 7, 9]
        sampler1 = DistributedBatchSampler(
            batch_sampler, num_data_shards=2, data_shard_id=1
        )
        self.assertEqual(list(sampler1), [1, 3, 5, 7, 9])
        self.assertEqual(len(sampler1), 5)

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_uneven_sharding(self):
        # 11 batches total
        batch_sampler = list(range(11))

        # Test with 3 replicas
        # Rank 0 should get [0, 3, 6, 9]
        sampler0 = DistributedBatchSampler(
            batch_sampler, num_data_shards=3, data_shard_id=0, drop_last=False
        )
        self.assertEqual(list(sampler0), [0, 3, 6, 9])
        self.assertEqual(len(sampler0), 4)

        # Rank 1 should get [1, 4, 7, 10]
        sampler1 = DistributedBatchSampler(
            batch_sampler, num_data_shards=3, data_shard_id=1, drop_last=False
        )
        self.assertEqual(list(sampler1), [1, 4, 7, 10])
        self.assertEqual(len(sampler1), 4)

        # Rank 2 should get [2, 5, 8]
        sampler2 = DistributedBatchSampler(
            batch_sampler, num_data_shards=3, data_shard_id=2, drop_last=False
        )
        self.assertEqual(list(sampler2), [2, 5, 8])
        self.assertEqual(len(sampler2), 3)

    @pytest.mark.skipif(backend.backend() != "torch", reason="torch only")
    def test_drop_last(self):
        # 11 batches total
        batch_sampler = list(range(11))

        # Test with 3 replicas and drop_last=True
        # num_batches = 11 // 3 = 3 batches per replica
        # Total used = 3 * 3 = 9 batches
        # Rank 0: [0, 3, 6]
        sampler0 = DistributedBatchSampler(
            batch_sampler, num_data_shards=3, data_shard_id=0, drop_last=True
        )
        self.assertEqual(list(sampler0), [0, 3, 6])
        self.assertEqual(len(sampler0), 3)

        # Rank 1: [1, 4, 7]
        sampler1 = DistributedBatchSampler(
            batch_sampler, num_data_shards=3, data_shard_id=1, drop_last=True
        )
        self.assertEqual(list(sampler1), [1, 4, 7])
        self.assertEqual(len(sampler1), 3)

        # Rank 2: [2, 5, 8]
        sampler2 = DistributedBatchSampler(
            batch_sampler, num_data_shards=3, data_shard_id=2, drop_last=True
        )
        self.assertEqual(list(sampler2), [2, 5, 8])
        self.assertEqual(len(sampler2), 3)


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
