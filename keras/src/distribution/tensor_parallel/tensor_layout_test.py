from keras.src import ops
from keras.src import testing
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap
from keras.src.distribution.tensor_parallel.tensor_layout import (
    split_tensor_for_parallelism,
)


class LayoutTest(testing.TestCase):
    """Test suite for tensor layout actions and mappings."""

    def test_split_with_even_division(self):
        """Tests splitting a tensor that divides evenly among workers."""
        device_count = 4
        dim = 0
        tensor = ops.reshape(ops.arange(16, dtype="float32"), (8, 2))

        expected_shard_0 = ops.array([[0.0, 1.0], [2.0, 3.0]])
        expected_shard_2 = ops.array([[8.0, 9.0], [10.0, 11.0]])

        shard_0 = split_tensor_for_parallelism(
            tensor, index=0, device_count=device_count, dim=dim
        )
        shard_2 = split_tensor_for_parallelism(
            tensor, index=2, device_count=device_count, dim=dim
        )

        self.assertAllClose(shard_0, expected_shard_0)
        self.assertAllClose(shard_2, expected_shard_2)
        self.assertEqual(shard_0.shape, (2, 2))

    def test_split_with_uneven_division(self):
        """Tests splitting tensor where remainder is distributed correctly."""
        device_count = 3
        dim = 0
        tensor = ops.reshape(ops.arange(10, dtype="float32"), (10, 1))

        shard_0 = split_tensor_for_parallelism(
            tensor, index=0, device_count=device_count, dim=dim
        )
        self.assertEqual(shard_0.shape, (4, 1))
        self.assertAllClose(shard_0, ops.array([[0.0], [1.0], [2.0], [3.0]]))

        shard_1 = split_tensor_for_parallelism(
            tensor, index=1, device_count=device_count, dim=dim
        )
        self.assertEqual(shard_1.shape, (3, 1))
        self.assertAllClose(shard_1, ops.array([[4.0], [5.0], [6.0]]))

        shard_2 = split_tensor_for_parallelism(
            tensor, index=2, device_count=device_count, dim=dim
        )
        self.assertEqual(shard_2.shape, (3, 1))
        self.assertAllClose(shard_2, ops.array([[7.0], [8.0], [9.0]]))

    def test_split_and_undo_cycle_even_removed(self):
        """
        Confirms that the original tensor can be reconstructed.
        """
        device_count = 2
        dim = 0
        original_tensor = ops.reshape(ops.arange(12, dtype="float32"), (6, 2))

        shards = [
            split_tensor_for_parallelism(
                original_tensor, index=i, device_count=device_count, dim=dim
            )
            for i in range(device_count)
        ]

        reconstructed_tensor = ops.concatenate(shards, axis=dim)

        self.assertAllClose(original_tensor, reconstructed_tensor)

    def test_split_and_undo_cycle_uneven_removed(self):
        """
        Confirms that original tensor can be reconstructed with uneven split.
        """
        device_count = 4
        dim = 0
        original_tensor = ops.reshape(ops.arange(22, dtype="float32"), (11, 2))

        shards = [
            split_tensor_for_parallelism(
                original_tensor, index=i, device_count=device_count, dim=dim
            )
            for i in range(device_count)
        ]

        self.assertEqual(shards[0].shape, (3, 2))
        self.assertEqual(shards[1].shape, (3, 2))
        self.assertEqual(shards[2].shape, (3, 2))
        self.assertEqual(shards[3].shape, (2, 2))

        reconstructed_tensor = ops.concatenate(shards, axis=dim)
        self.assertAllClose(original_tensor, reconstructed_tensor)

    def test_split_last_dimension(self):
        """Tests splitting on the last dimension."""
        device_count = 3
        dim = 2
        original_tensor = ops.reshape(
            ops.arange(30, dtype="float32"), (2, 5, 3)
        )

        shards = [
            split_tensor_for_parallelism(
                original_tensor, index=i, device_count=device_count, dim=dim
            )
            for i in range(device_count)
        ]

        self.assertEqual(shards[0].shape, (2, 5, 1))
        self.assertEqual(shards[1].shape, (2, 5, 1))
        self.assertEqual(shards[2].shape, (2, 5, 1))

    def test_split_with_sharding_type_hint(self):
        """Tests using 'row' and 'column' sharding hints for 2D tensors."""
        device_count = 2
        tensor = ops.reshape(ops.arange(16, dtype="float32"), (4, 4))

        row_dim = 0
        shard_row_0 = split_tensor_for_parallelism(
            tensor, index=0, device_count=device_count, dim=row_dim
        )
        self.assertAllClose(shard_row_0, tensor[:2, :])

        col_dim = 1
        shard_col_0 = split_tensor_for_parallelism(
            tensor, index=0, device_count=device_count, dim=col_dim
        )
        self.assertAllClose(shard_col_0, tensor[:, :2])

    def test_layout_map_namedtuple_behavior(self):
        """Tests basic behavior of the LayoutMap namedtuple."""

        def rule_kernel(tensor, index):
            return split_tensor_for_parallelism(
                tensor, index=index, device_count=2, dim=0
            )

        def rule_output(tensor, index):
            return split_tensor_for_parallelism(
                tensor, index=index, device_count=2, dim=-1
            )

        state_rules = {"kernel": rule_kernel}
        output_rules = {"output": rule_output}

        layout_map = LayoutMap(
            state_rules=state_rules, output_rules=output_rules
        )

        self.assertIs(layout_map.state_rules, state_rules)
        self.assertIs(layout_map.output_rules, output_rules)

        self.assertIs(layout_map[0], state_rules)
        self.assertIs(layout_map[1], output_rules)

        with self.assertRaises(AttributeError):
            layout_map.state_rules = {}

        self.assertTrue(callable(layout_map.state_rules["kernel"]))
