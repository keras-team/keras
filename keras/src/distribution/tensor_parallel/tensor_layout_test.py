import keras
from keras.src import testing
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap
from keras.src.distribution.tensor_parallel.tensor_layout import Split


class LayoutTest(testing.TestCase):
    """Test suite for tensor layout actions and mappings."""

    def test_split_with_even_division(self):
        """Tests splitting a tensor that divides evenly among workers."""
        device_count = 4
        tensor = keras.ops.reshape(
            keras.ops.arange(16, dtype="float32"), (8, 2)
        )
        action = Split(device_count=device_count, dim=0)

        expected_shard_0 = keras.ops.array([[0.0, 1.0], [2.0, 3.0]])
        expected_shard_2 = keras.ops.array([[8.0, 9.0], [10.0, 11.0]])

        shard_0 = action(tensor, rank=0)
        shard_2 = action(tensor, rank=2)

        self.assertAllClose(shard_0, expected_shard_0)
        self.assertAllClose(shard_2, expected_shard_2)
        self.assertEqual(shard_0.shape, (2, 2))

    def test_split_with_uneven_division(self):
        """Tests splitting tensor where remainder is distributed correctly."""
        device_count = 3
        tensor = keras.ops.reshape(
            keras.ops.arange(10, dtype="float32"), (10, 1)
        )
        action = Split(device_count=device_count, dim=0)

        shard_0 = action(tensor, rank=0)
        self.assertEqual(shard_0.shape, (4, 1))
        self.assertAllClose(
            shard_0, keras.ops.array([[0.0], [1.0], [2.0], [3.0]])
        )

        shard_1 = action(tensor, rank=1)
        self.assertEqual(shard_1.shape, (3, 1))
        self.assertAllClose(shard_1, keras.ops.array([[4.0], [5.0], [6.0]]))

        shard_2 = action(tensor, rank=2)
        self.assertEqual(shard_2.shape, (3, 1))
        self.assertAllClose(shard_2, keras.ops.array([[7.0], [8.0], [9.0]]))

    def test_split_and_undo_cycle_even_removed(self):
        """
        Confirms that the original tensor can be reconstructed.
        """
        device_count = 2
        original_tensor = keras.ops.reshape(
            keras.ops.arange(12, dtype="float32"), (6, 2)
        )
        action = Split(device_count=device_count, dim=0)

        shards = [action(original_tensor, rank=i) for i in range(device_count)]

        reconstructed_tensor = keras.ops.concatenate(shards, axis=action.dim)

        self.assertAllClose(original_tensor, reconstructed_tensor)

    def test_split_and_undo_cycle_uneven_removed(self):
        """
        Confirms that original tensor can be reconstructed with uneven split.
        """
        device_count = 4
        original_tensor = keras.ops.reshape(
            keras.ops.arange(22, dtype="float32"), (11, 2)
        )
        action = Split(device_count=device_count, dim=0)

        shards = [action(original_tensor, rank=i) for i in range(device_count)]

        self.assertEqual(shards[0].shape, (3, 2))
        self.assertEqual(shards[1].shape, (3, 2))
        self.assertEqual(shards[2].shape, (3, 2))
        self.assertEqual(shards[3].shape, (2, 2))

        reconstructed_tensor = keras.ops.concatenate(shards, axis=action.dim)
        self.assertAllClose(original_tensor, reconstructed_tensor)

    def test_split_last_dimension(self):
        """Tests splitting on the last dimension using dim=-1."""
        device_count = 3
        original_tensor = keras.ops.reshape(
            keras.ops.arange(30, dtype="float32"), (2, 5, 3)
        )
        action = Split(device_count=device_count, dim=-1)

        shards = [action(original_tensor, rank=i) for i in range(device_count)]

        self.assertEqual(shards[0].shape, (2, 5, 1))
        self.assertEqual(shards[1].shape, (2, 5, 1))
        self.assertEqual(shards[2].shape, (2, 5, 1))

    def test_split_with_sharding_type_hint(self):
        """Tests using 'row' and 'column' sharding hints for 2D tensors."""
        device_count = 2
        tensor = keras.ops.reshape(
            keras.ops.arange(16, dtype="float32"), (4, 4)
        )

        action_row = Split(
            device_count=device_count, dim=-1, sharding_type="row"
        )
        shard_row_0 = action_row(tensor, rank=0)
        self.assertAllClose(shard_row_0, tensor[:2, :])
        self.assertEqual(action_row.dim, 0)

        action_col = Split(
            device_count=device_count, dim=-1, sharding_type="column"
        )
        shard_col_0 = action_col(tensor, rank=0)
        self.assertAllClose(shard_col_0, tensor[:, :2])
        self.assertEqual(action_col.dim, 1)

    def test_layout_map_namedtuple_behavior(self):
        """Tests basic behavior of the LayoutMap namedtuple."""
        state_rules = {"kernel": Split(device_count=2, dim=0)}
        output_rules = {"output": Split(device_count=2, dim=-1)}

        layout_map = LayoutMap(
            state_rules=state_rules, output_rules=output_rules
        )

        self.assertIs(layout_map.state_rules, state_rules)
        self.assertIs(layout_map.output_rules, output_rules)

        self.assertIs(layout_map[0], state_rules)
        self.assertIs(layout_map[1], output_rules)

        with self.assertRaises(AttributeError):
            layout_map.state_rules = {}

        self.assertIsInstance(layout_map.state_rules["kernel"], Split)
