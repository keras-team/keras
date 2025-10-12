import pytest

import keras
from keras.src import backend
from keras.src import testing
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutAction
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap
from keras.src.distribution.tensor_parallel.tensor_layout import Split


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="Test requires JAX backend",
)
class LayoutTest(testing.TestCase):
    """Test suite for tensor layout actions and mappings."""

    def test_layout_action_abstract_methods_raise_error(self):
        """Ensures the base class methods raise NotImplementedError."""
        action = LayoutAction()
        with self.assertRaises(NotImplementedError):
            action(tensor=None, rank=0)
        with self.assertRaises(NotImplementedError):
            action.undo(tensors=None)

    # --- Split Action Tests ---

    def test_split_with_even_division(self):
        """Tests splitting a tensor that divides evenly among workers."""
        world_size = 4
        # Create a tensor of shape (8, 2)
        tensor = keras.ops.reshape(
            keras.ops.arange(16, dtype="float32"), (8, 2)
        )
        action = Split(world_size=world_size, dim=0)

        # Expected shard for rank 0 has shape (2, 2)
        expected_shard_0 = keras.ops.array([[0.0, 1.0], [2.0, 3.0]])
        # Expected shard for rank 2 has shape (2, 2)
        expected_shard_2 = keras.ops.array([[8.0, 9.0], [10.0, 11.0]])

        shard_0 = action(tensor, rank=0)
        shard_2 = action(tensor, rank=2)

        self.assertAllClose(shard_0, expected_shard_0)
        self.assertAllClose(shard_2, expected_shard_2)
        self.assertEqual(shard_0.shape, (2, 2))

    def test_split_with_uneven_division(self):
        """Tests splitting where the remainder is distributed correctly."""
        world_size = 3
        # Create a tensor of shape (10, 1). 10 / 3 = 3 with remainder 1.
        tensor = keras.ops.reshape(
            keras.ops.arange(10, dtype="float32"), (10, 1)
        )
        action = Split(world_size=world_size, dim=0)

        # Rank 0 should get 3 + 1 = 4 rows.
        shard_0 = action(tensor, rank=0)
        self.assertEqual(shard_0.shape, (4, 1))
        self.assertAllClose(
            shard_0, keras.ops.array([[0.0], [1.0], [2.0], [3.0]])
        )

        # Rank 1 should get 3 rows.
        shard_1 = action(tensor, rank=1)
        self.assertEqual(shard_1.shape, (3, 1))
        self.assertAllClose(shard_1, keras.ops.array([[4.0], [5.0], [6.0]]))

        # Rank 2 should get 3 rows.
        shard_2 = action(tensor, rank=2)
        self.assertEqual(shard_2.shape, (3, 1))
        self.assertAllClose(shard_2, keras.ops.array([[7.0], [8.0], [9.0]]))

    def test_split_and_undo_cycle_even(self):
        """Tests splitting and reconstructing evenly divisible tensor."""
        world_size = 2
        original_tensor = keras.ops.reshape(
            keras.ops.arange(12, dtype="float32"), (6, 2)
        )
        action = Split(world_size=world_size, dim=0)

        # Create all shards
        shards = [action(original_tensor, rank=i) for i in range(world_size)]

        # Reconstruct the tensor
        reconstructed_tensor = action.undo(shards)

        self.assertAllClose(original_tensor, reconstructed_tensor)

    def test_split_and_undo_cycle_uneven(self):
        """Tests the full cycle for an unevenly distributed tensor."""
        world_size = 4
        # 11 / 4 = 2 with a remainder of 3.
        original_tensor = keras.ops.reshape(
            keras.ops.arange(22, dtype="float32"), (11, 2)
        )
        action = Split(world_size=world_size, dim=0)

        shards = [action(original_tensor, rank=i) for i in range(world_size)]

        # Verify shard shapes: first 3 get 2+1=3 rows, last one gets 2.
        self.assertEqual(shards[0].shape, (3, 2))
        self.assertEqual(shards[1].shape, (3, 2))
        self.assertEqual(shards[2].shape, (3, 2))
        self.assertEqual(shards[3].shape, (2, 2))

        reconstructed_tensor = action.undo(shards)
        self.assertAllClose(original_tensor, reconstructed_tensor)

    def test_split_last_dimension_with_undo(self):
        """Tests splitting on the last dimension using dim=-1."""
        world_size = 3
        original_tensor = keras.ops.reshape(
            keras.ops.arange(30, dtype="float32"), (2, 5, 3)
        )
        action = Split(world_size=world_size, dim=-1)

        shards = [action(original_tensor, rank=i) for i in range(world_size)]

        # Each shard should have the last dimension split.
        self.assertEqual(shards[0].shape, (2, 5, 1))
        self.assertEqual(shards[1].shape, (2, 5, 1))
        self.assertEqual(shards[2].shape, (2, 5, 1))

        reconstructed_tensor = action.undo(shards)
        self.assertAllClose(original_tensor, reconstructed_tensor)

    def test_split_with_sharding_type_hint(self):
        """Tests using 'row' and 'column' sharding hints for 2D tensors."""
        world_size = 2
        tensor = keras.ops.reshape(
            keras.ops.arange(16, dtype="float32"), (4, 4)
        )

        # Row sharding should split along axis 0
        action_row = Split(world_size=world_size, dim=-1, sharding_type="row")
        shard_row_0 = action_row(tensor, rank=0)
        self.assertAllClose(shard_row_0, tensor[:2, :])
        self.assertEqual(action_row.dim, 0)

        # Column sharding should split along axis 1
        action_col = Split(
            world_size=world_size, dim=-1, sharding_type="column"
        )
        shard_col_0 = action_col(tensor, rank=0)
        self.assertAllClose(shard_col_0, tensor[:, :2])
        self.assertEqual(action_col.dim, 1)

    # --- LayoutMap Tests ---

    def test_layout_map_initialization_and_methods(self):
        """Tests basic initialization and method behavior of LayoutMap class."""
        state_rules = {"kernel": Split(world_size=2, dim=0)}
        output_rules = {"output": Split(world_size=2, dim=-1)}

        layout_map = LayoutMap(state_rules, output_rules)

        self.assertIs(layout_map.state_rules["kernel"], state_rules["kernel"])
        self.assertIs(layout_map.output_rules["output"], output_rules["output"])

        self.assertIs(
            layout_map.create_collective_ops(devices=["cpu:0"]), layout_map
        )
