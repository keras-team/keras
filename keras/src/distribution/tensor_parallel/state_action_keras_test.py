import keras
from keras.src import testing
from keras.src.distribution.tensor_parallel.state_action_keras import (
    GatherKeras,
)
from keras.src.distribution.tensor_parallel.state_action_keras import SplitKeras
from keras.src.distribution.tensor_parallel.state_action_keras import SumKeras


class TestStateActions(testing.TestCase):
    """Test suite for tensor distribution state actions."""

    def test_split_keras_even_split(self):
        """Tests SplitKeras with a tensor that divides evenly."""
        world_size = 4
        tensor = keras.ops.reshape(
            keras.ops.arange(16, dtype="float32"), (4, 4)
        )

        action_row = SplitKeras(world_size=world_size, dim=0)
        shards_row = [action_row(tensor, rank=i) for i in range(world_size)]

        self.assertEqual(shards_row[0].shape, (1, 4))
        self.assertAllClose(shards_row[0], tensor[0:1, :])
        self.assertAllClose(shards_row[3], tensor[3:4, :])

        reconstructed_row = action_row.undo(shards_row)
        self.assertAllClose(reconstructed_row, tensor)

        action_col = SplitKeras(world_size=world_size, dim=1)
        shards_col = [action_col(tensor, rank=i) for i in range(world_size)]

        self.assertEqual(shards_col[0].shape, (4, 1))
        self.assertAllClose(shards_col[0], tensor[:, 0:1])
        self.assertAllClose(shards_col[2], tensor[:, 2:3])

        reconstructed_col = action_col.undo(shards_col)
        self.assertAllClose(reconstructed_col, tensor)

    def test_split_keras_uneven_split(self):
        """Tests SplitKeras with a tensor that does not divide evenly."""
        world_size = 3
        tensor = keras.ops.reshape(
            keras.ops.arange(40, dtype="float32"), (4, 10)
        )

        action = SplitKeras(world_size=world_size, dim=1)
        shards = [action(tensor, rank=i) for i in range(world_size)]

        self.assertEqual(shards[0].shape, (4, 4))
        self.assertEqual(shards[1].shape, (4, 3))
        self.assertEqual(shards[2].shape, (4, 3))

        self.assertAllClose(shards[0], tensor[:, 0:4])
        self.assertAllClose(shards[1], tensor[:, 4:7])
        self.assertAllClose(shards[2], tensor[:, 7:10])

        reconstructed = action.undo(shards)
        self.assertAllClose(reconstructed, tensor)

    def test_split_keras_sharding_type_inference(self):
        """Tests that `sharding_type` correctly infers the split dimension."""
        action_row = SplitKeras(world_size=2, dim=-1, sharding_type="row")
        self.assertEqual(action_row.dim, 0)

        action_col = SplitKeras(world_size=2, dim=-1, sharding_type="column")
        self.assertEqual(action_col.dim, 1)

    def test_gather_keras(self):
        """Tests the GatherKeras action."""
        world_size = 4
        action = GatherKeras(world_size=world_size, dim=0)
        tensor = keras.ops.array([[1, 2], [3, 4]], dtype="float32")

        processed_tensor = action(tensor, rank=0)
        self.assertAllClose(processed_tensor, tensor)

        tensors_to_gather = [
            keras.ops.ones((2, 2)),
            keras.ops.zeros((2, 2)),
            keras.ops.ones((2, 2)),
        ]
        reconstructed = action.undo(tensors_to_gather)
        expected = keras.ops.concatenate(tensors_to_gather, axis=0)
        self.assertAllClose(reconstructed, expected)

    def test_sum_keras(self):
        """Tests the SumKeras action."""
        world_size = 2
        action = SumKeras(world_size=world_size)
        tensor = keras.ops.array([[1, 2], [3, 4]], dtype="float32")

        processed_tensor = action(tensor, rank=0)
        self.assertAllClose(processed_tensor, tensor)

        tensors_to_sum = [
            keras.ops.full((2, 3), 5.0),
            keras.ops.full((2, 3), 10.0),
        ]
        reconstructed = action.undo(tensors_to_sum)
        expected = keras.ops.full((2, 3), 15.0)
        self.assertAllClose(reconstructed, expected)
