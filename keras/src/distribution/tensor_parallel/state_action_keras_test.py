import numpy as np

import keras
from keras.src.distribution.tensor_parallel.state_action_keras import (
    GatherKeras,
)
from keras.src.distribution.tensor_parallel.state_action_keras import SplitKeras
from keras.src.distribution.tensor_parallel.state_action_keras import SumKeras


class TestSplitKeras:
    def test_split_call_even(self):
        """Tests SplitKeras.__call__ with an evenly divisible tensor."""
        action = SplitKeras(world_size=4, dim=1)
        tensor = keras.ops.reshape(
            keras.ops.arange(16, dtype="float32"), (2, 8)
        )

        shard = action(tensor, rank=2)
        expected_shard = np.array([[4.0, 5.0], [12.0, 13.0]])
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(shard), expected_shard
        )
        assert shard.shape == (2, 2)

    def test_split_call_uneven(self):
        """Tests SplitKeras.__call__ with a remainder."""
        action = SplitKeras(world_size=3, dim=0)
        tensor = keras.ops.reshape(
            keras.ops.arange(20, dtype="float32"), (10, 2)
        )

        shard_0 = action(tensor, rank=0)
        assert shard_0.shape == (4, 2)

        shard_1 = action(tensor, rank=1)
        assert shard_1.shape == (3, 2)


class TestGatherKeras:
    def test_gather_call(self):
        """Tests that GatherKeras.__call__ is an identity operation."""
        action = GatherKeras(world_size=4, dim=0)
        tensor = keras.ops.array([1, 2, 3])
        result = action(tensor, rank=0)
        assert result is tensor


class TestSumKeras:
    def test_sum_call(self):
        """Tests that SumKeras.__call__ is an identity operation."""
        action = SumKeras(world_size=4)
        tensor = keras.ops.array([1, 2, 3])
        result = action(tensor, rank=0)
        assert result is tensor

    def test_sum_undo(self):
        """Tests that SumKeras.undo correctly sums the tensors."""
        action = SumKeras(world_size=3)
        tensors = [
            keras.ops.array([1.0, 2.0]),
            keras.ops.array([3.0, 4.0]),
            keras.ops.array([5.0, 6.0]),
        ]

        result = action.undo(tensors)
        expected = np.array([9.0, 12.0])
        np.testing.assert_array_equal(
            keras.ops.convert_to_numpy(result), expected
        )
