import pytest

import keras
from keras.src import backend
from keras.src import testing
from keras.src.backend import distributed_backend
from keras.src.distribution.tensor_parallel.communications import AllGatherKeras
from keras.src.distribution.tensor_parallel.communications import AllReduceKeras
from keras.src.distribution.tensor_parallel.communications import BroadcastKeras
from keras.src.distribution.tensor_parallel.communications import (
    TensorParallelCommunicator,
)


@pytest.mark.skipif(
    backend.backend() not in ("torch", "jax"),
    reason="This test is for JAX/PyTorch backends.",
)
class TestCollectiveOps(testing.TestCase):
    """
    Tests collective communication ops on a JAX distributed backend.
    """

    def setUp(self):
        super().setUp()
        device_info = distributed_backend.get_device_info()
        self.world_size = device_info.get("device_count", 1)

        if not self.world_size:
            self.world_size = 1

        self.axis_name = "data"

    def test_all_reduce(self):
        """Tests the all-reduce operation."""
        all_reduce_op = AllReduceKeras(world_size=self.world_size, op="sum")
        local_tensor = keras.ops.array([1.0, 2.0, 3.0])

        result = all_reduce_op(local_tensor, axis_name=self.axis_name)

        expected_output = keras.ops.multiply(
            local_tensor, float(self.world_size)
        )
        self.assertAllClose(result, expected_output)

    def test_all_gather(self):
        """Tests the all-gather operation."""
        all_gather_op = AllGatherKeras(world_size=self.world_size, dim=0)
        local_slice = keras.ops.arange(6, dtype="float32").reshape((2, 3))
        result = all_gather_op(local_slice, axis_name=self.axis_name)

        expected_output = keras.ops.concatenate(
            [local_slice] * self.world_size, axis=0
        )
        self.assertAllClose(result, expected_output)

    def test_broadcast(self):
        """Tests the broadcast operation."""
        broadcast_op = BroadcastKeras(
            world_size=self.world_size, src_rank=0, rank=0
        )
        tensor_to_broadcast = keras.ops.array([5.0, 10.0, 15.0])
        result = broadcast_op(tensor_to_broadcast, axis_name=self.axis_name)

        self.assertAllClose(result, tensor_to_broadcast)

    def test_tensor_parallel_communicator_forward_column_parallel(self):
        """Tests the communicator's all-gather for column-parallel forward."""
        communicator = TensorParallelCommunicator(
            world_size=self.world_size, rank=0
        )

        local_slice = keras.ops.array([[0.0, 1.0], [2.0, 3.0]], dtype="float32")

        result = communicator.forward_column_parallel(
            partial_outputs=[local_slice],
            dim=0,
            axis_name=self.axis_name,
        )

        expected_output = keras.ops.concatenate(
            [local_slice] * self.world_size, axis=0
        )
        self.assertAllClose(result, expected_output)
