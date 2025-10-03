import pytest

import keras
from keras.src import testing
from keras.src.backend import distributed_backend
from keras.src.distribution.tensor_parallel.communications import AllGatherKeras
from keras.src.distribution.tensor_parallel.communications import AllReduceKeras
from keras.src.distribution.tensor_parallel.communications import BroadcastKeras
from keras.src.distribution.tensor_parallel.communications import (
    TensorParallelCommunicator,
)


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="This test suite requires a real JAX distributed backend.",
)
class TestCollectiveOpsSimulated(testing.TestCase):
    """
    Tests the simulated, single-device behavior of collective communication ops.
    This test is backend-agnostic.
    """

    def setUp(self):
        super().setUp()
        device_info = distributed_backend.get_device_info()
        self.world_size = device_info.get("device_count", 1)

        if self.world_size == 0:
            self.world_size = 1

        self.axis_name = "data"

    def test_all_reduce_simulation(self):
        """Tests the simulated all-reduce operation from multiple ranks."""
        
        local_tensors = [
            keras.ops.array([float(i + 1), float(i + 2), float(i + 3)])
            for i in range(self.world_size)
        ]
        expected_output = keras.ops.zeros_like(local_tensors[0])
        for tensor in local_tensors:
            expected_output = keras.ops.add(expected_output, tensor)

        results = []
        for rank in range(self.world_size):
            all_reduce_op = AllReduceKeras(
                world_size=self.world_size, op="sum", rank=rank
            )
            result = all_reduce_op(local_tensors[rank], axis_name=self.axis_name)
            results.append(result)

        for result in results:
            self.assertAllClose(result, expected_output)

    def test_all_gather_simulation(self):
        all_gather_op = AllGatherKeras(world_size=self.world_size, dim=0)

        local_slice = keras.ops.arange(6, dtype="float32").reshape((2, 3))
        result = all_gather_op(local_slice, axis_name=self.axis_name)

        expected_output = keras.ops.concatenate(
            [local_slice] * self.world_size, axis=0
        )

        self.assertAllClose(result, expected_output)

    def test_broadcast_simulation(self):
        """Tests the simulated broadcast operation."""
        broadcast_op = BroadcastKeras(
            world_size=self.world_size, src_rank=0, rank=0
        )

        tensor_to_broadcast = keras.ops.array([5.0, 10.0, 15.0])
        result = broadcast_op(tensor_to_broadcast, axis_name=self.axis_name)

        self.assertAllClose(result, tensor_to_broadcast)

    def test_tensor_parallel_communicator_simulation(self):
        """Tests the communicator's use of simulated collective ops."""

        local_slices = [
            keras.ops.array(
                [[float(rank), float(rank + 1)], [float(rank + 2), float(rank + 3)]]
            )
            for rank in range(self.world_size)
        ]
        expected_output = keras.ops.concatenate(local_slices, axis=0)

        results = []
        for rank in range(self.world_size):
            communicator = TensorParallelCommunicator(
                world_size=self.world_size, rank=rank
            )

            result = communicator.forward_column_parallel(
                partial_outputs=local_slices, dim=0, axis_name=self.axis_name
            )
            results.append(result)

        for result in results:
            self.assertAllClose(result, expected_output)
