import os

import pytest

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ['XLA_FLAGS'] = '--xla_force_host_platform_device_count=4'

import jax
from communications import AllGatherKeras
from communications import AllReduceKeras
from communications import BroadcastKeras
from communications import TensorParallelCommunicator

import keras
from keras.src import testing
from keras.src.backend.distributed import backend_resolver


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="This test suite requires a real JAX distributed backend.",
)
class TestCollectiveOps(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.world_size = jax.device_count()
        if self.world_size < 2:
            self.skipTest(
                "This test requires JAX to have at least 2 "
                "(real or virtual) devices."
            )
        self.axis_name = "i"

    def test_all_reduce(self):
        def parallel_fn(x):
            dist_backend = backend_resolver.get_distributed_backend()
            all_reduce_op = AllReduceKeras(
                world_size=self.world_size, backend=dist_backend, op="sum"
            )
            return all_reduce_op(x, axis_name=self.axis_name)

        data_to_distribute = keras.ops.ones(
            (self.world_size, 4), dtype="float32"
        )
        result = jax.pmap(parallel_fn, axis_name=self.axis_name)(
            data_to_distribute
        )
        expected_output = keras.ops.full(
            (4,), float(self.world_size), dtype="float32"
        )
        self.assertAllClose(result[0], expected_output)

    def test_all_gather(self):
        def parallel_fn(x_slice):
            dist_backend = backend_resolver.get_distributed_backend()
            all_gather_op = AllGatherKeras(
                world_size=self.world_size, backend=dist_backend, dim=0
            )
            return all_gather_op(x_slice, axis_name=self.axis_name)

        data_to_distribute = keras.ops.arange(
            self.world_size * 4, dtype="float32"
        ).reshape(self.world_size, 2, 2)
        result = jax.pmap(parallel_fn, axis_name=self.axis_name)(
            data_to_distribute
        )
        expected_output = keras.ops.arange(
            self.world_size * 4, dtype="float32"
        ).reshape(self.world_size * 2, 2)

        reshaped_result = keras.ops.reshape(result[0], (self.world_size * 2, 2))
        self.assertAllClose(reshaped_result, expected_output)

    def test_broadcast(self):
        def parallel_fn(rank_placeholder):
            rank = jax.lax.axis_index(self.axis_name)
            tensor_to_broadcast = jax.lax.cond(
                rank == 0,
                lambda: keras.ops.array([5.0, 10.0, 15.0]),
                lambda: keras.ops.zeros((3,), dtype="float32"),
            )
            dist_backend = backend_resolver.get_distributed_backend()
            broadcast_op = BroadcastKeras(
                world_size=self.world_size,
                backend=dist_backend,
                src_rank=0,
                rank=rank,
            )
            return broadcast_op(tensor_to_broadcast, axis_name=self.axis_name)

        dummy_input = keras.ops.zeros(self.world_size)
        result = jax.pmap(parallel_fn, axis_name=self.axis_name)(dummy_input)
        expected_output = keras.ops.array([5.0, 10.0, 15.0])
        self.assertAllClose(result[0], expected_output)
        self.assertAllClose(result[1], expected_output)

    def test_tensor_parallel_communicator_forward_column(self):
        def parallel_fn(x_slice):
            rank = jax.lax.axis_index(self.axis_name)
            communicator = TensorParallelCommunicator(
                world_size=self.world_size, rank=rank
            )
            return communicator.forward_column_parallel(
                x_slice, dim=0, axis_name=self.axis_name
            )

        data_to_distribute = keras.ops.arange(
            self.world_size * 4, dtype="float32"
        ).reshape(self.world_size, 2, 2)
        result = jax.pmap(parallel_fn, axis_name=self.axis_name)(
            data_to_distribute
        )
        expected_output = data_to_distribute.reshape(self.world_size * 2, 2)

        reshaped_result = keras.ops.reshape(result[0], (self.world_size * 2, 2))
        self.assertAllClose(reshaped_result, expected_output)
