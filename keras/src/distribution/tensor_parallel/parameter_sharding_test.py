import functools
import os

import numpy as np
import pytest

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

from keras.src import backend
from keras.src import distribution
from keras.src import ops
from keras.src.distribution.tensor_parallel.parameter_sharding import (
    ShardedWeight,
)
from keras.src.distribution.tensor_parallel.parameter_sharding import (
    make_parameter_sharded_model,
)
from keras.src.distribution.tensor_parallel.tensor_layout import LayoutMap
from keras.src.distribution.tensor_parallel.tensor_layout import (
    split_tensor_for_parallelism,
)
from keras.src.layers import Activation
from keras.src.layers import Dense
from keras.src.layers import Input
from keras.src.models import Model
from keras.src.testing import TestCase


def _create_simple_mlp():
    inputs = Input(shape=(16,), name="input")
    x = Dense(32, use_bias=True, name="up_proj")(inputs)
    x = Activation("relu")(x)
    outputs = Dense(8, use_bias=False, name="down_proj")(x)
    return Model(inputs=inputs, outputs=outputs, name="simple_mlp")


@pytest.mark.skipif(
    backend.backend() not in ("jax", "torch"),
    reason="Parameter sharding tests are supported for JAX and PyTorch.",
)
class UnifiedParameterShardingTest(TestCase):
    def setUp(self):
        super().setUp()

        self.all_devices = distribution.list_devices()
        self.device_count = (
            len(self.all_devices) if len(self.all_devices) > 0 else 1
        )
        self.devices = self.all_devices if self.all_devices else ["cpu"]

        self.original_model = _create_simple_mlp()
        self.original_model.build(input_shape=(None, 16))

        self.tp_config = LayoutMap(
            state_rules={
                "up_proj/kernel": functools.partial(
                    split_tensor_for_parallelism,
                    device_count=self.device_count,
                    dim=1,
                ),
                "down_proj/kernel": functools.partial(
                    split_tensor_for_parallelism,
                    device_count=self.device_count,
                    dim=0,
                ),
            },
            output_rules={"down_proj": "sum"},
        )
        self.input_data = np.random.rand(4, 16).astype("float32")

    def test_model_sharding_structure(self):
        """Verifies sharding logic is applied correctly to the weight shapes."""
        rank = 0
        sharded_model, modified_params = make_parameter_sharded_model(
            self.original_model,
            self.tp_config,
            rank=rank,
            device_count=self.device_count,
            device_id=self.devices[rank],
        )

        self.assertTrue(any("up_proj/kernel" in p for p in modified_params))

        sharded_weights_dict = {w.name: w for w in sharded_model.weights}

        expected_up_dim = 32 // self.device_count
        self.assertEqual(
            sharded_weights_dict["up_proj_kernel"].shape, (16, expected_up_dim)
        )

    def test_forward_pass_execution(self):
        """Tests the manual execution loop in the sharded wrapper."""
        rank = 0
        sharded_model, _ = make_parameter_sharded_model(
            self.original_model,
            self.tp_config,
            rank=rank,
            device_count=self.device_count,
            device_id=self.devices[rank],
        )

        input_tensor = ops.convert_to_tensor(self.input_data)

        try:
            output = sharded_model(input_tensor)
            self.assertEqual(output.shape, (4, 8))
        except Exception as e:
            msg = str(e).lower()
            if "process group" in msg or "logical devices" in msg:
                pytest.skip(f"Collective ops not available: {e}")
            else:
                raise e

    def test_device_placement(self):
        """Verifies ShardedWeight respects the requested device."""
        rank = 0
        target_device = self.devices[rank]

        sharded_model, _ = make_parameter_sharded_model(
            self.original_model,
            self.tp_config,
            rank=rank,
            device_count=self.device_count,
            device_id=target_device,
        )

        for w in sharded_model.weights:
            if isinstance(w, ShardedWeight):
                val = w.variable.value
                actual_device = str(getattr(val, "device", "cpu")).lower()

                target_type = str(target_device).split(":")[0].lower()
                self.assertIn(target_type, actual_device)
                break
