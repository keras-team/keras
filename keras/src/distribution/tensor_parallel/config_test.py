from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from keras.src.distribution.tensor_parallel.communications import AllGatherKeras
from keras.src.distribution.tensor_parallel.communications import AllReduceKeras
from keras.src.distribution.tensor_parallel.communications import BroadcastKeras
from keras.src.distribution.tensor_parallel.config import ConfigKeras


@pytest.fixture
def mock_backend():
    """Provides a mock backend object for tests."""
    return MagicMock()


@patch("keras.src.distribution.tensor_parallel.config.get_distributed_backend")
def test_create_collective_ops_parsing(mock_get_backend, mock_backend):
    """
    Tests that various rule strings are correctly parsed into collective op
    objects.
    """
    mock_get_backend.return_value = mock_backend
    devices = ["cpu:0", "cpu:1"]
    world_size = len(devices)

    input_rules = {
        "dense_layer": {
            "kernel": "sum",
            "bias": "broadcast",
        },
        "output_layer": {
            "output": "gather -2",
            "activation": None,
        },
    }

    config = ConfigKeras(state_rules={}, output_rules=input_rules)

    new_config = config.create_collective_ops(devices)
    rules = new_config.output_rules

    sum_op = rules["dense_layer"]["kernel"]
    assert isinstance(sum_op, AllReduceKeras)
    assert sum_op.op == "sum"
    assert sum_op.world_size == world_size
    assert sum_op.backend == mock_backend

    broadcast_op = rules["dense_layer"]["bias"]
    assert isinstance(broadcast_op, BroadcastKeras)
    assert broadcast_op.world_size == world_size

    gather_op = rules["output_layer"]["output"]
    assert isinstance(gather_op, AllGatherKeras)
    assert gather_op.dim == -2
    assert gather_op.world_size == world_size

    assert rules["output_layer"]["activation"] is None


@patch("keras.src.distribution.tensor_parallel.config.get_distributed_backend")
def test_create_collective_ops_with_default_gather(
    mock_get_backend, mock_backend
):
    """Tests the 'gather' rule without a specified dimension."""
    mock_get_backend.return_value = mock_backend
    devices = ["cpu:0", "cpu:1", "cpu:2"]
    input_rules = {"output": "gather"}
    config = ConfigKeras(state_rules={}, output_rules={"layer": input_rules})

    new_config = config.create_collective_ops(devices)
    gather_op = new_config.output_rules["layer"]["output"]

    assert isinstance(gather_op, AllGatherKeras)
    assert gather_op.dim == -1
