import pytest
import numpy as np
import keras
import tensorflow as tf
from unittest.mock import patch, MagicMock

from keras.src.distribution.tensor_parallel.tensor_parallel import TensorParallelKeras


INPUT_DIM = 10
HIDDEN_DIM = 17
OUTPUT_DIM = 5
BATCH_SIZE = 8

def create_simple_mlp():
    """Creates a simple Keras Sequential model for testing."""
    return keras.Sequential(
        [
            keras.Input(shape=(INPUT_DIM,), name="input_layer"),
            keras.layers.Dense(HIDDEN_DIM, activation="relu", name="dense_1"),
            keras.layers.Dense(OUTPUT_DIM, name="dense_2"),
        ]
    )

def count_params(model_or_weights):
    """Helper function to count the total number of parameters."""
    weights = []
    if isinstance(model_or_weights, keras.Model):
        weights = model_or_weights.weights
    else:
        weights = model_or_weights
        
    total_params = 0
    for p in weights:
        if hasattr(p, "shape"):
            total_params += np.prod(p.shape)
    return int(total_params)


class TestTensorParallelKeras:
    """Test suite for the TensorParallelKeras wrapper."""

    @pytest.fixture
    def mock_devices(self):
        """A pytest fixture to mock device discovery for a predictable environment."""
        with patch.object(
            TensorParallelKeras,
            "_discover_devices",
            return_value=["cpu:0", "cpu:1"],
        ) as mock:
            yield mock

    def test_initialization_and_sharding(self, mock_devices):
        """
        Tests if the model is correctly initialized and sharded for world_size > 1.
        """
        print("🚀 Testing model initialization and sharding...")
        original_model = create_simple_mlp()
        original_params = count_params(original_model)

        tp_model = TensorParallelKeras(model=original_model, world_size=2)

        assert tp_model.world_size == 2
        assert tp_model.distributed is True
        assert len(tp_model.model_shards) == 2, "Model should be split into 2 shards"

        shard1_params = count_params(tp_model.model_shards[0])
        shard2_params = count_params(tp_model.model_shards[1])

        assert shard1_params < original_params, "Shard 1 should have fewer params than the original"
        assert shard2_params < original_params, "Shard 2 should have fewer params than the original"
        assert shard1_params != shard2_params, "Shards should have different param counts"
        print("✅ Initialization and sharding successful.")


    def test_non_distributed_case_world_size_one(self, mock_devices):
        """
        Tests if the model behaves like a standard Keras model when world_size is 1.
        """
        print("\n🚀 Testing non-distributed case (world_size=1)...")
        original_model = create_simple_mlp()
        original_params = count_params(original_model)

        tp_model = TensorParallelKeras(model=original_model, world_size=1)

        assert tp_model.world_size == 1
        assert tp_model.distributed is False
        assert len(tp_model.model_shards) == 1
        assert tp_model.model_shards[0] == original_model
        assert count_params(tp_model.model_shards[0]) == original_params
        print("✅ Non-distributed case handled correctly.")


    def test_forward_pass_output_shape(self, mock_devices):
        """
        Tests if the forward pass of the sharded model executes and returns the correct shape.
        """
        print("\n🚀 Testing forward pass output shape...")
        original_model = create_simple_mlp()
        dummy_input = np.random.rand(BATCH_SIZE, INPUT_DIM).astype("float32")

        tp_model = TensorParallelKeras(model=original_model, world_size=2)
        output = tp_model(dummy_input)

        assert output is not None
        assert output.shape == (BATCH_SIZE, OUTPUT_DIM), "Output shape is incorrect"
        print("✅ Forward pass successful with correct output shape.")


    @patch("keras.src.distribution.tensor_parallel.communications.TensorParallelCommunicator")
    def test_gradient_slicing_logic(self, MockCommunicator, mock_devices):
        """
        Verifies that the correct upstream gradient slicing methods are called.
        """
        print("\n🚀 Testing upstream gradient slicing logic...")
        mock_communicator_instance = MagicMock()
        MockCommunicator.return_value = mock_communicator_instance
        
        original_model = create_simple_mlp()
        tp_model = TensorParallelKeras(model=original_model, world_size=2)
        
        dummy_full_gradients = tf.ones((BATCH_SIZE, OUTPUT_DIM))

        tp_model._slice_upstream_gradients_for_backward(dummy_full_gradients, "column_parallel")
        assert mock_communicator_instance.slice_upstream_gradient_for_column_parallel.call_count == 2
        
        tp_model._slice_upstream_gradients_for_backward(dummy_full_gradients, "row_parallel")
        assert mock_communicator_instance.slice_upstream_gradient_for_row_parallel.call_count == 2
        
        print("✅ Upstream gradient slicing calls verified.")


    @patch("keras.src.distribution.tensor_parallel.communications.TensorParallelCommunicator")
    def test_backward_communication_logic(self, MockCommunicator, mock_devices):
        """
        Verifies that the correct backward communication primitives (AllReduce/AllGather) are called.
        """
        print("\n🚀 Testing backward pass communication logic...")
        mock_communicator_instance = MagicMock()
        MockCommunicator.return_value = mock_communicator_instance

        original_model = create_simple_mlp()
        tp_model = TensorParallelKeras(model=original_model, world_size=2)
        
        dummy_gradients = [tf.ones((INPUT_DIM, HIDDEN_DIM)), tf.ones((HIDDEN_DIM,))]

        tp_model._apply_backward_communication(dummy_gradients, layer_type="column")
        mock_communicator_instance.backward_column_parallel.assert_called_once()
        
        tp_model._apply_backward_communication(dummy_gradients, layer_type="row")
        mock_communicator_instance.backward_row_parallel.assert_called_once()
        
        print("✅ Backward communication calls verified.")