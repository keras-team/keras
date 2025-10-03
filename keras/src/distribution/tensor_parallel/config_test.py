import pytest

import keras
from keras.src import testing
from keras.src.distribution.tensor_parallel.communications import AllGatherKeras
from keras.src.distribution.tensor_parallel.communications import AllReduceKeras
from keras.src.distribution.tensor_parallel.communications import BroadcastKeras
from keras.src.distribution.tensor_parallel.config import ConfigKeras
from keras.src.distribution.tensor_parallel.config import _create_ops_from_rules


@pytest.mark.skipif(
    keras.backend.backend() != "jax",
    reason="This test suite requires a real JAX distributed backend.",
)
class TestConfig(testing.TestCase):
    """Test suite for the tensor parallel configuration."""

    def test_create_ops_from_rules_helper(self):
        """
        Tests the private _create_ops_from_rules helper function directly
        to ensure it correctly parses various rule types.
        """
        devices = ["/gpu:0", "/gpu:1"]
        world_size = len(devices)
        rules = {
            "dense/kernel": {"forward": "sum", "backward": "mean"},
            "embedding/weight": {
                "forward": "gather 0",
                "backward": "gather -1",
            },
            "attention/dense/bias": {"forward": "broadcast"},
            "passthrough": {"action": 123},
            "no_dict_action": "identity",
        }

        processed_rules = _create_ops_from_rules(rules, world_size)

        sum_op = processed_rules["dense/kernel"]["forward"]
        self.assertIsInstance(sum_op, AllReduceKeras)
        self.assertEqual(sum_op.op, "sum")
        self.assertEqual(sum_op.world_size, world_size)

        mean_op = processed_rules["dense/kernel"]["backward"]
        self.assertIsInstance(mean_op, AllReduceKeras)
        self.assertEqual(mean_op.op, "mean")

        gather_op_0 = processed_rules["embedding/weight"]["forward"]
        self.assertIsInstance(gather_op_0, AllGatherKeras)
        self.assertEqual(gather_op_0.dim, 0)
        self.assertEqual(gather_op_0.world_size, world_size)

        gather_op_neg1 = processed_rules["embedding/weight"]["backward"]
        self.assertIsInstance(gather_op_neg1, AllGatherKeras)
        self.assertEqual(gather_op_neg1.dim, -1)

        broadcast_op = processed_rules["attention/dense/bias"]["forward"]
        self.assertIsInstance(broadcast_op, BroadcastKeras)
        self.assertEqual(broadcast_op.world_size, world_size)

        self.assertEqual(processed_rules["passthrough"]["action"], 123)
        self.assertEqual(processed_rules["no_dict_action"], "identity")

    def test_config_keras_create_collective_ops(self):
        """
        Tests the public create_collective_ops method of the ConfigKeras class.
        """
        devices = ["/gpu:0", "/gpu:1"]
        world_size = len(devices)

        state_rules = {"some_weight": "split"}
        output_rules = {
            "layer_1_output": {"activation": "sum"},
            "layer_2_output": {"activation": "gather -1"},
        }

        config = ConfigKeras(state_rules=state_rules, output_rules=output_rules)
        new_config = config.create_collective_ops(devices)

        self.assertIsNot(new_config, config)

        self.assertEqual(new_config.state_rules, state_rules)

        self.assertIsInstance(
            config.output_rules["layer_1_output"]["activation"], str
        )

        sum_op = new_config.output_rules["layer_1_output"]["activation"]
        self.assertIsInstance(sum_op, AllReduceKeras)
        self.assertEqual(sum_op.op, "sum")
        self.assertEqual(sum_op.world_size, world_size)

        gather_op = new_config.output_rules["layer_2_output"]["activation"]
        self.assertIsInstance(gather_op, AllGatherKeras)
        self.assertEqual(gather_op.dim, -1)
        self.assertEqual(gather_op.world_size, world_size)
