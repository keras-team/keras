import numpy as np
import pytest

from keras_core import backend
from keras_core import initializers
from keras_core import layers
from keras_core import testing


@pytest.mark.skipif(
    backend.backend() != "tensorflow",
    reason="Only implemented for TF for now.",
)
class SimpleRNNTest(testing.TestCase):
    def test_basics(self):
        self.run_layer_test(
            layers.SimpleRNN,
            init_kwargs={"units": 3, "dropout": 0.5, "recurrent_dropout": 0.5},
            input_shape=(3, 2, 4),
            call_kwargs={"training": True},
            expected_output_shape=(3, 3),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.SimpleRNN,
            init_kwargs={
                "units": 3,
                "return_sequences": True,
                "bias_regularizer": "l1",
                "kernel_regularizer": "l2",
                "recurrent_regularizer": "l2",
            },
            input_shape=(3, 2, 4),
            expected_output_shape=(3, 2, 3),
            expected_num_losses=3,
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )

    def test_correctness(self):
        sequence = np.arange(24).reshape((2, 3, 4)).astype("float32")
        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.405432, 0.405432, 0.405432, 0.405432],
                    [0.73605347, 0.73605347, 0.73605347, 0.73605347],
                ]
            ),
            output,
        )

    def test_statefulness(self):
        sequence = np.arange(24).reshape((2, 3, 4)).astype("float32")
        layer = layers.SimpleRNN(
            4,
            stateful=True,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        layer(sequence)
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.40559256, 0.40559256, 0.40559256, 0.40559256],
                    [0.7361247, 0.7361247, 0.7361247, 0.7361247],
                ]
            ),
            output,
        )
        layer.reset_state()
        layer(sequence)
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.40559256, 0.40559256, 0.40559256, 0.40559256],
                    [0.7361247, 0.7361247, 0.7361247, 0.7361247],
                ]
            ),
            output,
        )

    # TODO: test masking
