import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import initializers
from keras.src import layers
from keras.src import testing


class GRUTest(testing.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basics(self):
        self.run_layer_test(
            layers.GRU,
            init_kwargs={"units": 3, "dropout": 0.5, "recurrent_dropout": 0.5},
            input_shape=(3, 2, 4),
            call_kwargs={"training": True},
            expected_output_shape=(3, 3),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.GRU,
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

    @parameterized.parameters([1, 2])
    def test_correctness(self, implementation):
        sequence = np.arange(72).reshape((3, 6, 4)).astype("float32")
        layer = layers.GRU(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.5217289, 0.5217289, 0.5217289],
                    [0.6371659, 0.6371659, 0.6371659],
                    [0.39384964, 0.39384964, 0.3938496],
                ]
            ),
            output,
        )

        layer = layers.GRU(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.24406259, 0.24406259, 0.24406259],
                    [0.611516, 0.611516, 0.611516],
                    [0.3928808, 0.3928808, 0.3928808],
                ]
            ),
            output,
        )

        layer = layers.GRU(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            unroll=True,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.5217289, 0.5217289, 0.5217289],
                    [0.6371659, 0.6371659, 0.6371659],
                    [0.39384964, 0.39384964, 0.3938496],
                ]
            ),
            output,
        )

        layer = layers.GRU(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            reset_after=False,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.51447755, 0.51447755, 0.51447755],
                    [0.6426879, 0.6426879, 0.6426879],
                    [0.40208298, 0.40208298, 0.40208298],
                ]
            ),
            output,
        )

        layer = layers.GRU(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            use_bias=False,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.49988455, 0.49988455, 0.49988455],
                    [0.64701194, 0.64701194, 0.64701194],
                    [0.4103359, 0.4103359, 0.4103359],
                ]
            ),
            output,
        )

    def test_statefulness(self):
        sequence = np.arange(24).reshape((2, 3, 4)).astype("float32")
        layer = layers.GRU(
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
                    [0.29542392, 0.29542392, 0.29542392, 0.29542392],
                    [0.5885018, 0.5885018, 0.5885018, 0.5885018],
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
                    [0.29542392, 0.29542392, 0.29542392, 0.29542392],
                    [0.5885018, 0.5885018, 0.5885018, 0.5885018],
                ]
            ),
            output,
        )

    def test_pass_initial_state(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        initial_state = np.arange(4).reshape((2, 2)).astype("float32")
        layer = layers.GRU(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        output = layer(sequence, initial_state=initial_state)
        self.assertAllClose(
            np.array([[0.23774096, 0.33508456], [0.83659905, 1.0227708]]),
            output,
        )

        layer = layers.GRU(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        output = layer(sequence, initial_state=initial_state)
        self.assertAllClose(
            np.array([[0.13486053, 0.23261218], [0.78257304, 0.9691353]]),
            output,
        )

    def test_masking(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        mask = np.array([[True, True, False, True], [True, False, False, True]])
        layer = layers.GRU(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            unroll=True,
        )
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array([[0.19393763, 0.19393763], [0.30818558, 0.30818558]]),
            output,
        )

        layer = layers.GRU(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            return_sequences=True,
        )
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array(
                [
                    [0.03606692, 0.03606692],
                    [0.09497581, 0.09497581],
                    [0.09497581, 0.09497581],
                    [0.19393763, 0.19393763],
                ],
            ),
            output[0],
        )
        self.assertAllClose(
            np.array(
                [
                    [0.16051409, 0.16051409],
                    [0.16051409, 0.16051409],
                    [0.16051409, 0.16051409],
                    [0.30818558, 0.30818558],
                ],
            ),
            output[1],
        )

        layer = layers.GRU(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            return_sequences=True,
            zero_output_for_mask=True,
        )
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array(
                [
                    [0.03606692, 0.03606692],
                    [0.09497581, 0.09497581],
                    [0.0, 0.0],
                    [0.19393763, 0.19393763],
                ],
            ),
            output[0],
        )
        self.assertAllClose(
            np.array(
                [
                    [0.16051409, 0.16051409],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.30818558, 0.30818558],
                ],
            ),
            output[1],
        )

        layer = layers.GRU(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array([[0.11669192, 0.11669192], [0.28380975, 0.28380975]]),
            output,
        )

    def test_legacy_implementation_argument(self):
        sequence = np.arange(72).reshape((3, 6, 4)).astype("float32")
        layer = layers.GRU(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        config = layer.get_config()
        config["implementation"] = 0  # Add legacy argument
        layer = layers.GRU.from_config(config)
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.5217289, 0.5217289, 0.5217289],
                    [0.6371659, 0.6371659, 0.6371659],
                    [0.39384964, 0.39384964, 0.3938496],
                ]
            ),
            output,
        )
