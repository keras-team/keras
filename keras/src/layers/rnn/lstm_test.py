import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import initializers
from keras.src import layers
from keras.src import testing


class LSTMTest(testing.TestCase, parameterized.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basics(self):
        self.run_layer_test(
            layers.LSTM,
            init_kwargs={"units": 3, "dropout": 0.5, "recurrent_dropout": 0.5},
            input_shape=(3, 2, 4),
            call_kwargs={"training": True},
            expected_output_shape=(3, 3),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.LSTM,
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
        layer = layers.LSTM(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            implementation=implementation,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.6288687, 0.6288687, 0.6288687],
                    [0.86899155, 0.86899155, 0.86899155],
                    [0.9460773, 0.9460773, 0.9460773],
                ]
            ),
            output,
        )

        layer = layers.LSTM(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
            implementation=implementation,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.35622165, 0.35622165, 0.35622165],
                    [0.74789524, 0.74789524, 0.74789524],
                    [0.8872726, 0.8872726, 0.8872726],
                ]
            ),
            output,
        )

        layer = layers.LSTM(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            unroll=True,
            implementation=implementation,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.6288687, 0.6288687, 0.6288687],
                    [0.86899155, 0.86899155, 0.86899155],
                    [0.9460773, 0.9460773, 0.9460773],
                ]
            ),
            output,
        )

        layer = layers.LSTM(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            unit_forget_bias=False,
            implementation=implementation,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.57019705, 0.57019705, 0.57019705],
                    [0.8661914, 0.8661914, 0.8661914],
                    [0.9459622, 0.9459622, 0.9459622],
                ]
            ),
            output,
        )

        layer = layers.LSTM(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            use_bias=False,
            implementation=implementation,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.54986924, 0.54986924, 0.54986924],
                    [0.86226785, 0.86226785, 0.86226785],
                    [0.9443936, 0.9443936, 0.9443936],
                ]
            ),
            output,
        )

    def test_statefulness(self):
        sequence = np.arange(24).reshape((2, 3, 4)).astype("float32")
        layer = layers.LSTM(
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
                    [0.3124785, 0.3124785, 0.3124785, 0.3124785],
                    [0.6863672, 0.6863672, 0.6863672, 0.6863672],
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
                    [0.3124785, 0.3124785, 0.3124785, 0.3124785],
                    [0.6863672, 0.6863672, 0.6863672, 0.6863672],
                ]
            ),
            output,
        )

    def test_pass_initial_state(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        initial_state = [
            np.arange(4).reshape((2, 2)).astype("float32"),
            np.arange(4).reshape((2, 2)).astype("float32"),
        ]
        layer = layers.LSTM(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        output = layer(sequence, initial_state=initial_state)
        self.assertAllClose(
            np.array([[0.20574439, 0.3558822], [0.64930826, 0.66276]]),
            output,
        )

        layer = layers.LSTM(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        output = layer(sequence, initial_state=initial_state)
        self.assertAllClose(
            np.array([[0.13281618, 0.2790356], [0.5839337, 0.5992567]]),
            output,
        )

    def test_masking(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        mask = np.array([[True, True, False, True], [True, False, False, True]])
        layer = layers.LSTM(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            unroll=True,
        )
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array([[0.1524914, 0.1524914], [0.35969394, 0.35969394]]),
            output,
        )

        layer = layers.LSTM(
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
                    [0.0158891, 0.0158891],
                    [0.05552047, 0.05552047],
                    [0.05552047, 0.05552047],
                    [0.1524914, 0.1524914],
                ],
            ),
            output[0],
        )
        self.assertAllClose(
            np.array(
                [
                    [0.14185596, 0.14185596],
                    [0.14185596, 0.14185596],
                    [0.14185596, 0.14185596],
                    [0.35969394, 0.35969394],
                ],
            ),
            output[1],
        )

        layer = layers.LSTM(
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
                    [0.0158891, 0.0158891],
                    [0.05552047, 0.05552047],
                    [0.0, 0.0],
                    [0.1524914, 0.1524914],
                ],
            ),
            output[0],
        )
        self.assertAllClose(
            np.array(
                [
                    [0.14185596, 0.14185596],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.35969394, 0.35969394],
                ],
            ),
            output[1],
        )

        layer = layers.LSTM(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array([[0.10056866, 0.10056866], [0.31006062, 0.31006062]]),
            output,
        )
