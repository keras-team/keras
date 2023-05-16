import numpy as np
from absl.testing import parameterized

from keras_core import initializers
from keras_core import layers
from keras_core import testing


class LSTMTest(testing.TestCase, parameterized.TestCase):
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

    # TODO: test masking
