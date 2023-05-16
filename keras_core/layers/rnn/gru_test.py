import numpy as np
from absl.testing import parameterized

from keras_core import initializers
from keras_core import layers
from keras_core import testing


class GRUTest(testing.TestCase, parameterized.TestCase):
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

    # TODO: test masking
