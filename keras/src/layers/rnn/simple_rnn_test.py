import numpy as np
import pytest

from keras.src import initializers
from keras.src import layers
from keras.src import testing


class SimpleRNNTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basics(self):
        self.run_layer_test(
            layers.SimpleRNN,
            init_kwargs={"units": 3, "dropout": 0.5, "recurrent_dropout": 0.5},
            input_shape=(3, 2, 4),
            call_kwargs={"training": True},
            expected_output_shape=(3, 3),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            expected_num_non_trainable_variables=1,
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
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            unroll=True,
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
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.11144729, 0.11144729, 0.11144729, 0.11144729],
                    [0.5528889, 0.5528889, 0.5528889, 0.5528889],
                ]
            ),
            output,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
            unroll=True,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.11144729, 0.11144729, 0.11144729, 0.11144729],
                    [0.5528889, 0.5528889, 0.5528889, 0.5528889],
                ]
            ),
            output,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
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
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
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
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

    def test_pass_initial_state(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        initial_state = np.arange(8).reshape((2, 4)).astype("float32")
        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        output = layer(sequence, initial_state=initial_state)
        self.assertAllClose(
            np.array(
                [
                    [0.33621645, 0.33621645, 0.33621645, 0.33621645],
                    [0.6262637, 0.6262637, 0.6262637, 0.6262637],
                ]
            ),
            output,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        output = layer(sequence, initial_state=initial_state)
        self.assertAllClose(
            np.array(
                [
                    [0.07344437, 0.07344437, 0.07344437, 0.07344437],
                    [0.43043602, 0.43043602, 0.43043602, 0.43043602],
                ]
            ),
            output,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

    def test_masking(self):
        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        mask = np.array([[True, True, False, True], [True, False, False, True]])
        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            unroll=True,
        )
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array(
                [
                    [0.32951632, 0.32951632, 0.32951632, 0.32951632],
                    [0.61799484, 0.61799484, 0.61799484, 0.61799484],
                ]
            ),
            output,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

        layer = layers.SimpleRNN(
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
                    [0.0599281, 0.0599281],
                    [0.15122814, 0.15122814],
                    [0.15122814, 0.15122814],
                    [0.32394567, 0.32394567],
                ],
            ),
            output[0],
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        self.assertAllClose(
            np.array(
                [
                    [0.3969304, 0.3969304],
                    [0.3969304, 0.3969304],
                    [0.3969304, 0.3969304],
                    [0.608085, 0.608085],
                ],
            ),
            output[1],
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

        layer = layers.SimpleRNN(
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
                    [0.0599281, 0.0599281],
                    [0.15122814, 0.15122814],
                    [0.0, 0.0],
                    [0.32394567, 0.32394567],
                ],
            ),
            output[0],
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
        self.assertAllClose(
            np.array(
                [
                    [0.3969304, 0.3969304],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.608085, 0.608085],
                ],
            ),
            output[1],
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )

        layer = layers.SimpleRNN(
            4,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        output = layer(sequence, mask=mask)
        self.assertAllClose(
            np.array(
                [
                    [0.07376196, 0.07376196, 0.07376196, 0.07376196],
                    [0.43645123, 0.43645123, 0.43645123, 0.43645123],
                ]
            ),
            output,
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
