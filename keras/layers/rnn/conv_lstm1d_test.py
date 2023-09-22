import numpy as np
import pytest

from keras import initializers
from keras import layers
from keras import testing


class ConvLSTM1DTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basics(self):
        self.run_layer_test(
            layers.ConvLSTM1D,
            init_kwargs={"filters": 5, "kernel_size": 3, "padding": "same"},
            input_shape=(3, 2, 4, 3),
            expected_output_shape=(3, 4, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.ConvLSTM1D,
            init_kwargs={
                "filters": 5,
                "kernel_size": 3,
                "padding": "valid",
                "recurrent_dropout": 0.5,
            },
            input_shape=(3, 2, 8, 3),
            call_kwargs={"training": True},
            expected_output_shape=(3, 6, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.ConvLSTM1D,
            init_kwargs={
                "filters": 5,
                "kernel_size": 3,
                "padding": "valid",
                "return_sequences": True,
            },
            input_shape=(3, 2, 8, 3),
            expected_output_shape=(3, 2, 6, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )

    def test_correctness(self):
        sequence = np.arange(120).reshape((2, 3, 4, 5)).astype("float32") / 10
        layer = layers.ConvLSTM1D(
            filters=2,
            kernel_size=3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [[0.40807986, 0.40807986], [0.46421072, 0.46421072]],
                    [[0.80933154, 0.80933154], [0.8233646, 0.8233646]],
                ]
            ),
            output,
        )
