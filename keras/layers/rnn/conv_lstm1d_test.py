import numpy as np
import pytest

from keras import backend
from keras import initializers
from keras import layers
from keras import testing


class ConvLSTM1DTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basics(self):
        channels_last = backend.config.image_data_format() == "channels_last"
        self.run_layer_test(
            layers.ConvLSTM1D,
            init_kwargs={"filters": 5, "kernel_size": 3, "padding": "same"},
            input_shape=(3, 2, 4, 3) if channels_last else (3, 2, 3, 4),
            expected_output_shape=(3, 4, 5) if channels_last else (3, 5, 4),
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
            input_shape=(3, 2, 8, 3) if channels_last else (3, 2, 3, 8),
            call_kwargs={"training": True},
            expected_output_shape=(3, 6, 5) if channels_last else (3, 5, 6),
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
            input_shape=(3, 2, 8, 3) if channels_last else (3, 2, 3, 8),
            expected_output_shape=(3, 2, 6, 5)
            if channels_last
            else (3, 2, 5, 6),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )

    def test_correctness(self):
        sequence = np.arange(120).reshape((2, 3, 4, 5)).astype("float32") / 10
        expected_output = np.array(
            [
                [[0.40807986, 0.40807986], [0.46421072, 0.46421072]],
                [[0.80933154, 0.80933154], [0.8233646, 0.8233646]],
            ]
        )
        if backend.config.image_data_format() == "channels_first":
            sequence = sequence.transpose((0, 1, 3, 2))
            expected_output = expected_output.transpose((0, 2, 1))
        layer = layers.ConvLSTM1D(
            filters=2,
            kernel_size=3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        output = layer(sequence)
        self.assertAllClose(
            expected_output,
            output,
        )
