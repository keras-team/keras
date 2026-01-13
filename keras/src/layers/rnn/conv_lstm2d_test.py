import numpy as np
import pytest

from keras.src import backend
from keras.src import initializers
from keras.src import layers
from keras.src import testing


class ConvLSTM2DTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basics(self):
        channels_last = backend.config.image_data_format() == "channels_last"
        self.run_layer_test(
            layers.ConvLSTM2D,
            init_kwargs={"filters": 5, "kernel_size": 3, "padding": "same"},
            input_shape=(3, 2, 4, 4, 3) if channels_last else (3, 2, 3, 4, 4),
            expected_output_shape=(
                (3, 4, 4, 5) if channels_last else (3, 5, 4, 4)
            ),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.ConvLSTM2D,
            init_kwargs={
                "filters": 5,
                "kernel_size": 3,
                "padding": "valid",
                "recurrent_dropout": 0.5,
            },
            input_shape=(3, 2, 8, 8, 3) if channels_last else (3, 2, 3, 8, 8),
            call_kwargs={"training": True},
            expected_output_shape=(
                (3, 6, 6, 5) if channels_last else (3, 5, 6, 6)
            ),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.ConvLSTM2D,
            init_kwargs={
                "filters": 5,
                "kernel_size": 3,
                "padding": "valid",
                "return_sequences": True,
            },
            input_shape=(3, 2, 8, 8, 3) if channels_last else (3, 2, 3, 8, 8),
            expected_output_shape=(
                (3, 2, 6, 6, 5) if channels_last else (3, 2, 5, 6, 6)
            ),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )

    def test_correctness(self):
        sequence = (
            np.arange(480).reshape((2, 3, 4, 4, 5)).astype("float32") / 100
        )
        expected_output = np.array(
            [
                [
                    [[0.48694518, 0.48694518], [0.50237733, 0.50237733]],
                    [[0.5461202, 0.5461202], [0.5598283, 0.5598283]],
                ],
                [
                    [[0.8661607, 0.8661607], [0.86909103, 0.86909103]],
                    [[0.8774414, 0.8774414], [0.8800861, 0.8800861]],
                ],
            ]
        )
        if backend.config.image_data_format() == "channels_first":
            sequence = sequence.transpose((0, 1, 4, 2, 3))
            expected_output = expected_output.transpose((0, 3, 1, 2))
        layer = layers.ConvLSTM2D(
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
            tpu_atol=1e-3,
            tpu_rtol=1e-3,
        )
