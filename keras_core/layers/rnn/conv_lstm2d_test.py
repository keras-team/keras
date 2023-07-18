import numpy as np
import pytest

from keras_core import initializers
from keras_core import layers
from keras_core import testing


class ConvLSTM2DTest(testing.TestCase):
    @pytest.mark.requires_trainable_backend
    def test_basics(self):
        self.run_layer_test(
            layers.ConvLSTM2D,
            init_kwargs={"filters": 5, "kernel_size": 3, "padding": "same"},
            input_shape=(3, 2, 4, 4, 3),
            expected_output_shape=(3, 4, 4, 5),
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
            input_shape=(3, 2, 8, 8, 3),
            call_kwargs={"training": True},
            expected_output_shape=(3, 6, 6, 5),
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
            input_shape=(3, 2, 8, 8, 3),
            expected_output_shape=(3, 2, 6, 6, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )

    def test_correctness(self):
        sequence = (
            np.arange(480).reshape((2, 3, 4, 4, 5)).astype("float32") / 100
        )
        layer = layers.ConvLSTM2D(
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
                    [
                        [[0.48694518, 0.48694518], [0.50237733, 0.50237733]],
                        [[0.5461202, 0.5461202], [0.5598283, 0.5598283]],
                    ],
                    [
                        [[0.8661607, 0.8661607], [0.86909103, 0.86909103]],
                        [[0.8774414, 0.8774414], [0.8800861, 0.8800861]],
                    ],
                ]
            ),
            output,
        )
