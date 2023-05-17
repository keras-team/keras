from keras_core import layers
from keras_core import testing


class ConvLSTM1DTest(testing.TestCase):
    def test_basics(self):
        self.run_layer_test(
            layers.ConvLSTM3D,
            init_kwargs={"filters": 5, "kernel_size": 3, "padding": "same"},
            input_shape=(3, 2, 4, 4, 4, 3),
            expected_output_shape=(3, 4, 4, 4, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.ConvLSTM3D,
            init_kwargs={
                "filters": 5,
                "kernel_size": 3,
                "padding": "valid",
                "recurrent_dropout": 0.5,
            },
            input_shape=(3, 2, 8, 8, 8, 3),
            call_kwargs={"training": True},
            expected_output_shape=(3, 6, 6, 6, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )
        self.run_layer_test(
            layers.ConvLSTM3D,
            init_kwargs={
                "filters": 5,
                "kernel_size": 3,
                "padding": "valid",
                "return_sequences": True,
            },
            input_shape=(3, 2, 8, 8, 8, 3),
            expected_output_shape=(3, 2, 6, 6, 6, 5),
            expected_num_trainable_weights=3,
            expected_num_non_trainable_weights=0,
            supports_masking=True,
        )

    # TODO: correctness testing
