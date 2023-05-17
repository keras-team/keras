from keras_core import layers
from keras_core import testing


class ConvLSTM1DTest(testing.TestCase):
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

    # TODO: correctness testing

    # def test_correctness(self):
    #     sequence = np.arange(120).reshape((2, 3, 4, 5)).astype("float32")
    #     layer = layers.ConvLSTM1D(
    #         filters=2,
    #         kernel_size=3,
    #         kernel_initializer=initializers.Constant(0.001),
    #         recurrent_initializer=initializers.Constant(0.0),
    #         bias_initializer=initializers.Constant(0.3),
    #         use_bias=False,
    #     )
    #     output = layer(sequence)
    #     self.assertAllClose(
    #         np.array(
    #             [
    #                 [[0.49877906, 0.49877906], [0.5447451, 0.5447451]],
    #                 [[0.94260275, 0.94260275], [0.95974874, 0.95974874]],
    #             ]
    #         ),
    #         output,
    #     )
