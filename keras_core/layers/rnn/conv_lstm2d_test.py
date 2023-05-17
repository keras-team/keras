from keras_core import layers
from keras_core import testing


class ConvLSTM2DTest(testing.TestCase):
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

    # TODO: correctness testing

    # def test_correctness(self):
    #     sequence = np.arange(480).reshape((2, 3, 4, 4, 5)).astype("float32")
    #     layer = layers.ConvLSTM2D(
    #         filters=2,
    #         kernel_size=3,
    #         kernel_initializer=initializers.Constant(0.0001),
    #         recurrent_initializer=initializers.Constant(0.01),
    #         bias_initializer=initializers.Constant(0.01),
    #     )
    #     output = layer(sequence)
    #     self.assertAllClose(
    #         np.array(
    #             [
    #                 [
    #                     [[0.4320268, 0.4320268], [0.4475501, 0.4475501]],
    #                     [[0.49229687, 0.49229687], [0.50656533, 0.50656533]],
    #                 ],
    #                 [
    #                     [[0.8781725, 0.8781725], [0.88340145, 0.88340145]],
    #                     [[0.8988858, 0.8988858], [0.9039862, 0.9039862]],
    #                 ],
    #             ]
    #         ),
    #         output,
    #     )
