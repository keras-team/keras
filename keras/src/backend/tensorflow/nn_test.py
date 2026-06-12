from keras.src import testing


class ConvTest(testing.TestCase):
    def test_conv_stride_and_dilation(self):
        # Tests resolution of Issue #23028
        import numpy as np

        from keras.src import backend

        x = np.zeros((1, 8, 8, 3), "float32")
        k = np.zeros((3, 3, 3, 4), "float32")

        # This should execute without raising an error
        result = backend.nn.conv(
            x,
            k,
            strides=2,
            padding="valid",
            dilation_rate=2,
            data_format="channels_last",
        )
        self.assertEqual(result.shape, (1, 2, 2, 4))
