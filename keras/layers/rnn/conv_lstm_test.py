import numpy as np

from keras import backend
from keras import initializers
from keras import testing
from keras.layers.rnn.conv_lstm import ConvLSTM
from keras.layers.rnn.conv_lstm import ConvLSTMCell


class ConvLSTMCellTest(testing.TestCase):
    def test_correctness(self):
        x = np.arange(150).reshape((2, 5, 5, 3)).astype("float32") / 10
        s1 = np.arange(200).reshape((2, 5, 5, 4)).astype("float32") / 10
        s2 = np.arange(200).reshape((2, 5, 5, 4)).astype("float32") / 10

        layer = ConvLSTMCell(
            rank=2,
            filters=4,
            kernel_size=3,
            padding="same",
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
        )
        output = layer(x, [s1, s2])
        checksum_0 = np.sum(backend.convert_to_numpy(output[0]))
        self.assertAllClose(checksum_0, 188.89502)
        checksum_1 = np.sum(backend.convert_to_numpy(output[1][0]))
        self.assertAllClose(checksum_1, 188.89502)
        checksum_2 = np.sum(backend.convert_to_numpy(output[1][1]))
        self.assertAllClose(checksum_2, 2170.444)


class ConvLSTMTest(testing.TestCase):
    def test_correctness(self):
        x = np.arange(450).reshape((2, 3, 5, 5, 3)).astype("float32") / 100
        s1 = np.arange(200).reshape((2, 5, 5, 4)).astype("float32") / 100
        s2 = np.arange(200).reshape((2, 5, 5, 4)).astype("float32") / 100

        layer = ConvLSTM(
            rank=2,
            filters=4,
            kernel_size=3,
            padding="same",
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
        )
        output = layer(x, initial_state=[s1, s2])
        output = backend.convert_to_numpy(output)
        self.assertAllClose(np.sum(output), 119.812454)
