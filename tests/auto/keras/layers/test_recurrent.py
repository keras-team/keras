import unittest
import numpy as np

from keras.layers import recurrent, ntm
from keras import backend as K

nb_samples, timesteps, input_dim, output_dim = 3, 3, 10, 5


def _runner(layer_class):
    """
    All the recurrent layers share the same interface,
    so we can run through them with a single function.
    """
    for ret_seq in [True, False]:
        layer = layer_class(output_dim, return_sequences=ret_seq,
                            weights=None, input_shape=(None, input_dim))
        layer.input = K.variable(np.ones((nb_samples, timesteps, input_dim)))
        config = layer.get_config()

        for train in [True, False]:
            out = K.eval(layer.get_output(train))
            # Make sure the output has the desired shape
            if ret_seq:
                assert(out.shape == (nb_samples, timesteps, output_dim))
            else:
                assert(out.shape == (nb_samples, output_dim))

            mask = layer.get_output_mask(train)


class TestRNNS(unittest.TestCase):
    """
    Test all the RNNs using a generic test runner function defined above.
    """
    def test_simple(self):
        _runner(recurrent.SimpleRNN)

    def test_gru(self):
        _runner(recurrent.GRU)

    def test_lstm(self):
        _runner(recurrent.LSTM)

    # def test_jzs1(self):
    #     _runner(recurrent.JZS1)

    # def test_jzs2(self):
    #     _runner(recurrent.JZS2)

    # def test_jzs3(self):
    #     _runner(recurrent.JZS3)


if __name__ == '__main__':
    unittest.main()
