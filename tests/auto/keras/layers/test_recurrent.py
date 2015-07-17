import unittest
import numpy as np
import theano

from keras.layers import recurrent

nb_samples, timesteps, input_dim, output_dim = 3, 3, 10, 5


def _runner(layer_class):
    """
    All the recurrent layers share the same interface, so we can run through them with a single
    function.
    """
    for weights in [None, [np.ones((input_dim, output_dim))]]:
        for ret_seq in [True, False]:
            layer = layer_class(input_dim, output_dim, return_sequences=ret_seq, weights=weights)
            layer.input = theano.shared(value=np.ones((nb_samples, timesteps, input_dim)))
            config = layer.get_config()

            for train in [True, False]:
                out = layer.get_output(train).eval()
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

    def test_simple_deep(self):
        _runner(recurrent.SimpleDeepRNN)

    def test_gru(self):
        _runner(recurrent.GRU)

    def test_lstm(self):
        _runner(recurrent.LSTM)

    def test_jzs1(self):
        _runner(recurrent.JZS1)

    def test_jzs2(self):
        _runner(recurrent.JZS2)

    def test_jzs3(self):
        _runner(recurrent.JZS3)


if __name__ == '__main__':
    unittest.main()
