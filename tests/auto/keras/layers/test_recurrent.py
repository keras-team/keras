import unittest
import numpy as np
import theano

from keras.layers import recurrent

def recursive_runner(layer):
    layer.input = theano.shared(value=np.ones((10,10,10)))

    config = layer.get_config()

    for train in [True,False]:
        out = layer.get_output(train).eval()
        mask = layer.get_output_mask(train)

class TestRNNS(unittest.TestCase):
    def test_simple(self):
        for weights in [None, [np.ones((10,5))]]:
            for ret_seq in [True, False]:
                layer = recurrent.SimpleRNN(10, 5, return_sequences=ret_seq, weights=weights)
                recursive_runner(layer)

    def test_simple_deep(self):
        for weights in [None, [np.ones((10,5))]]:
            for ret_seq in [True, False]:
                layer = recurrent.SimpleDeepRNN(10, 5, return_sequences=ret_seq, weights=weights)
                recursive_runner(layer)

    def test_gru(self):
        for weights in [None, [np.ones((10,5))]]:
            for ret_seq in [True, False]:
                layer = recurrent.GRU(10, 5, return_sequences=ret_seq, weights=weights)
                recursive_runner(layer)

    def test_lstm(self):
        for weights in [None, [np.ones((10,5))]]:
            for ret_seq in [True, False]:
                layer = recurrent.LSTM(10, 5, return_sequences=ret_seq, weights=weights)
                recursive_runner(layer)


    def test_jzs1(self):
        for weights in [None, [np.ones((10,5))]]:
            for ret_seq in [True, False]:
                layer = recurrent.JZS1(10, 5, return_sequences=ret_seq, weights=weights)
                recursive_runner(layer)

    def test_jzs2(self):
        for weights in [None, [np.ones((10,5))]]:
            for ret_seq in [True, False]:
                layer = recurrent.JZS2(10, 5, return_sequences=ret_seq, weights=weights)
                recursive_runner(layer)

    def test_jzs3(self):
        for weights in [None, [np.ones((10,5))]]:
            for ret_seq in [True, False]:
                layer = recurrent.JZS3(10, 5, return_sequences=ret_seq, weights=weights)
                recursive_runner(layer)

if __name__ == '__main__':
    unittest.main()
