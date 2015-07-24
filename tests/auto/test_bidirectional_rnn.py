import unittest
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, JZS1, Bidirectional


class TestBidirectionalRNN(unittest.TestCase):
    def setUp(self):
        more_ones = np.array([
            [1, 0, 1, 1],
            [0, 1, 1, 1]
        ])
        more_zeros = np.array([
            [0, 0, 1, 0],
            [0, 1, 0, 0]
        ])
        self.X = np.concatenate([more_ones, more_zeros])
        self.Y = np.array([1, 1, 0, 0])

    def _make_bidirectional_model(self, rnn_class):
        model = Sequential()
        model.add(Embedding(2, 2))
        model.add(Bidirectional(rnn_class, input_dim=2, output_dim=5))
        model.add(Dense(5, 1))
        model.add(Activation("sigmoid"))
        model.compile("rmsprop", loss="mse")
        return model

    def test_bidirectional_lstm(self):
        model = self._make_bidirectional_model(LSTM)
        model.fit(self.X, self.Y)
        self.assertTrue(((model.predict(self.X) > 0.5) == self.Y).all())

    def test_bidirectional_gru(self):
        model = self._make_bidirectional_model(GRU)
        model.fit(self.X, self.Y)
        self.assertTrue(((model.predict(self.X) > 0.5) == self.Y).all())

    def test_bidrectional_jzs1(self):
        model = self._make_bidirectional_model(JZS1)
        model.fit(self.X, self.Y)
        self.assertTrue(((model.predict(self.X) > 0.5) == self.Y).all())

if __name__ == '__main__':
    unittest.main()
