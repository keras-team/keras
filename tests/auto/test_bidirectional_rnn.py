import unittest
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, JZS1, Bidirectional


class TestBidirectionalRNN(unittest.TestCase):
    def setUp(self):
        # positive sequences contain the symbol '2'
        positive_sequences = np.array([
            [1, 0, 2, 1],
            [0, 2, 1, 1],
            [2, 0, 0, 0],
            [1, 1, 1, 2],
        ])
        # negative sequence examples don't contain the symbol '2'
        negative_sequences = np.array([
            [0, 0, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ])
        self.X = np.concatenate([positive_sequences, negative_sequences])
        self.Y = np.array([1] * len(positive_sequences) + [0] * len(negative_sequences))

    def _make_bidirectional_model(self, rnn_class):
        model = Sequential()
        model.add(Embedding(3, 3))
        model.add(Bidirectional(rnn_class, input_dim=3, output_dim=5))
        model.add(Dense(10, 1))
        model.add(Activation("sigmoid"))
        model.compile("rmsprop", loss="mse")
        return model

    def test_bidirectional_lstm(self):
        model = self._make_bidirectional_model(LSTM)
        model.fit(self.X, self.Y)
        for pred, y in zip(model.predict(self.X), self.Y):
            self.assertEqual(pred > 0.5, y)

    def test_bidirectional_gru(self):
        model = self._make_bidirectional_model(GRU)
        model.fit(self.X, self.Y, verbose=0)
        for pred, y in zip(model.predict(self.X), self.Y):
            self.assertEqual(pred > 0.5, y)

    def test_bidrectional_jzs1(self):
        model = self._make_bidirectional_model(JZS1)
        model.fit(self.X, self.Y, verbose=0)
        for pred, y in zip(model.predict(self.X), self.Y):
            self.assertEqual(pred > 0.5, y)


if __name__ == '__main__':
    unittest.main()
