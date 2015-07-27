import unittest
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, Bidirectional

class TestBidirectionalRNN(unittest.TestCase):
    def setUp(self):
        # positive sequences contain the symbol '2'
        positive_sequences = np.array([
            [1, 0, 2, 1],
            [0, 2, 1, 1],
            [2, 0, 0, 0],
            [1, 1, 1, 2],
            [2, 2, 2, 2],
            [2, 2, 1, 1],
            [1, 1, 2, 1],
            [1, 2, 2, 0],
            [2, 0, 2, 0]
        ])
        # negative sequence examples don't contain the symbol '2'
        negative_sequences = np.array([
            [0, 0, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 1, 0, 1],
            [0, 0, 0, 1],
        ])
        self.X = np.concatenate([negative_sequences, positive_sequences])
        self.Y = np.array([0] * len(negative_sequences) + [1] * len(positive_sequences))

    def _make_bidirectional_model(self, rnn_class):
        model = Sequential()
        model.add(Embedding(3, 2))
        model.add(Bidirectional(rnn_class, input_dim=2, output_dim=4))
        model.add(Dense(4, 1))
        model.add(Activation("sigmoid"))
        model.compile("rmsprop", loss="binary_crossentropy")
        return model

    def test_bidirectional_lstm(self):
        model = self._make_bidirectional_model(LSTM)
        model.fit(self.X, self.Y, verbose=1, nb_epoch=1000)
        for pred, y in zip(model.predict(self.X), self.Y):
            self.assertEqual(pred > 0.5, y,
                "Wrong prediction by Bidirectional(LSTM), expected %s0.5 but got %s" % (
                    ">" if y else "<=",
                    pred))

if __name__ == '__main__':
    unittest.main()
