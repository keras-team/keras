import unittest
import numpy as np
from keras.models import Sequential
from keras.layers.core import Merge, Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from theano import function
from keras.constraints import unitnorm

class TestConcatenation(unittest.TestCase):

    def setUp(self):
        self.X1 = np.array([[1], [2]], dtype='int32')
        self.W1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype='float32')

    def test_unitnorm_constraint(self):
        lookup = Sequential()
        lookup.add(Embedding(3, 2, weights=[self.W1], W_constraint=unitnorm()))
        lookup.add(Flatten())
        lookup.add(Dense(2, 1))
        lookup.add(Activation('sigmoid'))
        lookup.compile(loss='binary_crossentropy', optimizer='sgd', class_mode='binary')
        lookup.train(self.X1, np.array([[1], [0]], dtype='int32'))
        norm = np.linalg.norm(lookup.params[0].get_value(), axis=1)
        self.assertTrue(np.allclose(norm, np.ones_like(norm).astype('float32')))

if __name__ == '__main__':
    unittest.main()