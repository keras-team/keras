import theano
import unittest
from numpy.testing import assert_allclose
import numpy as np
from keras.layers.recurrent import SimpleRNN
from mock import Mock

floatX = theano.config.floatX

__author__ = "Jeff Ye"


class TestSimpleRNN(unittest.TestCase):
    left_padding_data = np.array(
        [
            [  # batch 1
               [0], [1], [2], [3]
               ],
            [  # batch 2
               [0], [0], [1], [2]
               ]
        ], dtype=floatX)
    left_padding_mask = np.array(  # n_sample x n_time
        [
            [  # batch 1
               0, 1, 1, 1
               ],
            [  # batch 2
               0, 0, 1, 1
               ]
        ], dtype=np.int32)

    def setUp(self):
        W = np.array([[1]], dtype=floatX)
        U = np.array([[1]], dtype=floatX)
        b = np.array([0], dtype=floatX)
        weights = [W, U, b]
        self.forward = SimpleRNN(output_dim=1, activation='linear', weights=weights)
        self.backward = SimpleRNN(output_dim=1, activation='linear', weights=weights)

        previous = Mock()
        previous.nb_input = 1
        previous.nb_output = 1
        previous.output_shape = self.left_padding_data.shape
        previous.get_output_mask = Mock()
        self.previous = previous

    def test_left_padding(self):
        forward = self.forward
        forward.go_backwards = False
        forward.return_sequences = True
        self.previous.get_output.return_value = theano.shared(value=self.left_padding_data)
        self.previous.get_output_mask.return_value = theano.shared(value=self.left_padding_mask)
        forward.set_previous(self.previous)
        np.testing.assert_allclose(forward.get_output().eval(),
                                   np.array([
                                       [[0], [1], [3], [6]],
                                       [[0], [0], [1], [3]]]))

        backward = self.backward
        backward.go_backwards = True
        backward.return_sequences = True
        self.previous.get_output.return_value = theano.shared(value=self.left_padding_data)
        self.previous.get_output_mask.return_value = theano.shared(value=self.left_padding_mask)
        backward.set_previous(self.previous)
        np.testing.assert_allclose(backward.get_output().eval(),
                                   np.array([
                                       [[3], [5], [6], [0]],
                                       [[2], [3], [0], [0]]]))
