"""Tests for trainer functionality under tf implementation."""

import numpy as np
import tensorflow as tf

from keras.src import testing
from keras.src.backend.tensorflow.trainer import convert_to_np_if_not_ragged


class TrainerTest(testing.TestCase):
    def test_convert_to_np_if_not_ragged__ragged_input_should_return_ragged(self):
        rg_input = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
        output = convert_to_np_if_not_ragged(rg_input)
        self.assertIsInstance(output, tf.RaggedTensor)

    def test_convert_to_np_if_not_ragged__sparse_input_should_return_sparse(self):
        sp_input = tf.sparse.SparseTensor(indices=[[0, 3], [2, 4]],
                                          values=[10, 20],
                                          dense_shape=[3, 10])
        output = convert_to_np_if_not_ragged(sp_input)
        self.assertIsInstance(output, tf.SparseTensor)

    def test_convert_to_np_if_not_ragged__tftensor_input_should_return_numpy(self):
        tf_input = tf.constant([[3, 1, 4, 1], [5, 9, 2, 6]])
        output = convert_to_np_if_not_ragged(tf_input)
        self.assertIsInstance(output, np.ndarray)
