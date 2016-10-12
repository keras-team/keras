import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.models import Sequential
from keras.layers import recurrent_convolutional
from keras.utils.test_utils import layer_test


def test_recurrent_convolutional():
    # First test for ouptput shape:
    nb_row = 4
    nb_col = 4
    nb_filter = 20
    nb_samples = 5
    input_channel = 3
    input_nb_row = 30
    input_nb_col = 30
    sequence_len = 10
    for dim_ordering in ['th', 'tf']:

        if dim_ordering == 'th':
            input = np.random.rand(nb_samples, sequence_len,
                                   input_channel,
                                   input_nb_row, input_nb_col)
        else:  # tf
            input = np.random.rand(nb_samples, sequence_len,
                                   input_nb_row, input_nb_col,
                                   input_channel)

        for return_sequences in [True, False]:
            output = layer_test(recurrent_convolutional.LSTMConv2D,
                                kwargs={'dim_ordering': dim_ordering,
                                        'return_sequences': return_sequences,
                                        'nb_filter': nb_filter,
                                        'nb_row': nb_row,
                                        'nb_col': nb_col,
                                        'border_mode': "same"},
                                input_shape=input.shape)

            output_shape = [nb_samples, input_nb_row, input_nb_col]

            if dim_ordering == 'th':
                output_shape.insert(1, nb_filter)
            else:
                output_shape.insert(3, nb_filter)

            if return_sequences:
                output_shape.insert(1, sequence_len)

            assert output.shape == tuple(output_shape)

if __name__ == '__main__':
    pytest.main([__file__])
