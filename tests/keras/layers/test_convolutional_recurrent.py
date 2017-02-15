import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.models import Sequential
from keras.layers import convolutional_recurrent
from keras.utils.test_utils import layer_test
from keras import regularizers


def test_recurrent_convolutional():
    num_row = 3
    num_col = 3
    filters = 5
    num_samples = 2
    input_channel = 2
    input_num_row = 5
    input_num_col = 5
    sequence_len = 2
    for data_format in ['channels_first', 'channels_last']:

        if data_format == 'channels_first':
            input = np.random.rand(num_samples, sequence_len,
                                   input_channel,
                                   input_num_row, input_num_col)
        else:  # tf
            input = np.random.rand(num_samples, sequence_len,
                                   input_num_row, input_num_col,
                                   input_channel)

        for return_sequences in [True, False]:
            # test for ouptput shape:
            output = layer_test(convolutional_recurrent.ConvLSTM2D,
                                kwargs={'data_format': data_format,
                                        'return_sequences': return_sequences,
                                        'filters': filters,
                                        'num_row': num_row,
                                        'num_col': num_col,
                                        'border_mode': "same"},
                                input_shape=input.shape)

            output_shape = [num_samples, input_num_row, input_num_col]

            if data_format == 'channels_first':
                output_shape.insert(1, filters)
            else:
                output_shape.insert(3, filters)

            if return_sequences:
                output_shape.insert(1, sequence_len)

            assert output.shape == tuple(output_shape)

            # No need to check statefulness for both
            if data_format == 'channels_first' or return_sequences:
                continue

            # Tests for statefulness
            model = Sequential()
            kwargs = {'data_format': data_format,
                      'return_sequences': return_sequences,
                      'filters': filters,
                      'num_row': num_row,
                      'num_col': num_col,
                      'stateful': True,
                      'batch_input_shape': input.shape,
                      'border_mode': "same"}
            layer = convolutional_recurrent.ConvLSTM2D(**kwargs)

            model.add(layer)
            model.compile(optimizer='sgd', loss='mse')
            out1 = model.predict(np.ones_like(input))
            assert(out1.shape == tuple(output_shape))

            # train once so that the states change
            model.train_on_batch(np.ones_like(input),
                                 np.ones_like(output))
            out2 = model.predict(np.ones_like(input))

            # if the state is not reset, output should be different
            assert(out1.max() != out2.max())

            # check that output changes after states are reset
            # (even though the model itself didn't change)
            layer.reset_states()
            out3 = model.predict(np.ones_like(input))
            assert(out2.max() != out3.max())

            # check that container-level reset_states() works
            model.reset_states()
            out4 = model.predict(np.ones_like(input))
            assert_allclose(out3, out4, atol=1e-5)

            # check that the call to `predict` updated the states
            out5 = model.predict(np.ones_like(input))
            assert(out4.max() != out5.max())

            # check regularizers
            kwargs = {'data_format': data_format,
                      'return_sequences': return_sequences,
                      'filters': filters,
                      'num_row': num_row,
                      'num_col': num_col,
                      'stateful': True,
                      'batch_input_shape': input.shape,
                      'W_regularizer': regularizers.WeightRegularizer(l1=0.01),
                      'U_regularizer': regularizers.WeightRegularizer(l1=0.01),
                      'b_regularizer': 'l2',
                      'border_mode': "same"}

            layer = convolutional_recurrent.ConvLSTM2D(**kwargs)
            layer.build(input.shape)
            output = layer(K.variable(np.ones(input.shape)))
            K.eval(output)

            # check dropout
            layer_test(convolutional_recurrent.ConvLSTM2D,
                       kwargs={'data_format': data_format,
                               'return_sequences': return_sequences,
                               'filters': filters,
                               'num_row': num_row,
                               'num_col': num_col,
                               'border_mode': "same",
                               'dropout_W': 0.1,
                               'dropout_U': 0.1},
                       input_shape=input.shape)

if __name__ == '__main__':
    pytest.main([__file__])
