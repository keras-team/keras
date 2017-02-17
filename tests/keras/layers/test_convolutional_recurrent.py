import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.models import Sequential
from keras.layers import convolutional_recurrent
from keras.utils.test_utils import layer_test
from keras import regularizers


def test_convolutional_recurrent():
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
            inputs = np.random.rand(num_samples, sequence_len,
                                    input_channel,
                                    input_num_row, input_num_col)
        else:
            inputs = np.random.rand(num_samples, sequence_len,
                                    input_num_row, input_num_col,
                                    input_channel)

        for return_sequences in [True, False]:
            # test for output shape:
            output = layer_test(convolutional_recurrent.ConvLSTM2D,
                                kwargs={'data_format': data_format,
                                        'return_sequences': return_sequences,
                                        'filters': filters,
                                        'kernel_size': (num_row, num_col),
                                        'padding': 'valid'},
                                input_shape=inputs.shape)

            output_shape = [num_samples, input_num_row, input_num_col]

            if data_format == 'channels_first':
                output_shape.insert(1, filters)
            else:
                output_shape.insert(3, filters)

            if return_sequences:
                output_shape.insert(1, sequence_len)

            assert output.shape == tuple(output_shape)

            # No need to check following tests for both data formats
            if data_format == 'channels_first' or return_sequences:
                continue

            # Tests for statefulness
            model = Sequential()
            kwargs = {'data_format': data_format,
                      'return_sequences': return_sequences,
                      'filters': filters,
                      'kernel_size': (num_row, num_col),
                      'stateful': True,
                      'batch_input_shape': inputs.shape,
                      'padding': 'same'}
            layer = convolutional_recurrent.ConvLSTM2D(**kwargs)

            model.add(layer)
            model.compile(optimizer='sgd', loss='mse')
            out1 = model.predict(np.ones_like(inputs))
            assert(out1.shape == tuple(output_shape))

            # train once so that the states change
            model.train_on_batch(np.ones_like(inputs),
                                 np.ones_like(output))
            out2 = model.predict(np.ones_like(inputs))

            # if the state is not reset, output should be different
            assert(out1.max() != out2.max())

            # check that output changes after states are reset
            # (even though the model itself didn't change)
            layer.reset_states()
            out3 = model.predict(np.ones_like(inputs))
            assert(out2.max() != out3.max())

            # check that container-level reset_states() works
            model.reset_states()
            out4 = model.predict(np.ones_like(inputs))
            assert_allclose(out3, out4, atol=1e-5)

            # check that the call to `predict` updated the states
            out5 = model.predict(np.ones_like(inputs))
            assert(out4.max() != out5.max())

            # check regularizers
            kwargs = {'data_format': data_format,
                      'return_sequences': return_sequences,
                      'kernel_size': (num_row, num_col),
                      'stateful': True,
                      'batch_input_shape': inputs.shape,
                      'kernel_regularizer': regularizers.WeightRegularizer(l1=0.01),
                      'recurrent_regularizer': regularizers.WeightRegularizer(l1=0.01),
                      'bias_regularizer': 'l2',
                      'padding': 'same'}

            layer = convolutional_recurrent.ConvLSTM2D(**kwargs)
            layer.build(inputs.shape)
            assert len(layer.losses) == 3
            output = layer(K.variable(np.ones(inputs.shape)))
            assert len(layer.losses) == 3
            K.eval(output)

            # check dropout
            layer_test(convolutional_recurrent.ConvLSTM2D,
                       kwargs={'data_format': data_format,
                               'return_sequences': return_sequences,
                               'filters': filters,
                               'kernel_size': (num_row, num_col),
                               'padding': 'same',
                               'dropout': 0.1,
                               'recurrent_dropout': 0.1},
                       input_shape=inputs.shape)

if __name__ == '__main__':
    pytest.main([__file__])
