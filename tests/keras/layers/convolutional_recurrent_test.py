import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import convolutional_recurrent, Input
from keras.utils.test_utils import layer_test
from keras import regularizers

num_row = 3
num_col = 3
filters = 2
num_samples = 1
input_channel = 2
input_num_row = 5
input_num_col = 5
sequence_len = 2


def test_convolutional_recurrent():

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

            # test for return state:
            x = Input(batch_shape=inputs.shape)
            kwargs = {'data_format': data_format,
                      'return_sequences': return_sequences,
                      'return_state': True,
                      'stateful': True,
                      'filters': filters,
                      'kernel_size': (num_row, num_col),
                      'padding': 'valid'}
            layer = convolutional_recurrent.ConvLSTM2D(**kwargs)
            layer.build(inputs.shape)
            outputs = layer(x)
            output, states = outputs[0], outputs[1:]
            assert len(states) == 2
            model = Model(x, states[0])
            state = model.predict(inputs)
            np.testing.assert_allclose(
                K.eval(layer.states[0]), state, atol=1e-4)

            # test for output shape:
            output = layer_test(convolutional_recurrent.ConvLSTM2D,
                                kwargs={'data_format': data_format,
                                        'return_sequences': return_sequences,
                                        'filters': filters,
                                        'kernel_size': (num_row, num_col),
                                        'padding': 'valid'},
                                input_shape=inputs.shape)


def test_convolutional_recurrent_statefulness():

    data_format = 'channels_last'
    return_sequences = False
    inputs = np.random.rand(num_samples, sequence_len,
                            input_num_row, input_num_col,
                            input_channel)
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

    # train once so that the states change
    model.train_on_batch(np.ones_like(inputs),
                         np.random.random(out1.shape))
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

    # cntk doesn't support eval convolution with static
    # variable, will enable it later
    if K.backend() != 'cntk':
        # check regularizers
        kwargs = {'data_format': data_format,
                  'return_sequences': return_sequences,
                  'kernel_size': (num_row, num_col),
                  'stateful': True,
                  'filters': filters,
                  'batch_input_shape': inputs.shape,
                  'kernel_regularizer': regularizers.L1L2(l1=0.01),
                  'recurrent_regularizer': regularizers.L1L2(l1=0.01),
                  'bias_regularizer': 'l2',
                  'activity_regularizer': 'l2',
                  'kernel_constraint': 'max_norm',
                  'recurrent_constraint': 'max_norm',
                  'bias_constraint': 'max_norm',
                  'padding': 'same'}

        layer = convolutional_recurrent.ConvLSTM2D(**kwargs)
        layer.build(inputs.shape)
        assert len(layer.losses) == 3
        assert layer.activity_regularizer
        output = layer(K.variable(np.ones(inputs.shape)))
        assert len(layer.losses) == 4
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

    # check state initialization
    layer = convolutional_recurrent.ConvLSTM2D(
        filters=filters, kernel_size=(num_row, num_col),
        data_format=data_format, return_sequences=return_sequences)
    layer.build(inputs.shape)
    x = Input(batch_shape=inputs.shape)
    initial_state = layer.get_initial_state(x)
    y = layer(x, initial_state=initial_state)
    model = Model(x, y)
    assert (model.predict(inputs).shape ==
            layer.compute_output_shape(inputs.shape))


if __name__ == '__main__':
    pytest.main([__file__])
