import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test
from keras.layers import recurrent, embeddings
from keras.models import Sequential
from keras.layers.core import Masking
from keras import regularizers

from keras import backend as K

nb_samples, timesteps, embedding_dim, output_dim = 3, 5, 10, 5
embedding_num = 12


def _runner(layer_class):
    """
    All the recurrent layers share the same interface,
    so we can run through them with a single function.
    """
    # check return_sequences
    layer_test(layer_class,
               kwargs={'output_dim': output_dim,
                       'return_sequences': True},
               input_shape=(3, 2, 3))

    # check dropout
    layer_test(layer_class,
               kwargs={'output_dim': output_dim,
                       'dropout_U': 0.1,
                       'dropout_W': 0.1},
               input_shape=(3, 2, 3))

    # check implementation modes
    for mode in ['cpu', 'mem', 'gpu']:
        layer_test(layer_class,
                   kwargs={'output_dim': output_dim,
                           'consume_less': mode},
                   input_shape=(3, 2, 3))

    # check statefulness
    model = Sequential()
    model.add(embeddings.Embedding(embedding_num, embedding_dim,
                                   mask_zero=True,
                                   input_length=timesteps,
                                   batch_input_shape=(nb_samples, timesteps)))
    layer = layer_class(output_dim, return_sequences=False,
                        stateful=True,
                        weights=None)
    model.add(layer)
    model.compile(optimizer='sgd', loss='mse')
    out1 = model.predict(np.ones((nb_samples, timesteps)))
    assert(out1.shape == (nb_samples, output_dim))

    # train once so that the states change
    model.train_on_batch(np.ones((nb_samples, timesteps)),
                         np.ones((nb_samples, output_dim)))
    out2 = model.predict(np.ones((nb_samples, timesteps)))

    # if the state is not reset, output should be different
    assert(out1.max() != out2.max())

    # check that output changes after states are reset
    # (even though the model itself didn't change)
    layer.reset_states()
    out3 = model.predict(np.ones((nb_samples, timesteps)))
    assert(out2.max() != out3.max())

    # check that container-level reset_states() works
    model.reset_states()
    out4 = model.predict(np.ones((nb_samples, timesteps)))
    assert_allclose(out3, out4, atol=1e-5)

    # check that the call to `predict` updated the states
    out5 = model.predict(np.ones((nb_samples, timesteps)))
    assert(out4.max() != out5.max())

    # Check masking
    layer.reset_states()

    left_padded_input = np.ones((nb_samples, timesteps))
    left_padded_input[0, :1] = 0
    left_padded_input[1, :2] = 0
    left_padded_input[2, :3] = 0
    out6 = model.predict(left_padded_input)

    layer.reset_states()

    right_padded_input = np.ones((nb_samples, timesteps))
    right_padded_input[0, -1:] = 0
    right_padded_input[1, -2:] = 0
    right_padded_input[2, -3:] = 0
    out7 = model.predict(right_padded_input)

    assert_allclose(out7, out6, atol=1e-5)

    # check regularizers
    layer = layer_class(output_dim, return_sequences=False, weights=None,
                        batch_input_shape=(nb_samples, timesteps, embedding_dim),
                        W_regularizer=regularizers.WeightRegularizer(l1=0.01),
                        U_regularizer=regularizers.WeightRegularizer(l1=0.01),
                        b_regularizer='l2')
    shape = (nb_samples, timesteps, embedding_dim)
    layer.set_input(K.variable(np.ones(shape)),
                    shape=shape)
    K.eval(layer.output)


def test_SimpleRNN():
    _runner(recurrent.SimpleRNN)


def test_GRU():
    _runner(recurrent.GRU)


def test_LSTM():
    _runner(recurrent.LSTM)


def test_masking_layer():
    ''' This test based on a previously failing issue here:
    https://github.com/fchollet/keras/issues/1567

    '''
    model = Sequential()
    model.add(Masking(input_shape=(3, 4)))
    model.add(recurrent.LSTM(output_dim=5, return_sequences=True))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    I = np.random.random((6, 3, 4))
    V = np.abs(np.random.random((6, 3, 5)))
    V /= V.sum(axis=-1, keepdims=True)
    model.fit(I, V, nb_epoch=1, batch_size=100, verbose=1)


if __name__ == '__main__':
    pytest.main([__file__])
