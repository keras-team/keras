import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.layers import recurrent
from keras import backend as K
from keras.models import Sequential

nb_samples, timesteps, input_dim, output_dim = 3, 3, 10, 5


def _runner(layer_class):
    """
    All the recurrent layers share the same interface,
    so we can run through them with a single function.
    """
    for ret_seq in [True, False]:
        layer = layer_class(output_dim, return_sequences=ret_seq,
                            weights=None, input_shape=(timesteps, input_dim))
        layer.input = K.variable(np.ones((nb_samples, timesteps, input_dim)))
        layer.get_config()

        for train in [True, False]:
            out = K.eval(layer.get_output(train))
            # Make sure the output has the desired shape
            if ret_seq:
                assert(out.shape == (nb_samples, timesteps, output_dim))
            else:
                assert(out.shape == (nb_samples, output_dim))

            mask = layer.get_output_mask(train)

    # check statefulness
    layer = layer_class(output_dim, return_sequences=False,
                        stateful=True,
                        weights=None,
                        batch_input_shape=(nb_samples, timesteps, input_dim))
    model = Sequential()
    model.add(layer)
    model.compile(optimizer='sgd', loss='mse')
    out1 = model.predict(np.ones((nb_samples, timesteps, input_dim)))
    assert(out1.shape == (nb_samples, output_dim))

    # train once so that the states change
    model.train_on_batch(np.ones((nb_samples, timesteps, input_dim)),
                         np.ones((nb_samples, output_dim)))
    out2 = model.predict(np.ones((nb_samples, timesteps, input_dim)))

    # if the state is not reset, output should be different
    assert(out1.max() != out2.max())

    # check that output changes after states are reset
    # (even though the model itself didn't change)
    layer.reset_states()
    out3 = model.predict(np.ones((nb_samples, timesteps, input_dim)))
    assert(out2.max() != out3.max())

    # check that container-level reset_states() works
    model.reset_states()
    out4 = model.predict(np.ones((nb_samples, timesteps, input_dim)))
    assert_allclose(out3, out4, atol=1e-5)

    # check that the call to `predict` updated the states
    out5 = model.predict(np.ones((nb_samples, timesteps, input_dim)))
    assert(out4.max() != out5.max())


def test_SimpleRNN():
    _runner(recurrent.SimpleRNN)


def test_GRU():
    _runner(recurrent.GRU)


def test_LSTM():
    _runner(recurrent.LSTM)


if __name__ == '__main__':
    pytest.main([__file__])
