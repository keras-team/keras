import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test
from keras.utils.test_utils import keras_test
from keras.layers import recurrent
from keras.layers import embeddings
from keras.models import Sequential
from keras.models import Model
from keras.engine.topology import Input
from keras.layers.core import Masking
from keras import regularizers
from keras import backend as K

num_samples, timesteps, embedding_dim, units = 2, 5, 4, 3
embedding_num = 12


@keras_test
def rnn_test(f):
    """
    All the recurrent layers share the same interface,
    so we can run through them with a single function.
    """
    f = keras_test(f)
    return pytest.mark.parametrize('layer_class', [
        recurrent.SimpleRNN,
        recurrent.GRU,
        recurrent.LSTM
    ])(f)


@rnn_test
def test_return_sequences(layer_class):
    layer_test(layer_class,
               kwargs={'units': units,
                       'return_sequences': True},
               input_shape=(num_samples, timesteps, embedding_dim))


@rnn_test
def test_dynamic_behavior(layer_class):
    layer = layer_class(units, input_shape=(None, embedding_dim))
    model = Sequential()
    model.add(layer)
    model.compile('sgd', 'mse')
    x = np.random.random((num_samples, timesteps, embedding_dim))
    y = np.random.random((num_samples, units))
    model.train_on_batch(x, y)


@rnn_test
def test_stateful_invalid_use(layer_class):
    layer = layer_class(units,
                        stateful=True,
                        batch_input_shape=(num_samples, timesteps, embedding_dim))
    model = Sequential()
    model.add(layer)
    model.compile('sgd', 'mse')
    x = np.random.random((num_samples * 2, timesteps, embedding_dim))
    y = np.random.random((num_samples * 2, units))
    with pytest.raises(ValueError):
        model.fit(x, y)
    with pytest.raises(ValueError):
        model.predict(x, batch_size=num_samples + 1)


@rnn_test
def test_dropout(layer_class):
    layer_test(layer_class,
               kwargs={'units': units,
                       'dropout': 0.1,
                       'recurrent_dropout': 0.1},
               input_shape=(num_samples, timesteps, embedding_dim))
    # Test that dropout is not applied during testing
    x = np.random.random((num_samples, timesteps, embedding_dim))
    layer = layer_class(units, dropout=0.5, recurrent_dropout=0.5,
                        input_shape=(timesteps, embedding_dim))
    model = Sequential([layer])
    y1 = model.predict(x)
    y2 = model.predict(x)
    assert_allclose(y1, y2)


@rnn_test
def test_implementation_mode(layer_class):
    for mode in [0, 1, 2]:
        layer_test(layer_class,
                   kwargs={'units': units,
                           'implementation': mode},
                   input_shape=(num_samples, timesteps, embedding_dim))
    layer_test(layer_class,
               kwargs={'units': units,
                       'implementation': mode,
                       'dropout': 0.1,
                       'recurrent_dropout': 0.1},
               input_shape=(num_samples, timesteps, embedding_dim))


@rnn_test
def test_statefulness(layer_class):
    model = Sequential()
    model.add(embeddings.Embedding(embedding_num, embedding_dim,
                                   mask_zero=True,
                                   input_length=timesteps,
                                   batch_input_shape=(num_samples, timesteps)))
    layer = layer_class(units, return_sequences=False,
                        stateful=True,
                        weights=None)
    model.add(layer)
    model.compile(optimizer='sgd', loss='mse')
    out1 = model.predict(np.ones((num_samples, timesteps)))
    assert(out1.shape == (num_samples, units))

    # train once so that the states change
    model.train_on_batch(np.ones((num_samples, timesteps)),
                         np.ones((num_samples, units)))
    out2 = model.predict(np.ones((num_samples, timesteps)))

    # if the state is not reset, output should be different
    assert(out1.max() != out2.max())

    # check that output changes after states are reset
    # (even though the model itself didn't change)
    layer.reset_states()
    out3 = model.predict(np.ones((num_samples, timesteps)))
    assert(out2.max() != out3.max())

    # check that container-level reset_states() works
    model.reset_states()
    out4 = model.predict(np.ones((num_samples, timesteps)))
    assert_allclose(out3, out4, atol=1e-5)

    # check that the call to `predict` updated the states
    out5 = model.predict(np.ones((num_samples, timesteps)))
    assert(out4.max() != out5.max())

    # Check masking
    layer.reset_states()

    left_padded_input = np.ones((num_samples, timesteps))
    left_padded_input[0, :1] = 0
    left_padded_input[1, :2] = 0
    out6 = model.predict(left_padded_input)

    layer.reset_states()

    right_padded_input = np.ones((num_samples, timesteps))
    right_padded_input[0, -1:] = 0
    right_padded_input[1, -2:] = 0
    out7 = model.predict(right_padded_input)

    assert_allclose(out7, out6, atol=1e-5)


@rnn_test
def test_regularizer(layer_class):
    layer = layer_class(units, return_sequences=False, weights=None,
                        batch_input_shape=(num_samples, timesteps, embedding_dim),
                        kernel_regularizer=regularizers.l1(0.01),
                        recurrent_regularizer=regularizers.l1(0.01),
                        activity_regularizer='l1',
                        bias_regularizer='l2')
    layer.build((None, None, 2))
    assert len(layer.losses) == 3
    layer(K.variable(np.ones((2, 3, 2))))
    assert len(layer.losses) == 4


@keras_test
def test_masking_layer():
    ''' This test based on a previously failing issue here:
    https://github.com/fchollet/keras/issues/1567
    '''
    inputs = np.random.random((6, 3, 4))
    targets = np.abs(np.random.random((6, 3, 5)))
    targets /= targets.sum(axis=-1, keepdims=True)

    model = Sequential()
    model.add(Masking(input_shape=(3, 4)))
    model.add(recurrent.LSTM(units=5, return_sequences=True, unroll=False))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(inputs, targets, epochs=1, batch_size=100, verbose=1)

    model = Sequential()
    model.add(Masking(input_shape=(3, 4)))
    model.add(recurrent.LSTM(units=5, return_sequences=True, unroll=True))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(inputs, targets, epochs=1, batch_size=100, verbose=1)


@rnn_test
def test_from_config(layer_class):
    stateful_flags = (False, True)
    for stateful in stateful_flags:
        l1 = layer_class(units=1, stateful=stateful)
        l2 = layer_class.from_config(l1.get_config())
        assert l1.get_config() == l2.get_config()


@rnn_test
def test_specify_initial_state_keras_tensor(layer_class):
    num_states = 2 if layer_class is recurrent.LSTM else 1

    # Test with Keras tensor
    inputs = Input((timesteps, embedding_dim))
    initial_state = [Input((units,)) for _ in range(num_states)]
    layer = layer_class(units)
    if len(initial_state) == 1:
        output = layer(inputs, initial_state=initial_state[0])
    else:
        output = layer(inputs, initial_state=initial_state)
    assert initial_state[0] in layer.inbound_nodes[0].input_tensors

    model = Model([inputs] + initial_state, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    initial_state = [np.random.random((num_samples, units))
                     for _ in range(num_states)]
    targets = np.random.random((num_samples, units))
    model.fit([inputs] + initial_state, targets)


@rnn_test
def test_specify_initial_state_non_keras_tensor(layer_class):
    num_states = 2 if layer_class is recurrent.LSTM else 1

    # Test with non-Keras tensor
    inputs = Input((timesteps, embedding_dim))
    initial_state = [K.random_normal_variable((num_samples, units), 0, 1)
                     for _ in range(num_states)]
    layer = layer_class(units)
    output = layer(inputs, initial_state=initial_state)

    model = Model(inputs, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    targets = np.random.random((num_samples, units))
    model.fit(inputs, targets)


@rnn_test
def test_reset_states_with_values(layer_class):
    num_states = 2 if layer_class is recurrent.LSTM else 1

    layer = layer_class(units, stateful=True)
    layer.build((num_samples, timesteps, embedding_dim))
    layer.reset_states()
    assert len(layer.states) == num_states
    assert layer.states[0] is not None
    np.testing.assert_allclose(K.eval(layer.states[0]),
                               np.zeros(K.int_shape(layer.states[0])),
                               atol=1e-4)
    state_shapes = [K.int_shape(state) for state in layer.states]
    values = [np.ones(shape) for shape in state_shapes]
    if len(values) == 1:
        values = values[0]
    layer.reset_states(values)
    np.testing.assert_allclose(K.eval(layer.states[0]),
                               np.ones(K.int_shape(layer.states[0])),
                               atol=1e-4)

    # Test fit with invalid data
    with pytest.raises(ValueError):
        layer.reset_states([1] * (len(layer.states) + 1))


@rnn_test
def test_specify_state_with_masking(layer_class):
    ''' This test based on a previously failing issue here:
    https://github.com/fchollet/keras/issues/1567
    '''
    num_states = 2 if layer_class is recurrent.LSTM else 1

    inputs = Input((timesteps, embedding_dim))
    _ = Masking()(inputs)
    initial_state = [Input((units,)) for _ in range(num_states)]
    output = layer_class(units)(inputs, initial_state=initial_state)

    model = Model([inputs] + initial_state, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    initial_state = [np.random.random((num_samples, units))
                     for _ in range(num_states)]
    targets = np.random.random((num_samples, units))
    model.fit([inputs] + initial_state, targets)


@rnn_test
def test_return_state(layer_class):
    num_states = 2 if layer_class is recurrent.LSTM else 1

    inputs = Input(batch_shape=(num_samples, timesteps, embedding_dim))
    layer = layer_class(units, return_state=True, stateful=True)
    outputs = layer(inputs)
    output, state = outputs[0], outputs[1:]
    assert len(state) == num_states
    model = Model(inputs, state[0])

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    state = model.predict(inputs)
    np.testing.assert_allclose(K.eval(layer.states[0]), state, atol=1e-4)


@rnn_test
def test_state_reuse(layer_class):
    inputs = Input(batch_shape=(num_samples, timesteps, embedding_dim))
    layer = layer_class(units, return_state=True, return_sequences=True)
    outputs = layer(inputs)
    output, state = outputs[0], outputs[1:]
    output = layer_class(units)(output, initial_state=state)
    model = Model(inputs, output)

    inputs = np.random.random((num_samples, timesteps, embedding_dim))
    outputs = model.predict(inputs)


def test_unroll_true_throws_exception_with_one_timestep():
    model = Sequential()
    with pytest.raises(ValueError):
        model.add(recurrent.LSTM(units=5, batch_input_shape=(1, 1, 5), unroll=True))


if __name__ == '__main__':
    pytest.main([__file__])
