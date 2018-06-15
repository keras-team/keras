import pytest
import numpy as np
from numpy.testing import assert_allclose

import keras
from keras.utils.test_utils import layer_test
from keras.utils.test_utils import keras_test
from keras.layers import recurrent
from keras.layers import embeddings
from keras.models import Sequential
from keras.models import Model
from keras.engine import Input
from keras.layers import Masking
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


@keras_test
def rnn_cell_test(f):
    f = keras_test(f)
    return pytest.mark.parametrize('cell_class', [
        recurrent.SimpleRNNCell,
        recurrent.GRUCell,
        recurrent.LSTMCell
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
                        batch_input_shape=(num_samples,
                                           timesteps,
                                           embedding_dim))
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
@pytest.mark.skipif((K.backend() in ['theano']),
                    reason='Not supported.')
def test_dropout(layer_class):
    for unroll in [True, False]:
        layer_test(layer_class,
                   kwargs={'units': units,
                           'dropout': 0.1,
                           'recurrent_dropout': 0.1,
                           'unroll': unroll},
                   input_shape=(num_samples, timesteps, embedding_dim))

        # Test that dropout is applied during training
        x = K.ones((num_samples, timesteps, embedding_dim))
        layer = layer_class(units, dropout=0.5, recurrent_dropout=0.5,
                            input_shape=(timesteps, embedding_dim))
        y = layer(x)
        assert y._uses_learning_phase

        y = layer(x, training=True)
        assert not getattr(y, '_uses_learning_phase')

        # Test that dropout is not applied during testing
        x = np.random.random((num_samples, timesteps, embedding_dim))
        layer = layer_class(units, dropout=0.5, recurrent_dropout=0.5,
                            unroll=unroll,
                            input_shape=(timesteps, embedding_dim))
        model = Sequential([layer])
        assert model.uses_learning_phase
        y1 = model.predict(x)
        y2 = model.predict(x)
        assert_allclose(y1, y2)


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


@rnn_test
def test_masking_correctness(layer_class):
    # Check masking: output with left padding and right padding
    # should be the same.
    model = Sequential()
    model.add(embeddings.Embedding(embedding_num, embedding_dim,
                                   mask_zero=True,
                                   input_length=timesteps,
                                   batch_input_shape=(num_samples, timesteps)))
    layer = layer_class(units, return_sequences=False)
    model.add(layer)
    model.compile(optimizer='sgd', loss='mse')

    left_padded_input = np.ones((num_samples, timesteps))
    left_padded_input[0, :1] = 0
    left_padded_input[1, :2] = 0
    out6 = model.predict(left_padded_input)

    right_padded_input = np.ones((num_samples, timesteps))
    right_padded_input[0, -1:] = 0
    right_padded_input[1, -2:] = 0
    out7 = model.predict(right_padded_input)

    assert_allclose(out7, out6, atol=1e-5)


@rnn_test
def test_implementation_mode(layer_class):
    for mode in [1, 2]:
        # Without dropout
        layer_test(layer_class,
                   kwargs={'units': units,
                           'implementation': mode},
                   input_shape=(num_samples, timesteps, embedding_dim))
        # With dropout
        layer_test(layer_class,
                   kwargs={'units': units,
                           'implementation': mode,
                           'dropout': 0.1,
                           'recurrent_dropout': 0.1},
                   input_shape=(num_samples, timesteps, embedding_dim))
        # Without bias
        layer_test(layer_class,
                   kwargs={'units': units,
                           'implementation': mode,
                           'use_bias': False},
                   input_shape=(num_samples, timesteps, embedding_dim))


@rnn_test
def test_regularizer(layer_class):
    layer = layer_class(units, return_sequences=False, weights=None,
                        input_shape=(timesteps, embedding_dim),
                        kernel_regularizer=regularizers.l1(0.01),
                        recurrent_regularizer=regularizers.l1(0.01),
                        bias_regularizer='l2')
    layer.build((None, None, embedding_dim))
    assert len(layer.losses) == 3
    assert len(layer.cell.losses) == 3

    layer = layer_class(units, return_sequences=False, weights=None,
                        input_shape=(timesteps, embedding_dim),
                        activity_regularizer='l2')
    assert layer.activity_regularizer
    x = K.variable(np.ones((num_samples, timesteps, embedding_dim)))
    layer(x)
    assert len(layer.cell.get_losses_for(x)) == 0
    assert len(layer.get_losses_for(x)) == 1


@rnn_test
def test_trainability(layer_class):
    layer = layer_class(units)
    layer.build((None, None, embedding_dim))
    assert len(layer.weights) == 3
    assert len(layer.trainable_weights) == 3
    assert len(layer.non_trainable_weights) == 0
    layer.trainable = False
    assert len(layer.weights) == 3
    assert len(layer.trainable_weights) == 0
    assert len(layer.non_trainable_weights) == 3
    layer.trainable = True
    assert len(layer.weights) == 3
    assert len(layer.trainable_weights) == 3
    assert len(layer.non_trainable_weights) == 0


@keras_test
def test_masking_layer():
    ''' This test based on a previously failing issue here:
    https://github.com/keras-team/keras/issues/1567
    '''
    inputs = np.random.random((6, 3, 4))
    targets = np.abs(np.random.random((6, 3, 5)))
    targets /= targets.sum(axis=-1, keepdims=True)

    model = Sequential()
    model.add(Masking(input_shape=(3, 4)))
    model.add(recurrent.SimpleRNN(units=5, return_sequences=True, unroll=False))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(inputs, targets, epochs=1, batch_size=100, verbose=1)

    model = Sequential()
    model.add(Masking(input_shape=(3, 4)))
    model.add(recurrent.SimpleRNN(units=5, return_sequences=True, unroll=True))
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
    assert initial_state[0] in layer._inbound_nodes[0].input_tensors

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
def test_initial_states_as_other_inputs(layer_class):
    num_states = 2 if layer_class is recurrent.LSTM else 1

    # Test with Keras tensor
    main_inputs = Input((timesteps, embedding_dim))
    initial_state = [Input((units,)) for _ in range(num_states)]
    inputs = [main_inputs] + initial_state

    layer = layer_class(units)
    output = layer(inputs)
    assert initial_state[0] in layer._inbound_nodes[0].input_tensors

    model = Model(inputs, output)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    main_inputs = np.random.random((num_samples, timesteps, embedding_dim))
    initial_state = [np.random.random((num_samples, units))
                     for _ in range(num_states)]
    targets = np.random.random((num_samples, units))
    model.train_on_batch([main_inputs] + initial_state, targets)


@rnn_test
def test_specify_state_with_masking(layer_class):
    ''' This test based on a previously failing issue here:
    https://github.com/keras-team/keras/issues/1567
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


@rnn_test
@pytest.mark.skipif((K.backend() in ['theano']),
                    reason='Not supported.')
def test_state_reuse_with_dropout(layer_class):
    input1 = Input(batch_shape=(num_samples, timesteps, embedding_dim))
    layer = layer_class(units, return_state=True, return_sequences=True, dropout=0.2)
    state = layer(input1)[1:]

    input2 = Input(batch_shape=(num_samples, timesteps, embedding_dim))
    output = layer_class(units)(input2, initial_state=state)
    model = Model([input1, input2], output)

    inputs = [np.random.random((num_samples, timesteps, embedding_dim)),
              np.random.random((num_samples, timesteps, embedding_dim))]
    outputs = model.predict(inputs)


@keras_test
def test_minimal_rnn_cell_non_layer():

    class MinimalRNNCell(object):

        def __init__(self, units, input_dim):
            self.units = units
            self.state_size = units
            self.kernel = keras.backend.variable(
                np.random.random((input_dim, units)))

        def call(self, inputs, states):
            prev_output = states[0]
            output = keras.backend.dot(inputs, self.kernel) + prev_output
            return output, [output]

    # Basic test case.
    cell = MinimalRNNCell(32, 5)
    x = keras.Input((None, 5))
    layer = recurrent.RNN(cell)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test stacking.
    cells = [MinimalRNNCell(8, 5),
             MinimalRNNCell(32, 8),
             MinimalRNNCell(32, 32)]
    layer = recurrent.RNN(cells)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))


@keras_test
def test_minimal_rnn_cell_non_layer_multiple_states():

    class MinimalRNNCell(object):

        def __init__(self, units, input_dim):
            self.units = units
            self.state_size = (units, units)
            self.kernel = keras.backend.variable(
                np.random.random((input_dim, units)))

        def call(self, inputs, states):
            prev_output_1 = states[0]
            prev_output_2 = states[1]
            output = keras.backend.dot(inputs, self.kernel)
            output += prev_output_1
            output -= prev_output_2
            return output, [output * 2, output * 3]

    # Basic test case.
    cell = MinimalRNNCell(32, 5)
    x = keras.Input((None, 5))
    layer = recurrent.RNN(cell)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test stacking.
    cells = [MinimalRNNCell(8, 5),
             MinimalRNNCell(16, 8),
             MinimalRNNCell(32, 16)]
    layer = recurrent.RNN(cells)
    assert layer.cell.state_size == (32, 32, 16, 16, 8, 8)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))


@keras_test
def test_minimal_rnn_cell_layer():

    class MinimalRNNCell(keras.layers.Layer):

        def __init__(self, units, **kwargs):
            self.units = units
            self.state_size = units
            super(MinimalRNNCell, self).__init__(**kwargs)

        def build(self, input_shape):
            # no time axis in the input shape passed to RNN cells
            assert len(input_shape) == 2

            self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                          initializer='uniform',
                                          name='kernel')
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='uniform',
                name='recurrent_kernel')
            self.built = True

        def call(self, inputs, states):
            prev_output = states[0]
            h = keras.backend.dot(inputs, self.kernel)
            output = h + keras.backend.dot(prev_output, self.recurrent_kernel)
            return output, [output]

        def get_config(self):
            config = {'units': self.units}
            base_config = super(MinimalRNNCell, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    # Test basic case.
    x = keras.Input((None, 5))
    cell = MinimalRNNCell(32)
    layer = recurrent.RNN(cell)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test basic case serialization.
    x_np = np.random.random((6, 5, 5))
    y_np = model.predict(x_np)
    weights = model.get_weights()
    config = layer.get_config()
    with keras.utils.CustomObjectScope({'MinimalRNNCell': MinimalRNNCell}):
        layer = recurrent.RNN.from_config(config)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.set_weights(weights)
    y_np_2 = model.predict(x_np)
    assert_allclose(y_np, y_np_2, atol=1e-4)

    # Test stacking.
    cells = [MinimalRNNCell(8),
             MinimalRNNCell(12),
             MinimalRNNCell(32)]
    layer = recurrent.RNN(cells)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test stacked RNN serialization.
    x_np = np.random.random((6, 5, 5))
    y_np = model.predict(x_np)
    weights = model.get_weights()
    config = layer.get_config()
    with keras.utils.CustomObjectScope({'MinimalRNNCell': MinimalRNNCell}):
        layer = recurrent.RNN.from_config(config)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.set_weights(weights)
    y_np_2 = model.predict(x_np)
    assert_allclose(y_np, y_np_2, atol=1e-4)


@rnn_cell_test
def test_builtin_rnn_cell_layer(cell_class):
    # Test basic case.
    x = keras.Input((None, 5))
    cell = cell_class(32)
    layer = recurrent.RNN(cell)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test basic case serialization.
    x_np = np.random.random((6, 5, 5))
    y_np = model.predict(x_np)
    weights = model.get_weights()
    config = layer.get_config()
    layer = recurrent.RNN.from_config(config)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.set_weights(weights)
    y_np_2 = model.predict(x_np)
    assert_allclose(y_np, y_np_2, atol=1e-4)

    # Test stacking.
    cells = [cell_class(8),
             cell_class(12),
             cell_class(32)]
    layer = recurrent.RNN(cells)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(np.zeros((6, 5, 5)), np.zeros((6, 32)))

    # Test stacked RNN serialization.
    x_np = np.random.random((6, 5, 5))
    y_np = model.predict(x_np)
    weights = model.get_weights()
    config = layer.get_config()
    layer = recurrent.RNN.from_config(config)
    y = layer(x)
    model = keras.models.Model(x, y)
    model.set_weights(weights)
    y_np_2 = model.predict(x_np)
    assert_allclose(y_np, y_np_2, atol=1e-4)


@keras_test
@pytest.mark.skipif((K.backend() in ['cntk', 'theano']),
                    reason='Not supported.')
def test_stacked_rnn_dropout():
    cells = [recurrent.LSTMCell(3, dropout=0.1, recurrent_dropout=0.1),
             recurrent.LSTMCell(3, dropout=0.1, recurrent_dropout=0.1)]
    layer = recurrent.RNN(cells)

    x = keras.Input((None, 5))
    y = layer(x)
    model = keras.models.Model(x, y)
    model.compile('sgd', 'mse')
    x_np = np.random.random((6, 5, 5))
    y_np = np.random.random((6, 3))
    model.train_on_batch(x_np, y_np)


@keras_test
def test_stacked_rnn_attributes():
    cells = [recurrent.LSTMCell(3),
             recurrent.LSTMCell(3, kernel_regularizer='l2')]
    layer = recurrent.RNN(cells)
    layer.build((None, None, 5))

    # Test regularization losses
    assert len(layer.losses) == 1

    # Test weights
    assert len(layer.trainable_weights) == 6
    cells[0].trainable = False
    assert len(layer.trainable_weights) == 3
    assert len(layer.non_trainable_weights) == 3

    # Test `get_losses_for`
    x = keras.Input((None, 5))
    y = K.sum(x)
    cells[0].add_loss(y, inputs=x)
    assert layer.get_losses_for(x) == [y]


@keras_test
def test_stacked_rnn_compute_output_shape():
    cells = [recurrent.LSTMCell(3),
             recurrent.LSTMCell(6)]
    layer = recurrent.RNN(cells, return_state=True, return_sequences=True)
    output_shape = layer.compute_output_shape((None, timesteps, embedding_dim))
    expected_output_shape = [(None, timesteps, 6),
                             (None, 6),
                             (None, 6),
                             (None, 3),
                             (None, 3)]
    assert output_shape == expected_output_shape


@rnn_test
def test_batch_size_equal_one(layer_class):
    inputs = Input(batch_shape=(1, timesteps, embedding_dim))
    layer = layer_class(units)
    outputs = layer(inputs)
    model = Model(inputs, outputs)
    model.compile('sgd', 'mse')
    x = np.random.random((1, timesteps, embedding_dim))
    y = np.random.random((1, units))
    model.train_on_batch(x, y)


@keras_test
def test_rnn_cell_with_constants_layer():

    class RNNCellWithConstants(keras.layers.Layer):

        def __init__(self, units, **kwargs):
            self.units = units
            self.state_size = units
            super(RNNCellWithConstants, self).__init__(**kwargs)

        def build(self, input_shape):
            if not isinstance(input_shape, list):
                raise TypeError('expects constants shape')
            [input_shape, constant_shape] = input_shape
            # will (and should) raise if more than one constant passed

            self.input_kernel = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='uniform',
                name='kernel')
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='uniform',
                name='recurrent_kernel')
            self.constant_kernel = self.add_weight(
                shape=(constant_shape[-1], self.units),
                initializer='uniform',
                name='constant_kernel')
            self.built = True

        def call(self, inputs, states, constants):
            [prev_output] = states
            [constant] = constants
            h_input = keras.backend.dot(inputs, self.input_kernel)
            h_state = keras.backend.dot(prev_output, self.recurrent_kernel)
            h_const = keras.backend.dot(constant, self.constant_kernel)
            output = h_input + h_state + h_const
            return output, [output]

        def get_config(self):
            config = {'units': self.units}
            base_config = super(RNNCellWithConstants, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    # Test basic case.
    x = keras.Input((None, 5))
    c = keras.Input((3,))
    cell = RNNCellWithConstants(32)
    layer = recurrent.RNN(cell)
    y = layer(x, constants=c)
    model = keras.models.Model([x, c], y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(
        [np.zeros((6, 5, 5)), np.zeros((6, 3))],
        np.zeros((6, 32))
    )

    # Test basic case serialization.
    x_np = np.random.random((6, 5, 5))
    c_np = np.random.random((6, 3))
    y_np = model.predict([x_np, c_np])
    weights = model.get_weights()
    config = layer.get_config()
    custom_objects = {'RNNCellWithConstants': RNNCellWithConstants}
    with keras.utils.CustomObjectScope(custom_objects):
        layer = recurrent.RNN.from_config(config.copy())
    y = layer(x, constants=c)
    model = keras.models.Model([x, c], y)
    model.set_weights(weights)
    y_np_2 = model.predict([x_np, c_np])
    assert_allclose(y_np, y_np_2, atol=1e-4)

    # test flat list inputs
    with keras.utils.CustomObjectScope(custom_objects):
        layer = recurrent.RNN.from_config(config.copy())
    y = layer([x, c])
    model = keras.models.Model([x, c], y)
    model.set_weights(weights)
    y_np_3 = model.predict([x_np, c_np])
    assert_allclose(y_np, y_np_3, atol=1e-4)

    # Test stacking.
    cells = [recurrent.GRUCell(8),
             RNNCellWithConstants(12),
             RNNCellWithConstants(32)]
    layer = recurrent.RNN(cells)
    y = layer(x, constants=c)
    model = keras.models.Model([x, c], y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(
        [np.zeros((6, 5, 5)), np.zeros((6, 3))],
        np.zeros((6, 32))
    )

    # Test stacked RNN serialization.
    x_np = np.random.random((6, 5, 5))
    c_np = np.random.random((6, 3))
    y_np = model.predict([x_np, c_np])
    weights = model.get_weights()
    config = layer.get_config()
    with keras.utils.CustomObjectScope(custom_objects):
        layer = recurrent.RNN.from_config(config.copy())
    y = layer(x, constants=c)
    model = keras.models.Model([x, c], y)
    model.set_weights(weights)
    y_np_2 = model.predict([x_np, c_np])
    assert_allclose(y_np, y_np_2, atol=1e-4)


@keras_test
def test_rnn_cell_with_constants_layer_passing_initial_state():

    class RNNCellWithConstants(keras.layers.Layer):

        def __init__(self, units, **kwargs):
            self.units = units
            self.state_size = units
            super(RNNCellWithConstants, self).__init__(**kwargs)

        def build(self, input_shape):
            if not isinstance(input_shape, list):
                raise TypeError('expects constants shape')
            [input_shape, constant_shape] = input_shape
            # will (and should) raise if more than one constant passed

            self.input_kernel = self.add_weight(
                shape=(input_shape[-1], self.units),
                initializer='uniform',
                name='kernel')
            self.recurrent_kernel = self.add_weight(
                shape=(self.units, self.units),
                initializer='uniform',
                name='recurrent_kernel')
            self.constant_kernel = self.add_weight(
                shape=(constant_shape[-1], self.units),
                initializer='uniform',
                name='constant_kernel')
            self.built = True

        def call(self, inputs, states, constants):
            [prev_output] = states
            [constant] = constants
            h_input = keras.backend.dot(inputs, self.input_kernel)
            h_state = keras.backend.dot(prev_output, self.recurrent_kernel)
            h_const = keras.backend.dot(constant, self.constant_kernel)
            output = h_input + h_state + h_const
            return output, [output]

        def get_config(self):
            config = {'units': self.units}
            base_config = super(RNNCellWithConstants, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    # Test basic case.
    x = keras.Input((None, 5))
    c = keras.Input((3,))
    s = keras.Input((32,))
    cell = RNNCellWithConstants(32)
    layer = recurrent.RNN(cell)
    y = layer(x, initial_state=s, constants=c)
    model = keras.models.Model([x, s, c], y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(
        [np.zeros((6, 5, 5)), np.zeros((6, 32)), np.zeros((6, 3))],
        np.zeros((6, 32))
    )

    # Test basic case serialization.
    x_np = np.random.random((6, 5, 5))
    s_np = np.random.random((6, 32))
    c_np = np.random.random((6, 3))
    y_np = model.predict([x_np, s_np, c_np])
    weights = model.get_weights()
    config = layer.get_config()
    custom_objects = {'RNNCellWithConstants': RNNCellWithConstants}
    with keras.utils.CustomObjectScope(custom_objects):
        layer = recurrent.RNN.from_config(config.copy())
    y = layer(x, initial_state=s, constants=c)
    model = keras.models.Model([x, s, c], y)
    model.set_weights(weights)
    y_np_2 = model.predict([x_np, s_np, c_np])
    assert_allclose(y_np, y_np_2, atol=1e-4)

    # verify that state is used
    y_np_2_different_s = model.predict([x_np, s_np + 10., c_np])
    with pytest.raises(AssertionError):
        assert_allclose(y_np, y_np_2_different_s, atol=1e-4)

    # test flat list inputs
    with keras.utils.CustomObjectScope(custom_objects):
        layer = recurrent.RNN.from_config(config.copy())
    y = layer([x, s, c])
    model = keras.models.Model([x, s, c], y)
    model.set_weights(weights)
    y_np_3 = model.predict([x_np, s_np, c_np])
    assert_allclose(y_np, y_np_3, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__])
