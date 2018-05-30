import pytest
import numpy as np
import copy
from numpy.testing import assert_allclose
from keras.utils.test_utils import keras_test
from keras.utils import CustomObjectScope
from keras.layers import wrappers, Input, Layer
from keras.layers import RNN
from keras import layers
from keras.models import Sequential, Model, model_from_json
from keras import backend as K
from keras.utils.generic_utils import object_list_uid, to_list


@keras_test
def test_TimeDistributed():
    # first, test with Dense layer
    model = Sequential()
    model.add(wrappers.TimeDistributed(layers.Dense(2), input_shape=(3, 4)))
    model.add(layers.Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 2)),
              epochs=1,
              batch_size=10)

    # test config
    model.get_config()

    # test when specifying a batch_input_shape
    test_input = np.random.random((1, 3, 4))
    test_output = model.predict(test_input)
    weights = model.layers[0].get_weights()

    reference = Sequential()
    reference.add(wrappers.TimeDistributed(layers.Dense(2),
                                           batch_input_shape=(1, 3, 4)))
    reference.add(layers.Activation('relu'))
    reference.compile(optimizer='rmsprop', loss='mse')
    reference.layers[0].set_weights(weights)

    reference_output = reference.predict(test_input)
    assert_allclose(test_output, reference_output, atol=1e-05)

    # test with Embedding
    model = Sequential()
    model.add(wrappers.TimeDistributed(layers.Embedding(5, 6),
                                       batch_input_shape=(10, 3, 4),
                                       dtype='int32'))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(np.random.randint(5, size=(10, 3, 4), dtype='int32'),
              np.random.random((10, 3, 4, 6)), epochs=1, batch_size=10)

    # compare to not using batch_input_shape
    test_input = np.random.randint(5, size=(10, 3, 4), dtype='int32')
    test_output = model.predict(test_input)
    weights = model.layers[0].get_weights()

    reference = Sequential()
    reference.add(wrappers.TimeDistributed(layers.Embedding(5, 6),
                                           input_shape=(3, 4), dtype='int32'))
    reference.compile(optimizer='rmsprop', loss='mse')
    reference.layers[0].set_weights(weights)

    reference_output = reference.predict(test_input)
    assert_allclose(test_output, reference_output, atol=1e-05)

    # test with Conv2D
    model = Sequential()
    model.add(wrappers.TimeDistributed(layers.Conv2D(5, (2, 2),
                                                     padding='same'),
                                       input_shape=(2, 4, 4, 3)))
    model.add(layers.Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(np.random.random((1, 2, 4, 4, 3)),
                         np.random.random((1, 2, 4, 4, 5)))

    model = model_from_json(model.to_json())
    model.summary()

    # test stacked layers
    model = Sequential()
    model.add(wrappers.TimeDistributed(layers.Dense(2), input_shape=(3, 4)))
    model.add(wrappers.TimeDistributed(layers.Dense(3)))
    model.add(layers.Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')

    model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 3)),
              epochs=1, batch_size=10)

    # test wrapping Sequential model
    model = Sequential()
    model.add(layers.Dense(3, input_dim=2))
    outer_model = Sequential()
    outer_model.add(wrappers.TimeDistributed(model, input_shape=(3, 2)))
    outer_model.compile(optimizer='rmsprop', loss='mse')
    outer_model.fit(np.random.random((10, 3, 2)), np.random.random((10, 3, 3)),
                    epochs=1, batch_size=10)

    # test with functional API
    x = Input(shape=(3, 2))
    y = wrappers.TimeDistributed(model)(x)
    outer_model = Model(x, y)
    outer_model.compile(optimizer='rmsprop', loss='mse')
    outer_model.fit(np.random.random((10, 3, 2)), np.random.random((10, 3, 3)),
                    epochs=1, batch_size=10)

    # test with BatchNormalization
    model = Sequential()
    model.add(wrappers.TimeDistributed(
        layers.BatchNormalization(center=True, scale=True),
        name='bn', input_shape=(10, 2)))
    model.compile(optimizer='rmsprop', loss='mse')
    # Assert that mean and variance are 0 and 1.
    td = model.layers[0]
    assert np.array_equal(td.get_weights()[2], np.array([0, 0]))
    assert np.array_equal(td.get_weights()[3], np.array([1, 1]))
    # Train
    model.train_on_batch(np.random.normal(loc=2, scale=2, size=(1, 10, 2)),
                         np.broadcast_to(np.array([0, 1]), (1, 10, 2)))
    # Assert that mean and variance changed.
    assert not np.array_equal(td.get_weights()[2], np.array([0, 0]))
    assert not np.array_equal(td.get_weights()[3], np.array([1, 1]))
    # Verify input_map has one mapping from inputs to reshaped inputs.
    uid = object_list_uid(model.inputs)
    assert len(td._input_map.keys()) == 1
    assert uid in td._input_map
    assert K.int_shape(td._input_map[uid]) == (None, 2)


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason='Flaky with CNTK backend')
def test_TimeDistributed_learning_phase():
    # test layers that need learning_phase to be set
    np.random.seed(1234)
    x = Input(shape=(3, 2))
    y = wrappers.TimeDistributed(layers.Dropout(.999))(x, training=True)
    model = Model(x, y)
    y = model.predict(np.random.random((10, 3, 2)))
    assert_allclose(np.mean(y), 0., atol=1e-1, rtol=1e-1)


@keras_test
def test_TimeDistributed_trainable():
    # test layers that need learning_phase to be set
    x = Input(shape=(3, 2))
    layer = wrappers.TimeDistributed(layers.BatchNormalization())
    _ = layer(x)
    assert len(layer.updates) == 2
    assert len(layer.trainable_weights) == 2
    layer.trainable = False
    assert len(layer.updates) == 0
    assert len(layer.trainable_weights) == 0
    layer.trainable = True
    assert len(layer.updates) == 2
    assert len(layer.trainable_weights) == 2


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason='Unknown timestamps for RNN not supported in CNTK.')
def test_TimeDistributed_with_masked_embedding_and_unspecified_shape():
    # test with unspecified shape and Embeddings with mask_zero
    model = Sequential()
    model.add(wrappers.TimeDistributed(layers.Embedding(5, 6, mask_zero=True),
                                       input_shape=(None, None)))  # N by t_1 by t_2 by 6
    model.add(wrappers.TimeDistributed(layers.SimpleRNN(7, return_sequences=True)))
    model.add(wrappers.TimeDistributed(layers.SimpleRNN(8, return_sequences=False)))
    model.add(layers.SimpleRNN(1, return_sequences=False))
    model.compile(optimizer='rmsprop', loss='mse')
    model_input = np.random.randint(low=1, high=5, size=(10, 3, 4), dtype='int32')
    for i in range(4):
        model_input[i, i:, i:] = 0
    model.fit(model_input,
              np.random.random((10, 1)), epochs=1, batch_size=10)
    mask_outputs = [model.layers[0].compute_mask(model.input)]
    for layer in model.layers[1:]:
        mask_outputs.append(layer.compute_mask(layer.input, mask_outputs[-1]))
    func = K.function([model.input], mask_outputs[:-1])
    mask_outputs_val = func([model_input])
    ref_mask_val_0 = model_input > 0         # embedding layer
    ref_mask_val_1 = ref_mask_val_0          # first RNN layer
    ref_mask_val_2 = np.any(ref_mask_val_1, axis=-1)     # second RNN layer
    ref_mask_val = [ref_mask_val_0, ref_mask_val_1, ref_mask_val_2]
    for i in range(3):
        assert np.array_equal(mask_outputs_val[i], ref_mask_val[i])
    assert mask_outputs[-1] is None  # final layer


@keras_test
def test_TimeDistributed_with_masking_layer():
    # test with Masking layer
    model = Sequential()
    model.add(wrappers.TimeDistributed(layers.Masking(mask_value=0.,),
                                       input_shape=(None, 4)))
    model.add(wrappers.TimeDistributed(layers.Dense(5)))
    model.compile(optimizer='rmsprop', loss='mse')
    model_input = np.random.randint(low=1, high=5, size=(10, 3, 4))
    for i in range(4):
        model_input[i, i:, :] = 0.
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(model_input,
              np.random.random((10, 3, 5)), epochs=1, batch_size=6)
    mask_outputs = [model.layers[0].compute_mask(model.input)]
    mask_outputs += [model.layers[1].compute_mask(model.layers[1].input, mask_outputs[-1])]
    func = K.function([model.input], mask_outputs)
    mask_outputs_val = func([model_input])
    assert np.array_equal(mask_outputs_val[0], np.any(model_input, axis=-1))
    assert np.array_equal(mask_outputs_val[1], np.any(model_input, axis=-1))


@keras_test
def test_regularizers():
    model = Sequential()
    model.add(wrappers.TimeDistributed(
        layers.Dense(2, kernel_regularizer='l1'), input_shape=(3, 4)))
    model.add(layers.Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    assert len(model.layers[0].layer.losses) == 1
    assert len(model.layers[0].losses) == 1
    assert len(model.layers[0].get_losses_for(None)) == 1
    assert len(model.losses) == 1

    model = Sequential()
    model.add(wrappers.TimeDistributed(
        layers.Dense(2, activity_regularizer='l1'), input_shape=(3, 4)))
    model.add(layers.Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    assert len(model.losses) == 1


@keras_test
def test_Bidirectional():
    rnn = layers.SimpleRNN
    samples = 2
    dim = 2
    timesteps = 2
    output_dim = 2
    dropout_rate = 0.2
    for mode in ['sum', 'concat']:
        x = np.random.random((samples, timesteps, dim))
        target_dim = 2 * output_dim if mode == 'concat' else output_dim
        y = np.random.random((samples, target_dim))

        # test with Sequential model
        model = Sequential()
        model.add(wrappers.Bidirectional(rnn(output_dim, dropout=dropout_rate,
                                             recurrent_dropout=dropout_rate),
                                         merge_mode=mode,
                                         input_shape=(timesteps, dim)))
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, epochs=1, batch_size=1)

        # test config
        model.get_config()
        model = model_from_json(model.to_json())
        model.summary()

        # test stacked bidirectional layers
        model = Sequential()
        model.add(wrappers.Bidirectional(rnn(output_dim,
                                             return_sequences=True),
                                         merge_mode=mode,
                                         input_shape=(timesteps, dim)))
        model.add(wrappers.Bidirectional(rnn(output_dim), merge_mode=mode))
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, epochs=1, batch_size=1)

        # test with functional API
        inputs = Input((timesteps, dim))
        outputs = wrappers.Bidirectional(rnn(output_dim, dropout=dropout_rate,
                                             recurrent_dropout=dropout_rate),
                                         merge_mode=mode)(inputs)
        model = Model(inputs, outputs)
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, epochs=1, batch_size=1)

        # Bidirectional and stateful
        inputs = Input(batch_shape=(1, timesteps, dim))
        outputs = wrappers.Bidirectional(rnn(output_dim, stateful=True),
                                         merge_mode=mode)(inputs)
        model = Model(inputs, outputs)
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, epochs=1, batch_size=1)


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason='Unknown timestamps not supported in CNTK.')
def test_Bidirectional_dynamic_timesteps():
    # test with functional API with dynamic length
    rnn = layers.SimpleRNN
    samples = 2
    dim = 2
    timesteps = 2
    output_dim = 2
    dropout_rate = 0.2
    for mode in ['sum', 'concat']:
        x = np.random.random((samples, timesteps, dim))
        target_dim = 2 * output_dim if mode == 'concat' else output_dim
        y = np.random.random((samples, target_dim))

        inputs = Input((None, dim))
        outputs = wrappers.Bidirectional(rnn(output_dim, dropout=dropout_rate,
                                             recurrent_dropout=dropout_rate),
                                         merge_mode=mode)(inputs)
        model = Model(inputs, outputs)
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, epochs=1, batch_size=1)


@keras_test
@pytest.mark.parametrize('merge_mode', ['sum', 'mul', 'ave', 'concat', None])
def test_Bidirectional_merged_value(merge_mode):
    rnn = layers.LSTM
    samples = 2
    dim = 5
    timesteps = 3
    units = 3
    X = [np.random.rand(samples, timesteps, dim)]

    if merge_mode == 'sum':
        merge_func = lambda y, y_rev: y + y_rev
    elif merge_mode == 'mul':
        merge_func = lambda y, y_rev: y * y_rev
    elif merge_mode == 'ave':
        merge_func = lambda y, y_rev: (y + y_rev) / 2
    elif merge_mode == 'concat':
        merge_func = lambda y, y_rev: np.concatenate((y, y_rev), axis=-1)
    else:
        merge_func = lambda y, y_rev: [y, y_rev]

    # basic case
    inputs = Input((timesteps, dim))
    layer = wrappers.Bidirectional(rnn(units, return_sequences=True), merge_mode=merge_mode)
    f_merged = K.function([inputs], to_list(layer(inputs)))
    f_forward = K.function([inputs], [layer.forward_layer.call(inputs)])
    f_backward = K.function([inputs], [K.reverse(layer.backward_layer.call(inputs), 1)])

    y_merged = f_merged(X)
    y_expected = to_list(merge_func(f_forward(X)[0], f_backward(X)[0]))
    assert len(y_merged) == len(y_expected)
    for x1, x2 in zip(y_merged, y_expected):
        assert_allclose(x1, x2, atol=1e-5)

    # test return_state
    inputs = Input((timesteps, dim))
    layer = wrappers.Bidirectional(rnn(units, return_state=True), merge_mode=merge_mode)
    f_merged = K.function([inputs], layer(inputs))
    f_forward = K.function([inputs], layer.forward_layer.call(inputs))
    f_backward = K.function([inputs], layer.backward_layer.call(inputs))
    n_states = len(layer.layer.states)

    y_merged = f_merged(X)
    y_forward = f_forward(X)
    y_backward = f_backward(X)
    y_expected = to_list(merge_func(y_forward[0], y_backward[0]))
    assert len(y_merged) == len(y_expected) + n_states * 2
    for x1, x2 in zip(y_merged, y_expected):
        assert_allclose(x1, x2, atol=1e-5)

    # test if the state of a BiRNN is the concatenation of the underlying RNNs
    y_merged = y_merged[-n_states * 2:]
    y_forward = y_forward[-n_states:]
    y_backward = y_backward[-n_states:]
    for state_birnn, state_inner in zip(y_merged, y_forward + y_backward):
        assert_allclose(state_birnn, state_inner, atol=1e-5)


@keras_test
@pytest.mark.skipif(K.backend() == 'theano', reason='Not supported.')
@pytest.mark.parametrize('merge_mode', ['sum', 'concat', None])
def test_Bidirectional_dropout(merge_mode):
    rnn = layers.LSTM
    samples = 2
    dim = 5
    timesteps = 3
    units = 3
    X = [np.random.rand(samples, timesteps, dim)]

    inputs = Input((timesteps, dim))
    wrapped = wrappers.Bidirectional(rnn(units, dropout=0.2, recurrent_dropout=0.2),
                                     merge_mode=merge_mode)
    outputs = to_list(wrapped(inputs, training=True))
    assert all(not getattr(x, '_uses_learning_phase') for x in outputs)

    inputs = Input((timesteps, dim))
    wrapped = wrappers.Bidirectional(rnn(units, dropout=0.2, return_state=True),
                                     merge_mode=merge_mode)
    outputs = to_list(wrapped(inputs))
    assert all(x._uses_learning_phase for x in outputs)

    model = Model(inputs, outputs)
    assert model.uses_learning_phase
    y1 = to_list(model.predict(X))
    y2 = to_list(model.predict(X))
    for x1, x2 in zip(y1, y2):
        assert_allclose(x1, x2, atol=1e-5)


@keras_test
def test_Bidirectional_state_reuse():
    rnn = layers.LSTM
    samples = 2
    dim = 5
    timesteps = 3
    units = 3

    input1 = Input((timesteps, dim))
    layer = wrappers.Bidirectional(rnn(units, return_state=True, return_sequences=True))
    state = layer(input1)[1:]

    # test passing invalid initial_state: passing a tensor
    input2 = Input((timesteps, dim))
    with pytest.raises(ValueError):
        output = wrappers.Bidirectional(rnn(units))(input2, initial_state=state[0])

    # test valid usage: passing a list
    output = wrappers.Bidirectional(rnn(units))(input2, initial_state=state)
    model = Model([input1, input2], output)
    assert len(model.layers) == 4
    assert isinstance(model.layers[-1].input, list)
    inputs = [np.random.rand(samples, timesteps, dim),
              np.random.rand(samples, timesteps, dim)]
    outputs = model.predict(inputs)


@keras_test
def test_Bidirectional_with_constants():
    class RNNCellWithConstants(Layer):
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
            h_input = K.dot(inputs, self.input_kernel)
            h_state = K.dot(prev_output, self.recurrent_kernel)
            h_const = K.dot(constant, self.constant_kernel)
            output = h_input + h_state + h_const
            return output, [output]

        def get_config(self):
            config = {'units': self.units}
            base_config = super(RNNCellWithConstants, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    # Test basic case.
    x = Input((5, 5))
    c = Input((3,))
    cell = RNNCellWithConstants(32)
    custom_objects = {'RNNCellWithConstants': RNNCellWithConstants}
    with CustomObjectScope(custom_objects):
        layer = wrappers.Bidirectional(RNN(cell))
    y = layer(x, constants=c)
    model = Model([x, c], y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(
        [np.zeros((6, 5, 5)), np.zeros((6, 3))],
        np.zeros((6, 64))
    )

    # Test basic case serialization.
    x_np = np.random.random((6, 5, 5))
    c_np = np.random.random((6, 3))
    y_np = model.predict([x_np, c_np])
    weights = model.get_weights()
    config = layer.get_config()
    with CustomObjectScope(custom_objects):
        layer = wrappers.Bidirectional.from_config(copy.deepcopy(config))
    y = layer(x, constants=c)
    model = Model([x, c], y)
    model.set_weights(weights)
    y_np_2 = model.predict([x_np, c_np])
    assert_allclose(y_np, y_np_2, atol=1e-4)

    # test flat list inputs
    with CustomObjectScope(custom_objects):
        layer = wrappers.Bidirectional.from_config(copy.deepcopy(config))
    y = layer([x, c])
    model = Model([x, c], y)
    model.set_weights(weights)
    y_np_3 = model.predict([x_np, c_np])
    assert_allclose(y_np, y_np_3, atol=1e-4)


@keras_test
def test_Bidirectional_with_constants_layer_passing_initial_state():
    class RNNCellWithConstants(Layer):
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
            h_input = K.dot(inputs, self.input_kernel)
            h_state = K.dot(prev_output, self.recurrent_kernel)
            h_const = K.dot(constant, self.constant_kernel)
            output = h_input + h_state + h_const
            return output, [output]

        def get_config(self):
            config = {'units': self.units}
            base_config = super(RNNCellWithConstants, self).get_config()
            return dict(list(base_config.items()) + list(config.items()))

    # Test basic case.
    x = Input((5, 5))
    c = Input((3,))
    s_for = Input((32,))
    s_bac = Input((32,))
    cell = RNNCellWithConstants(32)
    custom_objects = {'RNNCellWithConstants': RNNCellWithConstants}
    with CustomObjectScope(custom_objects):
        layer = wrappers.Bidirectional(RNN(cell))
    y = layer(x, initial_state=[s_for, s_bac], constants=c)
    model = Model([x, s_for, s_bac, c], y)
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(
        [np.zeros((6, 5, 5)), np.zeros((6, 32)), np.zeros((6, 32)), np.zeros((6, 3))],
        np.zeros((6, 64))
    )

    # Test basic case serialization.
    x_np = np.random.random((6, 5, 5))
    s_fw_np = np.random.random((6, 32))
    s_bk_np = np.random.random((6, 32))
    c_np = np.random.random((6, 3))
    y_np = model.predict([x_np, s_fw_np, s_bk_np, c_np])
    weights = model.get_weights()
    config = layer.get_config()
    with CustomObjectScope(custom_objects):
        layer = wrappers.Bidirectional.from_config(copy.deepcopy(config))
    y = layer(x, initial_state=[s_for, s_bac], constants=c)
    model = Model([x, s_for, s_bac, c], y)
    model.set_weights(weights)
    y_np_2 = model.predict([x_np, s_fw_np, s_bk_np, c_np])
    assert_allclose(y_np, y_np_2, atol=1e-4)

    # verify that state is used
    y_np_2_different_s = model.predict([x_np, s_fw_np + 10., s_bk_np + 10., c_np])
    with pytest.raises(AssertionError):
        assert_allclose(y_np, y_np_2_different_s, atol=1e-4)

    # test flat list inputs
    with CustomObjectScope(custom_objects):
        layer = wrappers.Bidirectional.from_config(copy.deepcopy(config))
    y = layer([x, s_for, s_bac, c])
    model = Model([x, s_for, s_bac, c], y)
    model.set_weights(weights)
    y_np_3 = model.predict([x_np, s_fw_np, s_bk_np, c_np])
    assert_allclose(y_np, y_np_3, atol=1e-4)


@keras_test
def test_Bidirectional_trainable():
    # test layers that need learning_phase to be set
    x = Input(shape=(3, 2))
    layer = wrappers.Bidirectional(layers.SimpleRNN(3))
    _ = layer(x)
    assert len(layer.trainable_weights) == 6
    layer.trainable = False
    assert len(layer.trainable_weights) == 0
    layer.trainable = True
    assert len(layer.trainable_weights) == 6


@keras_test
def test_Bidirectional_updates():
    x = Input(shape=(3, 2))
    layer = wrappers.Bidirectional(layers.SimpleRNN(3))
    assert len(layer.updates) == 0
    assert len(layer.get_updates_for(None)) == 0
    assert len(layer.get_updates_for(x)) == 0
    layer.forward_layer.add_update(0, inputs=x)
    layer.forward_layer.add_update(1, inputs=None)
    layer.backward_layer.add_update(0, inputs=x)
    layer.backward_layer.add_update(1, inputs=None)
    assert len(layer.updates) == 4
    assert len(layer.get_updates_for(None)) == 2
    assert len(layer.get_updates_for(x)) == 2


@keras_test
def test_Bidirectional_losses():
    x = Input(shape=(3, 2))
    layer = wrappers.Bidirectional(
        layers.SimpleRNN(3, kernel_regularizer='l1', bias_regularizer='l1'))
    _ = layer(x)
    assert len(layer.losses) == 4
    assert len(layer.get_losses_for(None)) == 4
    assert len(layer.get_losses_for(x)) == 0
    layer.forward_layer.add_loss(0, inputs=x)
    layer.forward_layer.add_loss(1, inputs=None)
    layer.backward_layer.add_loss(0, inputs=x)
    layer.backward_layer.add_loss(1, inputs=None)
    assert len(layer.losses) == 8
    assert len(layer.get_losses_for(None)) == 6
    assert len(layer.get_losses_for(x)) == 2


if __name__ == '__main__':
    pytest.main([__file__])
