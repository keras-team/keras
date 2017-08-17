import pytest
import numpy as np
from numpy.testing import assert_allclose
from keras.utils.test_utils import keras_test
from keras.layers import wrappers, Input
from keras.layers import core, convolutional, recurrent, embeddings, normalization
from keras.models import Sequential, Model, model_from_json
from keras import backend as K
from keras.engine.topology import _object_list_uid


@keras_test
def test_TimeDistributed():
    # first, test with Dense layer
    model = Sequential()
    model.add(wrappers.TimeDistributed(core.Dense(2), input_shape=(3, 4)))
    model.add(core.Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 2)), epochs=1, batch_size=10)

    # test config
    model.get_config()

    # test when specifying a batch_input_shape
    test_input = np.random.random((1, 3, 4))
    test_output = model.predict(test_input)
    weights = model.layers[0].get_weights()

    reference = Sequential()
    reference.add(wrappers.TimeDistributed(core.Dense(2), batch_input_shape=(1, 3, 4)))
    reference.add(core.Activation('relu'))
    reference.compile(optimizer='rmsprop', loss='mse')
    reference.layers[0].set_weights(weights)

    reference_output = reference.predict(test_input)
    assert_allclose(test_output, reference_output, atol=1e-05)

    # test with Embedding
    model = Sequential()
    model.add(wrappers.TimeDistributed(embeddings.Embedding(5, 6), batch_input_shape=(10, 3, 4), dtype='int32'))
    model.compile(optimizer='rmsprop', loss='mse')
    model.fit(np.random.randint(5, size=(10, 3, 4), dtype='int32'), np.random.random((10, 3, 4, 6)), epochs=1, batch_size=10)

    # compare to not using batch_input_shape
    test_input = np.random.randint(5, size=(10, 3, 4), dtype='int32')
    test_output = model.predict(test_input)
    weights = model.layers[0].get_weights()

    reference = Sequential()
    reference.add(wrappers.TimeDistributed(embeddings.Embedding(5, 6), input_shape=(3, 4), dtype='int32'))
    reference.compile(optimizer='rmsprop', loss='mse')
    reference.layers[0].set_weights(weights)

    reference_output = reference.predict(test_input)
    assert_allclose(test_output, reference_output, atol=1e-05)

    # test with Conv2D
    model = Sequential()
    model.add(wrappers.TimeDistributed(convolutional.Conv2D(5, (2, 2), padding='same'), input_shape=(2, 4, 4, 3)))
    model.add(core.Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    model.train_on_batch(np.random.random((1, 2, 4, 4, 3)), np.random.random((1, 2, 4, 4, 5)))

    model = model_from_json(model.to_json())
    model.summary()

    # test stacked layers
    model = Sequential()
    model.add(wrappers.TimeDistributed(core.Dense(2), input_shape=(3, 4)))
    model.add(wrappers.TimeDistributed(core.Dense(3)))
    model.add(core.Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')

    model.fit(np.random.random((10, 3, 4)), np.random.random((10, 3, 3)), epochs=1, batch_size=10)

    # test wrapping Sequential model
    model = Sequential()
    model.add(core.Dense(3, input_dim=2))
    outer_model = Sequential()
    outer_model.add(wrappers.TimeDistributed(model, input_shape=(3, 2)))
    outer_model.compile(optimizer='rmsprop', loss='mse')
    outer_model.fit(np.random.random((10, 3, 2)), np.random.random((10, 3, 3)), epochs=1, batch_size=10)

    # test with functional API
    x = Input(shape=(3, 2))
    y = wrappers.TimeDistributed(model)(x)
    outer_model = Model(x, y)
    outer_model.compile(optimizer='rmsprop', loss='mse')
    outer_model.fit(np.random.random((10, 3, 2)), np.random.random((10, 3, 3)), epochs=1, batch_size=10)

    # test with BatchNormalization
    model = Sequential()
    model.add(wrappers.TimeDistributed(normalization.BatchNormalization(center=True, scale=True),
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
    uid = _object_list_uid(model.inputs)
    assert len(td._input_map.keys()) == 1
    assert uid in td._input_map
    assert K.int_shape(td._input_map[uid]) == (None, 2)


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason='cntk does not support dropout yet')
def test_TimeDistributed_learning_phase():
    # test layers that need learning_phase to be set
    np.random.seed(1234)
    x = Input(shape=(3, 2))
    y = wrappers.TimeDistributed(core.Dropout(.999))(x, training=True)
    model = Model(x, y)
    y = model.predict(np.random.random((10, 3, 2)))
    assert_allclose(np.mean(y), 0., atol=1e-1, rtol=1e-1)


@keras_test
def test_regularizers():
    model = Sequential()
    model.add(wrappers.TimeDistributed(
        core.Dense(2, kernel_regularizer='l1'), input_shape=(3, 4)))
    model.add(core.Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    assert len(model.layers[0].layer.losses) == 1
    assert len(model.layers[0].losses) == 1
    assert len(model.layers[0].get_losses_for(None)) == 1
    assert len(model.losses) == 1

    model = Sequential()
    model.add(wrappers.TimeDistributed(
        core.Dense(2, activity_regularizer='l1'), input_shape=(3, 4)))
    model.add(core.Activation('relu'))
    model.compile(optimizer='rmsprop', loss='mse')
    assert len(model.losses) == 1


@keras_test
def test_Bidirectional():
    rnn = recurrent.SimpleRNN
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
                                         merge_mode=mode, input_shape=(timesteps, dim)))
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, epochs=1, batch_size=1)

        # test config
        model.get_config()
        model = model_from_json(model.to_json())
        model.summary()

        # test stacked bidirectional layers
        model = Sequential()
        model.add(wrappers.Bidirectional(rnn(output_dim, return_sequences=True),
                                         merge_mode=mode, input_shape=(timesteps, dim)))
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
        outputs = wrappers.Bidirectional(rnn(output_dim, stateful=True), merge_mode=mode)(inputs)
        model = Model(inputs, outputs)
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, epochs=1, batch_size=1)


if __name__ == '__main__':
    pytest.main([__file__])
