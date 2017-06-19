import pytest
import numpy as np
from numpy.testing import assert_allclose
from keras.utils.test_utils import keras_test
from keras.layers import wrappers, Input
from keras.layers import core, convolutional, recurrent, embeddings
from keras.models import Sequential, Model, model_from_json
from keras import backend as K


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


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason='cntk does not support dropout yet')
def test_TimeDistributed_learning_phase():
    # test layers that need learning_phase to be set
    np.random.seed(1234)
    x = Input(shape=(3, 2))
    y = wrappers.TimeDistributed(core.Dropout(.5, seed=1234))(x, training=True)
    model = Model(x, y)
    y = model.predict(np.random.random((10, 3, 2)))

    if K.backend() == 'tensorflow':
        desired_y = np.asarray(
            [[[0.38303891, 0.],
              [0., 0.],
              [1.55995166, 0.]],
             [[0., 0.],
              [1.91627872, 0.],
              [0., 1.0019902]],
             [[1.36692584, 0.],
              [0., 0.],
              [0., 0.0275369]],
             [[1.54565322, 0.],
              [0., 1.2307924],
              [0., 0.]],
             [[0., 0.],
              [0.79440516, 1.57746029],
              [0., 0.]],
             [[1.73825479, 0.87234682],
              [1.60429525, 0.28753364],
              [1.40852189, 0.]],
             [[0.43758422, 0.],
              [0., 0.],
              [0., 0.]],
             [[0., 0.],
              [0., 0.],
              [0., 0.]],
             [[0.65933686, 1.00593364],
              [0.22378863, 0.],
              [0., 0.01352812]],
             [[0., 1.82424581],
              [0., 0.],
              [1.91760349, 1.58392823]]]
        )
    elif K.backend() == 'theano':
        desired_y = np.asarray(
            [[[0.38303891, 0.],
              [0., 1.57071722],
              [0., 0.54518521]],
             [[0., 1.60374439],
              [1.91627872, 0.],
              [0.71563452, 0.]],
             [[0., 1.42540407],
              [0., 1.12239242],
              [1.00616634, 0.0275369]],
             [[1.54565322, 0.],
              [0.72977197, 1.2307924],
              [0.15076248, 0.]],
             [[1.8662802, 0.],
              [0., 1.57746029],
              [0.63367224, 1.13619733]],
             [[1.73825479, 0.],
              [1.60429525, 0.28753364],
              [1.40852189, 0.]],
             [[0.43758422, 1.84973526],
              [0., 0.],
              [0., 0.]],
             [[0.09471056, 1.34976184],
              [0., 0.],
              [0., 0.]],
             [[0.65933686, 0.],
              [0., 0.],
              [0., 0.01352812]],
             [[1.23488343, 1.82424581],
              [0., 1.98416293],
              [1.91760349, 1.58392823]]])

    np.testing.assert_allclose(y, desired_y, atol=1e-1, rtol=1e-1)


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
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason='cntk does not support reverse yet')
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
        input = Input((timesteps, dim))
        output = wrappers.Bidirectional(rnn(output_dim, dropout=dropout_rate,
                                            recurrent_dropout=dropout_rate),
                                        merge_mode=mode)(input)
        model = Model(input, output)
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, epochs=1, batch_size=1)

        # Bidirectional and stateful
        input = Input(batch_shape=(1, timesteps, dim))
        output = wrappers.Bidirectional(rnn(output_dim, stateful=True), merge_mode=mode)(input)
        model = Model(input, output)
        model.compile(loss='mse', optimizer='sgd')
        model.fit(x, y, epochs=1, batch_size=1)


if __name__ == '__main__':
    pytest.main([__file__])
