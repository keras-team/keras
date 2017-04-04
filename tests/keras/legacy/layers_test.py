import pytest

from keras.utils.test_utils import keras_test
from keras.utils.test_utils import layer_test
from keras.legacy import layers as legacy_layers
from keras import layers
from keras import models
from keras import regularizers
from keras import constraints
from keras import backend as K
import numpy as np


@keras_test
def test_highway():
    layer_test(legacy_layers.Highway,
               kwargs={},
               input_shape=(3, 2))

    layer_test(legacy_layers.Highway,
               kwargs={'W_regularizer': regularizers.l2(0.01),
                       'b_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.l2(0.01),
                       'W_constraint': constraints.MaxNorm(1),
                       'b_constraint': constraints.MaxNorm(1)},
               input_shape=(3, 2))


@keras_test
def test_maxout_dense():
    layer_test(legacy_layers.MaxoutDense,
               kwargs={'output_dim': 3},
               input_shape=(3, 2))

    layer_test(legacy_layers.MaxoutDense,
               kwargs={'output_dim': 3,
                       'W_regularizer': regularizers.l2(0.01),
                       'b_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.l2(0.01),
                       'W_constraint': constraints.MaxNorm(1),
                       'b_constraint': constraints.MaxNorm(1)},
               input_shape=(3, 2))


@keras_test
def test_merge():
    # test modes: 'sum', 'mul', 'concat', 'ave', 'cos', 'dot'.
    input_shapes = [(3, 2), (3, 2)]
    inputs = [np.random.random(shape) for shape in input_shapes]

    # test functional API
    for mode in ['sum', 'mul', 'concat', 'ave', 'max']:
        print(mode)
        input_a = layers.Input(shape=input_shapes[0][1:])
        input_b = layers.Input(shape=input_shapes[1][1:])
        merged = legacy_layers.merge([input_a, input_b], mode=mode)
        model = models.Model([input_a, input_b], merged)
        model.compile('rmsprop', 'mse')

        expected_output_shape = model.compute_output_shape(input_shapes)
        actual_output_shape = model.predict(inputs).shape
        assert expected_output_shape == actual_output_shape

        config = model.get_config()
        model = models.Model.from_config(config)
        model.compile('rmsprop', 'mse')

        # test Merge (#2460)
        merged = legacy_layers.Merge(mode=mode)([input_a, input_b])
        model = models.Model([input_a, input_b], merged)
        model.compile('rmsprop', 'mse')

        expected_output_shape = model.compute_output_shape(input_shapes)
        actual_output_shape = model.predict(inputs).shape
        assert expected_output_shape == actual_output_shape

    # test lambda with output_shape lambda
    input_a = layers.Input(shape=input_shapes[0][1:])
    input_b = layers.Input(shape=input_shapes[1][1:])
    merged = legacy_layers.merge(
        [input_a, input_b],
        mode=lambda tup: K.concatenate([tup[0], tup[1]]),
        output_shape=lambda tup: tup[0][:-1] + (tup[0][-1] + tup[1][-1],))
    model = models.Model([input_a, input_b], merged)
    expected_output_shape = model.compute_output_shape(input_shapes)
    actual_output_shape = model.predict(inputs).shape
    assert expected_output_shape == actual_output_shape

    config = model.get_config()
    model = models.Model.from_config(config)
    model.compile('rmsprop', 'mse')

    # test function with output_shape function
    def fn_mode(tup):
        x, y = tup
        return K.concatenate([x, y], axis=1)

    def fn_output_shape(tup):
        s1, s2 = tup
        return (s1[0], s1[1] + s2[1]) + s1[2:]

    input_a = layers.Input(shape=input_shapes[0][1:])
    input_b = layers.Input(shape=input_shapes[1][1:])
    merged = legacy_layers.merge([input_a, input_b],
                                 mode=fn_mode,
                                 output_shape=fn_output_shape)
    model = models.Model([input_a, input_b], merged)
    expected_output_shape = model.compute_output_shape(input_shapes)
    actual_output_shape = model.predict(inputs).shape
    assert expected_output_shape == actual_output_shape

    config = model.get_config()
    model = models.Model.from_config(config)
    model.compile('rmsprop', 'mse')

    # test function with output_mask function
    # time dimension is required for masking
    input_shapes = [(4, 3, 2), (4, 3, 2)]
    inputs = [np.random.random(shape) for shape in input_shapes]

    def fn_output_mask(tup):
        x_mask, y_mask = tup
        return K.concatenate([x_mask, y_mask])

    input_a = layers.Input(shape=input_shapes[0][1:])
    input_b = layers.Input(shape=input_shapes[1][1:])
    a = layers.Masking()(input_a)
    b = layers.Masking()(input_b)
    merged = legacy_layers.merge([a, b], mode=fn_mode, output_shape=fn_output_shape, output_mask=fn_output_mask)
    model = models.Model([input_a, input_b], merged)
    expected_output_shape = model.compute_output_shape(input_shapes)
    actual_output_shape = model.predict(inputs).shape
    assert expected_output_shape == actual_output_shape

    config = model.get_config()
    model = models.Model.from_config(config)
    model.compile('rmsprop', 'mse')

    mask_inputs = (np.zeros(input_shapes[0][:-1]), np.ones(input_shapes[1][:-1]))
    expected_mask_output = np.concatenate(mask_inputs, axis=-1)
    mask_input_placeholders = [K.placeholder(shape=input_shape[:-1]) for input_shape in input_shapes]
    mask_output = model.layers[-1]._output_mask(mask_input_placeholders)
    assert np.all(K.function(mask_input_placeholders, [mask_output])(mask_inputs)[0] == expected_mask_output)

    # test lambda with output_mask lambda
    input_a = layers.Input(shape=input_shapes[0][1:])
    input_b = layers.Input(shape=input_shapes[1][1:])
    a = layers.Masking()(input_a)
    b = layers.Masking()(input_b)
    merged = legacy_layers.merge(
        [a, b], mode=lambda tup: K.concatenate([tup[0], tup[1]], axis=1),
        output_shape=lambda tup: (tup[0][0], tup[0][1] + tup[1][1]) + tup[0][2:],
        output_mask=lambda tup: K.concatenate([tup[0], tup[1]]))
    model = models.Model([input_a, input_b], merged)
    expected_output_shape = model.compute_output_shape(input_shapes)
    actual_output_shape = model.predict(inputs).shape
    assert expected_output_shape == actual_output_shape

    config = model.get_config()
    model = models.Model.from_config(config)
    model.compile('rmsprop', 'mse')

    mask_output = model.layers[-1]._output_mask(mask_input_placeholders)
    assert np.all(K.function(mask_input_placeholders, [mask_output])(mask_inputs)[0] == expected_mask_output)

    # test with arguments
    input_shapes = [(3, 2), (3, 2)]
    inputs = [np.random.random(shape) for shape in input_shapes]

    def fn_mode(tup, a, b):
        x, y = tup
        return x * a + y * b

    input_a = layers.Input(shape=input_shapes[0][1:])
    input_b = layers.Input(shape=input_shapes[1][1:])
    merged = legacy_layers.merge([input_a, input_b], mode=fn_mode, output_shape=lambda s: s[0], arguments={'a': 0.7, 'b': 0.3})
    model = models.Model([input_a, input_b], merged)
    output = model.predict(inputs)

    config = model.get_config()
    model = models.Model.from_config(config)

    assert np.all(model.predict(inputs) == output)


@keras_test
def test_merge_mask_2d():
    rand = lambda *shape: np.asarray(np.random.random(shape) > 0.5, dtype='int32')

    # inputs
    input_a = layers.Input(shape=(3,))
    input_b = layers.Input(shape=(3,))

    # masks
    masked_a = layers.Masking(mask_value=0)(input_a)
    masked_b = layers.Masking(mask_value=0)(input_b)

    # three different types of merging
    merged_sum = legacy_layers.merge([masked_a, masked_b], mode='sum')
    merged_concat = legacy_layers.merge([masked_a, masked_b], mode='concat', concat_axis=1)
    merged_concat_mixed = legacy_layers.merge([masked_a, input_b], mode='concat', concat_axis=1)

    # test sum
    model_sum = models.Model([input_a, input_b], [merged_sum])
    model_sum.compile(loss='mse', optimizer='sgd')
    model_sum.fit([rand(2, 3), rand(2, 3)], [rand(2, 3)], epochs=1)

    # test concatenation
    model_concat = models.Model([input_a, input_b], [merged_concat])
    model_concat.compile(loss='mse', optimizer='sgd')
    model_concat.fit([rand(2, 3), rand(2, 3)], [rand(2, 6)], epochs=1)

    # test concatenation with masked and non-masked inputs
    model_concat = models.Model([input_a, input_b], [merged_concat_mixed])
    model_concat.compile(loss='mse', optimizer='sgd')
    model_concat.fit([rand(2, 3), rand(2, 3)], [rand(2, 6)], epochs=1)


@keras_test
def test_merge_mask_3d():
    rand = lambda *shape: np.asarray(np.random.random(shape) > 0.5, dtype='int32')

    # embeddings
    input_a = layers.Input(shape=(3,), dtype='int32')
    input_b = layers.Input(shape=(3,), dtype='int32')
    embedding = layers.Embedding(3, 4, mask_zero=True)
    embedding_a = embedding(input_a)
    embedding_b = embedding(input_b)

    # rnn
    rnn = layers.SimpleRNN(3, return_sequences=True)
    rnn_a = rnn(embedding_a)
    rnn_b = rnn(embedding_b)

    # concatenation
    merged_concat = legacy_layers.merge([rnn_a, rnn_b], mode='concat', concat_axis=-1)
    model = models.Model([input_a, input_b], [merged_concat])
    model.compile(loss='mse', optimizer='sgd')
    model.fit([rand(2, 3), rand(2, 3)], [rand(2, 3, 6)])


@keras_test
def test_sequential_regression():
    # start with a basic example of using a Sequential model
    # inside the functional API
    seq = models.Sequential()
    seq.add(layers.Dense(10, input_shape=(10,)))

    x = layers.Input(shape=(10,))
    y = seq(x)
    model = models.Model(x, y)
    model.compile('rmsprop', 'mse')
    weights = model.get_weights()

    # test serialization
    config = model.get_config()
    model = models.Model.from_config(config)
    model.compile('rmsprop', 'mse')
    model.set_weights(weights)

    # more advanced model with multiple branches

    branch_1 = models.Sequential(name='branch_1')
    branch_1.add(layers.Embedding(input_dim=100,
                                  output_dim=10,
                                  input_length=2,
                                  name='embed_1'))
    branch_1.add(layers.LSTM(32, name='lstm_1'))

    branch_2 = models.Sequential(name='branch_2')
    branch_2.add(layers.Dense(32, input_shape=(8,), name='dense_2'))

    branch_3 = models.Sequential(name='branch_3')
    branch_3.add(layers.Dense(32, input_shape=(6,), name='dense_3'))

    branch_1_2 = models.Sequential([legacy_layers.Merge([branch_1, branch_2], mode='concat')], name='branch_1_2')
    branch_1_2.add(layers.Dense(16, name='dense_1_2-0'))
    # test whether impromtu input_shape breaks the model
    branch_1_2.add(layers.Dense(16, input_shape=(16,), name='dense_1_2-1'))

    model = models.Sequential([legacy_layers.Merge([branch_1_2, branch_3], mode='concat')], name='final')
    model.add(layers.Dense(16, name='dense_final'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    x = (100 * np.random.random((100, 2))).astype('int32')
    y = np.random.random((100, 8))
    z = np.random.random((100, 6))
    labels = np.random.random((100, 16))
    model.fit([x, y, z], labels, epochs=1)

    # test if Sequential can be called in the functional API

    a = layers.Input(shape=(2,), dtype='int32')
    b = layers.Input(shape=(8,))
    c = layers.Input(shape=(6,))
    o = model([a, b, c])

    outer_model = models.Model([a, b, c], o)
    outer_model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    outer_model.fit([x, y, z], labels, epochs=1)

    # test serialization
    config = outer_model.get_config()
    outer_model = models.Model.from_config(config)
    outer_model.compile(optimizer='rmsprop',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
    outer_model.fit([x, y, z], labels, epochs=1)


if __name__ == '__main__':
    pytest.main([__file__])
