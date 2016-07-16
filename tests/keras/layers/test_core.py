import pytest
import numpy as np

from keras import backend as K
from keras.layers import core
from keras.utils.test_utils import layer_test


def test_masking():
    layer_test(core.Masking,
               kwargs={},
               input_shape=(3, 2, 3))


def test_merge():
    from keras.layers import Input, merge, Merge
    from keras.models import Model

    # test modes: 'sum', 'mul', 'concat', 'ave', 'cos', 'dot'.
    input_shapes = [(3, 2), (3, 2)]
    inputs = [np.random.random(shape) for shape in input_shapes]

    # test functional API
    for mode in ['sum', 'mul', 'concat', 'ave', 'max']:
        print(mode)
        input_a = Input(shape=input_shapes[0][1:])
        input_b = Input(shape=input_shapes[1][1:])
        merged = merge([input_a, input_b], mode=mode)
        model = Model([input_a, input_b], merged)
        model.compile('rmsprop', 'mse')

        expected_output_shape = model.get_output_shape_for(input_shapes)
        actual_output_shape = model.predict(inputs).shape
        assert expected_output_shape == actual_output_shape

        config = model.get_config()
        model = Model.from_config(config)
        model.compile('rmsprop', 'mse')

        # test Merge (#2460)
        merged = Merge(mode=mode)([input_a, input_b])
        model = Model([input_a, input_b], merged)
        model.compile('rmsprop', 'mse')

        expected_output_shape = model.get_output_shape_for(input_shapes)
        actual_output_shape = model.predict(inputs).shape
        assert expected_output_shape == actual_output_shape

    # test lambda with output_shape lambda
    input_a = Input(shape=input_shapes[0][1:])
    input_b = Input(shape=input_shapes[1][1:])
    merged = merge([input_a, input_b],
                   mode=lambda tup: K.concatenate([tup[0], tup[1]]),
                   output_shape=lambda tup: (tup[0][:-1],) + (tup[0][-1] + tup[1][-1],))
    expected_output_shape = model.get_output_shape_for(input_shapes)
    actual_output_shape = model.predict(inputs).shape
    assert expected_output_shape == actual_output_shape

    config = model.get_config()
    model = Model.from_config(config)
    model.compile('rmsprop', 'mse')

    # test function with output_shape function
    def fn_mode(tup):
        x, y = tup
        return K.concatenate([x, y])

    def fn_output_shape(tup):
        s1, s2 = tup
        return (s1[:-1],) + (s1[-1] + s2[-1],)

    input_a = Input(shape=input_shapes[0][1:])
    input_b = Input(shape=input_shapes[1][1:])
    merged = merge([input_a, input_b],
                   mode=fn_mode,
                   output_shape=fn_output_shape)
    expected_output_shape = model.get_output_shape_for(input_shapes)
    actual_output_shape = model.predict(inputs).shape
    assert expected_output_shape == actual_output_shape

    config = model.get_config()
    model = Model.from_config(config)
    model.compile('rmsprop', 'mse')


def test_merge_mask_2d():
    from keras.layers import Input, merge, Masking
    from keras.models import Model

    rand = lambda *shape: np.asarray(np.random.random(shape) > 0.5, dtype='int32')

    # inputs
    input_a = Input(shape=(3,))
    input_b = Input(shape=(3,))

    # masks
    masked_a = Masking(mask_value=0)(input_a)
    masked_b = Masking(mask_value=0)(input_b)

    # two different types of merging
    merged_sum = merge([masked_a, masked_b], mode='sum')
    merged_concat = merge([masked_a, masked_b], mode='concat', concat_axis=1)

    # test sum
    model_sum = Model([input_a, input_b], [merged_sum])
    model_sum.compile(loss='mse', optimizer='sgd')
    model_sum.fit([rand(2,3), rand(2,3)], [rand(2,3)], nb_epoch=1)

    # test concatenation
    model_concat = Model([input_a, input_b], [merged_concat])
    model_concat.compile(loss='mse', optimizer='sgd')
    model_concat.fit([rand(2,3), rand(2,3)], [rand(2,6)], nb_epoch=1)


def test_merge_mask_3d():
    from keras.layers import Input, merge, Embedding, SimpleRNN
    from keras.models import Model

    rand = lambda *shape: np.asarray(np.random.random(shape) > 0.5, dtype='int32')

    # embeddings
    input_a = Input(shape=(3,), dtype='int32')
    input_b = Input(shape=(3,), dtype='int32')
    embedding = Embedding(3, 4, mask_zero=True)
    embedding_a = embedding(input_a)
    embedding_b = embedding(input_b)

    # rnn
    rnn = SimpleRNN(3, return_sequences=True)
    rnn_a = rnn(embedding_a)
    rnn_b = rnn(embedding_b)

    # concatenation
    merged_concat = merge([rnn_a, rnn_b], mode='concat', concat_axis=-1)
    model = Model([input_a, input_b], [merged_concat])
    model.compile(loss='mse', optimizer='sgd')
    model.fit([rand(2,3), rand(2,3)], [rand(2,3,6)])


def test_dropout():
    layer_test(core.Dropout,
               kwargs={'p': 0.5},
               input_shape=(3, 2))


def test_activation():
    # with string argument
    layer_test(core.Activation,
               kwargs={'activation': 'relu'},
               input_shape=(3, 2))

    # with function argument
    layer_test(core.Activation,
               kwargs={'activation': K.relu},
               input_shape=(3, 2))


def test_reshape():
    layer_test(core.Reshape,
               kwargs={'target_shape': (8, 1)},
               input_shape=(3, 2, 4))


def test_permute():
    layer_test(core.Permute,
               kwargs={'dims': (2, 1)},
               input_shape=(3, 2, 4))


def test_flatten():
    layer_test(core.Flatten,
               kwargs={},
               input_shape=(3, 2, 4))


def test_repeat_vector():
    layer_test(core.RepeatVector,
               kwargs={'n': 3},
               input_shape=(3, 2))


def test_lambda():
    from keras.utils.layer_utils import layer_from_config
    Lambda = core.Lambda

    layer_test(Lambda,
               kwargs={'function': lambda x: x + 1},
               input_shape=(3, 2))

    # test serialization with function
    def f(x):
        return x + 1

    ld = Lambda(f)
    config = ld.get_config()
    ld = layer_from_config({'class_name': 'Lambda', 'config': config})

    ld = Lambda(lambda x: K.concatenate([K.square(x), x]),
                output_shape=lambda s: tuple(list(s)[:-1] + [2 * s[-1]]))
    config = ld.get_config()
    ld = Lambda.from_config(config)

    # test serialization with output_shape function
    def f(x):
        return K.concatenate([K.square(x), x])

    def f_shape(s):
        return tuple(list(s)[:-1] + [2 * s[-1]])

    ld = Lambda(f, output_shape=f_shape)
    config = ld.get_config()
    ld = layer_from_config({'class_name': 'Lambda', 'config': config})


def test_dense():
    from keras import regularizers
    from keras import constraints

    layer_test(core.Dense,
               kwargs={'output_dim': 3},
               input_shape=(3, 2))

    layer_test(core.Dense,
               kwargs={'output_dim': 3,
                       'W_regularizer': regularizers.l2(0.01),
                       'b_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.activity_l2(0.01),
                       'W_constraint': constraints.MaxNorm(1),
                       'b_constraint': constraints.MaxNorm(1)},
               input_shape=(3, 2))


def test_activity_regularization():
    from keras.engine import Input, Model

    layer = core.ActivityRegularization(l1=0.01, l2=0.01)

    # test in functional API
    x = Input(shape=(3,))
    z = core.Dense(2)(x)
    y = layer(z)
    model = Model(input=x, output=y)
    model.compile('rmsprop', 'mse', mode='FAST_COMPILE')

    model.predict(np.random.random((2, 3)))

    # test serialization
    model_config = model.get_config()
    model = Model.from_config(model_config)
    model.compile('rmsprop', 'mse')


def test_maxout_dense():
    from keras import regularizers
    from keras import constraints

    layer_test(core.MaxoutDense,
               kwargs={'output_dim': 3},
               input_shape=(3, 2))

    layer_test(core.MaxoutDense,
               kwargs={'output_dim': 3,
                       'W_regularizer': regularizers.l2(0.01),
                       'b_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.activity_l2(0.01),
                       'W_constraint': constraints.MaxNorm(1),
                       'b_constraint': constraints.MaxNorm(1)},
               input_shape=(3, 2))


def test_highway():
    from keras import regularizers
    from keras import constraints

    layer_test(core.Highway,
               kwargs={},
               input_shape=(3, 2))

    layer_test(core.Highway,
               kwargs={'W_regularizer': regularizers.l2(0.01),
                       'b_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.activity_l2(0.01),
                       'W_constraint': constraints.MaxNorm(1),
                       'b_constraint': constraints.MaxNorm(1)},
               input_shape=(3, 2))


def test_timedistributeddense():
    from keras import regularizers
    from keras import constraints

    layer_test(core.TimeDistributedDense,
               kwargs={'output_dim': 2, 'input_length': 2},
               input_shape=(3, 2, 3))

    layer_test(core.TimeDistributedDense,
               kwargs={'output_dim': 3,
                       'W_regularizer': regularizers.l2(0.01),
                       'b_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.activity_l2(0.01),
                       'W_constraint': constraints.MaxNorm(1),
                       'b_constraint': constraints.MaxNorm(1)},
               input_shape=(3, 2, 3))


if __name__ == '__main__':
    pytest.main([__file__])
