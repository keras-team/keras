import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras import layers
from keras.models import Model
from keras.utils.test_utils import layer_test
from keras.utils.test_utils import keras_test
from keras import regularizers
from keras import constraints
from keras.layers import deserialize as deserialize_layer


@keras_test
def test_masking():
    layer_test(layers.Masking,
               kwargs={},
               input_shape=(3, 2, 3))


@keras_test
def test_dropout():
    layer_test(layers.Dropout,
               kwargs={'rate': 0.5},
               input_shape=(3, 2))

    layer_test(layers.Dropout,
               kwargs={'rate': 0.5, 'noise_shape': [3, 1]},
               input_shape=(3, 2))

    layer_test(layers.Dropout,
               kwargs={'rate': 0.5, 'noise_shape': [None, 1]},
               input_shape=(3, 2))

    layer_test(layers.SpatialDropout1D,
               kwargs={'rate': 0.5},
               input_shape=(2, 3, 4))

    for data_format in ['channels_last', 'channels_first']:
        for shape in [(4, 5), (4, 5, 6)]:
            if data_format == 'channels_last':
                input_shape = (2,) + shape + (3,)
            else:
                input_shape = (2, 3) + shape
            layer_test(layers.SpatialDropout2D if len(shape) == 2 else layers.SpatialDropout3D,
                       kwargs={'rate': 0.5,
                               'data_format': data_format},
                       input_shape=input_shape)

            # Test invalid use cases
            with pytest.raises(ValueError):
                layer_test(layers.SpatialDropout2D if len(shape) == 2 else layers.SpatialDropout3D,
                           kwargs={'rate': 0.5,
                                   'data_format': 'channels_middle'},
                           input_shape=input_shape)


@keras_test
def test_activation():
    # with string argument
    layer_test(layers.Activation,
               kwargs={'activation': 'relu'},
               input_shape=(3, 2))

    # with function argument
    layer_test(layers.Activation,
               kwargs={'activation': K.relu},
               input_shape=(3, 2))


@keras_test
def test_reshape():
    layer_test(layers.Reshape,
               kwargs={'target_shape': (8, 1)},
               input_shape=(3, 2, 4))

    layer_test(layers.Reshape,
               kwargs={'target_shape': (-1, 1)},
               input_shape=(3, 2, 4))

    layer_test(layers.Reshape,
               kwargs={'target_shape': (1, -1)},
               input_shape=(3, 2, 4))

    layer_test(layers.Reshape,
               kwargs={'target_shape': (-1, 1)},
               input_shape=(None, None, 4))


@keras_test
def test_permute():
    layer_test(layers.Permute,
               kwargs={'dims': (2, 1)},
               input_shape=(3, 2, 4))


@keras_test
def test_flatten():
    layer_test(layers.Flatten,
               kwargs={},
               input_shape=(3, 2, 4))


@keras_test
def test_repeat_vector():
    layer_test(layers.RepeatVector,
               kwargs={'n': 3},
               input_shape=(3, 2))


@keras_test
def test_lambda():
    layer_test(layers.Lambda,
               kwargs={'function': lambda x: x + 1},
               input_shape=(3, 2))

    layer_test(layers.Lambda,
               kwargs={'function': lambda x, a, b: x * a + b,
                       'arguments': {'a': 0.6, 'b': 0.4}},
               input_shape=(3, 2))

    def antirectifier(x):
        x -= K.mean(x, axis=1, keepdims=True)
        x = K.l2_normalize(x, axis=1)
        pos = K.relu(x)
        neg = K.relu(-x)
        return K.concatenate([pos, neg], axis=1)

    def antirectifier_output_shape(input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # only valid for 2D tensors
        shape[-1] *= 2
        return tuple(shape)

    layer_test(layers.Lambda,
               kwargs={'function': antirectifier,
                       'output_shape': antirectifier_output_shape},
               input_shape=(3, 2))

    # test layer with multiple outputs
    def test_multiple_outputs():
        def func(x):
            return [x * 0.2, x * 0.3]

        def output_shape(input_shape):
            return [input_shape, input_shape]

        def mask(inputs, mask=None):
            return [None, None]

        i = layers.Input(shape=(64, 64, 3))
        o = layers.Lambda(function=func,
                          output_shape=output_shape,
                          mask=mask)(i)

        o1, o2 = o
        assert o1._keras_shape == (None, 64, 64, 3)
        assert o2._keras_shape == (None, 64, 64, 3)

        model = Model(i, o)

        x = np.random.random((4, 64, 64, 3))
        out1, out2 = model.predict(x)
        assert out1.shape == (4, 64, 64, 3)
        assert out2.shape == (4, 64, 64, 3)
        assert_allclose(out1, x * 0.2, atol=1e-4)
        assert_allclose(out2, x * 0.3, atol=1e-4)

    test_multiple_outputs()

    # test serialization with function
    def f(x):
        return x + 1

    ld = layers.Lambda(f)
    config = ld.get_config()
    ld = deserialize_layer({'class_name': 'Lambda', 'config': config})

    # test with lambda
    ld = layers.Lambda(
        lambda x: K.concatenate([K.square(x), x]),
        output_shape=lambda s: tuple(list(s)[:-1] + [2 * s[-1]]))
    config = ld.get_config()
    ld = layers.Lambda.from_config(config)

    # test serialization with output_shape function
    def f(x):
        return K.concatenate([K.square(x), x])

    def f_shape(s):
        return tuple(list(s)[:-1] + [2 * s[-1]])

    ld = layers.Lambda(f, output_shape=f_shape)
    config = ld.get_config()
    ld = deserialize_layer({'class_name': 'Lambda', 'config': config})


@keras_test
def test_dense():
    layer_test(layers.Dense,
               kwargs={'units': 3},
               input_shape=(3, 2))

    layer_test(layers.Dense,
               kwargs={'units': 3},
               input_shape=(3, 4, 2))

    layer_test(layers.Dense,
               kwargs={'units': 3},
               input_shape=(None, None, 2))

    layer_test(layers.Dense,
               kwargs={'units': 3},
               input_shape=(3, 4, 5, 2))

    layer_test(layers.Dense,
               kwargs={'units': 3,
                       'kernel_regularizer': regularizers.l2(0.01),
                       'bias_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.L1L2(l1=0.01, l2=0.01),
                       'kernel_constraint': constraints.MaxNorm(1),
                       'bias_constraint': constraints.max_norm(1)},
               input_shape=(3, 2))

    layer = layers.Dense(3,
                         kernel_regularizer=regularizers.l1(0.01),
                         bias_regularizer='l1')
    layer.build((None, 4))
    assert len(layer.losses) == 2


@keras_test
def test_activity_regularization():
    layer = layers.ActivityRegularization(l1=0.01, l2=0.01)

    # test in functional API
    x = layers.Input(shape=(3,))
    z = layers.Dense(2)(x)
    y = layer(z)
    model = Model(x, y)
    model.compile('rmsprop', 'mse')

    model.predict(np.random.random((2, 3)))

    # test serialization
    model_config = model.get_config()
    model = Model.from_config(model_config)
    model.compile('rmsprop', 'mse')


if __name__ == '__main__':
    pytest.main([__file__])
