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

    def test_4d():
        np_inp_channels_last = np.arange(24, dtype='float32').reshape(
                                        (1, 4, 3, 2))

        np_output_cl = layer_test(layers.Flatten,
                                  kwargs={'data_format':
                                          'channels_last'},
                                  input_data=np_inp_channels_last)

        np_inp_channels_first = np.transpose(np_inp_channels_last,
                                             [0, 3, 1, 2])

        np_output_cf = layer_test(layers.Flatten,
                                  kwargs={'data_format':
                                          'channels_first'},
                                  input_data=np_inp_channels_first,
                                  expected_output=np_output_cl)

    def test_3d():
        np_inp_channels_last = np.arange(12, dtype='float32').reshape(
            (1, 4, 3))

        np_output_cl = layer_test(layers.Flatten,
                                  kwargs={'data_format':
                                          'channels_last'},
                                  input_data=np_inp_channels_last)

        np_inp_channels_first = np.transpose(np_inp_channels_last,
                                             [0, 2, 1])

        np_output_cf = layer_test(layers.Flatten,
                                  kwargs={'data_format':
                                          'channels_first'},
                                  input_data=np_inp_channels_first,
                                  expected_output=np_output_cl)

    def test_5d():
        np_inp_channels_last = np.arange(120, dtype='float32').reshape(
            (1, 5, 4, 3, 2))

        np_output_cl = layer_test(layers.Flatten,
                                  kwargs={'data_format':
                                          'channels_last'},
                                  input_data=np_inp_channels_last)

        np_inp_channels_first = np.transpose(np_inp_channels_last,
                                             [0, 4, 1, 2, 3])

        np_output_cf = layer_test(layers.Flatten,
                                  kwargs={'data_format':
                                          'channels_first'},
                                  input_data=np_inp_channels_first,
                                  expected_output=np_output_cl)
    test_3d()
    test_4d()
    test_5d()


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

        i = layers.Input(shape=(3, 2, 1))
        o = layers.Lambda(function=func,
                          output_shape=output_shape,
                          mask=mask)(i)

        o1, o2 = o
        assert o1._keras_shape == (None, 3, 2, 1)
        assert o2._keras_shape == (None, 3, 2, 1)

        model = Model(i, o)

        x = np.random.random((4, 3, 2, 1))
        out1, out2 = model.predict(x)
        assert out1.shape == (4, 3, 2, 1)
        assert out2.shape == (4, 3, 2, 1)
        assert_allclose(out1, x * 0.2, atol=1e-4)
        assert_allclose(out2, x * 0.3, atol=1e-4)

    test_multiple_outputs()

    # test layer with multiple outputs and no
    # explicit mask
    def test_multiple_outputs_no_mask():
        def func(x):
            return [x * 0.2, x * 0.3]

        def output_shape(input_shape):
            return [input_shape, input_shape]

        i = layers.Input(shape=(3, 2, 1))
        o = layers.Lambda(function=func,
                          output_shape=output_shape)(i)

        assert o[0]._keras_shape == (None, 3, 2, 1)
        assert o[1]._keras_shape == (None, 3, 2, 1)

        o = layers.add(o)
        model = Model(i, o)

        i2 = layers.Input(shape=(3, 2, 1))
        o2 = model(i2)
        model2 = Model(i2, o2)

        x = np.random.random((4, 3, 2, 1))
        out = model2.predict(x)
        assert out.shape == (4, 3, 2, 1)
        assert_allclose(out, x * 0.2 + x * 0.3, atol=1e-4)

    test_multiple_outputs_no_mask()

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
@pytest.mark.skipif((K.backend() == 'theano'),
                    reason="theano cannot compute "
                           "the output shape automatically.")
def test_lambda_output_shape():
    layer_test(layers.Lambda,
               kwargs={'function': lambda x: K.mean(x, axis=-1)},
               input_shape=(3, 2, 4))


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
