import numpy as np
import pytest

from keras.utils.test_utils import layer_test
from keras.layers import pooling
from keras.layers import Masking
from keras.layers import convolutional
from keras.models import Sequential


@pytest.mark.parametrize(
    'padding,stride,data_format',
    [(padding, stride, data_format)
     for padding in ['valid', 'same']
     for stride in [1, 2]
     for data_format in ['channels_first', 'channels_last']]
)
def test_maxpooling_1d(padding, stride, data_format):
    layer_test(convolutional.MaxPooling1D,
               kwargs={'strides': stride,
                       'padding': padding,
                       'data_format': data_format},
               input_shape=(3, 5, 4))


@pytest.mark.parametrize(
    'strides',
    [(1, 1), (2, 3)]
)
def test_maxpooling_2d(strides):
    pool_size = (3, 3)
    layer_test(convolutional.MaxPooling2D,
               kwargs={'strides': strides,
                       'padding': 'valid',
                       'pool_size': pool_size},
               input_shape=(3, 5, 6, 4))


@pytest.mark.parametrize(
    'strides,data_format,input_shape',
    [(2, None, (3, 11, 12, 10, 4)),
     (3, 'channels_first', (3, 4, 11, 12, 10))]
)
def test_maxpooling_3d(strides, data_format, input_shape):
    pool_size = (3, 3, 3)
    layer_test(convolutional.MaxPooling3D,
               kwargs={'strides': strides,
                       'padding': 'valid',
                       'data_format': data_format,
                       'pool_size': pool_size},
               input_shape=input_shape)


@pytest.mark.parametrize(
    'padding,stride,data_format',
    [(padding, stride, data_format)
     for padding in ['valid', 'same']
     for stride in [1, 2]
     for data_format in ['channels_first', 'channels_last']]
)
def test_averagepooling_1d(padding, stride, data_format):
    layer_test(convolutional.AveragePooling1D,
               kwargs={'strides': stride,
                       'padding': padding,
                       'data_format': data_format},
               input_shape=(3, 5, 4))


@pytest.mark.parametrize(
    'strides,padding,data_format,input_shape',
    [((2, 2), 'same', None, (3, 5, 6, 4)),
     ((2, 2), 'valid', None, (3, 5, 6, 4)),
     ((1, 1), 'valid', 'channels_first', (3, 4, 5, 6))]
)
def test_averagepooling_2d(strides, padding, data_format, input_shape):
    layer_test(convolutional.AveragePooling2D,
               kwargs={'strides': strides,
                       'padding': padding,
                       'pool_size': (2, 2),
                       'data_format': data_format},
               input_shape=input_shape)


@pytest.mark.parametrize(
    'strides,data_format,input_shape',
    [(2, None, (3, 11, 12, 10, 4)),
     (3, 'channels_first', (3, 4, 11, 12, 10))]
)
def test_averagepooling_3d(strides, data_format, input_shape):
    pool_size = (3, 3, 3)

    layer_test(convolutional.AveragePooling3D,
               kwargs={'strides': strides,
                       'padding': 'valid',
                       'data_format': data_format,
                       'pool_size': pool_size},
               input_shape=input_shape)


@pytest.mark.parametrize(
    'data_format,pooling_class',
    [(data_format, pooling_class)
     for data_format in ['channels_first', 'channels_last']
     for pooling_class in [pooling.GlobalMaxPooling1D,
                           pooling.GlobalAveragePooling1D]]
)
def test_globalpooling_1d(data_format, pooling_class):
    layer_test(pooling_class,
               kwargs={'data_format': data_format},
               input_shape=(3, 4, 5))


def test_globalpooling_1d_supports_masking():
    # Test GlobalAveragePooling1D supports masking
    model = Sequential()
    model.add(Masking(mask_value=0., input_shape=(3, 4)))
    model.add(pooling.GlobalAveragePooling1D())
    model.compile(loss='mae', optimizer='adam')

    model_input = np.random.randint(low=1, high=5, size=(2, 3, 4))
    model_input[0, 1:, :] = 0
    output = model.predict(model_input)
    assert np.array_equal(output[0], model_input[0, 0, :])


@pytest.mark.parametrize(
    'data_format,pooling_class',
    [(data_format, pooling_class)
     for data_format in ['channels_first', 'channels_last']
     for pooling_class in [pooling.GlobalMaxPooling2D,
                           pooling.GlobalAveragePooling2D]]
)
def test_globalpooling_2d(data_format, pooling_class):
    layer_test(pooling_class,
               kwargs={'data_format': data_format},
               input_shape=(3, 4, 5, 6))


@pytest.mark.parametrize(
    'data_format,pooling_class',
    [(data_format, pooling_class)
     for data_format in ['channels_first', 'channels_last']
     for pooling_class in [pooling.GlobalMaxPooling3D,
                           pooling.GlobalAveragePooling3D]]
)
def test_globalpooling_3d(data_format, pooling_class):
    layer_test(pooling_class,
               kwargs={'data_format': data_format},
               input_shape=(3, 4, 3, 4, 3))


if __name__ == '__main__':
    pytest.main([__file__])
