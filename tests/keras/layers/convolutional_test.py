import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test
from keras.utils.test_utils import keras_test
from keras import backend as K
from keras.engine.topology import InputLayer
from keras.layers import convolutional
from keras.layers import pooling
from keras.models import Sequential


# TensorFlow does not support full convolution.
if K.backend() == 'theano':
    _convolution_paddings = ['valid', 'same', 'full']
else:
    _convolution_paddings = ['valid', 'same']


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support dilated conv")
def test_causal_dilated_conv():
    # Causal:
    layer_test(convolutional.Conv1D,
               input_data=np.reshape(np.arange(4, dtype='float32'), (1, 4, 1)),
               kwargs={
                   'filters': 1,
                   'kernel_size': 2,
                   'dilation_rate': 1,
                   'padding': 'causal',
                   'kernel_initializer': 'ones',
                   'use_bias': False,
               },
               expected_output=[[[0], [1], [3], [5]]]
               )

    # Non-causal:
    layer_test(convolutional.Conv1D,
               input_data=np.reshape(np.arange(4, dtype='float32'), (1, 4, 1)),
               kwargs={
                   'filters': 1,
                   'kernel_size': 2,
                   'dilation_rate': 1,
                   'padding': 'valid',
                   'kernel_initializer': 'ones',
                   'use_bias': False,
               },
               expected_output=[[[1], [3], [5]]]
               )

    # Causal dilated with larger kernel size:
    layer_test(convolutional.Conv1D,
               input_data=np.reshape(np.arange(10, dtype='float32'), (1, 10, 1)),
               kwargs={
                   'filters': 1,
                   'kernel_size': 3,
                   'dilation_rate': 2,
                   'padding': 'causal',
                   'kernel_initializer': 'ones',
                   'use_bias': False,
               },
               expected_output=np.float32([[[0], [1], [2], [4], [6], [9], [12], [15], [18], [21]]])
               )


@keras_test
def test_conv_1d():
    batch_size = 2
    steps = 8
    input_dim = 2
    kernel_size = 3
    filters = 3

    for padding in _convolution_paddings:
        for strides in [1, 2]:
            if padding == 'same' and strides != 1:
                continue
            layer_test(convolutional.Conv1D,
                       kwargs={'filters': filters,
                               'kernel_size': kernel_size,
                               'padding': padding,
                               'strides': strides},
                       input_shape=(batch_size, steps, input_dim))

            layer_test(convolutional.Conv1D,
                       kwargs={'filters': filters,
                               'kernel_size': kernel_size,
                               'padding': padding,
                               'kernel_regularizer': 'l2',
                               'bias_regularizer': 'l2',
                               'activity_regularizer': 'l2',
                               'kernel_constraint': 'max_norm',
                               'bias_constraint': 'max_norm',
                               'strides': strides},
                       input_shape=(batch_size, steps, input_dim))

    # Test dilation
    layer_test(convolutional.Conv1D,
               kwargs={'filters': filters,
                       'kernel_size': kernel_size,
                       'padding': padding,
                       'dilation_rate': 2,
                       'activation': None},
               input_shape=(batch_size, steps, input_dim))

    convolutional.Conv1D(filters=filters,
                         kernel_size=kernel_size,
                         padding=padding,
                         input_shape=(input_dim,))


@keras_test
def test_maxpooling_1d():
    for padding in ['valid', 'same']:
        for stride in [1, 2]:
            layer_test(convolutional.MaxPooling1D,
                       kwargs={'strides': stride,
                               'padding': padding},
                       input_shape=(3, 5, 4))


@keras_test
def test_averagepooling_1d():
    for padding in ['valid', 'same']:
        for stride in [1, 2]:
            layer_test(convolutional.AveragePooling1D,
                       kwargs={'strides': stride,
                               'padding': padding},
                       input_shape=(3, 5, 4))


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support dilated conv")
def test_convolution_2d():
    num_samples = 2
    filters = 2
    stack_size = 3
    kernel_size = (3, 2)
    num_row = 7
    num_col = 6

    for padding in _convolution_paddings:
        for strides in [(1, 1), (2, 2)]:
            if padding == 'same' and strides != (1, 1):
                continue

            layer_test(convolutional.Conv2D,
                       kwargs={'filters': filters,
                               'kernel_size': kernel_size,
                               'padding': padding,
                               'strides': strides,
                               'data_format': 'channels_first'},
                       input_shape=(num_samples, stack_size, num_row, num_col))

    layer_test(convolutional.Conv2D,
               kwargs={'filters': filters,
                       'kernel_size': 3,
                       'padding': padding,
                       'data_format': 'channels_last',
                       'activation': None,
                       'kernel_regularizer': 'l2',
                       'bias_regularizer': 'l2',
                       'activity_regularizer': 'l2',
                       'kernel_constraint': 'max_norm',
                       'bias_constraint': 'max_norm',
                       'strides': strides},
               input_shape=(num_samples, num_row, num_col, stack_size))

    # Test dilation
    layer_test(convolutional.Conv2D,
               kwargs={'filters': filters,
                       'kernel_size': kernel_size,
                       'padding': padding,
                       'dilation_rate': (2, 2)},
               input_shape=(num_samples, num_row, num_col, stack_size))

    # Test invalid use case
    with pytest.raises(ValueError):
        model = Sequential([convolutional.Conv2D(filters=filters,
                                                 kernel_size=kernel_size,
                                                 padding=padding,
                                                 batch_input_shape=(None, None, 5, None))])


@keras_test
def test_conv2d_transpose():
    num_samples = 2
    filters = 2
    stack_size = 3
    num_row = 5
    num_col = 6

    for padding in _convolution_paddings:
        for strides in [(1, 1), (2, 2)]:
            if padding == 'same' and strides != (1, 1):
                continue
            layer_test(convolutional.Deconvolution2D,
                       kwargs={'filters': filters,
                               'kernel_size': 3,
                               'padding': padding,
                               'strides': strides,
                               'data_format': 'channels_last'},
                       input_shape=(num_samples, num_row, num_col, stack_size),
                       fixed_batch_size=True)

    layer_test(convolutional.Deconvolution2D,
               kwargs={'filters': filters,
                       'kernel_size': 3,
                       'padding': padding,
                       'data_format': 'channels_first',
                       'activation': None,
                       'kernel_regularizer': 'l2',
                       'bias_regularizer': 'l2',
                       'activity_regularizer': 'l2',
                       'kernel_constraint': 'max_norm',
                       'bias_constraint': 'max_norm',
                       'strides': strides},
               input_shape=(num_samples, stack_size, num_row, num_col),
               fixed_batch_size=True)

    # Test invalid use case
    with pytest.raises(ValueError):
        model = Sequential([convolutional.Conv2DTranspose(filters=filters,
                                                          kernel_size=3,
                                                          padding=padding,
                                                          batch_input_shape=(None, None, 5, None))])


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TF backend')
@keras_test
def test_separable_conv_1d():
    num_samples = 2
    filters = 6
    stack_size = 3
    num_step = 9

    for padding in _convolution_paddings:
        for strides in [1, 2]:
            for multiplier in [1, 2]:
                for dilation_rate in [1, 2]:
                    if padding == 'same' and strides != 1:
                        continue
                    if dilation_rate != 1 and strides != 1:
                        continue

                    layer_test(convolutional.SeparableConv1D,
                               kwargs={'filters': filters,
                                       'kernel_size': 3,
                                       'padding': padding,
                                       'strides': strides,
                                       'depth_multiplier': multiplier,
                                       'dilation_rate': dilation_rate},
                               input_shape=(num_samples, num_step, stack_size))

    layer_test(convolutional.SeparableConv1D,
               kwargs={'filters': filters,
                       'kernel_size': 3,
                       'padding': padding,
                       'data_format': 'channels_first',
                       'activation': None,
                       'depthwise_regularizer': 'l2',
                       'pointwise_regularizer': 'l2',
                       'bias_regularizer': 'l2',
                       'activity_regularizer': 'l2',
                       'pointwise_constraint': 'unit_norm',
                       'depthwise_constraint': 'unit_norm',
                       'strides': 1,
                       'depth_multiplier': multiplier},
               input_shape=(num_samples, stack_size, num_step))

    # Test invalid use case
    with pytest.raises(ValueError):
        model = Sequential([convolutional.SeparableConv1D(filters=filters,
                                                          kernel_size=3,
                                                          padding=padding,
                                                          batch_input_shape=(None, 5, None))])


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TF backend')
@keras_test
def test_separable_conv_2d():
    num_samples = 2
    filters = 6
    stack_size = 3
    num_row = 7
    num_col = 6

    for padding in _convolution_paddings:
        for strides in [(1, 1), (2, 2)]:
            for multiplier in [1, 2]:
                for dilation_rate in [(1, 1), (2, 2), (2, 1), (1, 2)]:
                    if padding == 'same' and strides != (1, 1):
                        continue
                    if dilation_rate != (1, 1) and strides != (1, 1):
                        continue

                    layer_test(convolutional.SeparableConv2D,
                               kwargs={'filters': filters,
                                       'kernel_size': (3, 3),
                                       'padding': padding,
                                       'strides': strides,
                                       'depth_multiplier': multiplier,
                                       'dilation_rate': dilation_rate},
                               input_shape=(num_samples, num_row, num_col, stack_size))

    layer_test(convolutional.SeparableConv2D,
               kwargs={'filters': filters,
                       'kernel_size': 3,
                       'padding': padding,
                       'data_format': 'channels_first',
                       'activation': None,
                       'depthwise_regularizer': 'l2',
                       'pointwise_regularizer': 'l2',
                       'bias_regularizer': 'l2',
                       'activity_regularizer': 'l2',
                       'pointwise_constraint': 'unit_norm',
                       'depthwise_constraint': 'unit_norm',
                       'strides': strides,
                       'depth_multiplier': multiplier},
               input_shape=(num_samples, stack_size, num_row, num_col))

    # Test invalid use case
    with pytest.raises(ValueError):
        model = Sequential([convolutional.SeparableConv2D(filters=filters,
                                                          kernel_size=3,
                                                          padding=padding,
                                                          batch_input_shape=(None, None, 5, None))])


@keras_test
def test_globalpooling_1d():
    layer_test(pooling.GlobalMaxPooling1D,
               input_shape=(3, 4, 5))
    layer_test(pooling.GlobalAveragePooling1D,
               input_shape=(3, 4, 5))


@keras_test
def test_globalpooling_2d():
    layer_test(pooling.GlobalMaxPooling2D,
               kwargs={'data_format': 'channels_first'},
               input_shape=(3, 4, 5, 6))
    layer_test(pooling.GlobalMaxPooling2D,
               kwargs={'data_format': 'channels_last'},
               input_shape=(3, 5, 6, 4))
    layer_test(pooling.GlobalAveragePooling2D,
               kwargs={'data_format': 'channels_first'},
               input_shape=(3, 4, 5, 6))
    layer_test(pooling.GlobalAveragePooling2D,
               kwargs={'data_format': 'channels_last'},
               input_shape=(3, 5, 6, 4))


@keras_test
def test_globalpooling_3d():
    layer_test(pooling.GlobalMaxPooling3D,
               kwargs={'data_format': 'channels_first'},
               input_shape=(3, 4, 3, 4, 3))
    layer_test(pooling.GlobalMaxPooling3D,
               kwargs={'data_format': 'channels_last'},
               input_shape=(3, 4, 3, 4, 3))
    layer_test(pooling.GlobalAveragePooling3D,
               kwargs={'data_format': 'channels_first'},
               input_shape=(3, 4, 3, 4, 3))
    layer_test(pooling.GlobalAveragePooling3D,
               kwargs={'data_format': 'channels_last'},
               input_shape=(3, 4, 3, 4, 3))


@keras_test
def test_maxpooling_2d():
    pool_size = (3, 3)

    for strides in [(1, 1), (2, 2)]:
        layer_test(convolutional.MaxPooling2D,
                   kwargs={'strides': strides,
                           'padding': 'valid',
                           'pool_size': pool_size},
                   input_shape=(3, 5, 6, 4))


@keras_test
def test_averagepooling_2d():
    layer_test(convolutional.AveragePooling2D,
               kwargs={'strides': (2, 2),
                       'padding': 'same',
                       'pool_size': (2, 2)},
               input_shape=(3, 5, 6, 4))
    layer_test(convolutional.AveragePooling2D,
               kwargs={'strides': (2, 2),
                       'padding': 'valid',
                       'pool_size': (3, 3)},
               input_shape=(3, 5, 6, 4))
    layer_test(convolutional.AveragePooling2D,
               kwargs={'strides': (1, 1),
                       'padding': 'valid',
                       'pool_size': (2, 2),
                       'data_format': 'channels_first'},
               input_shape=(3, 4, 5, 6))


@keras_test
def test_convolution_3d():
    num_samples = 2
    filters = 2
    stack_size = 3

    input_len_dim1 = 9
    input_len_dim2 = 8
    input_len_dim3 = 8

    for padding in _convolution_paddings:
        for strides in [(1, 1, 1), (2, 2, 2)]:
            if padding == 'same' and strides != (1, 1, 1):
                continue

            layer_test(convolutional.Convolution3D,
                       kwargs={'filters': filters,
                               'kernel_size': 3,
                               'padding': padding,
                               'strides': strides},
                       input_shape=(num_samples,
                                    input_len_dim1, input_len_dim2, input_len_dim3,
                                    stack_size))

    layer_test(convolutional.Convolution3D,
               kwargs={'filters': filters,
                       'kernel_size': (1, 2, 3),
                       'padding': padding,
                       'activation': None,
                       'kernel_regularizer': 'l2',
                       'bias_regularizer': 'l2',
                       'activity_regularizer': 'l2',
                       'kernel_constraint': 'max_norm',
                       'bias_constraint': 'max_norm',
                       'strides': strides},
               input_shape=(num_samples,
                            input_len_dim1, input_len_dim2, input_len_dim3,
                            stack_size))


@keras_test
def test_conv3d_transpose():
    filters = 2
    stack_size = 3
    num_depth = 7
    num_row = 5
    num_col = 6

    for padding in _convolution_paddings:
        for strides in [(1, 1, 1), (2, 2, 2)]:
            for data_format in ['channels_first', 'channels_last']:
                if padding == 'same' and strides != (1, 1, 1):
                    continue
                layer_test(convolutional.Conv3DTranspose,
                           kwargs={'filters': filters,
                                   'kernel_size': 3,
                                   'padding': padding,
                                   'strides': strides,
                                   'data_format': data_format},
                           input_shape=(None, num_depth, num_row, num_col, stack_size),
                           fixed_batch_size=True)

    layer_test(convolutional.Conv3DTranspose,
               kwargs={'filters': filters,
                       'kernel_size': 3,
                       'padding': padding,
                       'data_format': 'channels_first',
                       'activation': None,
                       'kernel_regularizer': 'l2',
                       'bias_regularizer': 'l2',
                       'activity_regularizer': 'l2',
                       'kernel_constraint': 'max_norm',
                       'bias_constraint': 'max_norm',
                       'strides': strides},
               input_shape=(None, stack_size, num_depth, num_row, num_col),
               fixed_batch_size=True)

    # Test invalid use case
    with pytest.raises(ValueError):
        model = Sequential([convolutional.Conv3DTranspose(filters=filters,
                                                          kernel_size=3,
                                                          padding=padding,
                                                          batch_input_shape=(None, None, 5, None, None))])


@keras_test
def test_maxpooling_3d():
    pool_size = (3, 3, 3)

    layer_test(convolutional.MaxPooling3D,
               kwargs={'strides': 2,
                       'padding': 'valid',
                       'pool_size': pool_size},
               input_shape=(3, 11, 12, 10, 4))
    layer_test(convolutional.MaxPooling3D,
               kwargs={'strides': 3,
                       'padding': 'valid',
                       'data_format': 'channels_first',
                       'pool_size': pool_size},
               input_shape=(3, 4, 11, 12, 10))


@keras_test
def test_averagepooling_3d():
    pool_size = (3, 3, 3)

    layer_test(convolutional.AveragePooling3D,
               kwargs={'strides': 2,
                       'padding': 'valid',
                       'pool_size': pool_size},
               input_shape=(3, 11, 12, 10, 4))
    layer_test(convolutional.AveragePooling3D,
               kwargs={'strides': 3,
                       'padding': 'valid',
                       'data_format': 'channels_first',
                       'pool_size': pool_size},
               input_shape=(3, 4, 11, 12, 10))


@keras_test
def test_zero_padding_1d():
    num_samples = 2
    input_dim = 2
    num_steps = 5
    shape = (num_samples, num_steps, input_dim)
    inputs = np.ones(shape)

    # basic test
    layer_test(convolutional.ZeroPadding1D,
               kwargs={'padding': 2},
               input_shape=inputs.shape)
    layer_test(convolutional.ZeroPadding1D,
               kwargs={'padding': (1, 2)},
               input_shape=inputs.shape)

    # correctness test
    layer = convolutional.ZeroPadding1D(padding=2)
    layer.build(shape)
    outputs = layer(K.variable(inputs))
    np_output = K.eval(outputs)
    for offset in [0, 1, -1, -2]:
        assert_allclose(np_output[:, offset, :], 0.)
    assert_allclose(np_output[:, 2:-2, :], 1.)

    layer = convolutional.ZeroPadding1D(padding=(1, 2))
    layer.build(shape)
    outputs = layer(K.variable(inputs))
    np_output = K.eval(outputs)
    for left_offset in [0]:
        assert_allclose(np_output[:, left_offset, :], 0.)
    for right_offset in [-1, -2]:
        assert_allclose(np_output[:, right_offset, :], 0.)
    assert_allclose(np_output[:, 1:-2, :], 1.)
    layer.get_config()


@keras_test
def test_zero_padding_2d():
    num_samples = 2
    stack_size = 2
    input_num_row = 4
    input_num_col = 5
    for data_format in ['channels_first', 'channels_last']:
        inputs = np.ones((num_samples, input_num_row, input_num_col, stack_size))
        inputs = np.ones((num_samples, stack_size, input_num_row, input_num_col))

        # basic test
        layer_test(convolutional.ZeroPadding2D,
                   kwargs={'padding': (2, 2), 'data_format': data_format},
                   input_shape=inputs.shape)
        layer_test(convolutional.ZeroPadding2D,
                   kwargs={'padding': ((1, 2), (3, 4)), 'data_format': data_format},
                   input_shape=inputs.shape)

        # correctness test
        layer = convolutional.ZeroPadding2D(padding=(2, 2),
                                            data_format=data_format)
        layer.build(inputs.shape)
        outputs = layer(K.variable(inputs))
        np_output = K.eval(outputs)
        if data_format == 'channels_last':
            for offset in [0, 1, -1, -2]:
                assert_allclose(np_output[:, offset, :, :], 0.)
                assert_allclose(np_output[:, :, offset, :], 0.)
            assert_allclose(np_output[:, 2:-2, 2:-2, :], 1.)
        elif data_format == 'channels_first':
            for offset in [0, 1, -1, -2]:
                assert_allclose(np_output[:, :, offset, :], 0.)
                assert_allclose(np_output[:, :, :, offset], 0.)
            assert_allclose(np_output[:, 2:-2, 2:-2, :], 1.)

        layer = convolutional.ZeroPadding2D(padding=((1, 2), (3, 4)),
                                            data_format=data_format)
        layer.build(inputs.shape)
        outputs = layer(K.variable(inputs))
        np_output = K.eval(outputs)
        if data_format == 'channels_last':
            for top_offset in [0]:
                assert_allclose(np_output[:, top_offset, :, :], 0.)
            for bottom_offset in [-1, -2]:
                assert_allclose(np_output[:, bottom_offset, :, :], 0.)
            for left_offset in [0, 1, 2]:
                assert_allclose(np_output[:, :, left_offset, :], 0.)
            for right_offset in [-1, -2, -3, -4]:
                assert_allclose(np_output[:, :, right_offset, :], 0.)
            assert_allclose(np_output[:, 1:-2, 3:-4, :], 1.)
        elif data_format == 'channels_first':
            for top_offset in [0]:
                assert_allclose(np_output[:, :, top_offset, :], 0.)
            for bottom_offset in [-1, -2]:
                assert_allclose(np_output[:, :, bottom_offset, :], 0.)
            for left_offset in [0, 1, 2]:
                assert_allclose(np_output[:, :, :, left_offset], 0.)
            for right_offset in [-1, -2, -3, -4]:
                assert_allclose(np_output[:, :, :, right_offset], 0.)
            assert_allclose(np_output[:, :, 1:-2, 3:-4], 1.)


def test_zero_padding_3d():
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 4
    input_len_dim2 = 5
    input_len_dim3 = 3

    inputs = np.ones((num_samples,
                     input_len_dim1, input_len_dim2, input_len_dim3,
                     stack_size))

    # basic test
    for data_format in ['channels_first', 'channels_last']:
        layer_test(convolutional.ZeroPadding3D,
                   kwargs={'padding': (2, 2, 2), 'data_format': data_format},
                   input_shape=inputs.shape)
        layer_test(convolutional.ZeroPadding3D,
                   kwargs={'padding': ((1, 2), (3, 4), (0, 2)), 'data_format': data_format},
                   input_shape=inputs.shape)

        # correctness test
        layer = convolutional.ZeroPadding3D(padding=(2, 2, 2),
                                            data_format=data_format)
        layer.build(inputs.shape)
        outputs = layer(K.variable(inputs))
        np_output = K.eval(outputs)
        if data_format == 'channels_last':
            for offset in [0, 1, -1, -2]:
                assert_allclose(np_output[:, offset, :, :, :], 0.)
                assert_allclose(np_output[:, :, offset, :, :], 0.)
                assert_allclose(np_output[:, :, :, offset, :], 0.)
            assert_allclose(np_output[:, 2:-2, 2:-2, 2:-2, :], 1.)
        elif data_format == 'channels_first':
            for offset in [0, 1, -1, -2]:
                assert_allclose(np_output[:, :, offset, :, :], 0.)
                assert_allclose(np_output[:, :, :, offset, :], 0.)
                assert_allclose(np_output[:, :, :, :, offset], 0.)
            assert_allclose(np_output[:, :, 2:-2, 2:-2, 2:-2], 1.)

        layer = convolutional.ZeroPadding3D(padding=((1, 2), (3, 4), (0, 2)),
                                            data_format=data_format)
        layer.build(inputs.shape)
        outputs = layer(K.variable(inputs))
        np_output = K.eval(outputs)
        if data_format == 'channels_last':
            for dim1_offset in [0, -1, -2]:
                assert_allclose(np_output[:, dim1_offset, :, :, :], 0.)
            for dim2_offset in [0, 1, 2, -1, -2, -3, -4]:
                assert_allclose(np_output[:, :, dim2_offset, :, :], 0.)
            for dim3_offset in [-1, -2]:
                assert_allclose(np_output[:, :, :, dim3_offset, :], 0.)
            assert_allclose(np_output[:, 1:-2, 3:-4, 0:-2, :], 1.)
        elif data_format == 'channels_first':
            for dim1_offset in [0, -1, -2]:
                assert_allclose(np_output[:, :, dim1_offset, :, :], 0.)
            for dim2_offset in [0, 1, 2, -1, -2, -3, -4]:
                assert_allclose(np_output[:, :, :, dim2_offset, :], 0.)
            for dim3_offset in [-1, -2]:
                assert_allclose(np_output[:, :, :, :, dim3_offset], 0.)
            assert_allclose(np_output[:, :, 1:-2, 3:-4, 0:-2], 1.)


@keras_test
def test_upsampling_1d():
    layer_test(convolutional.UpSampling1D,
               kwargs={'size': 2},
               input_shape=(3, 5, 4))


@keras_test
def test_upsampling_2d():
    num_samples = 2
    stack_size = 2
    input_num_row = 11
    input_num_col = 12

    for data_format in ['channels_first', 'channels_last']:
        if data_format == 'channels_first':
            inputs = np.random.rand(num_samples, stack_size, input_num_row,
                                    input_num_col)
        else:  # tf
            inputs = np.random.rand(num_samples, input_num_row, input_num_col,
                                    stack_size)

        # basic test
        layer_test(convolutional.UpSampling2D,
                   kwargs={'size': (2, 2), 'data_format': data_format},
                   input_shape=inputs.shape)

        for length_row in [2]:
            for length_col in [2, 3]:
                layer = convolutional.UpSampling2D(
                    size=(length_row, length_col),
                    data_format=data_format)
                layer.build(inputs.shape)
                outputs = layer(K.variable(inputs))
                np_output = K.eval(outputs)
                if data_format == 'channels_first':
                    assert np_output.shape[2] == length_row * input_num_row
                    assert np_output.shape[3] == length_col * input_num_col
                else:  # tf
                    assert np_output.shape[1] == length_row * input_num_row
                    assert np_output.shape[2] == length_col * input_num_col

                # compare with numpy
                if data_format == 'channels_first':
                    expected_out = np.repeat(inputs, length_row, axis=2)
                    expected_out = np.repeat(expected_out, length_col, axis=3)
                else:  # tf
                    expected_out = np.repeat(inputs, length_row, axis=1)
                    expected_out = np.repeat(expected_out, length_col, axis=2)

                assert_allclose(np_output, expected_out)


@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support it yet")
def test_upsampling_3d():
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 10
    input_len_dim2 = 11
    input_len_dim3 = 12

    for data_format in ['channels_first', 'channels_last']:
        if data_format == 'channels_first':
            inputs = np.random.rand(num_samples,
                                    stack_size,
                                    input_len_dim1, input_len_dim2, input_len_dim3)
        else:  # tf
            inputs = np.random.rand(num_samples,
                                    input_len_dim1, input_len_dim2, input_len_dim3,
                                    stack_size)

        # basic test
        layer_test(convolutional.UpSampling3D,
                   kwargs={'size': (2, 2, 2), 'data_format': data_format},
                   input_shape=inputs.shape)

        for length_dim1 in [2, 3]:
            for length_dim2 in [2]:
                for length_dim3 in [3]:
                    layer = convolutional.UpSampling3D(
                        size=(length_dim1, length_dim2, length_dim3),
                        data_format=data_format)
                    layer.build(inputs.shape)
                    outputs = layer(K.variable(inputs))
                    np_output = K.eval(outputs)
                    if data_format == 'channels_first':
                        assert np_output.shape[2] == length_dim1 * input_len_dim1
                        assert np_output.shape[3] == length_dim2 * input_len_dim2
                        assert np_output.shape[4] == length_dim3 * input_len_dim3
                    else:  # tf
                        assert np_output.shape[1] == length_dim1 * input_len_dim1
                        assert np_output.shape[2] == length_dim2 * input_len_dim2
                        assert np_output.shape[3] == length_dim3 * input_len_dim3

                    # compare with numpy
                    if data_format == 'channels_first':
                        expected_out = np.repeat(inputs, length_dim1, axis=2)
                        expected_out = np.repeat(expected_out, length_dim2, axis=3)
                        expected_out = np.repeat(expected_out, length_dim3, axis=4)
                    else:  # tf
                        expected_out = np.repeat(inputs, length_dim1, axis=1)
                        expected_out = np.repeat(expected_out, length_dim2, axis=2)
                        expected_out = np.repeat(expected_out, length_dim3, axis=3)

                    assert_allclose(np_output, expected_out)


@keras_test
@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support slice to 0 dimension")
def test_cropping_1d():
    num_samples = 2
    time_length = 4
    input_len_dim1 = 2
    inputs = np.random.rand(num_samples, time_length, input_len_dim1)

    layer_test(convolutional.Cropping1D,
               kwargs={'cropping': (2, 2)},
               input_shape=inputs.shape)


def test_cropping_2d():
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 9
    input_len_dim2 = 9
    cropping = ((2, 2), (3, 3))

    for data_format in ['channels_first', 'channels_last']:
        if data_format == 'channels_first':
            inputs = np.random.rand(num_samples, stack_size,
                                    input_len_dim1, input_len_dim2)
        else:
            inputs = np.random.rand(num_samples,
                                    input_len_dim1, input_len_dim2,
                                    stack_size)
        # basic test
        layer_test(convolutional.Cropping2D,
                   kwargs={'cropping': cropping,
                           'data_format': data_format},
                   input_shape=inputs.shape)
        # correctness test
        layer = convolutional.Cropping2D(cropping=cropping,
                                         data_format=data_format)
        layer.build(inputs.shape)
        outputs = layer(K.variable(inputs))
        np_output = K.eval(outputs)
        # compare with numpy
        if data_format == 'channels_first':
            expected_out = inputs[:,
                                  :,
                                  cropping[0][0]: -cropping[0][1],
                                  cropping[1][0]: -cropping[1][1]]
        else:
            expected_out = inputs[:,
                                  cropping[0][0]: -cropping[0][1],
                                  cropping[1][0]: -cropping[1][1],
                                  :]
        assert_allclose(np_output, expected_out)

    for data_format in ['channels_first', 'channels_last']:
        if data_format == 'channels_first':
            inputs = np.random.rand(num_samples, stack_size,
                                    input_len_dim1, input_len_dim2)
        else:
            inputs = np.random.rand(num_samples,
                                    input_len_dim1, input_len_dim2,
                                    stack_size)
        # another correctness test (no cropping)
        cropping = ((0, 0), (0, 0))
        layer = convolutional.Cropping2D(cropping=cropping,
                                         data_format=data_format)
        layer.build(inputs.shape)
        outputs = layer(K.variable(inputs))
        np_output = K.eval(outputs)
        # compare with input
        assert_allclose(np_output, inputs)

    # Test invalid use cases
    with pytest.raises(ValueError):
        layer = convolutional.Cropping2D(cropping=((1, 1),))
    with pytest.raises(ValueError):
        layer = convolutional.Cropping2D(cropping=lambda x: x)


def test_cropping_3d():
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 8
    input_len_dim2 = 8
    input_len_dim3 = 8
    cropping = ((2, 2), (3, 3), (2, 3))

    for data_format in ['channels_last', 'channels_first']:
        if data_format == 'channels_first':
            inputs = np.random.rand(num_samples, stack_size,
                                    input_len_dim1, input_len_dim2, input_len_dim3)
        else:
            inputs = np.random.rand(num_samples,
                                    input_len_dim1, input_len_dim2,
                                    input_len_dim3, stack_size)
        # basic test
        layer_test(convolutional.Cropping3D,
                   kwargs={'cropping': cropping,
                           'data_format': data_format},
                   input_shape=inputs.shape)
        # correctness test
        layer = convolutional.Cropping3D(cropping=cropping,
                                         data_format=data_format)
        layer.build(inputs.shape)
        outputs = layer(K.variable(inputs))
        np_output = K.eval(outputs)
        # compare with numpy
        if data_format == 'channels_first':
            expected_out = inputs[:,
                                  :,
                                  cropping[0][0]: -cropping[0][1],
                                  cropping[1][0]: -cropping[1][1],
                                  cropping[2][0]: -cropping[2][1]]
        else:
            expected_out = inputs[:,
                                  cropping[0][0]: -cropping[0][1],
                                  cropping[1][0]: -cropping[1][1],
                                  cropping[2][0]: -cropping[2][1],
                                  :]
        assert_allclose(np_output, expected_out)

    for data_format in ['channels_last', 'channels_first']:
        if data_format == 'channels_first':
            inputs = np.random.rand(num_samples, stack_size,
                                    input_len_dim1, input_len_dim2, input_len_dim3)
        else:
            inputs = np.random.rand(num_samples,
                                    input_len_dim1, input_len_dim2,
                                    input_len_dim3, stack_size)
        # another correctness test (no cropping)
        cropping = ((0, 0), (0, 0), (0, 0))
        layer = convolutional.Cropping3D(cropping=cropping,
                                         data_format=data_format)
        layer.build(inputs.shape)
        outputs = layer(K.variable(inputs))
        np_output = K.eval(outputs)
        # compare with input
        assert_allclose(np_output, inputs)

    # Test invalid use cases
    with pytest.raises(ValueError):
        layer = convolutional.Cropping3D(cropping=((1, 1),))
    with pytest.raises(ValueError):
        layer = convolutional.Cropping3D(cropping=lambda x: x)

if __name__ == '__main__':
    pytest.main([__file__])
