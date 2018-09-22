import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test
from keras import backend as K
from keras.layers import convolutional
from keras.models import Sequential


# TensorFlow does not support full convolution.
if K.backend() == 'theano':
    _convolution_paddings = ['valid', 'same', 'full']
else:
    _convolution_paddings = ['valid', 'same']


@pytest.mark.skipif((K.backend() == 'cntk' and K.dev.type() == 0),
                    reason='cntk only support dilated conv on GPU')
@pytest.mark.parametrize(
    'layer_kwargs,input_length,expected_output',
    [
        # Causal
        ({'filters': 1, 'kernel_size': 2, 'dilation_rate': 1, 'padding': 'causal',
          'kernel_initializer': 'ones', 'use_bias': False},
         4, [[[0], [1], [3], [5]]]),
        # Non-causal
        ({'filters': 1, 'kernel_size': 2, 'dilation_rate': 1, 'padding': 'valid',
          'kernel_initializer': 'ones', 'use_bias': False},
         4, [[[1], [3], [5]]]),
        # Causal dilated with larger kernel size
        ({'filters': 1, 'kernel_size': 3, 'dilation_rate': 2, 'padding': 'causal',
          'kernel_initializer': 'ones', 'use_bias': False},
         10, np.float32([[[0], [1], [2], [4], [6], [9], [12], [15], [18], [21]]])),
    ]
)
def test_causal_dilated_conv(layer_kwargs, input_length, expected_output):
    input_data = np.reshape(np.arange(input_length, dtype='float32'),
                            (1, input_length, 1))
    layer_test(convolutional.Conv1D, input_data=input_data,
               kwargs=layer_kwargs, expected_output=expected_output)


@pytest.mark.parametrize(
    'padding,strides',
    [(padding, strides)
     for padding in _convolution_paddings
     for strides in [1, 2]
     if not (padding == 'same' and strides != 1)]
)
def test_conv_1d(padding, strides):
    batch_size = 2
    steps = 8
    input_dim = 2
    kernel_size = 3
    filters = 3

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


@pytest.mark.skipif((K.backend() == 'cntk' and K.dev.type() == 0),
                    reason='cntk only support dilated conv on GPU')
def test_conv_1d_dilation():
    batch_size = 2
    steps = 8
    input_dim = 2
    kernel_size = 3
    filters = 3
    padding = _convolution_paddings[-1]

    layer_test(convolutional.Conv1D,
               kwargs={'filters': filters,
                       'kernel_size': kernel_size,
                       'padding': padding,
                       'dilation_rate': 2},
               input_shape=(batch_size, steps, input_dim))


def test_conv_1d_channels_first():
    batch_size = 2
    steps = 8
    input_dim = 2
    kernel_size = 3
    filters = 3

    layer_test(convolutional.Conv1D,
               kwargs={'filters': filters,
                       'kernel_size': kernel_size,
                       'data_format': 'channels_first'},
               input_shape=(batch_size, input_dim, steps))


@pytest.mark.parametrize(
    'strides,padding',
    [(strides, padding)
     for padding in _convolution_paddings
     for strides in [(1, 1), (2, 2)]
     if not (padding == 'same' and strides != (1, 1))]
)
def test_convolution_2d(strides, padding):
    num_samples = 2
    filters = 2
    stack_size = 3
    kernel_size = (3, 2)
    num_row = 7
    num_col = 6

    layer_test(convolutional.Conv2D,
               kwargs={'filters': filters,
                       'kernel_size': kernel_size,
                       'padding': padding,
                       'strides': strides,
                       'data_format': 'channels_first'},
               input_shape=(num_samples, stack_size, num_row, num_col))


def test_convolution_2d_channels_last():
    num_samples = 2
    filters = 2
    stack_size = 3
    num_row = 7
    num_col = 6
    padding = 'valid'
    strides = (2, 2)

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


@pytest.mark.skipif((K.backend() == 'cntk' and K.dev.type() == 0),
                    reason='cntk only supports dilated conv on GPU')
def test_convolution_2d_dilation():
    num_samples = 2
    filters = 2
    stack_size = 3
    kernel_size = (3, 2)
    num_row = 7
    num_col = 6
    padding = 'valid'

    layer_test(convolutional.Conv2D,
               kwargs={'filters': filters,
                       'kernel_size': kernel_size,
                       'padding': padding,
                       'dilation_rate': (2, 2)},
               input_shape=(num_samples, num_row, num_col, stack_size))


def test_convolution_2d_invalid():
    filters = 2
    padding = _convolution_paddings[-1]
    kernel_size = (3, 2)

    with pytest.raises(ValueError):
        model = Sequential([convolutional.Conv2D(
            filters=filters, kernel_size=kernel_size, padding=padding,
            batch_input_shape=(None, None, 5, None))])


@pytest.mark.parametrize(
    'padding,out_padding,strides',
    [(padding, out_padding, strides)
     for padding in _convolution_paddings
     for out_padding in [None, (0, 0), (1, 1)]
     for strides in [(1, 1), (2, 2)]
     if (not (padding == 'same' and strides != (1, 1))
         and not(strides == (1, 1) and out_padding == (1, 1)))]
)
def test_conv2d_transpose(padding, out_padding, strides):
    num_samples = 2
    filters = 2
    stack_size = 3
    num_row = 5
    num_col = 6

    layer_test(convolutional.Conv2DTranspose,
               kwargs={'filters': filters,
                       'kernel_size': 3,
                       'padding': padding,
                       'output_padding': out_padding,
                       'strides': strides,
                       'data_format': 'channels_last'},
               input_shape=(num_samples, num_row, num_col, stack_size),
               fixed_batch_size=True)


@pytest.mark.skipif((K.backend() == 'cntk' and K.dev.type() == 0),
                    reason='cntk only supports dilated conv transpose on GPU')
def test_conv2d_transpose_dilation():

    layer_test(convolutional.Conv2DTranspose,
               kwargs={'filters': 2,
                       'kernel_size': 3,
                       'padding': 'same',
                       'data_format': 'channels_last',
                       'dilation_rate': (2, 2)},
               input_shape=(2, 5, 6, 3))

    # Check dilated conv transpose returns expected output
    input_data = np.arange(48).reshape((1, 4, 4, 3)).astype(np.float32)
    expected_output = np.float32([[192, 228, 192, 228],
                                  [336, 372, 336, 372],
                                  [192, 228, 192, 228],
                                  [336, 372, 336, 372]]).reshape((1, 4, 4, 1))

    layer_test(convolutional.Conv2DTranspose,
               input_data=input_data,
               kwargs={'filters': 1,
                       'kernel_size': 3,
                       'padding': 'same',
                       'data_format': 'channels_last',
                       'dilation_rate': (2, 2),
                       'kernel_initializer': 'ones'},
               expected_output=expected_output)


def test_conv2d_transpose_channels_first():
    num_samples = 2
    filters = 2
    stack_size = 3
    num_row = 5
    num_col = 6
    padding = 'valid'
    strides = (2, 2)

    layer_test(convolutional.Conv2DTranspose,
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


def test_conv2d_transpose_invalid():
    filters = 2
    stack_size = 3
    num_row = 5
    num_col = 6
    padding = 'valid'

    with pytest.raises(ValueError):
        model = Sequential([convolutional.Conv2DTranspose(
            filters=filters,
            kernel_size=3,
            padding=padding,
            use_bias=True,
            batch_input_shape=(None, None, 5, None))])

    # Test invalid output padding for given stride. Output padding equal to stride
    with pytest.raises(ValueError):
        model = Sequential([convolutional.Conv2DTranspose(
            filters=filters,
            kernel_size=3,
            padding=padding,
            output_padding=(0, 3),
            strides=(1, 3),
            batch_input_shape=(None, num_row, num_col, stack_size))])

    # Output padding greater than stride
    with pytest.raises(ValueError):
        model = Sequential([convolutional.Conv2DTranspose(
            filters=filters,
            kernel_size=3,
            padding=padding,
            output_padding=(2, 2),
            strides=(1, 3),
            batch_input_shape=(None, num_row, num_col, stack_size))])


@pytest.mark.parametrize(
    'padding,strides,multiplier,dilation_rate',
    [(padding, strides, multiplier, dilation_rate)
     for padding in _convolution_paddings
     for strides in [1, 2]
     for multiplier in [1, 2]
     for dilation_rate in [1, 2]
     if (not (padding == 'same' and strides != 1)
         and not (dilation_rate != 1 and strides != 1)
         and not (dilation_rate != 1 and K.backend() == 'cntk'))]
)
def test_separable_conv_1d(padding, strides, multiplier, dilation_rate):
    num_samples = 2
    filters = 6
    stack_size = 3
    num_step = 9

    layer_test(convolutional.SeparableConv1D,
               kwargs={'filters': filters,
                       'kernel_size': 3,
                       'padding': padding,
                       'strides': strides,
                       'depth_multiplier': multiplier,
                       'dilation_rate': dilation_rate},
               input_shape=(num_samples, num_step, stack_size))


def test_separable_conv_1d_additional_args():
    num_samples = 2
    filters = 6
    stack_size = 3
    num_step = 9
    padding = 'valid'
    multiplier = 2

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
                       'use_bias': True,
                       'depth_multiplier': multiplier},
               input_shape=(num_samples, stack_size, num_step))


def test_separable_conv_1d_invalid():
    filters = 6
    padding = 'valid'
    with pytest.raises(ValueError):
        model = Sequential([convolutional.SeparableConv1D(
            filters=filters, kernel_size=3, padding=padding,
            batch_input_shape=(None, 5, None))])


@pytest.mark.parametrize(
    'padding,strides,multiplier,dilation_rate',
    [(padding, strides, multiplier, dilation_rate)
     for padding in _convolution_paddings
     for strides in [(1, 1), (2, 2)]
     for multiplier in [1, 2]
     for dilation_rate in [(1, 1), (2, 2), (2, 1), (1, 2)]
     if (not (padding == 'same' and strides != (1, 1))
         and not (dilation_rate != (1, 1) and strides != (1, 1))
         and not (dilation_rate != (1, 1) and multiplier == dilation_rate[0])
         and not (dilation_rate != (1, 1) and K.backend() == 'cntk'))]
)
def test_separable_conv_2d(padding, strides, multiplier, dilation_rate):
    num_samples = 2
    filters = 6
    stack_size = 3
    num_row = 7
    num_col = 6

    layer_test(
        convolutional.SeparableConv2D,
        kwargs={'filters': filters,
                'kernel_size': (3, 3),
                'padding': padding,
                'strides': strides,
                'depth_multiplier': multiplier,
                'dilation_rate': dilation_rate},
        input_shape=(num_samples, num_row, num_col, stack_size))


def test_separable_conv_2d_additional_args():
    num_samples = 2
    filters = 6
    stack_size = 3
    num_row = 7
    num_col = 6
    padding = 'valid'
    strides = (2, 2)
    multiplier = 2

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


def test_separable_conv_2d_invalid():
    filters = 6
    padding = 'valid'
    with pytest.raises(ValueError):
        model = Sequential([convolutional.SeparableConv2D(
            filters=filters, kernel_size=3, padding=padding,
            batch_input_shape=(None, None, 5, None))])


@pytest.mark.parametrize(
    'padding,strides,multiplier',
    [(padding, strides, multiplier)
     for padding in _convolution_paddings
     for strides in [(1, 1), (2, 2)]
     for multiplier in [1, 2]
     if not (padding == 'same' and strides != (1, 1))]
)
def test_depthwise_conv_2d(padding, strides, multiplier):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6

    layer_test(convolutional.DepthwiseConv2D,
               kwargs={'kernel_size': (3, 3),
                       'padding': padding,
                       'strides': strides,
                       'depth_multiplier': multiplier},
               input_shape=(num_samples,
                            num_row,
                            num_col,
                            stack_size))


def test_depthwise_conv_2d_additional_args():
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6
    padding = 'valid'
    strides = (2, 2)
    multiplier = 2

    layer_test(convolutional.DepthwiseConv2D,
               kwargs={'kernel_size': 3,
                       'padding': padding,
                       'data_format': 'channels_first',
                       'activation': None,
                       'depthwise_regularizer': 'l2',
                       'bias_regularizer': 'l2',
                       'activity_regularizer': 'l2',
                       'depthwise_constraint': 'unit_norm',
                       'use_bias': True,
                       'strides': strides,
                       'depth_multiplier': multiplier},
               input_shape=(num_samples, stack_size, num_row, num_col))


def test_depthwise_conv_2d_invalid():
    padding = 'valid'
    with pytest.raises(ValueError):
        Sequential([convolutional.DepthwiseConv2D(
            kernel_size=3,
            padding=padding,
            batch_input_shape=(None, None, 5, None))])


@pytest.mark.parametrize(
    'padding,strides',
    [(padding, strides)
     for padding in _convolution_paddings
     for strides in [(1, 1, 1), (2, 2, 2)]
     if not (padding == 'same' and strides != (1, 1, 1))]
)
def test_convolution_3d(padding, strides):
    num_samples = 2
    filters = 2
    stack_size = 3

    input_len_dim1 = 9
    input_len_dim2 = 8
    input_len_dim3 = 8

    layer_test(convolutional.Convolution3D,
               kwargs={'filters': filters,
                       'kernel_size': 3,
                       'padding': padding,
                       'strides': strides},
               input_shape=(num_samples,
                            input_len_dim1, input_len_dim2, input_len_dim3,
                            stack_size))


def test_convolution_3d_additional_args():
    num_samples = 2
    filters = 2
    stack_size = 3
    padding = 'valid'
    strides = (2, 2, 2)

    input_len_dim1 = 9
    input_len_dim2 = 8
    input_len_dim3 = 8

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


@pytest.mark.parametrize(
    'padding,out_padding,strides,data_format',
    [(padding, out_padding, strides, data_format)
     for padding in _convolution_paddings
     for out_padding in [None, (0, 0, 0), (1, 1, 1)]
     for strides in [(1, 1, 1), (2, 2, 2)]
     for data_format in ['channels_first', 'channels_last']
     if (not (padding == 'same' and strides != (1, 1, 1))
         and not (strides == (1, 1, 1) and out_padding == (1, 1, 1)))]
)
def test_conv3d_transpose(padding, out_padding, strides, data_format):
    filters = 2
    stack_size = 3
    num_depth = 7
    num_row = 5
    num_col = 6

    layer_test(
        convolutional.Conv3DTranspose,
        kwargs={'filters': filters,
                'kernel_size': 3,
                'padding': padding,
                'output_padding': out_padding,
                'strides': strides,
                'data_format': data_format},
        input_shape=(None, num_depth, num_row, num_col, stack_size),
        fixed_batch_size=True)


def test_conv3d_transpose_additional_args():
    filters = 2
    stack_size = 3
    num_depth = 7
    num_row = 5
    num_col = 6
    padding = 'valid'
    strides = (2, 2, 2)

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
                       'use_bias': True,
                       'strides': strides},
               input_shape=(None, stack_size, num_depth, num_row, num_col),
               fixed_batch_size=True)


def test_conv3d_transpose_invalid():
    filters = 2
    stack_size = 3
    num_depth = 7
    num_row = 5
    num_col = 6
    padding = 'valid'

    # Test invalid use case
    with pytest.raises(ValueError):
        model = Sequential([convolutional.Conv3DTranspose(
            filters=filters,
            kernel_size=3,
            padding=padding,
            batch_input_shape=(None, None, 5, None, None))])

    # Test invalid output padding for given stride. Output padding equal
    # to stride
    with pytest.raises(ValueError):
        model = Sequential([convolutional.Conv3DTranspose(
            filters=filters,
            kernel_size=3,
            padding=padding,
            output_padding=(0, 3, 3),
            strides=(1, 3, 4),
            batch_input_shape=(None, num_depth, num_row, num_col, stack_size))])

    # Output padding greater than stride
    with pytest.raises(ValueError):
        model = Sequential([convolutional.Conv3DTranspose(
            filters=filters,
            kernel_size=3,
            padding=padding,
            output_padding=(2, 2, 3),
            strides=(1, 3, 4),
            batch_input_shape=(None, num_depth, num_row, num_col, stack_size))])


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


@pytest.mark.parametrize(
    'data_format,padding',
    [(data_format, padding)
     for data_format in ['channels_first', 'channels_last']
     for padding in [(2, 2), ((1, 2), (3, 4))]]
)
def test_zero_padding_2d(data_format, padding):
    num_samples = 2
    stack_size = 2
    input_num_row = 4
    input_num_col = 5

    if data_format == 'channels_last':
        inputs = np.ones((num_samples, input_num_row, input_num_col, stack_size))
    else:
        inputs = np.ones((num_samples, stack_size, input_num_row, input_num_col))

    layer_test(convolutional.ZeroPadding2D,
               kwargs={'padding': padding, 'data_format': data_format},
               input_shape=inputs.shape)


def test_zero_padding_2d_correctness():
    num_samples = 2
    stack_size = 2
    input_num_row = 4
    input_num_col = 5
    inputs = np.ones((num_samples, stack_size, input_num_row, input_num_col))

    for data_format in ['channels_first', 'channels_last']:
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


@pytest.mark.parametrize(
    'data_format,padding',
    [(data_format, padding)
     for data_format in ['channels_first', 'channels_last']
     for padding in [(2, 2, 2), ((1, 2), (3, 4), (0, 2))]]
)
def test_zero_padding_3d(data_format, padding):
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 4
    input_len_dim2 = 5
    input_len_dim3 = 3
    inputs = np.ones((num_samples,
                     input_len_dim1, input_len_dim2, input_len_dim3,
                     stack_size))

    layer_test(convolutional.ZeroPadding3D,
               kwargs={'padding': padding, 'data_format': data_format},
               input_shape=inputs.shape)


def test_zero_padding_3d_correctness():
    num_samples = 2
    stack_size = 2
    input_len_dim1 = 4
    input_len_dim2 = 5
    input_len_dim3 = 3
    inputs = np.ones((num_samples,
                      input_len_dim1, input_len_dim2, input_len_dim3,
                      stack_size))

    for data_format in ['channels_first', 'channels_last']:
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


def test_upsampling_1d():
    layer_test(convolutional.UpSampling1D,
               kwargs={'size': 2},
               input_shape=(3, 5, 4))


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
                    reason='cntk does not support it yet')
@pytest.mark.parametrize('data_format',
                         ['channels_first', 'channels_last'])
def test_upsampling_2d_bilinear(data_format):
    num_samples = 2
    stack_size = 2
    input_num_row = 11
    input_num_col = 12

    if data_format == 'channels_first':
        inputs = np.random.rand(num_samples, stack_size, input_num_row,
                                input_num_col)
    else:  # tf
        inputs = np.random.rand(num_samples, input_num_row, input_num_col,
                                stack_size)

    # basic test
    layer_test(convolutional.UpSampling2D,
               kwargs={'size': (2, 2),
                       'data_format': data_format,
                       'interpolation': 'bilinear'},
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


@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason='CNTK does not support float64')
@pytest.mark.parametrize(
    'input_shape,conv_class',
    [((2, 4, 2), convolutional.Conv1D),
     ((2, 4, 4, 2), convolutional.Conv2D),
     ((2, 4, 4, 4, 2), convolutional.Conv3D)]
)
def test_conv_float64(input_shape, conv_class):
    kernel_size = 3
    strides = 1
    filters = 3
    K.set_floatx('float64')
    layer_test(conv_class,
               kwargs={'filters': filters,
                       'kernel_size': kernel_size,
                       'padding': 'valid',
                       'strides': strides},
               input_shape=input_shape)
    K.set_floatx('float32')


if __name__ == '__main__':
    pytest.main([__file__])
