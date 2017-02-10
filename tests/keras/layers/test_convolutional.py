import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test, keras_test
from keras.utils.np_utils import conv_input_length
from keras import backend as K
from keras.layers import convolutional, pooling


# TensorFlow does not support full convolution.
if K.backend() == 'theano':
    _convolution_paddings = ['valid', 'same', 'full']
else:
    _convolution_paddings = ['valid', 'same']


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
                               'size': kernel_size,
                               'padding': padding,
                               'kernel_regularizer': 'l2',
                               'bias_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'strides': strides},
                       input_shape=(batch_size, steps, input_dim))


@keras_test
def test_maxpooling_1d():
    for padding in ['valid', 'same']:
        for stride in [1, 2]:
            layer_test(convolutional.MaxPooling1D,
                       kwargs={'stride': stride,
                               'padding': padding},
                       input_shape=(3, 5, 4))


@keras_test
def test_averagepooling_1d():
    for stride in [1, 2]:
        layer_test(convolutional.AveragePooling1D,
                   kwargs={'stride': stride,
                           'padding': 'valid'},
                   input_shape=(3, 5, 4))


@keras_test
def test_convolution_2d():
    nb_samples = 2
    filters = 2
    stack_size = 3
    nb_row = 10
    nb_col = 6

    for padding in _convolution_paddings:
        for subsample in [(1, 1), (2, 2)]:
            if padding == 'same' and subsample != (1, 1):
                continue

            layer_test(convolutional.Convolution2D,
                       kwargs={'filters': filters,
                               'nb_row': 3,
                               'nb_col': 3,
                               'padding': padding,
                               'subsample': subsample},
                       input_shape=(nb_samples, nb_row, nb_col, stack_size))

            layer_test(convolutional.Convolution2D,
                       kwargs={'filters': filters,
                               'nb_row': 3,
                               'nb_col': 3,
                               'padding': padding,
                               'W_regularizer': 'l2',
                               'b_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'subsample': subsample},
                       input_shape=(nb_samples, nb_row, nb_col, stack_size))


@keras_test
def test_deconvolution_2d():
    nb_samples = 2
    filters = 2
    stack_size = 3
    nb_row = 10
    nb_col = 6

    for batch_size in [None, nb_samples]:
        for padding in _convolution_paddings:
            for subsample in [(1, 1), (2, 2)]:
                if padding == 'same' and subsample != (1, 1):
                    continue

                rows = conv_input_length(nb_row, 3, padding, subsample[0])
                cols = conv_input_length(nb_col, 3, padding, subsample[1])
                layer_test(convolutional.Deconvolution2D,
                           kwargs={'filters': filters,
                                   'nb_row': 3,
                                   'nb_col': 3,
                                   'output_shape': (batch_size, filters, rows, cols),
                                   'padding': padding,
                                   'subsample': subsample,
                                   'data_format': 'channels_first'},
                           input_shape=(nb_samples, stack_size, nb_row, nb_col),
                           fixed_batch_size=True)

                layer_test(convolutional.Deconvolution2D,
                           kwargs={'filters': filters,
                                   'nb_row': 3,
                                   'nb_col': 3,
                                   'output_shape': (batch_size, filters, rows, cols),
                                   'padding': padding,
                                   'data_format': 'channels_first',
                                   'W_regularizer': 'l2',
                                   'b_regularizer': 'l2',
                                   'activity_regularizer': 'activity_l2',
                                   'subsample': subsample},
                           input_shape=(nb_samples, stack_size, nb_row, nb_col),
                           fixed_batch_size=True)


@keras_test
def test_atrous_conv_2d():
    nb_samples = 2
    filters = 2
    stack_size = 3
    nb_row = 10
    nb_col = 6

    for padding in _convolution_paddings:
        for subsample in [(1, 1), (2, 2)]:
            for atrous_rate in [(1, 1), (2, 2)]:
                if padding == 'same' and subsample != (1, 1):
                    continue
                if subsample != (1, 1) and atrous_rate != (1, 1):
                    continue

                layer_test(convolutional.AtrousConv2D,
                           kwargs={'filters': filters,
                                   'nb_row': 3,
                                   'nb_col': 3,
                                   'padding': padding,
                                   'subsample': subsample,
                                   'atrous_rate': atrous_rate},
                           input_shape=(nb_samples, nb_row, nb_col, stack_size))

                layer_test(convolutional.AtrousConv2D,
                           kwargs={'filters': filters,
                                   'nb_row': 3,
                                   'nb_col': 3,
                                   'padding': padding,
                                   'W_regularizer': 'l2',
                                   'b_regularizer': 'l2',
                                   'activity_regularizer': 'activity_l2',
                                   'subsample': subsample,
                                   'atrous_rate': atrous_rate},
                           input_shape=(nb_samples, nb_row, nb_col, stack_size))


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TF backend')
@keras_test
def test_separable_conv_2d():
    nb_samples = 2
    filters = 6
    stack_size = 3
    nb_row = 10
    nb_col = 6

    for padding in _convolution_paddings:
        for subsample in [(1, 1), (2, 2)]:
            for multiplier in [1, 2]:
                if padding == 'same' and subsample != (1, 1):
                    continue

                layer_test(convolutional.SeparableConv2D,
                           kwargs={'filters': filters,
                                   'nb_row': 3,
                                   'nb_col': 3,
                                   'padding': padding,
                                   'subsample': subsample,
                                   'depth_multiplier': multiplier},
                           input_shape=(nb_samples, nb_row, nb_col, stack_size))

                layer_test(convolutional.SeparableConv2D,
                           kwargs={'filters': filters,
                                   'nb_row': 3,
                                   'nb_col': 3,
                                   'padding': padding,
                                   'depthwise_regularizer': 'l2',
                                   'pointwise_regularizer': 'l2',
                                   'b_regularizer': 'l2',
                                   'activity_regularizer': 'activity_l2',
                                   'pointwise_constraint': 'unitnorm',
                                   'depthwise_constraint': 'unitnorm',
                                   'subsample': subsample,
                                   'depth_multiplier': multiplier},
                           input_shape=(nb_samples, nb_row, nb_col, stack_size))


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
                   input_shape=(3, 11, 12, 4))


@keras_test
def test_averagepooling_2d():
    for padding in ['valid', 'same']:
        for pool_size in [(2, 2), (3, 3), (4, 4), (5, 5)]:
            for strides in [(1, 1), (2, 2)]:
                layer_test(convolutional.AveragePooling2D,
                           kwargs={'strides': strides,
                                   'padding': padding,
                                   'pool_size': pool_size},
                           input_shape=(3, 11, 12, 4))


@keras_test
def test_convolution_3d():
    nb_samples = 2
    filters = 2
    stack_size = 3
    kernel_dim1 = 2
    kernel_dim2 = 3
    kernel_dim3 = 1

    input_len_dim1 = 10
    input_len_dim2 = 11
    input_len_dim3 = 12

    for padding in _convolution_paddings:
        for subsample in [(1, 1, 1), (2, 2, 2)]:
            if padding == 'same' and subsample != (1, 1, 1):
                continue

            layer_test(convolutional.Convolution3D,
                       kwargs={'filters': filters,
                               'kernel_dim1': kernel_dim1,
                               'kernel_dim2': kernel_dim2,
                               'kernel_dim3': kernel_dim3,
                               'padding': padding,
                               'subsample': subsample},
                       input_shape=(nb_samples,
                                    input_len_dim1, input_len_dim2, input_len_dim3,
                                    stack_size))

            layer_test(convolutional.Convolution3D,
                       kwargs={'filters': filters,
                               'kernel_dim1': kernel_dim1,
                               'kernel_dim2': kernel_dim2,
                               'kernel_dim3': kernel_dim3,
                               'padding': padding,
                               'W_regularizer': 'l2',
                               'b_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'subsample': subsample},
                       input_shape=(nb_samples,
                                    input_len_dim1, input_len_dim2, input_len_dim3,
                                    stack_size))


@keras_test
def test_maxpooling_3d():
    pool_size = (3, 3, 3)

    for strides in [(1, 1, 1), (2, 2, 2)]:
        layer_test(convolutional.MaxPooling3D,
                   kwargs={'strides': strides,
                           'padding': 'valid',
                           'pool_size': pool_size},
                   input_shape=(3, 4, 11, 12, 10))


@keras_test
def test_averagepooling_3d():
    pool_size = (3, 3, 3)

    for strides in [(1, 1, 1), (2, 2, 2)]:
        layer_test(convolutional.AveragePooling3D,
                   kwargs={'strides': strides,
                           'padding': 'valid',
                           'pool_size': pool_size},
                   input_shape=(3, 4, 11, 12, 10))


@keras_test
def test_zero_padding_1d():
    nb_samples = 2
    input_dim = 2
    nb_steps = 5
    shape = (nb_samples, nb_steps, input_dim)
    input = np.ones(shape)

    # basic test
    layer_test(convolutional.ZeroPadding1D,
               kwargs={'padding': 2},
               input_shape=input.shape)
    layer_test(convolutional.ZeroPadding1D,
               kwargs={'padding': (1, 2)},
               input_shape=input.shape)
    layer_test(convolutional.ZeroPadding1D,
               kwargs={'padding': {'left_pad': 1, 'right_pad': 2}},
               input_shape=input.shape)

    # correctness test
    layer = convolutional.ZeroPadding1D(padding=2)
    layer.build(shape)
    output = layer(K.variable(input))
    np_output = K.eval(output)
    for offset in [0, 1, -1, -2]:
        assert_allclose(np_output[:, offset, :], 0.)
    assert_allclose(np_output[:, 2:-2, :], 1.)

    layer = convolutional.ZeroPadding1D(padding=(1, 2))
    layer.build(shape)
    output = layer(K.variable(input))
    np_output = K.eval(output)
    for left_offset in [0]:
        assert_allclose(np_output[:, left_offset, :], 0.)
    for right_offset in [-1, -2]:
        assert_allclose(np_output[:, right_offset, :], 0.)
    assert_allclose(np_output[:, 1:-2, :], 1.)
    layer.get_config()


@keras_test
def test_zero_padding_2d():
    nb_samples = 2
    stack_size = 2
    input_nb_row = 4
    input_nb_col = 5
    data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {"channels_last", "channels_first"}'

    if data_format == 'channels_last':
        input = np.ones((nb_samples, input_nb_row, input_nb_col, stack_size))
    elif data_format == 'channels_first':
        input = np.ones((nb_samples, stack_size, input_nb_row, input_nb_col))

    # basic test
    layer_test(convolutional.ZeroPadding2D,
               kwargs={'padding': (2, 2)},
               input_shape=input.shape)
    layer_test(convolutional.ZeroPadding2D,
               kwargs={'padding': (1, 2, 3, 4)},
               input_shape=input.shape)
    layer_test(convolutional.ZeroPadding2D,
               kwargs={'padding': {'top_pad': 1, 'bottom_pad': 2, 'left_pad': 3, 'right_pad': 4}},
               input_shape=input.shape)

    # correctness test
    layer = convolutional.ZeroPadding2D(padding=(2, 2))
    layer.build(input.shape)
    output = layer(K.variable(input))
    np_output = K.eval(output)
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

    layer = convolutional.ZeroPadding2D(padding=(1, 2, 3, 4))
    layer.build(input.shape)
    output = layer(K.variable(input))
    np_output = K.eval(output)
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
    layer.get_config()


def test_zero_padding_3d():
    nb_samples = 2
    stack_size = 2
    input_len_dim1 = 4
    input_len_dim2 = 5
    input_len_dim3 = 3

    input = np.ones((nb_samples,
                     input_len_dim1, input_len_dim2, input_len_dim3,
                     stack_size))

    # basic test
    layer_test(convolutional.ZeroPadding3D,
               kwargs={'padding': (2, 2, 2)},
               input_shape=input.shape)

    # correctness test
    layer = convolutional.ZeroPadding3D(padding=(2, 2, 2))
    layer.build(input.shape)
    output = layer(K.variable(input))
    np_output = K.eval(output)
    for offset in [0, 1, -1, -2]:
        assert_allclose(np_output[:, offset, :, :, :], 0.)
        assert_allclose(np_output[:, :, offset, :, :], 0.)
        assert_allclose(np_output[:, :, :, offset, :], 0.)
    assert_allclose(np_output[:, 2:-2, 2:-2, 2:-2, :], 1.)
    layer.get_config()


@keras_test
def test_upsampling_1d():
    layer_test(convolutional.UpSampling1D,
               kwargs={'length': 2},
               input_shape=(3, 5, 4))


@keras_test
def test_upsampling_2d():
    nb_samples = 2
    stack_size = 2
    input_nb_row = 11
    input_nb_col = 12

    for data_format in ['channels_first', 'channels_last']:
        if data_format == 'channels_first':
            input = np.random.rand(nb_samples, stack_size, input_nb_row,
                                   input_nb_col)
        else:  # tf
            input = np.random.rand(nb_samples, input_nb_row, input_nb_col,
                                   stack_size)

        for length_row in [2, 3, 9]:
            for length_col in [2, 3, 9]:
                layer = convolutional.UpSampling2D(
                    size=(length_row, length_col),
                    data_format=data_format)
                layer.build(input.shape)
                output = layer(K.variable(input))
                np_output = K.eval(output)
                if data_format == 'channels_first':
                    assert np_output.shape[2] == length_row * input_nb_row
                    assert np_output.shape[3] == length_col * input_nb_col
                else:  # tf
                    assert np_output.shape[1] == length_row * input_nb_row
                    assert np_output.shape[2] == length_col * input_nb_col

                # compare with numpy
                if data_format == 'channels_first':
                    expected_out = np.repeat(input, length_row, axis=2)
                    expected_out = np.repeat(expected_out, length_col, axis=3)
                else:  # tf
                    expected_out = np.repeat(input, length_row, axis=1)
                    expected_out = np.repeat(expected_out, length_col, axis=2)

                assert_allclose(np_output, expected_out)


def test_upsampling_3d():
    nb_samples = 2
    stack_size = 2
    input_len_dim1 = 10
    input_len_dim2 = 11
    input_len_dim3 = 12

    for data_format in ['channels_first', 'channels_last']:
        if data_format == 'channels_first':
            input = np.random.rand(nb_samples, stack_size, input_len_dim1, input_len_dim2,
                                   input_len_dim3)
        else:  # tf
            input = np.random.rand(nb_samples, input_len_dim1, input_len_dim2, input_len_dim3,
                                   stack_size)
        for length_dim1 in [2, 3, 9]:
            for length_dim2 in [2, 3, 9]:
                for length_dim3 in [2, 3, 9]:
                    layer = convolutional.UpSampling3D(
                        size=(length_dim1, length_dim2, length_dim3),
                        data_format=data_format)
                    layer.build(input.shape)
                    output = layer(K.variable(input))
                    np_output = K.eval(output)
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
                        expected_out = np.repeat(input, length_dim1, axis=2)
                        expected_out = np.repeat(expected_out, length_dim2, axis=3)
                        expected_out = np.repeat(expected_out, length_dim3, axis=4)
                    else:  # tf
                        expected_out = np.repeat(input, length_dim1, axis=1)
                        expected_out = np.repeat(expected_out, length_dim2, axis=2)
                        expected_out = np.repeat(expected_out, length_dim3, axis=3)

                    assert_allclose(np_output, expected_out)


@keras_test
def test_cropping_1d():
    nb_samples = 2
    time_length = 4
    input_len_dim1 = 2
    input = np.random.rand(nb_samples, time_length, input_len_dim1)

    layer_test(convolutional.Cropping1D,
               kwargs={'cropping': (2, 2)},
               input_shape=input.shape)


def test_cropping_2d():
    nb_samples = 2
    stack_size = 2
    input_len_dim1 = 8
    input_len_dim2 = 8
    cropping = ((2, 2), (3, 3))
    data_format = K.image_data_format()

    if data_format == 'channels_first':
        input = np.random.rand(nb_samples, stack_size,
                               input_len_dim1, input_len_dim2)
    else:
        input = np.random.rand(nb_samples,
                               input_len_dim1, input_len_dim2,
                               stack_size)
    # basic test
    layer_test(convolutional.Cropping2D,
               kwargs={'cropping': cropping,
                       'data_format': data_format},
               input_shape=input.shape)
    # correctness test
    layer = convolutional.Cropping2D(cropping=cropping,
                                     data_format=data_format)
    layer.build(input.shape)
    output = layer(K.variable(input))
    np_output = K.eval(output)
    # compare with numpy
    if data_format == 'channels_first':
        expected_out = input[:,
                             :,
                             cropping[0][0]: -cropping[0][1],
                             cropping[1][0]: -cropping[1][1]]
    else:
        expected_out = input[:,
                             cropping[0][0]: -cropping[0][1],
                             cropping[1][0]: -cropping[1][1],
                             :]
    assert_allclose(np_output, expected_out)
    # another correctness test (no cropping)
    cropping = ((0, 0), (0, 0))
    layer = convolutional.Cropping2D(cropping=cropping,
                                     data_format=data_format)
    layer.build(input.shape)
    output = layer(K.variable(input))
    np_output = K.eval(output)
    # compare with input
    assert_allclose(np_output, input)


def test_cropping_3d():
    nb_samples = 2
    stack_size = 2
    input_len_dim1 = 8
    input_len_dim2 = 8
    input_len_dim3 = 8
    cropping = ((2, 2), (3, 3), (2, 3))
    data_format = K.image_data_format()

    if data_format == 'channels_first':
        input = np.random.rand(nb_samples, stack_size,
                               input_len_dim1, input_len_dim2, input_len_dim3)
    else:
        input = np.random.rand(nb_samples,
                               input_len_dim1, input_len_dim2,
                               input_len_dim3, stack_size)
    # basic test
    layer_test(convolutional.Cropping3D,
               kwargs={'cropping': cropping,
                       'data_format': data_format},
               input_shape=input.shape)
    # correctness test
    layer = convolutional.Cropping3D(cropping=cropping,
                                     data_format=data_format)
    layer.build(input.shape)
    output = layer(K.variable(input))
    np_output = K.eval(output)
    # compare with numpy
    if data_format == 'channels_first':
        expected_out = input[:,
                             :,
                             cropping[0][0]: -cropping[0][1],
                             cropping[1][0]: -cropping[1][1],
                             cropping[2][0]: -cropping[2][1]]
    else:
        expected_out = input[:,
                             cropping[0][0]: -cropping[0][1],
                             cropping[1][0]: -cropping[1][1],
                             cropping[2][0]: -cropping[2][1],
                             :]
    assert_allclose(np_output, expected_out)
    # another correctness test (no cropping)
    cropping = ((0, 0), (0, 0), (0, 0))
    layer = convolutional.Cropping3D(cropping=cropping,
                                     data_format=data_format)
    layer.build(input.shape)
    output = layer(K.variable(input))
    np_output = K.eval(output)
    # compare with input
    assert_allclose(np_output, input)

if __name__ == '__main__':
    pytest.main([__file__])
