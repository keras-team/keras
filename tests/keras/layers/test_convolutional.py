import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test
from keras import backend as K
from keras.layers import convolutional


def test_convolution_1d():
    nb_samples = 2
    nb_steps = 8
    input_dim = 5
    filter_length = 3
    nb_filter = 4

    for border_mode in ['valid', 'same']:
        for subsample_length in [1]:
            if border_mode == 'same' and subsample_length != 1:
                continue
            layer_test(convolutional.Convolution1D,
                       kwargs={'nb_filter': nb_filter,
                               'filter_length': filter_length,
                               'border_mode': border_mode,
                               'subsample_length': subsample_length},
                       input_shape=(nb_samples, nb_steps, input_dim))

            layer_test(convolutional.Convolution1D,
                       kwargs={'nb_filter': nb_filter,
                               'filter_length': filter_length,
                               'border_mode': border_mode,
                               'W_regularizer': 'l2',
                               'b_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'subsample_length': subsample_length},
                       input_shape=(nb_samples, nb_steps, input_dim))


def test_maxpooling_1d():
    for stride in [1, 2]:
        layer_test(convolutional.MaxPooling1D,
                   kwargs={'stride': stride,
                           'border_mode': 'valid'},
                   input_shape=(3, 5, 4))


def test_averagepooling_1d():
    for stride in [1, 2]:
        layer_test(convolutional.AveragePooling1D,
                   kwargs={'stride': stride,
                           'border_mode': 'valid'},
                   input_shape=(3, 5, 4))


def test_convolution_2d():
    nb_samples = 8
    nb_filter = 3
    stack_size = 4
    nb_row = 10
    nb_col = 6

    for border_mode in ['valid', 'same']:
        for subsample in [(1, 1), (2, 2)]:
            if border_mode == 'same' and subsample != (1, 1):
                continue

            layer_test(convolutional.Convolution2D,
                       kwargs={'nb_filter': nb_filter,
                               'nb_row': 3,
                               'nb_col': 3,
                               'border_mode': border_mode,
                               'subsample': subsample},
                       input_shape=(nb_samples, stack_size, nb_row, nb_col))

            layer_test(convolutional.Convolution2D,
                       kwargs={'nb_filter': nb_filter,
                               'nb_row': 3,
                               'nb_col': 3,
                               'border_mode': border_mode,
                               'W_regularizer': 'l2',
                               'b_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'subsample': subsample},
                       input_shape=(nb_samples, stack_size, nb_row, nb_col))


def test_maxpooling_2d():
    pool_size = (3, 3)

    for strides in [(1, 1), (2, 2)]:
        layer_test(convolutional.MaxPooling2D,
                   kwargs={'strides': strides,
                           'border_mode': 'valid',
                           'pool_size': pool_size},
                   input_shape=(3, 4, 11, 12))


def test_averagepooling_2d():
    pool_size = (3, 3)

    for border_mode in ['valid', 'same']:
        for pool_size in [(2, 2), (3, 3), (4, 4), (5, 5)]:
            for strides in [(1, 1), (2, 2)]:
                layer_test(convolutional.MaxPooling2D,
                           kwargs={'strides': strides,
                                   'border_mode': border_mode,
                                   'pool_size': pool_size},
                           input_shape=(3, 4, 11, 12))


def test_convolution_3d():
    nb_samples = 2
    nb_filter = 5
    stack_size = 4
    kernel_dim1 = 2
    kernel_dim2 = 3
    kernel_dim3 = 1

    input_len_dim1 = 10
    input_len_dim2 = 11
    input_len_dim3 = 12

    for border_mode in ['same', 'valid']:
        for subsample in [(1, 1, 1), (2, 2, 2)]:
            if border_mode == 'same' and subsample != (1, 1, 1):
                continue

            layer_test(convolutional.Convolution3D,
                       kwargs={'nb_filter': nb_filter,
                               'kernel_dim1': kernel_dim1,
                               'kernel_dim2': kernel_dim2,
                               'kernel_dim3': kernel_dim3,
                               'border_mode': border_mode,
                               'subsample': subsample},
                       input_shape=(nb_samples, stack_size,
                                    input_len_dim1, input_len_dim2, input_len_dim3))

            layer_test(convolutional.Convolution3D,
                       kwargs={'nb_filter': nb_filter,
                               'kernel_dim1': kernel_dim1,
                               'kernel_dim2': kernel_dim2,
                               'kernel_dim3': kernel_dim3,
                               'border_mode': border_mode,
                               'W_regularizer': 'l2',
                               'b_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'subsample': subsample},
                       input_shape=(nb_samples, stack_size,
                                    input_len_dim1, input_len_dim2, input_len_dim3))


def test_maxpooling_3d():
    pool_size = (3, 3, 3)

    for strides in [(1, 1, 1), (2, 2, 2)]:
        layer_test(convolutional.MaxPooling3D,
                   kwargs={'strides': strides,
                           'border_mode': 'valid',
                           'pool_size': pool_size},
                   input_shape=(3, 4, 11, 12, 10))


def test_averagepooling_3d():
    pool_size = (3, 3, 3)

    for strides in [(1, 1, 1), (2, 2, 2)]:
        layer_test(convolutional.AveragePooling3D,
                   kwargs={'strides': strides,
                           'border_mode': 'valid',
                           'pool_size': pool_size},
                   input_shape=(3, 4, 11, 12, 10))


def test_zero_padding_2d():
    nb_samples = 9
    stack_size = 7
    input_nb_row = 11
    input_nb_col = 12

    input = np.ones((nb_samples, stack_size, input_nb_row, input_nb_col))

    # basic test
    layer_test(convolutional.ZeroPadding2D,
               kwargs={'padding': (2, 2)},
               input_shape=input.shape)

    # correctness test
    layer = convolutional.ZeroPadding2D(padding=(2, 2))
    layer.set_input(K.variable(input), shape=input.shape)

    out = K.eval(layer.output)
    for offset in [0, 1, -1, -2]:
        assert_allclose(out[:, :, offset, :], 0.)
        assert_allclose(out[:, :, :, offset], 0.)
    assert_allclose(out[:, :, 2:-2, 2:-2], 1.)
    layer.get_config()


@pytest.mark.skipif(K._BACKEND != 'theano', reason="Requires Theano backend")
def test_zero_padding_3d():
    nb_samples = 9
    stack_size = 7
    input_len_dim1 = 10
    input_len_dim2 = 11
    input_len_dim3 = 12

    input = np.ones((nb_samples, stack_size, input_len_dim1,
                     input_len_dim2, input_len_dim3))

    # basic test
    layer_test(convolutional.ZeroPadding3D,
               kwargs={'padding': (2, 2, 2)},
               input_shape=input.shape)

    # correctness test
    layer = convolutional.ZeroPadding3D(padding=(2, 2, 2))
    layer.set_input(K.variable(input), shape=input.shape)
    out = K.eval(layer.output)
    for offset in [0, 1, -1, -2]:
        assert_allclose(out[:, :, offset, :, :], 0.)
        assert_allclose(out[:, :, :, offset, :], 0.)
        assert_allclose(out[:, :, :, :, offset], 0.)
    assert_allclose(out[:, :, 2:-2, 2:-2, 2:-2], 1.)
    layer.get_config()


def test_upsampling_1d():
    layer_test(convolutional.UpSampling1D,
               kwargs={'length': 2},
               input_shape=(3, 5, 4))


def test_upsampling_2d():
    nb_samples = 9
    stack_size = 7
    input_nb_row = 11
    input_nb_col = 12

    for dim_ordering in ['th', 'tf']:
        if dim_ordering == 'th':
            input = np.random.rand(nb_samples, stack_size, input_nb_row,
                                   input_nb_col)
        else:  # tf
            input = np.random.rand(nb_samples, input_nb_row, input_nb_col,
                                   stack_size)

        for length_row in [2, 3, 9]:
            for length_col in [2, 3, 9]:
                layer = convolutional.UpSampling2D(
                    size=(length_row, length_col),
                    dim_ordering=dim_ordering)
                layer.set_input(K.variable(input), shape=input.shape)

                out = K.eval(layer.output)
                if dim_ordering == 'th':
                    assert out.shape[2] == length_row * input_nb_row
                    assert out.shape[3] == length_col * input_nb_col
                else:  # tf
                    assert out.shape[1] == length_row * input_nb_row
                    assert out.shape[2] == length_col * input_nb_col

                # compare with numpy
                if dim_ordering == 'th':
                    expected_out = np.repeat(input, length_row, axis=2)
                    expected_out = np.repeat(expected_out, length_col, axis=3)
                else:  # tf
                    expected_out = np.repeat(input, length_row, axis=1)
                    expected_out = np.repeat(expected_out, length_col, axis=2)

                assert_allclose(out, expected_out)


@pytest.mark.skipif(K._BACKEND != 'theano', reason="Requires Theano backend")
def test_upsampling_3d():
    nb_samples = 9
    stack_size = 7
    input_len_dim1 = 10
    input_len_dim2 = 11
    input_len_dim3 = 12

    for dim_ordering in ['th', 'tf']:
        if dim_ordering == 'th':
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
                        dim_ordering=dim_ordering)
                    layer.set_input(K.variable(input), shape=input.shape)

                    out = K.eval(layer.output)
                    if dim_ordering == 'th':
                        assert out.shape[2] == length_dim1 * input_len_dim1
                        assert out.shape[3] == length_dim2 * input_len_dim2
                        assert out.shape[4] == length_dim3 * input_len_dim3
                    else:  # tf
                        assert out.shape[1] == length_dim1 * input_len_dim1
                        assert out.shape[2] == length_dim2 * input_len_dim2
                        assert out.shape[3] == length_dim3 * input_len_dim3

                    # compare with numpy
                    if dim_ordering == 'th':
                        expected_out = np.repeat(input, length_dim1, axis=2)
                        expected_out = np.repeat(expected_out, length_dim2, axis=3)
                        expected_out = np.repeat(expected_out, length_dim3, axis=4)
                    else:  # tf
                        expected_out = np.repeat(input, length_dim1, axis=1)
                        expected_out = np.repeat(expected_out, length_dim2, axis=2)
                        expected_out = np.repeat(expected_out, length_dim3, axis=3)

                    assert_allclose(out, expected_out)


if __name__ == '__main__':
    # pytest.main([__file__])
    test_convolution_3d()
