import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers import convolutional


def test_convolution_1d():
    nb_samples = 9
    nb_steps = 7
    input_dim = 10
    filter_length = 6
    nb_filter = 5

    weights_in = [np.ones((nb_filter, input_dim, filter_length, 1)),
                  np.ones(nb_filter)]

    input = np.ones((nb_samples, nb_steps, input_dim))
    for weight in [None, weights_in]:
        for border_mode in ['valid', 'same']:
            for subsample_length in [1]:
                if border_mode == 'same' and subsample_length != 1:
                    continue
                for W_regularizer in [None, 'l2']:
                    for b_regularizer in [None, 'l2']:
                        for act_regularizer in [None, 'l2']:
                            layer = convolutional.Convolution1D(
                                nb_filter, filter_length,
                                weights=weight,
                                border_mode=border_mode,
                                W_regularizer=W_regularizer,
                                b_regularizer=b_regularizer,
                                activity_regularizer=act_regularizer,
                                subsample_length=subsample_length,
                                input_shape=(None, input_dim))

                        layer.input = K.variable(input)
                        for train in [True, False]:
                            out = K.eval(layer.get_output(train))
                            assert input.shape[0] == out.shape[0]
                            if border_mode == 'same' and subsample_length == 1:
                                assert input.shape[1] == out.shape[1]
                        layer.get_config()


def test_maxpooling_1d():
    nb_samples = 9
    nb_steps = 7
    input_dim = 10

    input = np.ones((nb_samples, nb_steps, input_dim))
    for stride in [1, 2]:
        layer = convolutional.MaxPooling1D(stride=stride,
                                           border_mode='valid')
        layer.input = K.variable(input)
        for train in [True, False]:
            K.eval(layer.get_output(train))
        layer.get_config()


def test_averagepooling_1d():
    nb_samples = 9
    nb_steps = 7
    input_dim = 10

    input = np.ones((nb_samples, nb_steps, input_dim))
    for stride in [1, 2]:
        layer = convolutional.AveragePooling1D(stride=stride,
                                               border_mode='valid')
        layer.input = K.variable(input)
        for train in [True, False]:
            K.eval(layer.get_output(train))
        layer.get_config()


def test_convolution_2d():
    nb_samples = 8
    nb_filter = 9
    stack_size = 7
    nb_row = 10
    nb_col = 6

    input_nb_row = 11
    input_nb_col = 12

    weights_in = [np.ones((nb_filter, stack_size, nb_row, nb_col)), np.ones(nb_filter)]

    input = np.ones((nb_samples, stack_size, input_nb_row, input_nb_col))
    for weight in [None, weights_in]:
        for border_mode in ['valid', 'same']:
            for subsample in [(1, 1), (2, 2)]:
                if border_mode == 'same' and subsample != (1, 1):
                    continue
                for W_regularizer in [None, 'l2']:
                    for b_regularizer in [None, 'l2']:
                        for act_regularizer in [None, 'l2']:
                            layer = convolutional.Convolution2D(
                                nb_filter, nb_row, nb_col,
                                weights=weight,
                                border_mode=border_mode,
                                W_regularizer=W_regularizer,
                                b_regularizer=b_regularizer,
                                activity_regularizer=act_regularizer,
                                subsample=subsample,
                                input_shape=(stack_size, None, None))

                            layer.input = K.variable(input)
                            for train in [True, False]:
                                out = K.eval(layer.get_output(train))
                                if border_mode == 'same' and subsample == (1, 1):
                                    assert out.shape[2:] == input.shape[2:]
                            layer.get_config()


def test_convolution_2d_dim_ordering():
    nb_filter = 4
    nb_row = 3
    nb_col = 2
    stack_size = 3

    np.random.seed(1337)
    weights = [np.random.random((nb_filter, stack_size, nb_row, nb_col)),
               np.random.random(nb_filter)]
    input = np.random.random((1, stack_size, 10, 10))

    layer = convolutional.Convolution2D(
        nb_filter, nb_row, nb_col,
        weights=weights,
        input_shape=input.shape[1:],
        dim_ordering='th')
    layer.input = K.variable(input)
    out_th = K.eval(layer.get_output(False))

    input = np.transpose(input, (0, 2, 3, 1))
    weights[0] = np.transpose(weights[0], (2, 3, 1, 0))
    layer = convolutional.Convolution2D(
        nb_filter, nb_row, nb_col,
        weights=weights,
        input_shape=input.shape[1:],
        dim_ordering='tf')
    layer.input = K.variable(input)
    out_tf = K.eval(layer.get_output(False))

    assert_allclose(out_tf, np.transpose(out_th, (0, 2, 3, 1)), atol=1e-05)


def test_maxpooling_2d():
    nb_samples = 9
    stack_size = 7
    input_nb_row = 11
    input_nb_col = 12
    pool_size = (3, 3)

    input = np.ones((nb_samples, stack_size, input_nb_row, input_nb_col))
    for strides in [(1, 1), (2, 2)]:
        layer = convolutional.MaxPooling2D(strides=strides,
                                           border_mode='valid',
                                           pool_size=pool_size)
        layer.input = K.variable(input)
        for train in [True, False]:
            K.eval(layer.get_output(train))
        layer.get_config()


def test_maxpooling_2d_dim_ordering():
    stack_size = 3

    input = np.random.random((1, stack_size, 10, 10))

    layer = convolutional.MaxPooling2D(
        (2, 2),
        input_shape=input.shape[1:],
        dim_ordering='th')
    layer.input = K.variable(input)
    out_th = K.eval(layer.get_output(False))

    input = np.transpose(input, (0, 2, 3, 1))
    layer = convolutional.MaxPooling2D(
        (2, 2),
        input_shape=input.shape[1:],
        dim_ordering='tf')
    layer.input = K.variable(input)
    out_tf = K.eval(layer.get_output(False))

    assert_allclose(out_tf, np.transpose(out_th, (0, 2, 3, 1)), atol=1e-05)


def test_averagepooling_2d():
    nb_samples = 9
    stack_size = 7
    input_nb_row = 11
    input_nb_col = 12
    pool_size = (3, 3)

    input = np.ones((nb_samples, stack_size, input_nb_row, input_nb_col))
    for strides in [(1, 1), (2, 2)]:
        layer = convolutional.AveragePooling2D(strides=strides,
                                               border_mode='valid',
                                               pool_size=pool_size)
        layer.input = K.variable(input)
        for train in [True, False]:
            K.eval(layer.get_output(train))
        layer.get_config()


@pytest.mark.skipif(K._BACKEND != 'theano', reason="Requires Theano backend")
def test_convolution_3d():
    nb_samples = 8
    nb_filter = 9
    stack_size = 7
    len_conv_dim1 = 2
    len_conv_dim2 = 10
    len_conv_dim3 = 6

    input_len_dim1 = 10
    input_len_dim2 = 11
    input_len_dim3 = 12

    weights_in = [np.ones((nb_filter, stack_size, len_conv_dim1, len_conv_dim2, len_conv_dim3)),
                  np.ones(nb_filter)]

    input = np.ones((nb_samples, stack_size, input_len_dim1,
                     input_len_dim2, input_len_dim3))
    for weight in [None, weights_in]:
        for border_mode in ['same', 'valid']:
            for subsample in [(1, 1, 1), (2, 2, 2)]:
                if border_mode == 'same' and subsample != (1, 1, 1):
                    continue
                for W_regularizer in [None, 'l2']:
                    for b_regularizer in [None, 'l2']:
                        for act_regularizer in [None, 'l2']:
                            layer = convolutional.Convolution3D(
                                nb_filter, len_conv_dim1, len_conv_dim2, len_conv_dim3,
                                weights=weight,
                                border_mode=border_mode,
                                W_regularizer=W_regularizer,
                                b_regularizer=b_regularizer,
                                activity_regularizer=act_regularizer,
                                subsample=subsample,
                                input_shape=(stack_size, None, None, None))

                            layer.input = K.variable(input)
                            for train in [True, False]:
                                out = K.eval(layer.get_output(train))
                                if border_mode == 'same' and subsample == (1, 1, 1):
                                    assert out.shape[2:] == input.shape[2:]
                            layer.get_config()


@pytest.mark.skipif(K._BACKEND != 'theano', reason="Requires Theano backend")
def test_maxpooling_3d():
    nb_samples = 9
    stack_size = 7
    input_len_dim1 = 10
    input_len_dim2 = 11
    input_len_dim3 = 12
    pool_size = (3, 3, 3)

    input = np.ones((nb_samples, stack_size, input_len_dim1,
                     input_len_dim2, input_len_dim3))
    for strides in [(1, 1, 1), (2, 2, 2)]:
        layer = convolutional.MaxPooling3D(strides=strides,
                                           border_mode='valid',
                                           pool_size=pool_size)
        layer.input = K.variable(input)
        for train in [True, False]:
            K.eval(layer.get_output(train))
        layer.get_config()


@pytest.mark.skipif(K._BACKEND != 'theano', reason="Requires Theano backend")
def test_averagepooling_3d():
    nb_samples = 9
    stack_size = 7
    input_len_dim1 = 10
    input_len_dim2 = 11
    input_len_dim3 = 12
    pool_size = (3, 3, 3)

    input = np.ones((nb_samples, stack_size, input_len_dim1,
                     input_len_dim2, input_len_dim3))
    for strides in [(1, 1, 1), (2, 2, 2)]:
        layer = convolutional.AveragePooling3D(strides=strides,
                                               border_mode='valid',
                                               pool_size=pool_size)
        layer.input = K.variable(input)
        for train in [True, False]:
            K.eval(layer.get_output(train))
        layer.get_config()


def test_zero_padding_2d():
    nb_samples = 9
    stack_size = 7
    input_nb_row = 11
    input_nb_col = 12

    input = np.ones((nb_samples, stack_size, input_nb_row, input_nb_col))
    layer = convolutional.ZeroPadding2D(padding=(2, 2))
    layer.input = K.variable(input)
    for train in [True, False]:
        out = K.eval(layer.get_output(train))
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
    layer = convolutional.ZeroPadding3D(padding=(2, 2, 2))
    layer.input = K.variable(input)
    for train in [True, False]:
        out = K.eval(layer.get_output(train))
        for offset in [0, 1, -1, -2]:
            assert_allclose(out[:, :, offset, :, :], 0.)
            assert_allclose(out[:, :, :, offset, :], 0.)
            assert_allclose(out[:, :, :, :, offset], 0.)
        assert_allclose(out[:, :, 2:-2, 2:-2, 2:-2], 1.)
    layer.get_config()


def test_upsampling_1d():
    nb_samples = 9
    nb_steps = 7
    input_dim = 10

    input = np.ones((nb_samples, nb_steps, input_dim))
    for length in [2, 3, 9]:
        layer = convolutional.UpSampling1D(length=length)
        layer.input = K.variable(input)
        for train in [True, False]:
            out = K.eval(layer.get_output(train))
            assert out.shape[1] == length * nb_steps
        layer.get_config()


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
                    input_shape=input.shape[1:],
                    dim_ordering=dim_ordering)
                layer.input = K.variable(input)
                for train in [True, False]:
                    out = K.eval(layer.get_output(train))
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

                layer.get_config()


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
                        input_shape=input.shape[1:],
                        dim_ordering=dim_ordering)
                    layer.input = K.variable(input)
                    for train in [True, False]:
                        out = K.eval(layer.get_output(train))
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

                    layer.get_config()


if __name__ == '__main__':
    pytest.main([__file__])
