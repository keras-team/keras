import unittest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers import convolutional


class TestConvolutions(unittest.TestCase):
    def test_convolution_1d(self):
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

    def test_maxpooling_1d(self):
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

    def test_convolution_2d(self):
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

    def test_maxpooling_2d(self):
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

    def test_zero_padding_2d(self):
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

    def test_upsampling_1d(self):
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

    def test_upsampling_2d(self):
        nb_samples = 9
        stack_size = 7
        input_nb_row = 11
        input_nb_col = 12

        input = np.ones((nb_samples, stack_size, input_nb_row, input_nb_col))

        for length_row in [2, 3, 9]:
            for length_col in [2, 3, 9]:
                layer = convolutional.UpSampling2D(size=(length_row, length_col))
                layer.input = K.variable(input)
                for train in [True, False]:
                    out = K.eval(layer.get_output(train))
                    assert out.shape[2] == length_row * input_nb_row
                    assert out.shape[3] == length_col * input_nb_col
            layer.get_config()

if __name__ == '__main__':
    unittest.main()
