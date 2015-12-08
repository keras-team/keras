import unittest
import numpy as np

from keras import backend as K
from keras.layers.core import *
from keras.layers.convolutional import *
from keras.layers.recurrent import SimpleRNN


def check_layer_output_shape(layer, input_data):
    ndim = len(input_data.shape)
    layer.input = K.placeholder(ndim=ndim)
    layer.set_input_shape(input_data.shape)
    expected_output_shape = layer.output_shape[1:]

    function = K.function([layer.input], [layer.get_output()])
    output = function([input_data])[0]
    assert output.shape[1:] == expected_output_shape, str(output.shape[1:]) + ' != ' + str(expected_output_shape)


class TestShapeInference(unittest.TestCase):
    # ########
    # # Core #
    # ########
    def test_Reshape(self):
        layer = Reshape(dims=(2, 3))
        input_data = np.random.random((2, 6))
        check_layer_output_shape(layer, input_data)

    def test_Permute(self):
        layer = Permute(dims=(1, 3, 2))
        input_data = np.random.random((2, 2, 4, 3))
        check_layer_output_shape(layer, input_data)

    def test_Flatten(self):
        layer = Flatten()
        input_data = np.random.random((2, 2, 3))
        check_layer_output_shape(layer, input_data)

    def test_RepeatVector(self):
        layer = RepeatVector(2)
        input_data = np.random.random((2, 2))
        check_layer_output_shape(layer, input_data)

    def test_Dense(self):
        layer = Dense(3)
        input_data = np.random.random((2, 2))
        check_layer_output_shape(layer, input_data)

    def test_TimeDistributedDense(self):
        layer = TimeDistributedDense(2)
        input_data = np.random.random((2, 2, 3))
        check_layer_output_shape(layer, input_data)

    #################
    # Convolutional #
    #################
    def test_Convolution1D(self):
        for border_mode in ['same', 'valid']:
            for filter_length in [2, 3]:
                for subsample_length in [1]:
                    if subsample_length > 1 and border_mode == 'same':
                        continue
                    for input_data_shape in [(2, 3, 4), (2, 4, 4)]:
                        layer = Convolution1D(nb_filter=1,
                                              filter_length=filter_length,
                                              border_mode=border_mode,
                                              subsample_length=subsample_length)
                        input_data = np.random.random(input_data_shape)
                        check_layer_output_shape(layer, input_data)

    def test_Convolution2D(self):
        for border_mode in ['same', 'valid']:
            for nb_row, nb_col in [(2, 2), (3, 3)]:
                for subsample in [(1, 1), (2, 2)]:
                    if (subsample[0] > 1 or subsample[1] > 1) and border_mode == 'same':
                        continue
                    for input_data_shape in [(2, 1, 3, 3), (2, 1, 4, 4)]:
                        layer = Convolution2D(nb_filter=1, nb_row=nb_row,
                                              nb_col=nb_row,
                                              border_mode=border_mode,
                                              subsample=subsample,
                                              dim_ordering='th')
                        input_data = np.random.random(input_data_shape)
                        check_layer_output_shape(layer, input_data)

                    for input_data_shape in [(2, 3, 3, 1)]:
                        layer = Convolution2D(nb_filter=1, nb_row=nb_row,
                                              nb_col=nb_row,
                                              border_mode=border_mode,
                                              subsample=subsample,
                                              dim_ordering='tf')
                        input_data = np.random.random(input_data_shape)
                        check_layer_output_shape(layer, input_data)

    def test_MaxPooling1D(self):
        for ignore_border in [True, False]:
            for pool_length in [1, 2]:
                for stride in [1]:
                    for input_data_shape in [(2, 3, 4), (2, 4, 4)]:
                        layer = MaxPooling1D(pool_length=pool_length,
                                             stride=stride,
                                             border_mode='valid')
                        input_data = np.random.random(input_data_shape)
                        check_layer_output_shape(layer, input_data)

    def test_MaxPooling2D(self):
        for ignore_border in [True, False]:
            for strides in [(1, 1), (2, 2)]:
                for pool_size in [(2, 2), (3, 3), (4, 4)]:
                    for input_data_shape in [(2, 1, 4, 4), (2, 1, 5, 5), (2, 1, 6, 6)]:
                        layer = MaxPooling2D(pool_size=pool_size,
                                             strides=strides,
                                             border_mode='valid',
                                             dim_ordering='th')
                        input_data = np.random.random(input_data_shape)
                        check_layer_output_shape(layer, input_data)

                    for input_data_shape in [(2, 4, 4, 1)]:
                        layer = MaxPooling2D(pool_size=pool_size,
                                             strides=strides,
                                             border_mode='valid',
                                             dim_ordering='tf')
                        input_data = np.random.random(input_data_shape)
                        check_layer_output_shape(layer, input_data)

    def test_UpSampling1D(self):
        layer = UpSampling1D(length=2)
        input_data = np.random.random((2, 2, 3))
        check_layer_output_shape(layer, input_data)

    def test_UpSampling2D(self):
        layer = UpSampling2D(size=(2, 2), dim_ordering='th')
        input_data = np.random.random((2, 1, 2, 3))
        check_layer_output_shape(layer, input_data)

        layer = UpSampling2D(size=(2, 2), dim_ordering='tf')
        input_data = np.random.random((2, 2, 3, 1))
        check_layer_output_shape(layer, input_data)

    def test_ZeroPadding1D(self):
        layer = ZeroPadding1D(1)
        input_data = np.random.random((2, 2, 1))
        check_layer_output_shape(layer, input_data)

    def test_ZeroPadding2D(self):
        layer = ZeroPadding2D((1, 2), dim_ordering='th')
        input_data = np.random.random((2, 1, 2, 3))
        check_layer_output_shape(layer, input_data)

        layer = ZeroPadding2D((1, 2), dim_ordering='tf')
        input_data = np.random.random((2, 2, 3, 1))
        check_layer_output_shape(layer, input_data)

    # #############
    # # Recurrent #
    # #############
    def test_SimpleRNN(self):
        # all recurrent layers inherit output_shape
        # from the same base recurrent layer
        layer = SimpleRNN(2)
        input_data = np.random.random((2, 2, 3))
        check_layer_output_shape(layer, input_data)


if __name__ == "__main__":
    unittest.main()
