import pytest
from keras.utils.test_utils import layer_test
from keras.utils.test_utils import keras_test
from keras import layers


@keras_test
def test_leaky_relu():
    for alpha in [0., .5, -1.]:
        layer_test(layers.LeakyReLU, kwargs={'alpha': alpha},
                   input_shape=(2, 3, 4))


@keras_test
def test_prelu():
    layer_test(layers.PReLU, kwargs={},
               input_shape=(2, 3, 4))


@keras_test
def test_prelu_share():
    layer_test(layers.PReLU, kwargs={'shared_axes': 1},
               input_shape=(2, 3, 4))


@keras_test
def test_elu():
    for alpha in [0., .5, -1.]:
        layer_test(layers.ELU, kwargs={'alpha': alpha},
                   input_shape=(2, 3, 4))


@keras_test
def test_thresholded_relu():
    layer_test(layers.ThresholdedReLU, kwargs={'theta': 0.5},
               input_shape=(2, 3, 4))


@keras_test
def test_softmax():
    for axis in [1, -1]:
        layer_test(layers.Softmax, kwargs={'axis': axis},
                   input_shape=(2, 3, 4))


if __name__ == '__main__':
    pytest.main([__file__])
