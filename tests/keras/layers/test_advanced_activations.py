import pytest
from keras.utils.test_utils import test_layer


def test_leaky_relu():
    from keras.layers.advanced_activations import LeakyReLU
    for alpha in [0., .5, -1.]:
        test_layer(LeakyReLU, kwargs={'alpha': alpha},
                   input_shape=(2, 3, 4))


def test_prelu():
    from keras.layers.advanced_activations import PReLU
    test_layer(PReLU, kwargs={},
               input_shape=(2, 3, 4))


def test_elu():
    from keras.layers.advanced_activations import ELU
    for alpha in [0., .5, -1.]:
        test_layer(ELU, kwargs={'alpha': alpha},
                   input_shape=(2, 3, 4))


def test_parametric_softplus():
    from keras.layers.advanced_activations import ParametricSoftplus
    for alpha in [0., .5, -1.]:
        test_layer(ParametricSoftplus,
                   kwargs={'alpha_init': 1.,
                           'beta_init': -1},
                   input_shape=(2, 3, 4))


def test_thresholded_linear():
    from keras.layers.advanced_activations import ThresholdedLinear
    test_layer(ThresholdedLinear, kwargs={'theta': 0.5},
               input_shape=(2, 3, 4))


def test_thresholded_relu():
    from keras.layers.advanced_activations import ThresholdedReLU
    test_layer(ThresholdedReLU, kwargs={'theta': 0.5},
               input_shape=(2, 3, 4))


def test_srelu():
    from keras.layers.advanced_activations import SReLU
    test_layer(SReLU, kwargs={},
               input_shape=(2, 3, 4))


if __name__ == '__main__':
    # pytest.main([__file__])
    test_srelu()
