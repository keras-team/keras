import pytest
from numpy.testing import assert_allclose
import numpy as np
from keras import backend as K


def get_standard_values():
    '''
    These are just a set of floats used for testing the activation
    functions, and are useful in multiple tests.

    The values should all be non-negative because they and their negatives
    are used to test the ReLU derivates in this file.
    '''
    return np.array([[0, 0.1, 0.5, 0.9, 1.0, 10, 1e2, 0.01]], dtype=K.floatx())


def test_leaky_relu():
    np.random.seed(1337)
    from keras.layers.advanced_activations import LeakyReLU
    inp = get_standard_values()
    for alpha in [0., .5, -1.]:
        layer = LeakyReLU(alpha=alpha)
        layer.input = K.variable(inp)
        for train in [True, False]:
            outp = K.eval(layer.get_output(train))
            assert_allclose(outp, inp)

        layer.input = K.variable(-inp)
        for train in [True, False]:
            outp = K.eval(layer.get_output(train))
            assert_allclose(outp, -inp*alpha)

        config = layer.get_config()
        assert config['alpha'] == alpha


def test_prelu():
    from keras.layers.advanced_activations import PReLU
    np.random.seed(1337)
    inp = get_standard_values()

    for train in [True, False]:
        # test with custom weights
        alphas = np.random.random(inp.shape)
        layer = PReLU(weights=alphas, input_shape=inp.flatten().shape)
        # calling build here causes an error, unclear if this is a bug
        # layer.build()

        layer.input = K.variable(inp)
        outp = K.eval(layer.get_output(train))
        assert_allclose(inp, outp)

        layer.input = K.variable(-inp)
        outp = K.eval(layer.get_output(train))
        assert_allclose(-alphas*inp, outp)

        # test with default weights
        layer = PReLU(input_shape=inp.flatten().shape)
        # layer.build()
        layer.input = K.variable(inp)
        outp = K.eval(layer.get_output(train))
        assert_allclose(inp, outp)

        layer.input = K.variable(-inp)
        outp = K.eval(layer.get_output(train))

        assert_allclose(0., alphas*outp)

        layer.get_config()


def test_elu():
    from keras.layers.advanced_activations import ELU
    np.random.seed(1337)
    inp = get_standard_values()
    for alpha in [0.1, .5, -1., 1.]:
        layer = ELU(alpha=alpha)
        layer.input = K.variable(inp)
        for train in [True, False]:
            outp = K.eval(layer.get_output(train))
            assert_allclose(outp, inp, rtol=1e-3)

        layer.input = K.variable(-inp)
        for train in [True, False]:
            outp = K.eval(layer.get_output(train))
            assert_allclose(outp, alpha*(np.exp(-inp)-1.), rtol=1e-3)

        config = layer.get_config()
        assert config['alpha'] == alpha


@pytest.mark.skipif(K._BACKEND == 'tensorflow',
                    reason='currently not working with TensorFlow')
def test_parametric_softplus():
    from keras.layers.advanced_activations import ParametricSoftplus
    np.random.seed(1337)
    inp = np.vstack((get_standard_values(), -get_standard_values()))
    # large values cause overflow in exp
    inp = inp[:-2]
    for alpha in [.5, -1., 1., 5]:
        for beta in [.5, -1., 1., 2]:
            layer = ParametricSoftplus(alpha_init=alpha,
                                       beta_init=beta,
                                       input_shape=inp.shape)
            layer.input = K.variable(inp)
            layer.build()
            for train in [True, False]:
                outp = K.eval(layer.get_output(train))
                assert_allclose(outp, alpha*np.log(1.+np.exp(beta*inp)),
                                atol=1e-3)

            config = layer.get_config()
            assert config['alpha_init'] == alpha
            assert config['beta_init'] == beta


@pytest.mark.skipif(K._BACKEND == 'tensorflow',
                    reason='currently not working with TensorFlow')
def test_thresholded_linear():
    from keras.layers.advanced_activations import ThresholdedLinear
    np.random.seed(1337)
    inp = get_standard_values()
    for theta in [0., .5, 1.]:
        layer = ThresholdedLinear(theta=theta)
        layer.input = K.variable(inp)
        for train in [True, False]:
            outp = K.eval(layer.get_output(train))
            assert_allclose(outp, inp*(np.abs(inp) >= theta))

        layer.input = K.variable(-inp)
        for train in [True, False]:
            outp = K.eval(layer.get_output(train))
            assert_allclose(outp, -inp*(np.abs(inp) >= theta))

        config = layer.get_config()
        assert config['theta'] == theta


@pytest.mark.skipif(K._BACKEND == 'tensorflow',
                    reason='currently not working with TensorFlow')
def test_thresholded_relu():
    from keras.layers.advanced_activations import ThresholdedReLU
    np.random.seed(1337)
    inp = get_standard_values()
    for theta in [-1, 0., .5, 1.]:
        layer = ThresholdedReLU(theta=theta)
        layer.input = K.variable(inp)
        for train in [True, False]:
            outp = K.eval(layer.get_output(train))
            assert_allclose(outp, inp*(inp > theta))

        layer.input = K.variable(-inp)
        for train in [True, False]:
            outp = K.eval(layer.get_output(train))
            assert_allclose(outp, -inp*(-inp > theta))

        config = layer.get_config()
        assert config['theta'] == theta


if __name__ == '__main__':
    pytest.main([__file__])
