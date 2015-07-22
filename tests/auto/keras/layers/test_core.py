import unittest
import numpy as np
from numpy.testing import assert_allclose
import theano

from keras.layers import core


class TestLayerBase(unittest.TestCase):
    def test_input_output(self):
        nb_samples = 10
        input_dim = 5
        layer = core.Layer()

        # As long as there is no input, an error should be raised.
        for train in [True, False]:
            self.assertRaises(AttributeError, layer.get_input, train)
            self.assertRaises(AttributeError, layer.get_output, train)

        # Once an input is provided, it should be reachable through the
        # appropriate getters
        input = np.ones((nb_samples, input_dim))
        layer.input = theano.shared(value=input)
        for train in [True, False]:
            assert_allclose(layer.get_input(train).eval(), input)
            assert_allclose(layer.get_output(train).eval(), input)

    def test_connections(self):
        nb_samples = 10
        input_dim = 5
        layer1 = core.Layer()
        layer2 = core.Layer()

        input = np.ones((nb_samples, input_dim))
        layer1.input = theano.shared(value=input)

        # As long as there is no previous layer, an error should be raised.
        for train in [True, False]:
            self.assertRaises(AttributeError, layer2.get_input, train)

        # After connecting, input of layer1 should be passed through
        layer2.set_previous(layer1)
        for train in [True, False]:
            assert_allclose(layer2.get_input(train).eval(), input)
            assert_allclose(layer2.get_output(train).eval(), input)


class TestConfigParams(unittest.TestCase):
    """
    Test the constructor, config and params functions of all layers in core.
    """

    def _runner(self, layer):
        conf = layer.get_config()
        assert (type(conf) == dict)

        param = layer.get_params()
        # Typically a list or a tuple, but may be any iterable
        assert hasattr(param, '__iter__')

    def test_base(self):
        layer = core.Layer()
        self._runner(layer)

    def test_masked(self):
        layer = core.MaskedLayer()
        self._runner(layer)

    def test_merge(self):
        layer_1 = core.Layer()
        layer_2 = core.Layer()
        layer = core.Merge([layer_1, layer_2])
        self._runner(layer)

    def test_dropout(self):
        layer = core.Dropout(0.5)
        self._runner(layer)

    def test_activation(self):
        layer = core.Activation('linear')
        self._runner(layer)

    def test_reshape(self):
        layer = core.Reshape(10, 10)
        self._runner(layer)

    def test_flatten(self):
        layer = core.Flatten()
        self._runner(layer)

    def test_repeat_vector(self):
        layer = core.RepeatVector(10)
        self._runner(layer)

    def test_dense(self):
        layer = core.Dense(10, 10)
        self._runner(layer)

    def test_act_reg(self):
        layer = core.ActivityRegularization(0.5, 0.5)
        self._runner(layer)

    def test_time_dist_dense(self):
        layer = core.TimeDistributedDense(10, 10)
        self._runner(layer)

    def test_autoencoder(self):
        layer_1 = core.Layer()
        layer_2 = core.Layer()

        layer = core.AutoEncoder(layer_1, layer_2)
        self._runner(layer)

    def test_maxout_dense(self):
        layer = core.MaxoutDense(10, 10)
        self._runner(layer)


if __name__ == '__main__':
    unittest.main()
