import unittest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers import core


class TestLayerBase(unittest.TestCase):
    def test_input_output(self):
        nb_samples = 10
        input_dim = 5
        layer = core.Layer()

        # Once an input is provided, it should be reachable through the
        # appropriate getters
        input = np.ones((nb_samples, input_dim))
        layer.input = K.variable(input)
        for train in [True, False]:
            assert_allclose(K.eval(layer.get_input(train)), input)
            assert_allclose(K.eval(layer.get_output(train)), input)

    def test_connections(self):
        nb_samples = 10
        input_dim = 5
        layer1 = core.Layer()
        layer2 = core.Layer()

        input = np.ones((nb_samples, input_dim))
        layer1.input = K.variable(input)

        # After connecting, input of layer1 should be passed through
        layer2.set_previous(layer1)
        for train in [True, False]:
            assert_allclose(K.eval(layer2.get_input(train)), input)
            assert_allclose(K.eval(layer2.get_output(train)), input)


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
        layer_1.set_input_shape((None,))
        layer_2.set_input_shape((None,))
        layer = core.Merge([layer_1, layer_2])
        self._runner(layer)

    def test_dropout(self):
        layer = core.Dropout(0.5)
        self._runner(layer)

    def test_activation(self):
        layer = core.Activation('linear')
        self._runner(layer)

    def test_reshape(self):
        layer = core.Reshape(dims=(10, 10))
        self._runner(layer)

    def test_flatten(self):
        layer = core.Flatten()
        self._runner(layer)

    def test_repeat_vector(self):
        layer = core.RepeatVector(10)
        self._runner(layer)

    def test_dense(self):
        layer = core.Dense(10, input_shape=(10,))
        self._runner(layer)

    def test_act_reg(self):
        layer = core.ActivityRegularization(0.5, 0.5)
        self._runner(layer)

    def test_time_dist_dense(self):
        layer = core.TimeDistributedDense(10, input_shape=(None, 10))
        self._runner(layer)

    def test_time_dist_merge(self):
        layer = core.TimeDistributedMerge()
        self._runner(layer)

    def test_autoencoder(self):
        layer_1 = core.Layer()
        layer_2 = core.Layer()

        layer = core.AutoEncoder(layer_1, layer_2)
        self._runner(layer)

    def test_maxout_dense(self):
        layer = core.MaxoutDense(10, 10)
        self._runner(layer)


class TestMasking(unittest.TestCase):
    """Test the Masking class"""

    def test_sequences(self):
        """Test masking sequences with zeroes as padding"""
        if K._BACKEND == "tensorflow":
            return
        # integer inputs, one per timestep, like embeddings
        layer = core.Masking()
        func = K.function([layer.input], [layer.get_output_mask()])
        input_data = np.array([[[1], [2], [3], [0]],
                              [[0], [4], [5], [0]]], dtype=np.int32)

        # This is the expected output mask, one dimension less
        expected = np.array([[1, 1, 1, 0], [0, 1, 1, 0]])

        # get mask for this input
        output = func([input_data])[0]
        self.assertTrue(np.all(output == expected))

    def test_non_zero(self):
        """Test masking with non-zero mask value"""
        if K._BACKEND == "tensorflow":
            return
        layer = core.Masking(5)
        func = K.function([layer.input], [layer.get_output_mask()])
        input_data = np.array([[[1, 1], [2, 1], [3, 1], [5, 5]],
                              [[1, 5], [5, 0], [0, 0], [0, 0]]],
                              dtype=np.int32)
        output = func([input_data])[0]
        expected = np.array([[1, 1, 1, 0], [1, 1, 1, 1]])
        self.assertTrue(np.all(output == expected))

    def test_non_zero_output(self):
        """Test output of masking layer with non-zero mask value"""
        if K._BACKEND == "tensorflow":
            return
        layer = core.Masking(5)
        func = K.function([layer.input], [layer.get_output()])

        input_data = np.array([[[1, 1], [2, 1], [3, 1], [5, 5]],
                              [[1, 5], [5, 0], [0, 0], [0, 0]]],
                              dtype=np.int32)
        output = func([input_data])[0]
        expected = np.array([[[1, 1], [2, 1], [3, 1], [0, 0]],
                            [[1, 5], [5, 0], [0, 0], [0, 0]]])
        self.assertTrue(np.all(output == expected))


if __name__ == '__main__':
    unittest.main()
