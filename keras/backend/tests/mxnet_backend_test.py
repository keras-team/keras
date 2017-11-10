import pytest
from keras.backend import mxnet_backend as K
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

class TestKerasSymbol(object):

    def test_symbol_addtion(self):
        symbol1 = K.placeholder(shape=(2,))
        symbol2 = K.placeholder(shape=(2,))
        symbol3 = symbol1 + symbol2
        assert symbol1 in symbol3.get_neighbor()
        assert symbol2 in symbol3.get_neighbor()
        assert symbol3 in symbol1.get_neighbor()
        assert symbol3 in symbol2.get_neighbor()

    def test_variable_addition(self):
        var1 = K.variable([1, 2, 3])
        var2 = K.variable([2, 3, 4])
        var3 = var1 + var2
        assert var1 in var3.get_neighbor()
        assert var2 in var3.get_neighbor()
        assert var3 in var1.get_neighbor()
        assert var3 in var2.get_neighbor()
        assert_array_equal(K.eval(var3), np.array([3, 5, 7]))


if __name__ == '__main__':
    pytest.main([__file__])
