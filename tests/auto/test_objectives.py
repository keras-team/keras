import unittest
import numpy as np
from numpy.testing import assert_allclose
from keras.objectives import mean_squared_error, root_mean_squared_error, mean_absolute_error, \
    mean_squared_logarithmic_error
from keras import backend as K

y_true = np.random.rand(250).astype(np.float32)
y_pred = np.random.rand(250).astype(np.float32)


class TestObjectives(unittest.TestCase):
    def test_mse(self):
        assert K.eval(mean_squared_error(y_true, y_true)) == 0.0
        assert_allclose(K.eval(mean_squared_error(y_true, y_pred)), np.mean((y_true - y_pred) ** 2), atol=1e-06)

    def test_rmse(self):
        assert K.eval(root_mean_squared_error(y_true, y_true)) == 0.0
        assert_allclose(K.eval(root_mean_squared_error(y_true, y_pred)), np.sqrt(np.mean((y_true - y_pred) ** 2)),
                        atol=1e-06)

    def test_mae(self):
        assert K.eval(mean_absolute_error(y_true, y_true)) == 0.0
        assert_allclose(K.eval(mean_absolute_error(y_true, y_pred)), np.abs(y_true - y_pred).mean(), atol=1e-06)

    def test_msle(self):
        assert K.eval(mean_squared_logarithmic_error(y_true, y_true)) == 0.0
        assert_allclose(K.eval(mean_squared_logarithmic_error(y_true, y_pred)),
                        np.mean((np.log(y_true + 1) - np.log(y_pred + 1)) ** 2), atol=1e-06)


if __name__ == '__main__':
    print('Test objectives')
    unittest.main()
