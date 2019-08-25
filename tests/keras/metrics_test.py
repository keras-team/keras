"""Tests for Keras metrics classes."""
import pytest
import numpy as np

from keras import metrics
from keras import backend as K


if K.backend() != 'tensorflow':
    # Need TensorFlow to use metric.__call__
    pytestmark = pytest.mark.skip


class TestSum(object):

    def test_sum(self):
        m = metrics.Sum(name='my_sum', dtype='float32')

        # check config
        assert m.name == 'my_sum'
        assert m.stateful
        assert m.dtype == 'float32'
        assert len(m.weights) == 1

        # check initial state
        assert K.eval(m.total) == 0

        # check __call__
        assert K.eval(m(100.0)) == 100
        assert K.eval(m.total) == 100

        # check update_state() and result() + state accumulation + tensor input
        K.eval(m.update_state([1, 5]))
        assert np.isclose(K.eval(m.result()), 106)
        assert K.eval(m.total) == 106  # 100 + 1 + 5

        # check reset_states()
        m.reset_states()
        assert K.eval(m.total) == 0

    def test_sum_with_sample_weight(self):
        m = metrics.Sum(dtype='float64')
        assert m.dtype == 'float64'

        # check scalar weight
        result_t = m(100, sample_weight=0.5)
        assert K.eval(result_t) == 50
        assert K.eval(m.total) == 50

        # check weights not scalar and weights rank matches values rank
        result_t = m([1, 5], sample_weight=[1, 0.2])
        result = K.eval(result_t)
        assert np.isclose(result, 52.)  # 50 + 1 + 5 * 0.2
        assert np.isclose(K.eval(m.total), 52.)

        # check weights broadcast
        result_t = m([1, 2], sample_weight=0.5)
        assert np.isclose(K.eval(result_t), 53.5)  # 52 + 0.5 + 1
        assert np.isclose(K.eval(m.total), 53.5)

        # check weights squeeze
        result_t = m([1, 5], sample_weight=[[1], [0.2]])
        assert np.isclose(K.eval(result_t), 55.5)  # 53.5 + 1 + 1
        assert np.isclose(K.eval(m.total), 55.5)

        # check weights expand
        result_t = m([[1], [5]], sample_weight=[1, 0.2])
        assert np.isclose(K.eval(result_t), 57.5, 2)  # 55.5 + 1 + 1
        assert np.isclose(K.eval(m.total), 57.5, 1)

        # check values reduced to the dimensions of weight
        result_t = m([[[1., 2.], [3., 2.], [0.5, 4.]]], sample_weight=[0.5])
        result = np.round(K.eval(result_t), decimals=2)
        # result = (prev: 57.5) + 0.5 + 1 + 1.5 + 1 + 0.25 + 2
        assert np.isclose(result, 63.75, 2)
        assert np.isclose(K.eval(m.total), 63.75, 2)


class TestMean(object):

    def test_mean(self):
        m = metrics.Mean(name='my_mean')

        # check config
        assert m.name == 'my_mean'
        assert m.stateful
        assert m.dtype == 'float32'
        assert len(m.weights) == 2

        # check initial state
        assert K.eval(m.total) == 0
        assert K.eval(m.count) == 0

        # check __call__()
        assert K.eval(m(100)) == 100
        assert K.eval(m.total) == 100
        assert K.eval(m.count) == 1

        # check update_state() and result()
        update_op = m.update_state([1, 5])
        K.eval(update_op)
        assert np.isclose(K.eval(m.result()), 106 / 3)
        assert K.eval(m.total) == 106  # 100 + 1 + 5
        assert K.eval(m.count) == 3

        # check reset_states()
        m.reset_states()
        assert K.eval(m.total) == 0
        assert K.eval(m.count) == 0

        # Check save and restore config
        m2 = metrics.Mean.from_config(m.get_config())
        assert m2.name == 'my_mean'
        assert m2.stateful
        assert m2.dtype == 'float32'
        assert len(m2.weights) == 2

    def test_mean_with_sample_weight(self):
        m = metrics.Mean(dtype='float64')
        assert m.dtype == 'float64'

        # check scalar weight
        result_t = m(100, sample_weight=0.5)
        assert K.eval(result_t) == 50 / 0.5
        assert K.eval(m.total) == 50
        assert K.eval(m.count) == 0.5

        # check weights not scalar and weights rank matches values rank
        result_t = m([1, 5], sample_weight=[1, 0.2])
        result = K.eval(result_t)
        assert np.isclose(result, 52 / 1.7)
        assert np.isclose(K.eval(m.total), 52)  # 50 + 1 + 5 * 0.2
        assert np.isclose(K.eval(m.count), 1.7)  # 0.5 + 1.2

        # check weights broadcast
        result_t = m([1, 2], sample_weight=0.5)
        assert np.isclose(K.eval(result_t), 53.5 / 2.7, rtol=3)
        assert np.isclose(K.eval(m.total), 53.5, rtol=3)  # 52 + 0.5 + 1
        assert np.isclose(K.eval(m.count), 2.7, rtol=3)  # 1.7 + 0.5 + 0.5

        # check weights squeeze
        result_t = m([1, 5], sample_weight=[[1], [0.2]])
        assert np.isclose(K.eval(result_t), 55.5 / 3.9, rtol=3)
        assert np.isclose(K.eval(m.total), 55.5, rtol=3)  # 53.5 + 1 + 1
        assert np.isclose(K.eval(m.count), 3.9, rtol=3)  # 2.7 + 1.2

        # check weights expand
        result_t = m([[1], [5]], sample_weight=[1, 0.2])
        assert np.isclose(K.eval(result_t), 57.5 / 5.1, rtol=3)
        assert np.isclose(K.eval(m.total), 57.5, rtol=3)  # 55.5 + 1 + 1
        assert np.isclose(K.eval(m.count), 5.1, rtol=3)  # 3.9 + 1.2

    def test_multiple_instances(self):
        m = metrics.Mean()
        m2 = metrics.Mean()

        assert m.name == 'mean'
        assert m2.name == 'mean'

        # check initial state
        assert K.eval(m.total) == 0
        assert K.eval(m.count) == 0
        assert K.eval(m2.total) == 0
        assert K.eval(m2.count) == 0

        # check __call__()
        assert K.eval(m(100)) == 100
        assert K.eval(m.total) == 100
        assert K.eval(m.count) == 1
        assert K.eval(m2.total) == 0
        assert K.eval(m2.count) == 0

        assert K.eval(m2([63, 10])) == 36.5
        assert K.eval(m2.total) == 73
        assert K.eval(m2.count) == 2
        assert K.eval(m.result()) == 100
        assert K.eval(m.total) == 100
        assert K.eval(m.count) == 1


class TestMeanSquaredErrorTest(object):

    def test_config(self):
        mse_obj = metrics.MeanSquaredError(name='my_mse', dtype='int32')
        assert mse_obj.name == 'my_mse'
        assert mse_obj.dtype == 'int32'

        # Check save and restore config
        mse_obj2 = metrics.MeanSquaredError.from_config(mse_obj.get_config())
        assert mse_obj2.name == 'my_mse'
        assert mse_obj2.dtype == 'int32'

    def test_unweighted(self):
        mse_obj = metrics.MeanSquaredError()
        y_true = ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        y_pred = ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))

        update_op = mse_obj.update_state(y_true, y_pred)
        K.eval(update_op)
        result = mse_obj.result()
        np.isclose(0.5, K.eval(result), atol=1e-5)

    def test_weighted(self):
        mse_obj = metrics.MeanSquaredError()
        y_true = ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1), (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        y_pred = ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1), (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        sample_weight = (1., 1.5, 2., 2.5)
        result = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        np.isclose(0.54285, K.eval(result), atol=1e-5)
