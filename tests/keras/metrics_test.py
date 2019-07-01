"""Tests for Keras metrics classes."""
import pytest
import numpy as np

from keras import metrics
from keras import backend as K


class TestKerasSum:

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
