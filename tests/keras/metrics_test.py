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
