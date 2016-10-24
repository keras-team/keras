import pytest
import numpy as np

from keras import objectives
from keras import backend as K


allobj = [objectives.mean_squared_error,
          objectives.mean_absolute_error,
          objectives.mean_absolute_percentage_error,
          objectives.mean_squared_logarithmic_error,
          objectives.squared_hinge,
          objectives.hinge, objectives.categorical_crossentropy,
          objectives.binary_crossentropy,
          objectives.kullback_leibler_divergence,
          objectives.poisson,
          objectives.cosine_proximity,
          objectives.q1,
          objectives.q5,
          objectives.q10,
          objectives.q50,
          objectives.q90,
          objectives.q95,
          objectives.q99]


def test_objective_shapes_3d():
    y_a = K.variable(np.random.random((5, 6, 7)))
    y_b = K.variable(np.random.random((5, 6, 7)))
    for obj in allobj:
        objective_output = obj(y_a, y_b)
        assert K.eval(objective_output).shape == (5, 6)


def test_objective_shapes_2d():
    y_a = K.variable(np.random.random((6, 7)))
    y_b = K.variable(np.random.random((6, 7)))
    for obj in allobj:
        objective_output = obj(y_a, y_b)
        assert K.eval(objective_output).shape == (6,)


def test_cce_one_hot():
    y_a = K.variable(np.random.randint(0, 7, (5, 6)))
    y_b = K.variable(np.random.random((5, 6, 7)))
    objective_output = objectives.sparse_categorical_crossentropy(y_a, y_b)
    assert K.eval(objective_output).shape == (5, 6)

    y_a = K.variable(np.random.randint(0, 7, (6,)))
    y_b = K.variable(np.random.random((6, 7)))
    assert K.eval(objectives.sparse_categorical_crossentropy(y_a, y_b)).shape == (6,)

def test_q50_abs():
    """
    This function tests if the output of the mean absolute and the 50th quantile
    are the same.
    """
    y_a = K.variable((6, 1))
    y_b = K.variable((6, 1))
    q50_out = objectives.q50(y_a, y_b)
    abs_out = objectives.mean_absolute_error(y_a, y_b)

    assert np.abs(K.eval(q50_out - abs_out))<1e-06


if __name__ == "__main__":
    pytest.main([__file__])
