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
          objectives.cosine_proximity]


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


def test_sparse_softmax_categorical_crossentropy():
    y_labels = K.variable(np.random.randint(0, 7, (5, 6)))
    y_logits = K.variable(np.random.random((5, 6, 7)) * 12)
    objective_value = K.eval(objectives.sparse_softmax_categorical_crossentropy(y_labels, y_logits))
    assert objective_value.shape == (5, 6)

    y_a = K.variable(np.random.randint(0, 7, (6,)))
    y_b = K.variable(np.random.random((6, 7)))
    objective_value = K.eval(objectives.sparse_softmax_categorical_crossentropy(y_a, y_b))
    assert objective_value.shape == (6,)


if __name__ == "__main__":
    pytest.main([__file__])
