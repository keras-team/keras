import pytest
import numpy as np

from keras import losses
from keras import backend as K
from keras.utils.generic_utils import custom_object_scope


allobj = [losses.mean_squared_error,
          losses.mean_absolute_error,
          losses.mean_absolute_percentage_error,
          losses.mean_squared_logarithmic_error,
          losses.squared_hinge,
          losses.hinge,
          losses.categorical_crossentropy,
          losses.binary_crossentropy,
          losses.kullback_leibler_divergence,
          losses.poisson,
          losses.cosine_proximity,
          losses.logcosh,
          losses.categorical_hinge]


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
    objective_output = losses.sparse_categorical_crossentropy(y_a, y_b)
    assert K.eval(objective_output).shape == (5, 6)

    y_a = K.variable(np.random.randint(0, 7, (6,)))
    y_b = K.variable(np.random.random((6, 7)))
    assert K.eval(losses.sparse_categorical_crossentropy(y_a, y_b)).shape == (6,)


def test_categorical_hinge():
    y_pred = K.variable(np.array([[0.3, 0.2, 0.1],
                                  [0.1, 0.2, 0.7]]))
    y_true = K.variable(np.array([[0, 1, 0],
                                  [1, 0, 0]]))
    expected_loss = ((0.3 - 0.2 + 1) + (0.7 - 0.1 + 1)) / 2.0
    loss = K.eval(losses.categorical_hinge(y_true, y_pred))
    assert np.isclose(expected_loss, np.mean(loss))


def test_sparse_categorical_crossentropy():
    y_pred = K.variable(np.array([[0.3, 0.6, 0.1],
                                  [0.1, 0.2, 0.7]]))
    y_true = K.variable(np.array([1, 2]))
    expected_loss = - (np.log(0.6) + np.log(0.7)) / 2
    loss = K.eval(losses.sparse_categorical_crossentropy(y_true, y_pred))
    assert np.isclose(expected_loss, np.mean(loss))


def test_serializing_loss_class():
    class MSE_MAE_loss:
        def __init__(self, mse_fraction):
            self.mse_fraction = mse_fraction

        def __call__(self, y_true, y_pred):
            return (self.mse_fraction * losses.mse(y_true, y_pred) +
                    (1 - self.mse_fraction) * losses.mae(y_true, y_pred))

        def get_config(self):
            return {'mse_fraction': self.mse_fraction}

    orig_loss_class = MSE_MAE_loss(0.3)
    with custom_object_scope({'MSE_MAE_loss': MSE_MAE_loss}):
        serialized = losses.serialize(orig_loss_class)
    
    with custom_object_scope({'MSE_MAE_loss': MSE_MAE_loss}):
        deserialized = losses.deserialize(serialized)
    assert isinstance(deserialized, MSE_MAE_loss)
    assert deserialized.mse_fraction == 0.3


if __name__ == '__main__':
    pytest.main([__file__])
