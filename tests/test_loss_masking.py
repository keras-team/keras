import numpy as np
import pytest

from keras.models import Sequential
from keras.engine.training import weighted_objective
from keras.layers.core import TimeDistributedDense, Masking
from keras import objectives
from keras import backend as K


def test_masking():
    np.random.seed(1337)
    X = np.array([[[1], [1]],
                  [[0], [0]]])
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(2, 1)))
    model.add(TimeDistributedDense(1, init='one'))
    model.compile(loss='mse', optimizer='sgd')
    y = np.array([[[1], [1]],
                  [[1], [1]]])
    loss = model.train_on_batch(X, y)
    assert loss == 0


def test_loss_masking():
    weighted_loss = weighted_objective(objectives.get('mae'))
    shape = (3, 4, 2)
    X = np.arange(24).reshape(shape)
    Y = 2 * X

    # Normally the trailing 1 is added by standardize_weights
    weights = np.ones((3,))
    mask = np.ones((3, 4))
    mask[1, 0] = 0

    out = K.eval(weighted_loss(K.variable(X),
                               K.variable(Y),
                               K.variable(weights),
                               K.variable(mask)))


if __name__ == '__main__':
    pytest.main([__file__])
