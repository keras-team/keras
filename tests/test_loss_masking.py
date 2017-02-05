import numpy as np
import pytest

from keras.models import Sequential
from keras.engine.training import weighted_objective, masked_tensor
from keras.layers.core import Dense, Masking
from keras.layers.wrappers import TimeDistributed
from keras.utils.test_utils import keras_test
from keras import objectives
from keras import backend as K


@keras_test
def test_masking():
    np.random.seed(1337)
    X = np.array([[[1], [1]],
                  [[0], [0]]])
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(2, 1)))
    model.add(TimeDistributed(Dense(1, init='one')))
    model.compile(loss='mse', optimizer='sgd', metrics=['accuracy'])
    y = np.array([[[1], [1]],
                  [[1], [1]]])
    (loss, acc) = model.train_on_batch(X, y)

    assert loss == 0
    assert acc == 1


@keras_test
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


@keras_test
def test_masked_tensor():
    x = np.random.randint(0, 10, size=(5, 10, 5))
    mask = np.random.randint(0, 2, size=(5, 10))
    i = np.nonzero(mask)
    exp_out = x[i[0], i[1], :]

    k_out = K.eval(masked_tensor(K.variable(x, dtype='int32'), K.variable(mask, dtype='int32')))

    assert np.all(k_out == exp_out)


if __name__ == '__main__':
    pytest.main([__file__])
