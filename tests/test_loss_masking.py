import numpy as np
import pytest

from keras.models import Sequential
from keras.engine.training import _weighted_masked_objective
from keras.layers import TimeDistributed, Masking, Dense
from keras.utils.test_utils import keras_test
from keras import losses
from keras import backend as K


@keras_test
def test_masking():
    np.random.seed(1337)
    x = np.array([[[1], [1]],
                  [[0], [0]]])
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(2, 1)))
    model.add(TimeDistributed(Dense(1, kernel_initializer='one')))
    model.compile(loss='mse', optimizer='sgd')
    y = np.array([[[1], [1]],
                  [[1], [1]]])
    loss = model.train_on_batch(x, y)
    assert loss == 0


@keras_test
def test_loss_masking():
    weighted_loss = _weighted_masked_objective(losses.get('mae'))
    shape = (3, 4, 2)
    x = np.arange(24).reshape(shape)
    y = 2 * x

    # Normally the trailing 1 is added by standardize_weights
    weights = np.ones((3,))
    mask = np.ones((3, 4))
    mask[1, 0] = 0

    out = K.eval(weighted_loss(K.variable(x),
                               K.variable(y),
                               K.variable(weights),
                               K.variable(mask)))


if __name__ == '__main__':
    pytest.main([__file__])
