import numpy as np
import pytest

from keras.models import Sequential
from keras.layers import TimeDistributed, Masking, Dense
from keras import losses
from keras import backend as K


def create_masking_model():
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(None, 1)))
    model.add(TimeDistributed(Dense(1, kernel_initializer='one')))
    model.compile(loss='mse', optimizer='sgd')
    return model


def test_masking():
    np.random.seed(1337)
    x = np.array([[[1], [1]],
                  [[0], [0]]])
    model = create_masking_model()
    y = np.array([[[1], [1]],
                  [[1], [1]]])
    loss = model.train_on_batch(x, y)
    assert loss == 0


def test_masking_is_all_zeros():
    x = y = np.array([[[0], [0]]])
    model = create_masking_model()
    loss = model.train_on_batch(x, y)
    assert loss == 0


if __name__ == '__main__':
    pytest.main([__file__])
