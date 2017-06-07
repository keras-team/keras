import pytest
import numpy as np
from keras.layers import Input, LSTM, TimeStepLSTM, Reshape
from keras.layers.merge import concatenate
from keras.models import Model
import keras.backend as K
from keras.utils.test_utils import keras_test


@keras_test
def test_timesteplstm():
    i1 = Input(shape=(10, 16))
    lstm1 = TimeStepLSTM(32)

    state_list1 = lstm1(i1, timepoint=0)
    assert state_list1[0]._keras_shape == (None, 32)
    assert len(state_list1) == 3

    state_list = []
    for t in range(K.int_shape(i1)[1]):
        cur_state_list = lstm1(i1, timepoint=t)
        state_list.append(Reshape((1, 32))(cur_state_list[0]))

    states = concatenate(state_list, axis=1)
    assert states._keras_shape == (None, 10, 32)

    lstm2 = LSTM(32, return_sequences=True)(i1)
    assert lstm2._keras_shape == (None, 10, 32)
