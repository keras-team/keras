import pytest
import os
import sys
import numpy as np
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.utils import vis_utils


def test_plot_model():
    model = Sequential()
    model.add(Conv2D(2, kernel_size=(2, 3), input_shape=(3, 5, 5), name='conv'))
    model.add(Flatten(name='flat'))
    model.add(Dense(5, name='dense1'))
    vis_utils.plot_model(model, to_file='model1.png', show_layer_names=False)
    os.remove('model1.png')

    model = Sequential()
    model.add(LSTM(16, return_sequences=True, input_shape=(2, 3), name='lstm'))
    model.add(TimeDistributed(Dense(5, name='dense2')))
    vis_utils.plot_model(model, to_file='model2.png', show_shapes=True)
    os.remove('model2.png')


if __name__ == '__main__':
    pytest.main([__file__])
