import pytest
import os
import sys
import numpy as np
from keras import Input, Model

from keras.layers import Conv2D, Bidirectional
from keras.layers import Dense
from keras.layers import Embedding
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

    inner_input = Input(shape=(2, 3), dtype='float32', name='inner_input')
    inner_lstm = Bidirectional(LSTM(16, name='inner_lstm'), name='bd')(inner_input)
    encoder = Model(inner_input, inner_lstm, name='Encoder_Model')
    outer_input = Input(shape=(5, 2, 3), dtype='float32', name='input')
    inner_encoder = TimeDistributed(encoder, name='td_encoder')(outer_input)
    lstm = LSTM(16, name='outer_lstm')(inner_encoder)
    preds = Dense(5, activation='softmax', name='predictions')(lstm)
    model = Model(outer_input, preds)
    vis_utils.plot_model(model, to_file='model3.png', show_shapes=True,
                         expand_nested=True, dpi=300)
    os.remove('model3.png')


def test_plot_sequential_embedding():
    """Fixes #11376"""
    model = Sequential()
    model.add(Embedding(10000, 256, input_length=400, name='embed'))
    vis_utils.plot_model(model,
                         to_file='model1.png',
                         show_shapes=True,
                         show_layer_names=True)
    os.remove('model1.png')


if __name__ == '__main__':
    pytest.main([__file__])
