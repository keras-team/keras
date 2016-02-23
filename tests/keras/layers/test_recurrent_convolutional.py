import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.models import Sequential
from keras.layers.recurrent_convolutional import LSTMConv2D


def test_shape2():
    #With return_sequences = True
    input_shape=[10,30,30,3]
    batch = 5
    nfilter = 20
    input_a = np.zeros([batch]+input_shape)
    gt = np.zeros([batch]+[input_shape[0],input_shape[1],input_shape[2],nfilter])
    seq = Sequential()
    seq.add(LSTMConv2D(nb_filter=20,nb_row=4,nb_col=4,input_shape=input_shape,
            border_mode="same",return_sequences=True))
    #seq.add(Reshape((27,27)))
    #seq.add(Convolution2D(1, 3, 3,  activation='sigmoid', border_mode='same', subsample=(1, 1), dim_ordering='tf'))
    seq.compile(loss="binary_crossentropy",optimizer="rmsprop")
    
    assert seq.predict(input_a).shape == (batch,input_shape[0],input_shape[1],input_shape[2],nfilter)
    
    seq.fit(input_a,gt,nb_epoch=1)
    
def test_shape_th_return_sequences():
    input_shape=[10,3,30,30]
    batch = 5
    nfilter = 20
    input_a = np.zeros([batch]+input_shape)
    gt = np.zeros([batch]+[input_shape[0],nfilter,input_shape[2],input_shape[3]])
    seq = Sequential()
    seq.add(LSTMConv2D(nb_filter=nfilter,nb_row=4,nb_col=4,dim_ordering="th",input_shape=input_shape,
            border_mode="same",return_sequences=True))
    #seq.add(Reshape((27,27)))
    #seq.add(Convolution2D(1, 3, 3,  activation='sigmoid', border_mode='same', subsample=(1, 1), dim_ordering='tf'))
    seq.compile(loss="binary_crossentropy",optimizer="rmsprop")
    assert seq.predict(input_a).shape == (batch,input_shape[0],nfilter,input_shape[2],input_shape[3])
    
    seq.fit(input_a,gt,nb_epoch=1)
    
def test_shape_th():
    input_shape=[10,3,30,30]
    batch = 5
    nfilter = 20
    input_a = np.zeros([batch]+input_shape)
    gt = np.zeros([batch]+[nfilter,input_shape[2],input_shape[3]])
    seq = Sequential()
    seq.add(LSTMConv2D(nb_filter=nfilter,nb_row=4,nb_col=4,dim_ordering="th",input_shape=input_shape,
            border_mode="same",return_sequences=False))
    #seq.add(Reshape((27,27)))
    #seq.add(Convolution2D(1, 3, 3,  activation='sigmoid', border_mode='same', subsample=(1, 1), dim_ordering='tf'))
    seq.compile(loss="binary_crossentropy",optimizer="rmsprop")
    assert seq.predict(input_a).shape == (batch,nfilter,input_shape[2],input_shape[3])
    
    seq.fit(input_a,gt,nb_epoch=1)
def test_shape():
    input_shape=[10,30,30,3]
    batch = 5
    nfilter = 20
    input_a = np.zeros([batch]+input_shape)
    gt = np.zeros([batch]+[input_shape[1],input_shape[2],nfilter])
    seq = Sequential()
    seq.add(LSTMConv2D(nb_filter=nfilter,nb_row=4,nb_col=4,input_shape=input_shape,
            border_mode="same",return_sequences=False))
    #seq.add(Reshape((27,27)))
    #seq.add(Convolution2D(1, 3, 3,  activation='sigmoid', border_mode='same', subsample=(1, 1), dim_ordering='tf'))
    seq.compile(loss="binary_crossentropy",optimizer="rmsprop")
    assert seq.predict(input_a).shape == (batch,input_shape[1],input_shape[2],nfilter)
    
    seq.fit(input_a,gt,nb_epoch=1)
if __name__ == '__main__':
    pytest.main([__file__])
