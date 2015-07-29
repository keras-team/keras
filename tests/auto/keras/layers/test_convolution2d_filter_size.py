
# coding: utf-8

# In[1]:

#import sys
#sys.path.append("./")


# In[29]:

from __future__ import absolute_import
from keras.layers.convolutional import Convolution2D
from keras.models import Sequential
import numpy as np
import unittest

class TestFilterSizeConvolution2D(unittest.TestCase):
    def test_shape_default_valid(self):

        d = 4
        depth = 1
        nfilter = 2
        model = Sequential()
        model.add(Convolution2D(nfilter, depth, d, d, border_mode='valid')) 
        model.compile(loss='mean_squared_error', optimizer="rmsprop")
        I = np.ones([d,d])
        I = np.array([[I]])
        #print I.shape
        R = model.predict(I)
        assert(R.shape == (1,2,1,1))

    def test_shape_default_full(self):

        d = 4
        depth = 1
        nfilter = 2
        model = Sequential()
        model.add(Convolution2D(nfilter, depth, d, d, border_mode='full')) 
        model.compile(loss='mean_squared_error', optimizer="rmsprop")
        I = np.ones([d,d])
        I = np.array([[I]])
        #print I.shape
        R = model.predict(I)
        assert(R.shape == (1,2,2*d-1,2*d-1))

    def test_shape_default_same(self):

        d = 4
        depth = 1
        nfilter = 2
        model = Sequential()
        model.add(Convolution2D(nfilter, depth, d, d, border_mode='same')) 
        model.compile(loss='mean_squared_error', optimizer="rmsprop")
        I = np.ones([d,d])
        I = np.array([[I]])
        #print I.shape
        R = model.predict(I)
        #print R.shape
        assert(R.shape == (1,2,d,d))


    def test_shape_filter_size_valid(self):

        d = 4
        depth = 1
        nfilter = 2
        model = Sequential()
        model.add(Convolution2D(nfilter, depth, d, d, border_mode='valid',filter_size=[2,2])) 
        model.compile(loss='mean_squared_error', optimizer="rmsprop")
        I = np.ones([d,d])
        I = np.array([[I]])
        #print I.shape
        R = model.predict(I)
        #print R.shape
        #print R
        assert(R.shape == (1,2,3,3))

    def test_shape_filter_size_full(self):

        d = 4
        depth = 1
        nfilter = 2
        sfilter=2
        model = Sequential()
        model.add(Convolution2D(nfilter, depth, d, d, border_mode='full',filter_size=[sfilter,sfilter])) 
        model.compile(loss='mean_squared_error', optimizer="rmsprop")
        I = np.ones([d,d])
        I = np.array([[I]])
        #print I.shape
        R = model.predict(I)
        #print R.shape
        assert(R.shape == (1,2,d+sfilter-1,d+sfilter-1))

    def test_shape_filter_size_same(self):

        d = 4
        depth = 1
        nfilter = 2
        sfilter=2
        model = Sequential()
        model.add(Convolution2D(nfilter, depth, d, d, border_mode='same',filter_size=[sfilter,sfilter]))
        model.compile(loss='mean_squared_error', optimizer="rmsprop")
        I = np.ones([d,d])
        I = np.array([[I]])
        #print I.shape
        R = model.predict(I)
        #print R.shape
        assert(R.shape == (1,2,d,d))

if __name__ == "__main__":
    print('Test Filter size')
    unittest.main()


# In[ ]:



