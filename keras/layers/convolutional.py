# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros
from ..layers.core import Layer


# class Convolution1D(Layer): TODO

# class MaxPooling1D(Layer): TODO


class Convolution2D(Layer):
    def __init__(self, nb_filter, stack_size, nb_row, nb_col, 
        init='uniform', activation='linear', weights=None, 
        image_shape=None, border_mode='valid', subsample=(1,1)):

        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = subsample
        self.border_mode = border_mode
        self.image_shape = image_shape
        
        self.input = T.tensor4()
        self.W_shape = (nb_filter, stack_size, nb_row, nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((nb_filter,))

        self.params = [self.W, self.b]

        if weights is not None:
            self.set_weights(weights)

    def output(self, train):
        X = self.get_input(train)

        conv_out = theano.tensor.nnet.conv.conv2d(X, self.W, 
            border_mode=self.border_mode, subsample=self.subsample, image_shape=self.image_shape)
        output = self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return output


class MaxPooling2D(Layer):
    def __init__(self, poolsize=(2, 2), ignore_border=True):
        self.input = T.tensor4()
        self.poolsize = poolsize
        self.ignore_border = ignore_border
        self.params = []

    def output(self, train):
        X = self.get_input(train)
        output = downsample.max_pool_2d(X, self.poolsize, ignore_border=self.ignore_border)
        return output


# class ZeroPadding2D(Layer): TODO
        
