# -*- coding: utf-8 -*-
from __future__ import absolute_import

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros
from ..layers.core import Layer

import math

class Convolution1D(Layer):
    def __init__(self, nb_filter,  stack_size, filter_length,
        init='uniform', activation='linear', weights=None,
<<<<<<< HEAD
        image_shape=None, border_mode='valid', subsample_length=1, name=None, prev=None):
=======
        border_mode='valid', subsample_length=1,
        W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None):
>>>>>>> upstream/master

        super(Convolution1D,self).__init__(name, prev)
        self.filter_length = filter_length
        self.subsample_length = subsample_length
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = (1,subsample_length)
        self.border_mode = border_mode
<<<<<<< HEAD
        self.image_shape = image_shape
        self.nb_filter = nb_filter
        self.stack_size = stack_size

        self.nb_row = 1
        self.nb_col = filter_length
=======
>>>>>>> upstream/master

        self.input = T.tensor4()
        self.W_shape = (self.nb_filter, self.stack_size, self.nb_row, self.nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((self.nb_filter,))

        self.params = [self.W, self.b]

<<<<<<< HEAD
=======
        self.regularizers = []
        if W_regularizer:
            W_regularizer.set_param(self.W)
            self.regularizers.append(W_regularizer)
        if b_regularizer:
            b_regularizer.set_param(self.b)
            self.regularizers.append(b_regularizer)
        if activity_regularizer:
            activity_regularizer.set_layer(self)
            self.regularizers.append(activity_regularizer)

        self.constraints = [W_constraint, b_constraint]

>>>>>>> upstream/master
        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train):
        X = self.get_input(train)

        conv_out = theano.tensor.nnet.conv.conv2d(X, self.W,
            border_mode=self.border_mode, subsample=self.subsample)
        output = self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
<<<<<<< HEAD
                "nb_filter":self.nb_filter,
                "stack_size":self.stack_size,
                "filter_length":self.filter_length,
                "init":self.init.__name__,
                "activation":self.activation.__name__,
                "image_shape":self.image_shape,
                "border_mode":self.border_mode,
                "subsample_length":self.subsample_length}
=======
            "nb_filter":self.nb_filter,
            "stack_size":self.stack_size,
            "filter_length":self.filter_length,
            "init":self.init.__name__,
            "activation":self.activation.__name__,
            "border_mode":self.border_mode,
            "subsample_length":self.subsample_length}
>>>>>>> upstream/master


class MaxPooling1D(Layer):
    def __init__(self, pool_length=2, stride=None, ignore_border=True, name=None, prev=None):
        super(MaxPooling1D,self).__init__(name, prev)
        self.pool_length = pool_length

        if stride is not None:
            self.stride = (1, stride)

        self.input = T.tensor4()
        self.poolsize = (1, pool_length)
        self.ignore_border = ignore_border

    def get_output(self, train):
        X = self.get_input(train)
        output = downsample.max_pool_2d(X, ds=self.poolsize, st=self.stride, ignore_border=self.ignore_border)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
                "pool_length":self.pool_length,
                "ignore_border":self.ignore_border,
                "subsample_length": self.subsample_length}



class Convolution2D(Layer):
<<<<<<< HEAD
    def __init__(self, nb_filter, stack_size, nb_row, nb_col,
        init='glorot_uniform', activation='linear', weights=None,
        image_shape=None, border_mode='valid', subsample=(1,1), name=None, prev=None):

        super(Convolution2D,self).__init__(name, prev)
=======
    def __init__(self, nb_filter, stack_size, nb_row, nb_col, 
        init='glorot_uniform', activation='linear', weights=None, 
        border_mode='valid', subsample=(1, 1),
        W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None):
        super(Convolution2D,self).__init__()
>>>>>>> upstream/master

        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = subsample
        self.border_mode = border_mode
        self.nb_filter = nb_filter
        self.stack_size = stack_size

        self.nb_row = nb_row
        self.nb_col = nb_col

        self.input = T.tensor4()
        self.W_shape = (self.nb_filter, self.stack_size, self.nb_row, self.nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((self.nb_filter,))

        self.params = [self.W, self.b]

<<<<<<< HEAD
=======
        self.regularizers = []
        if W_regularizer:
            W_regularizer.set_param(self.W)
            self.regularizers.append(W_regularizer)
        if b_regularizer:
            b_regularizer.set_param(self.b)
            self.regularizers.append(b_regularizer)
        if activity_regularizer:
            activity_regularizer.set_layer(self)
            self.regularizers.append(activity_regularizer)
            
        self.constraints = [W_constraint, b_constraint]

>>>>>>> upstream/master
        if weights is not None:
            self.set_weights(weights)

    def get_output(self, train):
        X = self.get_input(train)

<<<<<<< HEAD
        conv_out = theano.tensor.nnet.conv.conv2d(X, self.W,
            border_mode=self.border_mode, subsample=self.subsample, image_shape=self.image_shape)
=======
        conv_out = theano.tensor.nnet.conv.conv2d(X, self.W, 
            border_mode=self.border_mode, subsample=self.subsample)
>>>>>>> upstream/master
        output = self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
<<<<<<< HEAD
                "nb_filter":self.nb_filter,
                "stack_size":self.stack_size,
                "nb_row":self.nb_row,
                "nb_col":self.nb_col,
                "init":self.init.__name__,
                "activation":self.activation.__name__,
                "image_shape":self.image_shape,
                "border_mode":self.border_mode,
                "subsample":self.subsample}
=======
            "nb_filter":self.nb_filter,
            "stack_size":self.stack_size,
            "nb_row":self.nb_row,
            "nb_col":self.nb_col,
            "init":self.init.__name__,
            "activation":self.activation.__name__,
            "border_mode":self.border_mode,
            "subsample":self.subsample}
>>>>>>> upstream/master


class MaxPooling2D(Layer):
    def __init__(self, poolsize=(2, 2), stride=None, ignore_border=True, name=None, prev=None):
        super(MaxPooling2D,self).__init__(name, prev)
        self.poolsize = poolsize
        self.stride = stride
        self.ignore_border = ignore_border
        self.input = T.tensor4()

    def get_output(self, train):
        X = self.get_input(train)
        output = downsample.max_pool_2d(X, ds=self.poolsize, st=self.stride, ignore_border=self.ignore_border)
        return output

    def get_config(self):
        return {"name":self.__class__.__name__,
                "poolsize":self.poolsize,
                "ignore_border":self.ignore_border,
                "stride": self.stride}


class ZeroPadding2D(Layer):
    def __init__(self, width=1, name=None, prev=None):
        super(ZeroPadding2D, self).__init__(name, prev)
        self.width = width
        self.input = T.tensor4()

    def get_output(self, train):
        X = self.get_input(train)
        width =  self.width
        in_shape = X.shape
        out_shape = (in_shape[0], in_shape[1], in_shape[2] + 2 * width, in_shape[3] + 2 * width)
        out = T.zeros(out_shape)
        indices = (slice(None), slice(None), slice(width, in_shape[2] + width),slice(width, in_shape[3] + width))
        return T.set_subtensor(out[indices], X)

    def get_config(self):
        return {"name":self.__class__.__name__,
                "width":self.width}



# class Convolution3D: TODO

# class MaxPooling3D: TODO
