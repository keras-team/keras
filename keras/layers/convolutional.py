# -*- coding: utf-8 -*-
from __future__ import absolute_import

import theano
import theano.tensor as T
from theano.tensor.signal import downsample

from .. import activations, initializations
from ..utils.theano_utils import shared_zeros
from ..layers.core import Layer

import math


def convolution_output_dim(input_dim, filter_dim, subsample_dim, border_mode):
    if border_mode == 'valid':
        output_dim = input_dim - filter_dim + 1
    elif border_mode == 'full':
        output_dim = input_dim + filter_dim - 1
    output_dim = math.ceil(output_dim) / subsample_dim
    return int(output_dim)

def pooling_output_dim(input_dim, pooling_dim, subsample_dim, ignore_border=True):
    # as given in theano source
    if subsample_dim is None:
        subsample_dim = pooling_dim

    if ignore_border:
        output_dim = (input_dim - pooling_dim + subsample_dim) // subsample_dim
    else:
        if subsample_dim >= pooling_dim:
            output_dim = (input_dim + subsample_dim - 1) // subsample_dim
        else:
            output_dim = max(0, (input_dim - pooling_dim + subsample_dim - 1) // subsample_dim) + 1

    return output_dim


class Convolution1D(Layer):
    def __init__(self, nb_filter,  filter_length, stack_size=None,
        init='uniform', activation='linear', weights=None,
        image_shape=None, border_mode='valid', subsample_length=1, name=None, prev=None, input_dim=(None, )):

        if stack_size is not None:
            nb_filter, stack_size, filter_length = nb_filter, filter_length, stack_size

        super(Convolution1D,self).__init__(name, prev, input_dim)
        self.filter_length = filter_length
        self.subsample_length = subsample_length
        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = (1,subsample_length)
        self.border_mode = border_mode
        self.image_shape = image_shape
        self.nb_filter = nb_filter
        self.stack_size = stack_size

        self.weights = weights

        self.nb_row = 1
        self.nb_col = filter_length

    def setup(self):
        if self.stack_size is None:
            self.stack_size = self.input_dim[0]

        self.input = T.tensor4()
        self.W_shape = (self.nb_filter, self.stack_size, self.nb_row, self.nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((self.nb_filter,))

        self.params = [self.W, self.b]

        if self.weights is not None:
            self.set_weights(self.weights)

    def get_output(self, train):
        X = self.get_input(train)

        conv_out = theano.tensor.nnet.conv.conv2d(X, self.W,
            border_mode=self.border_mode, subsample=self.subsample, image_shape=self.image_shape)
        output = self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return output

    def get_output_dim(self, input_dim):
        output_dim = convolution_output_dim(input_dim[2], self.filter_length, self.subsample[1], self.border_mode)
        return (self.nb_filter, input_dim[1], output_dim)

    def get_config(self):
        return {"name":self.__class__.__name__,
                "nb_filter":self.nb_filter,
                "stack_size":self.stack_size,
                "filter_length":self.filter_length,
                "init":self.init.__name__,
                "activation":self.activation.__name__,
                "image_shape":self.image_shape,
                "border_mode":self.border_mode,
                "subsample_length":self.subsample_length}


class MaxPooling1D(Layer):
    def __init__(self, pool_length=2, stride=None, ignore_border=True, name=None, prev=None, input_dim=(None, )):
        super(MaxPooling1D,self).__init__(name, prev, input_dim)
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

    def get_output_dim(self, input_dim):
        output_dim = pooling_output_dim(input_dim[2], self.pool_length, self.stride, self.ignore_border)
        return (input_dim[0], input_dim[1], output_dim)


    def get_config(self):
        return {"name":self.__class__.__name__,
                "pool_length":self.pool_length,
                "ignore_border":self.ignore_border,
                "subsample_length": self.subsample_length}



class Convolution2D(Layer):
    def __init__(self, nb_filter,  nb_row, nb_col,stack_size=None,
        init='glorot_uniform', activation='linear', weights=None,
        image_shape=None, border_mode='valid', subsample=(1,1), name=None, prev=None, input_dim=(None, )):

        if stack_size is not None:
            nb_filter, stack_size, nb_row, nb_col = nb_filter, nb_row, nb_col, stack_size

        super(Convolution2D,self).__init__(name, prev, input_dim)

        self.init = initializations.get(init)
        self.activation = activations.get(activation)
        self.subsample = subsample
        self.border_mode = border_mode
        self.image_shape = image_shape
        self.nb_filter = nb_filter
        self.stack_size = stack_size

        self.nb_row = nb_row
        self.nb_col = nb_col

        self.weights = weights


    def setup(self):
        if self.stack_size is None:
            self.stack_size = self.input_dim[0]

        self.input = T.tensor4()
        self.W_shape = (self.nb_filter, self.stack_size, self.nb_row, self.nb_col)
        self.W = self.init(self.W_shape)
        self.b = shared_zeros((self.nb_filter,))

        self.params = [self.W, self.b]

        if self.weights is not None:
            self.set_weights(self.weights)

    def get_output(self, train):
        X = self.get_input(train)

        conv_out = theano.tensor.nnet.conv.conv2d(X, self.W,
            border_mode=self.border_mode, subsample=self.subsample, image_shape=self.image_shape)
        output = self.activation(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        return output

    def get_output_dim(self, input_dim):
        output_rows = convolution_output_dim(input_dim[1], self.nb_row, self.subsample[0], self.border_mode)
        output_columns = convolution_output_dim(input_dim[2], self.nb_col, self.subsample[1], self.border_mode)
        return (self.nb_filter, output_rows, output_columns)

    def get_config(self):
        return {"name":self.__class__.__name__,
                "nb_filter":self.nb_filter,
                "stack_size":self.stack_size,
                "nb_row":self.nb_row,
                "nb_col":self.nb_col,
                "init":self.init.__name__,
                "activation":self.activation.__name__,
                "image_shape":self.image_shape,
                "border_mode":self.border_mode,
                "subsample":self.subsample}




class MaxPooling2D(Layer):
    def __init__(self, poolsize=(2, 2), stride=None, ignore_border=True, name=None, prev=None, input_dim=(None, )):
        super(MaxPooling2D,self).__init__(name, prev, input_dim)

        self.poolsize = poolsize

        self.stride = stride
        if stride is None:
            self.stride = (stride, stride)

        self.ignore_border = ignore_border

        self.input = T.tensor4()

    def get_output(self, train):
        X = self.get_input(train)
        output = downsample.max_pool_2d(X, ds=self.poolsize, st=self.stride, ignore_border=self.ignore_border)
        return output

    def get_output_dim(self, input_dim):
        output_rows = pooling_output_dim(input_dim[1], self.poolsize[0], self.stride[0], self.ignore_border)
        output_columns = pooling_output_dim(input_dim[2], self.poolsize[1], self.stride[1], self.ignore_border)
        return (input_dim[0], output_rows, output_columns)


    def get_config(self):
        return {"name":self.__class__.__name__,
                "poolsize":self.poolsize,
                "ignore_border":self.ignore_border,
                "stride": self.stride}


class ZeroPadding2D(Layer):
    def __init__(self, width=1):
        super(ZeroPadding2D, self).__init__()
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

    def get_output_dim(self, input_dim):
        return (input_dim[0], input_dim[1] + 2 * self.width, input_dim[2] + 2 * self.width)

    def get_config(self):
        return {"name":self.__class__.__name__,
                "width":self.width}



# class Convolution3D: TODO

# class MaxPooling3D: TODO
