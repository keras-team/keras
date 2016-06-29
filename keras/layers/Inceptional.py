# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .. import backend as K
from ..engine import Layer

from ..layers.convolutional import Convolution2D
from ..layers import merge,BatchNormalization,ZeroPadding2D,MaxPooling2D,AveragePooling2D

import numpy as np

class FireModule(Layer):
    '''
    FireModule from the SqueezeNet paper
    http://arxiv.org/pdf/1602.07360v3.pdf

    # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.
    '''
    def __init__(self, kernel_set=None,dim_ordering=K.image_dim_ordering()):
        assert len(kernel_set)==3,"Valid kernel_set should be [nb_s1x1,nb_e1x1,nb_ex3x3]"
        self.nb_s1x1 = kernel_set[0]
        self.nb_e1x1 = kernel_set[1]
        self.nb_e3x3 = kernel_set[2]
        self.dim_ordering=dim_ordering
        super(FireModule, self).__init__()

    def get_output_shape_for(self, input_shape):
        
         if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
            input_dim=input_shape[1]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
            input_dim=input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        #determine output_dims
        output_dims=(self.nb_e1x1+self.nb_e3x3)
        
        if self.dim_ordering == 'th':
            return (input_shape[0], output_dims, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, output_dims)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):

        #squeeze by 1x1 filter
        squeeze  = Convolution2D(self.s1x1, 1, 1, border_mode='same', activation='relu')(x)
        #expand by 1x1 filter
        expand_1x1 = Convolution2D(self.e1x1, 1, 1, border_mode='same', activation='relu')(squeeze)
        #expand by 3x3 filter
        expend_3x3 = Convolution2D(self.e3x3, 3, 3, border_mode='same', activation='relu')(squeeze)

        output_layer = merge([expand_1x1,expend_3x3], mode='concat', concat_axis=1)
        return output_layer
        
    def get_config(self):
        config = {"nb_s1x1": self.nb_e1x1,
                  "nb_e1x1": self.nb_e1x1,
                  "nb_e3x3": self.nb_e3x3,
                  "dim_ordering": self.dim_ordering}
        base_config = super(FireModule, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
def Conv_pool(input_layer,PoolMethod,subsample):
    if subsample==(1,1):
        border_mode="same"
    else:
        border_mode="valid"
        
    if PoolMethod =="max":
        return MaxPooling2D((3, 3), strides=subsample, border_mode=border_mode)(input_layer)
    else:
        return AveragePooling2D((3, 3), strides=subsample, border_mode=border_mode)(input_layer)
        
def Conv_batch(input_layer, nb_filter, nb_row, nb_col, subsample, BatchNorm, border_mode="same"): 
        
    if BatchNorm==False:
        return Convolution2D(nb_filter, nb_row, nb_col,subsample=subsample, border_mode=border_mode, activation='relu')(input_layer)
    else:
        conv_1=Convolution2D(nb_filter, nb_row, nb_col,subsample=subsample, border_mode=border_mode, activation='relu')(input_layer)
        conv_1=BatchNormalization()(conv_1)
        return conv_1
    
class Inception(Layer):
    '''
     # Input shape
        4D tensor with shape:
        `(samples, channels, rows, cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, rows, cols, channels)` if dim_ordering='tf'.
    # Output shape
        4D tensor with shape:
        `(samples, nb_filter, new_rows, new_cols)` if dim_ordering='th'
        or 4D tensor with shape:
        `(samples, new_rows, new_cols, nb_filter)` if dim_ordering='tf'.
        `rows` and `cols` values might have changed due to padding.
    '''
    def __init__(self,ver="ver1",BatchNorm=False,subsample=(1,1),
                 dim_ordering=K.image_dim_ordering(),kernel_set=None,PoolMethod="Max"):
    
        assert ver in {"ver1","ver2","ver3","ver4"}, 'ver should be in {ver1,ver2,ver3,ver4}'
        self.ver=ver
        assert PoolMethod in {"max","avg"}, "PoolMethod should be in {max,avg}"
        self.PoolMethod=PoolMethod
        self.BatchNorm=BatchNorm
        self.kernel_set = kernel_set
        self.subsample = tuple(subsample)
        if self.subsample !=(1,1):
            assert len(self.kernel_set)==4, 'kernel_set must hold four for ({0},{1})subsample'.format(self.subsample) 
        else:
            assert len(self.kernel_set)==6, 'kernel_set must hold six for None subsample'
        self.dim_ordering=dim_ordering   
        assert self.dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        
        super(Inception, self).__init__()

    def get_output_shape_for(self, input_shape):

        if self.dim_ordering == 'th':
            rows = input_shape[2]
            cols = input_shape[3]
            input_dim=input_shape[1]
        elif self.dim_ordering == 'tf':
            rows = input_shape[1]
            cols = input_shape[2]
            input_dim=input_shape[3]
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

        #determine output_dims
        if self.subsample==(1,1):
            if self.ver=="ver1" or self.ver=="ver2":
                # total 4 towers will merge
                # remove temporary filters of 3x3 reduce,5x5 reduce
                output_dims=np.sum(self.kernel_set)-self.kernel_set[1]-self.kernel_set[3]
            else:
                output_dims=( self.kernel_set[0]+2*self.kernel_set[2]+
                                2*self.kernel_set[4]+self.kernel_set[5])
        else:
            if self.ver=="ver1" or self.ver=="ver2":
                # total 3 towers will merge
                # remove temporary filters of 3x3 reduce,5x5 reduce
                output_dims=(self.kernel_set[1]+self.kernel_set[3]+input_dim)
            else:
                output_dims=(2*self.kernel_set[1]+2*self.kernel_set[3]+input_dim)
                
            #in inception module,filter_size==3
            rows = (rows + self.subsample[0] - 3) //self.subsample[0]
            cols = (cols + self.subsample[1] - 3) //self.subsample[1]
            
        if self.dim_ordering == 'th':
            return (input_shape[0], output_dims, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0], rows, cols, output_dims)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

       

    def call(self, x, mask=None):
        nb_kernel=self.kernel_set
        if self.ver=="ver1" and self.subsample==(1,1):
            # ver1
            # without stride,total number of tower is four
            tower_1 = Conv_batch(x, nb_kernel[0], 1, 1, self.subsample, self.BatchNorm, border_mode="same")
            
            tower_2 = Conv_batch(x, nb_kernel[1], 1, 1, self.subsample, self.BatchNorm, border_mode="same")
            tower_2 = Conv_batch(tower_2, nb_kernel[2], 3, 3, self.subsample, self.BatchNorm, border_mode="same")

            tower_3 = Conv_batch(x, nb_kernel[3], 1, 1, self.subsample, self.BatchNorm, border_mode="same")
            tower_3 = Conv_batch(tower_3, nb_kernel[4], 5, 5, self.subsample, self.BatchNorm, border_mode="same")

            tower_4 = Conv_pool(x,self.PoolMethod,self.subsample)    
            tower_4 = Conv_batch(tower_4, nb_kernel[5], 1, 1, self.subsample, self.BatchNorm, border_mode="same")

            output_layer = merge([tower_1, tower_2, tower_3, tower_4], mode='concat', concat_axis=1)
            return output_layer
        
        elif self.ver=="ver1" and self.subsample!=(1,1):
            # ver1
            # with stride,total number of tower is three
            tower_1 = Conv_batch(x, nb_kernel[0], 1, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_1 = Conv_batch(x, nb_kernel[1], 3, 3, self.subsample, self.BatchNorm, border_mode="valid")

            tower_2 = Conv_batch(x, nb_kernel[2], 1, 1,(1,1), self.BatchNorm, border_mode="same")
            tower_2 = ZeroPadding2D((1,1))(tower_2)
            tower_2 = Conv_batch(tower_2, nb_kernel[3], 5, 5, self.subsample, self.BatchNorm, border_mode="valid")

            tower_3 = Conv_pool(x,self.PoolMethod,self.subsample)

            output_layer = merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=1)
            return output_layer
 
                    
        elif self.ver=="ver2" and self.subsample==(1,1):
            # ver2
            # without stride,total number of tower is four
            tower_1 = Conv_batch(x,nb_kernel[0], 1, 1, self.subsample, self.BatchNorm, border_mode="same")

            tower_2 = Conv_batch(x,nb_kernel[1], 1, 1, self.subsample, self.BatchNorm, border_mode="same")
            tower_2 = Conv_batch(tower_2, nb_kernel[2], 3, 3, self.subsample, self.BatchNorm, border_mode="same")

            tower_3 = Conv_batch(x,nb_kernel[3], 1, 1, self.subsample, self.BatchNorm, border_mode="same")
            tower_3 = Conv_batch(tower_3,nb_kernel[4], 3, 3, self.subsample, self.BatchNorm, border_mode="same")
            tower_3 = Conv_batch(tower_3,nb_kernel[4], 3, 3, self.subsample, self.BatchNorm, border_mode="same")

            tower_4 = Conv_pool(x,self.PoolMethod,self.subsample)
            tower_4 = Conv_batch(tower_4,nb_kernel[5], 1, 1, self.subsample, self.BatchNorm, border_mode="same")

            output_layer = merge([tower_1, tower_2, tower_3, tower_4], mode='concat', concat_axis=1)
            return output_layer
        
        elif self.ver=="ver2" and self.subsample!=(1,1):
            # ver2
            # with stride,total number of tower is Three
            tower_1 = Conv_batch(x,nb_kernel[0], 1, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_1 = Conv_batch(tower_1,nb_kernel[1], 3, 3, self.subsample, self.BatchNorm, border_mode="valid")

            tower_2 = Conv_batch(x,nb_kernel[2], 1, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_2 = Conv_batch(tower_2,nb_kernel[3], 1, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_2 = Conv_batch(tower_2,nb_kernel[3], 3, 3, self.subsample, self.BatchNorm, border_mode="valid")
            
            tower_3 = Conv_pool(x,self.PoolMethod,self.subsample)

            output_layer = merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=1)
            return output_layer

        elif self.ver=="ver3" and self.subsample==(1,1):
            # ver3
            # without stride,total number of tower is four
            tower_1 = Conv_batch(x,nb_kernel[0], 1, 1, (1,1), self.BatchNorm, border_mode="same")

            tower_2 = Conv_batch(x,nb_kernel[1], 1, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_2_left = Conv_batch(tower_2,nb_kernel[2], 3, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_2_right = Conv_batch(tower_2,nb_kernel[2], 1, 3, (1,1), self.BatchNorm, border_mode="same")
            
            tower_3 = Conv_batch(x,nb_kernel[3], 1, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_3 = Conv_batch(tower_3,nb_kernel[4], 3, 3, (1,1), self.BatchNorm, border_mode="same")
            tower_3_left = Conv_batch(tower_3,nb_kernel[4], 3, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_3_right = Conv_batch(tower_3,nb_kernel[4], 3, 1, (1,1), self.BatchNorm, border_mode="same")

            tower_4 = Conv_pool(x,self.PoolMethod,self.subsample)
            tower_4 = Conv_batch(x,nb_kernel[5], 1, 1, (1,1), self.BatchNorm, border_mode="same")

            output_layer = merge([tower_1, tower_2_left,tower_2_right, tower_3_left,tower_3_right, tower_4], 
                                 mode='concat', concat_axis=1)
            return output_layer
        
        elif self.ver=="ver3" and self.subsample!=(1,1):
            # ver3
            # with stride,total number of tower is Three
            tower_1 = Conv_batch(x,nb_kernel[0], 3, 3, self.subsample, self.BatchNorm, border_mode="valid")
            tower_1_left = Conv_batch(tower_1,nb_kernel[1], 3, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_1_right = Conv_batch(tower_1,nb_kernel[1], 1, 3, (1,1), self.BatchNorm, border_mode="same")

            tower_2 = Conv_batch(x,nb_kernel[2], 1, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_2 = Conv_batch(tower_2,nb_kernel[3], 3, 3, self.subsample, self.BatchNorm, border_mode="valid")
            tower_2_left = Conv_batch(tower_2,nb_kernel[2], 3, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_2_right = Conv_batch(tower_2,nb_kernel[2], 1, 3, (1,1), self.BatchNorm, border_mode="same")

            tower_3 = Conv_pool(x,self.PoolMethod,self.subsample)

            output_layer = merge([tower_1_left,tower_1_right, tower_2_left,tower_2_right,tower_3], 
                                 mode='concat', concat_axis=1)
            return output_layer
                    
        
        elif self.ver=="ver4" and self.subsample==(1,1):
            # ver4
            # without stride,total number of tower is four
            tower_1 = Conv_batch(x,nb_kernel[0], 1, 1, (1,1), self.BatchNorm, border_mode="same")

            tower_2 = Conv_batch(x,nb_kernel[1], 1, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_2_left = Conv_batch(tower_2,nb_kernel[2], 3, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_2_right = Conv_batch(tower_2,nb_kernel[2], 1, 3, (1,1), self.BatchNorm, border_mode="same")
            
            tower_3 = Conv_batch(x,nb_kernel[3], 1, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_3 = Conv_batch(tower_3,nb_kernel[4], 3, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_3 = Conv_batch(tower_3,nb_kernel[4], 1, 3, (1,1), self.BatchNorm, border_mode="same")
            tower_3_left = Conv_batch(tower_3,nb_kernel[4], 3, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_3_right = Conv_batch(tower_3,nb_kernel[4], 1, 3, (1,1), self.BatchNorm, border_mode="same")
            
            tower_4 = Conv_pool(x,self.PoolMethod,self.subsample)
            tower_4 = Conv_batch(tower_4,nb_kernel[5], 1, 1, (1,1), self.BatchNorm, border_mode="same")

            output_layer = merge([tower_1, tower_2_left,tower_2_right, tower_3_left,tower_3_right, tower_4], 
                                 mode='concat', concat_axis=1)
            return output_layer
        
        elif self.ver=="ver4" and self.subsample !=(1,1):
            # ver4
            # with stride,total number of tower is Three
            tower_1 = Conv_batch(x,nb_kernel[0], 3, 3, self.subsample, self.BatchNorm, border_mode="valid")
            tower_1_left = Conv_batch(tower_1,nb_kernel[1], 3, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_1_right = Conv_batch(tower_1,nb_kernel[1], 1, 3, (1,1), self.BatchNorm, border_mode="same")

            tower_2 = Conv_batch(tower_1,nb_kernel[2], 3, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_2 = Conv_batch(tower_2,nb_kernel[2], 1, 3, (1,1), self.BatchNorm, border_mode="same")
            tower_2_left = Conv_batch(tower_2,nb_kernel[3], 3, 1, (1,1), self.BatchNorm, border_mode="same")
            tower_2_right = Conv_batch(tower_2,nb_kernel[3], 1, 3, (1,1), self.BatchNorm, border_mode="same")
            
            tower_3 = Conv_pool(x,self.PoolMethod,self.subsample)

            output_layer = merge([tower_1_left,tower_1_right, tower_2_left,tower_2_right,tower_3], 
                                 mode='concat', concat_axis=1)
            return output_layer
                    
        
        else:
            raise Exception("Invalid Inception module",self)
            
            
