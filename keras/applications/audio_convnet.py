# -*- coding: utf-8 -*-
'''AudioConvnet model for Keras.

# Reference:

- [Automatic tagging using deep convolutional neural networks](https://arxiv.org/abs/1606.00298)

'''
from __future__ import print_function
from __future__ import absolute_import

from .. import backend as K
from ..layers import Input, Dense
from ..models import Model
from ..layers import Dense, Dropout, Flatten
from ..layers.convolutional import Convolution2D
from ..layers.convolutional import MaxPooling2D, ZeroPadding2D
from ..layers.normalization import BatchNormalization
from ..layers.advanced_activations import ELU
from ..utils.data_utils import get_file
from ..layers import Input, Dense

TH_WEIGHTS_PATH = 'https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/cnn_weights_theano.h5'
TF_WEIGHTS_PATH = 'https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/data/cnn_weights_tensorflow.h5'


def AudioConvnet(weights='msd', input_tensor=None):
    '''Instantiate the AudioConvnet architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.


    # Returns
        A Keras model instance.
    '''
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    # Determine proper input shape
    if K.image_dim_ordering() == 'th':
        input_shape = (1, 96, 1366)
    else:
        input_shape = (96, 1366, 1)

    if input_tensor is None:
        melgram_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            melgram_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            melgram_input = input_tensor

    # Determine input axis
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(melgram_input)

    # Conv block 1
    x = Convolution2D(32, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=2, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool1')(x)
    x = Dropout(0.5, name='dropout1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=2, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool2')(x)
    x = Dropout(0.5, name='dropout2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=2, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 4), name='pool3')(x)
    x = Dropout(0.5, name='dropout3')(x)

    # Conv block 4
    x = Convolution2D(192, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=2, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 5), name='pool4')(x)
    x = Dropout(0.5, name='dropout4')(x)

    # Conv block 5
    x = Convolution2D(256, 3, 3, border_mode='same', name='conv5')(x)
    x = BatchNormalization(axis=channel_axis, mode=2, name='bn5')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), name='pool5')(x)
    x = Dropout(0.5, name='dropout5')(x)

    # Output
    x = Flatten()(x)
    x = Dense(50, activation='sigmoid', name='output')(x)

    # Create model
    model = Model(melgram_input, x)

    # Load weights
    if K._BACKEND == 'theano':
        weights_path = get_file('rnn_weights_theano.h5',
                                TH_WEIGHTS_PATH,
                                cache_subdir='models')
    else:
        weights_path = get_file('rnn_weights_tensorflow.h5',
                                TF_WEIGHTS_PATH,
                                cache_subdir='models')

    model.load_weights(weights_path)
    return model
