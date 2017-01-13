# -*- coding: utf-8 -*-
"""MusicTaggerCRNN model for Keras.

# Reference:

- [Music-auto_tagging-keras](https://github.com/keunwoochoi/music-auto_tagging-keras)

"""
from __future__ import print_function
from __future__ import absolute_import

from .. import backend as K
from ..layers import Input, Dense
from ..models import Model
from ..layers import Dense, Dropout, Reshape, Permute
from ..layers.convolutional import Convolution2D
from ..layers.convolutional import MaxPooling2D, ZeroPadding2D
from ..layers.normalization import BatchNormalization
from ..layers.advanced_activations import ELU
from ..layers.recurrent import GRU
from ..engine.topology import get_source_inputs
from ..utils.data_utils import get_file
from ..utils.layer_utils import convert_all_kernels_in_model
from .audio_conv_utils import decode_predictions, preprocess_input

TH_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.3/music_tagger_crnn_weights_tf_kernels_th_dim_ordering.h5'
TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.3/music_tagger_crnn_weights_tf_kernels_tf_dim_ordering.h5'


def MusicTaggerCRNN(weights='msd', input_tensor=None,
                    include_top=True, classes=50):
    """Instantiate the MusicTaggerCRNN architecture,
    optionally loading weights pre-trained
    on Million Song Dataset. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    For preparing mel-spectrogram input, see
    `audio_conv_utils.py` in [applications](https://github.com/fchollet/keras/tree/master/keras/applications).
    You will need to install [Librosa](http://librosa.github.io/librosa/)
    to use it.

    # Arguments
        weights: one of `None` (random initialization)
            or "msd" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        include_top: whether to include the 1 fully-connected
            layer (output layer) at the top of the network.
            If False, the network outputs 32-dim features.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.
    """
    if weights not in {'msd', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `msd` '
                         '(pre-training on Million Song Dataset).')

    if weights == 'msd' and include_top and classes != 50:
        raise ValueError('If using `weights` as msd with `include_top`'
                         ' as true, `classes` should be 50')
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
    x = ZeroPadding2D(padding=(0, 37))(melgram_input)
    x = BatchNormalization(axis=time_axis, name='bn_0_freq')(x)

    # Conv block 1
    x = Convolution2D(64, 3, 3, border_mode='same', name='conv1')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1')(x)

    # Conv block 2
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv2')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(3, 3), name='pool2')(x)

    # Conv block 3
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv3')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool3')(x)

    # Conv block 4
    x = Convolution2D(128, 3, 3, border_mode='same', name='conv4')(x)
    x = BatchNormalization(axis=channel_axis, mode=0, name='bn4')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool4')(x)

    # reshaping
    if K.image_dim_ordering() == 'th':
        x = Permute((3, 1, 2))(x)
    x = Reshape((15, 128))(x)

    # GRU block 1, 2, output
    x = GRU(32, return_sequences=True, name='gru1')(x)
    x = GRU(32, return_sequences=False, name='gru2')(x)

    if include_top:
        x = Dense(classes, activation='sigmoid', name='output')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = melgram_input
    # Create model.
    model = Model(inputs, x, name='music_tagger_crnn')

    if weights is None:
        return model
    else:
        # Load weights
        if K.image_dim_ordering() == 'tf':
            weights_path = get_file('music_tagger_crnn_weights_tf_kernels_tf_dim_ordering.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('music_tagger_crnn_weights_tf_kernels_th_dim_ordering.h5',
                                    TH_WEIGHTS_PATH,
                                    cache_subdir='models')
        model.load_weights(weights_path, by_name=True)
        if K.backend() == 'theano':
            convert_all_kernels_in_model(model)
        return model
