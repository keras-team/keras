# -*- coding: utf-8 -*-
"""SqueezeNet-Residual model for Keras.

Do note that the input image format for this model is (227x227).

# Reference

- [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
  <0.5MB model size](https://arxiv.org/abs/1602.07360)

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from ..models import Model
from ..layers import Input, merge
from ..layers import Dropout, Activation
from ..layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from ..engine.topology import get_source_inputs
from ..utils.layer_utils import convert_all_kernels_in_model
from ..utils.data_utils import get_file
from .. import backend as K
from .imagenet_utils import decode_predictions, _obtain_input_shape


TH_WEIGHTS_PATH = ''
TF_WEIGHTS_PATH = ''
TH_WEIGHTS_PATH_NO_TOP = ''
TF_WEIGHTS_PATH_NO_TOP = ''

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
nb_classes = 1000


# function for fire node
def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
    else:
        channel_axis = 3

    x = Convolution2D(squeeze, 1, 1, border_mode='valid',
                      activation='relu', name=s_id + sq1x1)(x)

    left = Convolution2D(expand, 1, 1, border_mode='valid',
                         activation='relu', name=s_id + exp1x1)(x)

    right = Convolution2D(expand, 3, 3, border_mode='same',
                          activation='relu', name=s_id + exp3x3)(x)

    # SqueezeNet-Residual
    x = merge([left, right, x], mode='concat', concat_axis=channel_axis,
              name=s_id + 'concat')
    return x


# SqueezeNet architecture
def SqueezeNet(include_top=True, weights='imagenet',
               input_tensor=None, input_shape=None):

    """Instantiate the SqueezeNet architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    Note that the default input image size for this model is 227x227.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(227, 227, 3)` (with `tf` dim ordering)
            or `(3, 227, 227)` (with `th` dim ordering).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 127.
            E.g. `(150, 150, 3)` would be one valid value.

    # Returns
        A Keras model instance.
    """

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=127,
                                      dim_ordering=K.image_dim_ordering(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Convolution2D(64, 3, 3, subsample=(2, 2), border_mode='valid',
                      activation='relu', name='conv1')(img_input)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    fire_id = 2
    squeeze = [16, 32]
    expand = [64, 128]
    pool_id = [3, 5]
    # 2 * fire_module + maxpool
    for num in range(2):
        x = fire_module(x, fire_id=fire_id, squeeze=squeeze[num],
                        expand=expand[num])
        x = fire_module(x, fire_id=fire_id + 1, squeeze=squeeze[num],
                        expand=expand[num])
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                         name='pool' + str(pool_id[num]))(x)

        fire_id += 2

    squeeze = [48, 48, 64, 64]
    expand = [192, 192, 256, 256]
    # 4 * fire_module
    for num in range(4):
        x = fire_module(x, fire_id=fire_id, squeeze=squeeze[num],
                        expand=expand[num])

        fire_id += 1

    # housekeeping unecessary variables
    del fire_id, pool_id, squeeze, expand

    x = Dropout(0.5, name='drop9')(x)

    if include_top:
        x = Convolution2D(nb_classes, 1, 1, border_mode='valid',
                          activation='relu', name='conv10')(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax', name='predictions')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model
    model = Model(inputs, x, name='squeezenet')

    # load weights
    if weights == 'imagenet':
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file(
                    'squeezenet_weights_th_dim_ordering_th_kernels.h5',
                    TH_WEIGHTS_PATH,
                    cache_subdir='models',
                    md5_hash=''
                )
            else:
                weights_path = get_file(
                    'squeezenet_weights_th_dim_ordering_th_kernels_notop.h5',
                    TH_WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    md5_hash=''
                )

            model.load_weights(weights_path)

            if K.backend() == 'tensorflow':
                warnings.warn('You are using the TensorFlow backend, yet you '
                              'are using the Theano '
                              'image dimension ordering convention '
                              '(`image_dim_ordering="th"`). '
                              'For best performance, set '
                              '`image_dim_ordering="tf"` in '
                              'your Keras config '
                              'at ~/.keras/keras.json.')

                convert_all_kernels_in_model(model)
        else:
            if include_top:
                weights_path = get_file(
                    'squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
                    TF_WEIGHTS_PATH,
                    cache_subdir='models',
                    md5_hash=''
                )
            else:
                weights_path = get_file(
                    'squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    TF_WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    md5_hash=''
                )

            model.load_weights(weights_path)

            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)

    return model


# Remove image mean
def preprocess_input(x):
    x[:, :, 0] -= 104.006
    x[:, :, 1] -= 116.669
    x[:, :, 2] -= 122.679
    return x
