# -*- coding: utf-8 -*-
"""Inception V4 model for Keras.

Do note that the input image format for this model is different than
for the VGG16 and ResNet models (299x299 instead of 224x224), and that
the input preprocessing function is also different (same as Xception).

# Reference

- [Rethinking the Inception Architecture for Computer
  Vision](https://arxiv.org/abs/1602.07261)

"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from ..models import Model
from ..layers import Flatten, Dense, Input, BatchNormalization, merge
from ..layers import Dropout, Activation
from ..layers import Convolution2D, MaxPooling2D, AveragePooling2D
from ..engine.topology import get_source_inputs
from ..utils.layer_utils import convert_all_kernels_in_model
from ..utils.data_utils import get_file
from .. import backend as K
from .imagenet_utils import decode_predictions, _obtain_input_shape


TH_WEIGHTS_PATH = ''
TF_WEIGHTS_PATH = ''
TH_WEIGHTS_PATH_NO_TOP = ''
TF_WEIGHTS_PATH_NO_TOP = ''


def conv2d_bn(x, nb_filter, kW, kH, dW, dH, border, name=None):
    """Utility funciton to apply conv + BN"""
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3

    x = Convolution2D(nb_filter, kW, kH, activation='relu',
                      subsample=(dW, dH), border_mode=border,
                      name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)

    return x


def stem(x, channel_axis):
    if K.image_dim_ordering() == 'th':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(x, 32, 3, 3, 2, 2, 'valid')
    x = conv2d_bn(x, 32, 3, 3, 1, 1, 'valid')
    x = conv2d_bn(x, 64, 3, 3, 1, 1, 'same')

    p1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)
    conv_bn1 = conv2d_bn(x, 96, 3, 3, 2, 2, 'valid')

    cat1 = merge([p1, conv_bn1], mode='concat', concat_axis=channel_axis)

    p1 = conv2d_bn(cat1, 64, 1, 1, 1, 1, 'same')
    p1 = conv2d_bn(p1, 96, 3, 3, 1, 1, 'valid')

    conv_bn1 = conv2d_bn(cat1, 64, 1, 1, 1, 1, 'same')
    conv_bn1 = conv2d_bn(conv_bn1, 64, 7, 1, 1, 1, 'same')
    conv_bn1 = conv2d_bn(conv_bn1, 64, 1, 7, 1, 1, 'same')
    conv_bn1 = conv2d_bn(conv_bn1, 96, 3, 3, 1, 1, 'valid')

    cat2 = merge([p1, conv_bn1], mode='concat', concat_axis=channel_axis)

    p2 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(cat2)
    conv_bn2 = conv2d_bn(cat2, 192, 3, 3, 2, 2, 'valid')

    cat3 = merge([p2, conv_bn2], mode='concat', concat_axis=channel_axis)
    cat3 = Activation('relu')(cat3)

    return cat3


def reduction_A(x, channel_axis, k=192, l=224, m=256, n=384):
    r1 = MaxPooling2D((3, 3), strides=(2, 2), border_mode='valid')(x)
    conv_bn1 = conv2d_bn(x, n, 3, 3, 2, 2, 'valid')
    conv_bn2 = conv2d_bn(x, k, 1, 1, 1, 1, 'same')
    conv_bn2 = conv2d_bn(conv_bn2, l, 3, 3, 1, 1, 'same')
    conv_bn2 = conv2d_bn(conv_bn2, m, 3, 3, 2, 2, 'valid')

    cat1 = merge([r1, conv_bn1, conv_bn2],
                 mode='concat', concat_axis=channel_axis)

    cat1 = Activation('relu')(cat1)

    return cat1


def reduction_B(x, channel_axis):
    r1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid')(x)

    conv_bn1 = conv2d_bn(x, 192, 1, 1, 1, 1, 'same')
    conv_bn1 = conv2d_bn(conv_bn1, 192, 3, 3, 2, 2, 'valid')

    conv_bn2 = conv2d_bn(x, 256, 1, 1, 1, 1, 'same')
    conv_bn2 = conv2d_bn(conv_bn2, 256, 1, 7, 1, 1, 'same')
    conv_bn2 = conv2d_bn(conv_bn2, 320, 7, 1, 1, 1, 'same')
    conv_bn2 = conv2d_bn(conv_bn2, 320, 3, 3, 2, 2, 'valid')

    cat1 = merge([r1, conv_bn1, conv_bn2],
                 mode='concat', concat_axis=channel_axis)

    cat1 = Activation('relu')(cat1)

    return cat1


def inception_A(x, channel_axis):
    a1 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    a1 = conv2d_bn(a1, 96, 1, 1, 1, 1, 'same')

    conv_bn2 = conv2d_bn(x, 96, 1, 1, 1, 1, 'same')

    conv_bn3 = conv2d_bn(x, 64, 1, 1, 1, 1, 'same')
    conv_bn3 = conv2d_bn(conv_bn3, 96, 3, 3, 1, 1, 'same')

    conv_bn4 = conv2d_bn(x, 64, 1, 1, 1, 1, 'same')
    conv_bn4 = conv2d_bn(conv_bn4, 96, 3, 3, 1, 1, 'same')
    conv_bn4 = conv2d_bn(conv_bn4, 96, 3, 3, 1, 1, 'same')

    cat1 = merge([a1, conv_bn2, conv_bn3, conv_bn4],
                 mode='concat', concat_axis=channel_axis)
    cat1 = Activation('relu')(cat1)

    return cat1


def inception_B(x, channel_axis):
    b1 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    b1 = conv2d_bn(b1, 128, 1, 1, 1, 1, 'same')

    conv_bn2 = conv2d_bn(x, 384, 1, 1, 1, 1, 'same')

    conv_bn3 = conv2d_bn(x, 192, 1, 1, 1, 1, 'same')
    conv_bn3 = conv2d_bn(conv_bn3, 224, 1, 7, 1, 1, 'same')
    conv_bn3 = conv2d_bn(conv_bn3, 256, 7, 1, 1, 1, 'same')

    conv_bn4 = conv2d_bn(x, 192, 1, 1, 1, 1, 'same')
    conv_bn4 = conv2d_bn(conv_bn4, 192, 1, 7, 1, 1, 'same')
    conv_bn4 = conv2d_bn(conv_bn4, 224, 7, 1, 1, 1, 'same')
    conv_bn4 = conv2d_bn(conv_bn4, 224, 1, 7, 1, 1, 'same')
    conv_bn4 = conv2d_bn(conv_bn4, 256, 7, 1, 1, 1, 'same')

    cat1 = merge([b1, conv_bn2, conv_bn3, conv_bn4],
                 mode='concat', concat_axis=channel_axis)

    cat1 = Activation('relu')(cat1)

    return cat1


def inception_C(x, channel_axis):
    c1 = AveragePooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
    c1 = conv2d_bn(c1, 256, 1, 1, 1, 1, 'same')

    conv_bn2 = conv2d_bn(x, 256, 1, 1, 1, 1, 'same')

    conv_bn3 = conv2d_bn(x, 384, 1, 1, 1, 1, 'same')
    conv_bn3_1 = conv2d_bn(conv_bn3, 256, 3, 1, 1, 1, 'same')
    conv_bn3_2 = conv2d_bn(conv_bn3, 256, 1, 3, 1, 1, 'same')

    conv_bn4 = conv2d_bn(x, 384, 1, 1, 1, 1, 'same')
    conv_bn4 = conv2d_bn(conv_bn4, 448, 1, 3, 1, 1, 'same')
    conv_bn4 = conv2d_bn(conv_bn4, 512, 3, 1, 1, 1, 'same')

    conv_bn4_1 = conv2d_bn(conv_bn4, 256, 3, 1, 1, 1, 'same')
    conv_bn_4_2 = conv2d_bn(conv_bn4, 256, 1, 3, 1, 1, 'same')

    cat1 = merge([c1, conv_bn2, conv_bn3_1, conv_bn3_2, conv_bn4_1,
                  conv_bn_4_2],
                 mode='concat', concat_axis=channel_axis)

    cat1 = Activation('relu')(cat1)

    return cat1


def auxiliary_logits(x):
    """auxiliary head logits"""
    x = AveragePooling2D((5, 5), strides=(3, 3))(x)
    x = Convolution2D(128, 1, 1, 1, 1, 'same')(x)
    x = Flatten()(x)
    x = Dense(768, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation='softmax')(x)

    return x


def top_layer(x):
    """final pooling and prediction"""
    x = AveragePooling2D((8, 8), strides=(1, 1), border_mode='same')(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    x = Dense(1000, activation='softmax')(x)

    return x


def InceptionV4(include_top=True, weights='imagenet',
                input_tensor=None, input_shape=None):

    """Instantiate the Inception v4 architecture,
    optionally loading weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_dim_ordering="tf"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The dimension ordering
    convention used by the model is the one
    specified in your Keras config file.

    Note that the default input image size for this model is 299x299.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)` (with `tf` dim ordering)
            or `(3, 299, 299)` (with `th` dim ordering).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 139.
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
                                      default_size=299,
                                      min_size=139,
                                      dim_ordering=K.image_dim_ordering(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    if K.image_dim_ordering() == 'th':
        channel_axis = 1
    else:
        channel_axis = 3

    # Input Shape is 299 x 299 x 3 (tf) or 3 x 299 x 299 (th)
    model = stem(img_input, channel_axis)

    # 4 x Inception A
    for i in range(4):
        model = inception_A(model, channel_axis)

    # Reduction A
    model = reduction_A(model, channel_axis)

    # 7 x Inception B
    for i in range(7):
        model = inception_B(model, channel_axis)

    # Reduction B
    model = reduction_B(model, channel_axis)

    # 3 x Inception C
    for i in range(3):
        model = inception_C(model, channel_axis)

    if include_top:
        # classification block
        model = top_layer(model)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, model, name='inception_v4')

    # load weights
    if weights == 'imagenet':
        if K.image_dim_ordering() == 'th':
            if include_top:
                weights_path = get_file(
                    'inception_v4_weights_th_dim_ordering_th_kernels.h5',
                    TH_WEIGHTS_PATH,
                    cache_subdir='models',
                    md5_hash=''
                )
            else:
                weights_path = get_file(
                    'inception_v4_weights_th_dim_ordering_th_kernels_notop.h5',
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
                    'inception_v4_weights_tf_dim_ordering_tf_kernels.h5',
                    TF_WEIGHTS_PATH,
                    cache_subdir='models',
                    md5_hash=''
                )
            else:
                weights_path = get_file(
                    'inception_v4_weights_tf_dim_ordering_tf_kernels_notop.h5',
                    TF_WEIGHTS_PATH_NO_TOP,
                    cache_subdir='models',
                    md5_hash=''
                )

            model.load_weights(weights_path)

            if K.backend() == 'theano':
                convert_all_kernels_in_model(model)

    return model


def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x
