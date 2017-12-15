# -*- coding: utf-8 -*-
"""DenseNet models for Keras.

# Reference paper:

- [Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)

# Reference implementation:

- [Torch DenseNets](https://github.com/liuzhuang13/DenseNet/blob/master/models/densenet.lua)
- [TensorNets](https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
"""
from __future__ import print_function
from __future__ import absolute_import

import os

from .. import backend as K
from ..models import Model
from ..layers import Activation
from ..layers import AveragePooling2D
from ..layers import BatchNormalization
from ..layers import Concatenate
from ..layers import Conv2D
from ..layers import Dense
from ..layers import Flatten
from ..layers import GlobalAveragePooling2D
from ..layers import Input
from ..layers import MaxPooling2D
from ..layers import ZeroPadding2D
from ..utils.data_utils import get_file
from ..engine.topology import get_source_inputs
from .imagenet_utils import decode_predictions
from .imagenet_utils import _obtain_input_shape


__url_prefix__ = 'https://github.com/taehoonlee/deep-learning-models/' \
                 'releases/download/densenet/'


def dense(x, blocks, scope):
    """A dense block.

    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        scope: string, block label.

    # Returns
        output tensor for the block.
    """
    for i in range(blocks):
        x = block(x, 32, scope="%s/block%d" % (scope, i + 1))
    return x


def transition(x, reduction, scope):
    """A transition block.

    # Arguments
        x: input tensor.
        reduction: float, compression rate at transition layers.
        scope: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5,
                           name="%s/bn" % scope)(x)
    x = Activation('relu', name="%s/relu" % scope)(x)
    x = Conv2D(int(x._keras_shape[bn_axis] * reduction), 1, use_bias=False,
               name="%s/conv" % scope)(x)
    x = AveragePooling2D(2, strides=2, name="%s/pool" % scope)(x)
    return x


def block(x, growth_rate, scope):
    """A building block for a dense block.

    # Arguments
        x: input tensor.
        growth_rate: float, growth rate at dense layers.
        scope: string, block label.

    # Returns
        output tensor for the block.
    """
    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1
    x1 = BatchNormalization(axis=bn_axis, epsilon=1e-5,
                            name="%s/0/bn" % scope)(x)
    x1 = Activation('relu', name="%s/0/relu" % scope)(x1)
    x1 = Conv2D(4 * growth_rate, 1, use_bias=False,
                name="%s/1/conv" % scope)(x1)
    x1 = BatchNormalization(axis=bn_axis, epsilon=1e-5,
                            name="%s/1/bn" % scope)(x1)
    x1 = Activation('relu', name="%s/1/relu" % scope)(x1)
    x1 = Conv2D(growth_rate, 3, padding='same', use_bias=False,
                name="%s/2/conv" % scope)(x1)
    x = Concatenate(axis=bn_axis, name="%s/concat" % scope)([x, x1])
    return x


def DenseNet(blocks, include_top, weights, input_tensor, input_shape,
             pooling, classes):
    """Instantiates the DenseNet architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format='channels_last'` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        blocks: numbers of building blocks for the four dense layers.
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels.
        pooling: optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=224,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if K.image_data_format() == 'channels_last' else 1

    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x = Conv2D(64, 7, strides=2, use_bias=False, name='conv1/conv')(x)
    x = BatchNormalization(axis=bn_axis, epsilon=1e-5,
                           name='conv1/bn')(x)
    x = Activation('relu', name='conv1/relu')(x)
    x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense(x, blocks[0], scope='conv2')
    x = transition(x, 0.5, scope='pool2')
    x = dense(x, blocks[1], scope='conv3')
    x = transition(x, 0.5, scope='pool3')
    x = dense(x, blocks[2], scope='conv4')
    x = transition(x, 0.5, scope='pool4')
    x = dense(x, blocks[3], scope='conv5')

    x = BatchNormalization(axis=bn_axis, epsilon=1e-5,
                           name='bn')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = AveragePooling2D(7, name='avg_pool')(x)
        elif pooling == 'max':
            x = MaxPooling2D(7, name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = Model(inputs, x, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = Model(inputs, x, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = Model(inputs, x, name='densenet201')
    else:
        model = Model(inputs, x, name='densenet')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            if blocks == [6, 12, 24, 16]:
                filename = 'densenet121_weights_tf_dim_ordering_tf_kernels.h5'
                weights_path = get_file(
                    filename, __url_prefix__ + filename,
                    cache_subdir='models',
                    file_hash='0962ca643bae20f9b6771cb844dca3b0')
            elif blocks == [6, 12, 32, 32]:
                filename = 'densenet169_weights_tf_dim_ordering_tf_kernels.h5'
                weights_path = get_file(
                    filename, __url_prefix__ + filename,
                    cache_subdir='models',
                    file_hash='bcf9965cf5064a5f9eb6d7dc69386f43')
            elif blocks == [6, 12, 48, 32]:
                filename = 'densenet201_weights_tf_dim_ordering_tf_kernels.h5'
                weights_path = get_file(
                    filename, __url_prefix__ + filename,
                    cache_subdir='models',
                    file_hash='7bb75edd58cb43163be7e0005fbe95ef')
        else:
            if blocks == [6, 12, 24, 16]:
                filename = 'densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
                weights_path = get_file(
                    filename, __url_prefix__ + filename,
                    cache_subdir='models',
                    file_hash='4912a53fbd2a69346e7f2c0b5ec8c6d3')
            elif blocks == [6, 12, 32, 32]:
                filename = 'densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5'
                weights_path = get_file(
                    filename, __url_prefix__ + filename,
                    cache_subdir='models',
                    file_hash='50662582284e4cf834ce40ab4dfa58c6')
            elif blocks == [6, 12, 48, 32]:
                filename = 'densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5'
                weights_path = get_file(
                    filename, __url_prefix__ + filename,
                    cache_subdir='models',
                    file_hash='1c2de60ee40562448dbac34a0737e798')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def DenseNet121(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return DenseNet([6, 12, 24, 16],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


def DenseNet169(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return DenseNet([6, 12, 32, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


def DenseNet201(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000):
    return DenseNet([6, 12, 48, 32],
                    include_top, weights,
                    input_tensor, input_shape,
                    pooling, classes)


def preprocess_input(x):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    x = x.copy()
    x /= 255.
    x[:, :, :, 0] -= 0.485
    x[:, :, :, 1] -= 0.456
    x[:, :, :, 2] -= 0.406
    x[:, :, :, 0] /= 0.229
    x[:, :, :, 1] /= 0.224
    x[:, :, :, 2] /= 0.225
    return x


setattr(DenseNet121, '__doc__', DenseNet.__doc__)
setattr(DenseNet169, '__doc__', DenseNet.__doc__)
setattr(DenseNet201, '__doc__', DenseNet.__doc__)
