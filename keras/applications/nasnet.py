"""NASNet-A models for Keras

NASNet refers to Neural Architecture Search Network, a family of models
that were designed automatically by learning the model architectures
directly on the dataset of interest.

Here we consider NASNet-A, the highest performance model that was found
for the CIFAR-10 dataset, and then extended to ImageNet 2012 dataset,
obtaining state of the art performance on CIFAR-10 and ImageNet 2012.
Only the NASNet-A models, and their respective weights, which are suited
for ImageNet 2012 are provided.

The below table describes the performance on ImageNet 2012:
------------------------------------------------------------------------------------
      Architecture       | Top-1 Acc | Top-5 Acc |  Multiply-Adds |  Params (M)
------------------------------------------------------------------------------------
|   NASNet-A (4 @ 1056)  |   74.0 %  |   91.6 %  |       564 M    |     5.3        |
|   NASNet-A (6 @ 4032)  |   82.7 %  |   96.2 %  |      23.8 B    |    88.9        |
------------------------------------------------------------------------------------

Weights obtained from the official Tensorflow repository found at
https://github.com/tensorflow/models/tree/master/research/slim/nets/nasnet

# References:
 - [Learning Transferable Architectures for Scalable Image Recognition]
    (https://arxiv.org/abs/1707.07012)

Based on the following implementations:
 - [TF Slim Implementation]
   (https://github.com/tensorflow/models/blob/master/research/slim/nets/nasnet/nasnet.)
 - [TensorNets implementation]
   (https://github.com/taehoonlee/tensornets/blob/master/tensornets/nasnets.py)
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from ..models import Model
from ..layers import Input
from ..layers import Activation
from ..layers import Dense
from ..layers import Dropout
from ..layers import BatchNormalization
from ..layers import MaxPooling2D
from ..layers import AveragePooling2D
from ..layers import GlobalAveragePooling2D
from ..layers import GlobalMaxPooling2D
from ..layers import Conv2D
from ..layers import SeparableConv2D
from ..layers import ZeroPadding2D
from ..layers import Cropping2D
from ..layers import concatenate
from ..layers import add
from ..regularizers import l2
from ..utils.data_utils import get_file
from ..engine.topology import get_source_inputs
from ..applications.imagenet_utils import _obtain_input_shape
from ..applications.inception_v3 import preprocess_input
from ..applications.imagenet_utils import decode_predictions
from .. import backend as K

_BN_DECAY = 0.9997
_BN_EPSILON = 1e-3

NASNET_MOBILE_WEIGHT_PATH = "https://github.com/titu1994/Keras-NASNet/releases/download/v1.0/NASNet-mobile.h5"
NASNET_MOBILE_WEIGHT_PATH_NO_TOP = "https://github.com/titu1994/Keras-NASNet/releases/download/v1.0/NASNet-mobile-no-top.h5"
NASNET_LARGE_WEIGHT_PATH = "https://github.com/titu1994/Keras-NASNet/releases/download/v1.1/NASNet-large.h5"
NASNET_LARGE_WEIGHT_PATH_NO_TOP = "https://github.com/titu1994/Keras-NASNet/releases/download/v1.1/NASNet-large-no-top.h5"


def NASNet(input_shape=None,
           penultimate_filters=4032,
           nb_blocks=6,
           stem_filters=96,
           skip_reduction=True,
           filters_multiplier=2,
           dropout=0.5,
           weight_decay=5e-5,
           include_top=True,
           weights=None,
           input_tensor=None,
           pooling=None,
           classes=1000,
           default_size=None):
    """Instantiates a NASNet model.
    Note that only TensorFlow is supported for now,
    therefore it only works with the data format
    `image_data_format='channels_last'` in your Keras config
    at `~/.keras/keras.json`.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(331, 331, 3)` for NASNetLarge or
            `(224, 224, 3)` for NASNetMobile
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        penultimate_filters: number of filters in the penultimate layer.
            NASNet models use the notation `NASNet (N @ P)`, where:
                -   N is the number of blocks
                -   P is the number of penultimate filters
        nb_blocks: number of repeated blocks of the NASNet model.
            NASNet models use the notation `NASNet (N @ P)`, where:
                -   N is the number of blocks
                -   P is the number of penultimate filters
        stem_filters: number of filters in the initial stem block
        skip_reduction: Whether to skip the reduction step at the tail
            end of the network. Set to `False` for CIFAR models.
        filters_multiplier: controls the width of the network.
            - If `filters_multiplier` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `filters_multiplier` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `filters_multiplier` = 1, default number of filters from the paper
                 are used at each layer.
        dropout: dropout rate
        weight_decay: l2 regularization weight
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        default_size: specifies the default image size of the model

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    if K.backend() != 'tensorflow':
        raise RuntimeError('Only Tensorflow backend is currently supported, '
                           'as other backends do not support '
                           'separable convolution.')

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top` '
                         'as true, `classes` should be 1000')

    if default_size is None:
        default_size = 331

    # Determine proper input shape and default size.
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_size,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top or weights)

    if K.image_data_format() != 'channels_last':
        warnings.warn('The NASNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height).'
                      ' You should set `image_data_format="channels_last"` '
                      'in your Keras config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    assert penultimate_filters % 24 == 0, "`penultimate_filters` needs to be divisible " \
                                          "by 24."

    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    filters = penultimate_filters // 24

    if not skip_reduction:
        x = Conv2D(stem_filters, (3, 3), strides=(2, 2), padding='valid', use_bias=False, name='stem_conv1',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)
    else:
        x = Conv2D(stem_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False, name='stem_conv1',
                   kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(img_input)

    x = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                           name='stem_bn1')(x)

    p = None
    if not skip_reduction:  # imagenet / mobile mode
        x, p = _reduction_A(x, p, filters // (filters_multiplier ** 2), weight_decay, id='stem_1')
        x, p = _reduction_A(x, p, filters // filters_multiplier, weight_decay, id='stem_2')

    for i in range(nb_blocks):
        x, p = _normal_A(x, p, filters, weight_decay, id='%d' % (i))

    x, p0 = _reduction_A(x, p, filters * filters_multiplier, weight_decay, id='reduce_%d' % (nb_blocks))

    p = p0 if not skip_reduction else p

    for i in range(nb_blocks):
        x, p = _normal_A(x, p, filters * filters_multiplier, weight_decay, id='%d' % (nb_blocks + i + 1))

    x, p0 = _reduction_A(x, p, filters * filters_multiplier ** 2, weight_decay, id='reduce_%d' % (2 * nb_blocks))

    p = p0 if not skip_reduction else p

    for i in range(nb_blocks):
        x, p = _normal_A(x, p, filters * filters_multiplier ** 2, weight_decay, id='%d' % (2 * nb_blocks + i + 1))

    x = Activation('relu')(x)

    if include_top:
        x = GlobalAveragePooling2D()(x)
        x = Dropout(dropout)(x)
        x = Dense(classes, activation='softmax', kernel_regularizer=l2(weight_decay), name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    model = Model(inputs, x, name='NASNet')

    # load weights
    if weights == 'imagenet':
        if default_size == 224:  # mobile version
            if include_top:
                weight_path = NASNET_MOBILE_WEIGHT_PATH
                model_name = 'nasnet_mobile.h5'
            else:
                weight_path = NASNET_MOBILE_WEIGHT_PATH_NO_TOP
                model_name = 'nasnet_mobile_no_top.h5'

            weights_file = get_file(model_name, weight_path, cache_subdir='models')
            model.load_weights(weights_file, by_name=True)

        elif default_size == 331:  # large version
            if include_top:
                weight_path = NASNET_LARGE_WEIGHT_PATH
                model_name = 'nasnet_large.h5'
            else:
                weight_path = NASNET_LARGE_WEIGHT_PATH_NO_TOP
                model_name = 'nasnet_large_no_top.h5'

            weights_file = get_file(model_name, weight_path, cache_subdir='models')
            model.load_weights(weights_file, by_name=True)
        else:
            raise ValueError('ImageNet weights can only be loaded on NASNetLarge or NASNetMobile')

    if old_data_format:
        K.set_image_data_format(old_data_format)

    return model


def NASNetLarge(input_shape=None,
                dropout=0.5,
                include_top=True,
                weights='imagenet',
                input_tensor=None,
                pooling=None,
                classes=1000):
    """Instantiates a NASNet model in ImageNet mode.
    Note that only TensorFlow is supported for now,
    therefore it only works with the data format
    `image_data_format='channels_last'` in your Keras config
    at `~/.keras/keras.json`.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(331, 331, 3)` for NASNetLarge.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        dropout: dropout rate
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        default_size: specifies the default image size of the model

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    return NASNet(input_shape,
                  penultimate_filters=4032,
                  nb_blocks=6,
                  stem_filters=96,
                  skip_reduction=False,
                  filters_multiplier=2,
                  dropout=dropout,
                  weight_decay=5e-5,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  default_size=331)


def NASNetMobile(input_shape=None,
                 dropout=0.5,
                 include_top=True,
                 weights='imagenet',
                 input_tensor=None,
                 pooling=None,
                 classes=1000):
    """Instantiates a Mobile NASNet model in ImageNet mode.
    Note that only TensorFlow is supported for now,
    therefore it only works with the data format
    `image_data_format='channels_last'` in your Keras config
    at `~/.keras/keras.json`.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` for NASNetMobile
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(224, 224, 3)` would be one valid value.
        dropout: dropout rate
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        default_size: specifies the default image size of the model
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    return NASNet(input_shape,
                  penultimate_filters=1056,
                  nb_blocks=4,
                  stem_filters=32,
                  skip_reduction=False,
                  filters_multiplier=2,
                  dropout=dropout,
                  weight_decay=4e-5,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  default_size=224)


def NASNetCIFAR(input_shape=None,
                dropout=0.0,
                include_top=True,
                weights=None,
                input_tensor=None,
                pooling=None,
                classes=10):
    """Instantiates a NASNet architecture in CIFAR mode.
    Note that only TensorFlow is supported for now,
    therefore it only works with the data format
    `image_data_format='channels_last'` in your Keras config
    at `~/.keras/keras.json`.

    # Arguments
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(32, 32, 3)` for NASNetMobile
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 32.
            E.g. `(32, 32, 3)` would be one valid value.
        dropout: dropout rate
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: `None` (random initialization) or
            `imagenet` (ImageNet weights)
        input_tensor: optional Keras tensor (i.e. output of
            `layers.Input()`)
            to use as image input for the model.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model
                will be the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a
                2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
        default_size: specifies the default image size of the model

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    global _BN_DECAY, _BN_EPSILON
    _BN_DECAY = 0.9
    _BN_EPSILON = 1e-5

    return NASNet(input_shape,
                  penultimate_filters=768,
                  nb_blocks=6,
                  stem_filters=32,
                  skip_reduction=True,
                  filters_multiplier=2,
                  dropout=dropout,
                  weight_decay=5e-4,
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  pooling=pooling,
                  classes=classes,
                  default_size=224)


def _separable_conv_block(ip, filters, kernel_size=(3, 3), strides=(1, 1), weight_decay=5e-5, id=None):
    '''Adds 2 blocks of [relu-separable conv-batchnorm]

    # Arguments:
        ip: input tensor
        filters: number of output filters per layer
        kernel_size: kernel size of separable convolutions
        strides: strided convolution for downsampling
        weight_decay: l2 regularization weight
        id: string id

    # Returns:
        a Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('separable_conv_block_%s' % id):
        x = Activation('relu')(ip)
        x = SeparableConv2D(filters, kernel_size, strides=strides, name='separable_conv_1_%s' % id,
                            padding='same', use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                               name="separable_conv_1_bn_%s" % (id))(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, kernel_size, name='separable_conv_2_%s' % id,
                            padding='same', use_bias=False, kernel_initializer='he_normal',
                            kernel_regularizer=l2(weight_decay))(x)
        x = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                               name="separable_conv_2_bn_%s" % (id))(x)
    return x


def _adjust_block(p, ip, filters, weight_decay=5e-5, id=None):
    '''
    Adjusts the input `previou path` to match the shape of the `input`
    or situations where the output number of filters needs to be changed

    # Arguments:
        p: input tensor which needs to be modified
        ip: input tensor whose shape needs to be matched
        filters: number of output filters to be matched
        weight_decay: l2 regularization weight
        id: string id

    # Returns:
        an adjusted Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1
    img_dim = 2 if K.image_data_format() == 'channels_first' else -2

    with K.name_scope('adjust_block'):
        if p is None:
            p = ip

        elif p._keras_shape[img_dim] != ip._keras_shape[img_dim]:
            with K.name_scope('adjust_reduction_block_%s' % id):
                p = Activation('relu', name='adjust_relu_1_%s' % id)(p)

                p1 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid', name='adjust_avg_pool_1_%s' % id)(p)
                p1 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                            name='adjust_conv_1_%s' % id, kernel_initializer='he_normal')(p1)

                p2 = ZeroPadding2D(padding=((0, 1), (0, 1)))(p)
                p2 = Cropping2D(cropping=((1, 0), (1, 0)))(p2)
                p2 = AveragePooling2D((1, 1), strides=(2, 2), padding='valid', name='adjust_avg_pool_2_%s' % id)(p2)
                p2 = Conv2D(filters // 2, (1, 1), padding='same', use_bias=False, kernel_regularizer=l2(weight_decay),
                            name='adjust_conv_2_%s' % id, kernel_initializer='he_normal')(p2)

                p = concatenate([p1, p2], axis=channel_dim)
                p = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                                       name='adjust_bn_%s' % id)(p)

        elif p._keras_shape[channel_dim] != filters:
            with K.name_scope('adjust_projection_block_%s' % id):
                p = Activation('relu')(p)
                p = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='adjust_conv_projection_%s' % id,
                           use_bias=False, kernel_regularizer=l2(weight_decay), kernel_initializer='he_normal')(p)
                p = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                                       name='adjust_bn_%s' % id)(p)
    return p


def _normal_A(ip, p, filters, weight_decay=5e-5, id=None):
    '''Adds a Normal cell for NASNet-A (Fig. 4 in the paper)

    # Arguments:
        ip: input tensor `x`
        p: input tensor `p`
        filters: number of output filters
        weight_decay: l2 regularization weight
        id: string id

    # Returns:
        a Keras tensor
    '''
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('normal_A_block_%s' % id):
        p = _adjust_block(p, ip, filters, weight_decay, id)

        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='normal_conv_1_%s' % id,
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(h)
        h = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                               name='normal_bn_1_%s' % id)(h)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, kernel_size=(5, 5), weight_decay=weight_decay,
                                         id='normal_left1_%s' % id)
            x1_2 = _separable_conv_block(p, filters, weight_decay=weight_decay, id='normal_right1_%s' % id)
            x1 = add([x1_1, x1_2], name='normal_add_1_%s' % id)

        with K.name_scope('block_2'):
            x2_1 = _separable_conv_block(p, filters, (5, 5), weight_decay=weight_decay, id='normal_left2_%s' % id)
            x2_2 = _separable_conv_block(p, filters, (3, 3), weight_decay=weight_decay, id='normal_right2_%s' % id)
            x2 = add([x2_1, x2_2], name='normal_add_2_%s' % id)

        with K.name_scope('block_3'):
            x3 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left3_%s' % (id))(h)
            x3 = add([x3, p], name='normal_add_3_%s' % id)

        with K.name_scope('block_4'):
            x4_1 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_left4_%s' % (id))(p)
            x4_2 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='normal_right4_%s' % (id))(p)
            x4 = add([x4_1, x4_2], name='normal_add_4_%s' % id)

        with K.name_scope('block_5'):
            x5 = _separable_conv_block(h, filters, weight_decay=weight_decay, id='normal_left5_%s' % id)
            x5 = add([x5, h], name='normal_add_5_%s' % id)

        x = concatenate([p, x1, x2, x3, x4, x5], axis=channel_dim, name='normal_concat_%s' % id)
    return x, ip


def _reduction_A(ip, p, filters, weight_decay=5e-5, id=None):
    '''Adds a Reduction cell for NASNet-A (Fig. 4 in the paper)

    # Arguments:
        ip: input tensor `x`
        p: input tensor `p`
        filters: number of output filters
        weight_decay: l2 regularization weight
        id: string id

    # Returns:
        a Keras tensor
    '''
    """"""
    channel_dim = 1 if K.image_data_format() == 'channels_first' else -1

    with K.name_scope('reduction_A_block_%s' % id):
        p = _adjust_block(p, ip, filters, weight_decay, id)

        h = Activation('relu')(ip)
        h = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='reduction_conv_1_%s' % id,
                   use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay))(h)
        h = BatchNormalization(axis=channel_dim, momentum=_BN_DECAY, epsilon=_BN_EPSILON,
                               name='reduction_bn_1_%s' % id)(h)

        with K.name_scope('block_1'):
            x1_1 = _separable_conv_block(h, filters, (5, 5), strides=(2, 2), weight_decay=weight_decay,
                                         id='reduction_left1_%s' % id)
            x1_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), weight_decay=weight_decay,
                                         id='reduction_1_%s' % id)
            x1 = add([x1_1, x1_2], name='reduction_add_1_%s' % id)

        with K.name_scope('block_2'):
            x2_1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_left2_%s' % id)(h)
            x2_2 = _separable_conv_block(p, filters, (7, 7), strides=(2, 2), weight_decay=weight_decay,
                                         id='reduction_right2_%s' % id)
            x2 = add([x2_1, x2_2], name='reduction_add_2_%s' % id)

        with K.name_scope('block_3'):
            x3_1 = AveragePooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_left3_%s' % id)(h)
            x3_2 = _separable_conv_block(p, filters, (5, 5), strides=(2, 2), weight_decay=weight_decay,
                                         id='reduction_right3_%s' % id)
            x3 = add([x3_1, x3_2], name='reduction_add3_%s' % id)

        with K.name_scope('block_4'):
            x4 = AveragePooling2D((3, 3), strides=(1, 1), padding='same', name='reduction_left4_%s' % id)(x1)
            x4 = add([x2, x4])

        with K.name_scope('block_5'):
            x5_1 = _separable_conv_block(x1, filters, (3, 3), weight_decay=weight_decay, id='reduction_left4_%s' % id)
            x5_2 = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='reduction_right5_%s' % id)(h)
            x5 = add([x5_1, x5_2], name='reduction_add4_%s' % id)

        x = concatenate([x2, x3, x4, x5], axis=channel_dim, name='reduction_concat_%s' % id)
        return x, ip
