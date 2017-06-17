'''MobileNet models for Keras.
# Reference
- [MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf))
'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import warnings

from keras.models import Model
from keras.layers.core import Activation, Dropout, Reshape
from keras.activations import relu
from keras.layers.convolutional import Convolution2D, DepthwiseConvolution2D
from keras.layers.pooling import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
import keras.backend as K


BASE_WEIGHT_PATH = 'https://github.com/titu1994/MobileNetworks/releases/download/v1.0/'


def MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1,
              dropout=1e-3, include_top=True, weights='imagenet',
              input_tensor=None, pooling=None, classes=1000):
    ''' Instantiate the MobileNet architecture.
        Note that only TensorFlow is supported for now,
        therefore it only works with the data format
        `image_data_format='channels_last'` in your Keras config
        at ~/.keras/keras.json.
        # Arguments
            input_shape: optional shape tuple, only to be specified
                if `include_top` is False (otherwise the input shape
                has to be `(224, 224, 3)` (with `channels_last` data format)
                or (3, 224, 224) (with `channels_first` data format).
                It should have exactly 3 inputs channels,
                and width and height should be no smaller than 32.
                E.g. `(200, 200, 3)` would be one valid value.
            alpha: width multiplier of the MobileNet.
            depth_multiplier: depth multiplier for depthwise convolution
                (also called the resolution multiplier)
            dropout: dropout rate
            include_top: whether to include the fully-connected
                layer at the top of the network.
            weights: `None` (random initialization) or
                `imagenet` (ImageNet weights)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            pooling: Optional pooling mode for feature extraction
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
        '''

    if K.backend() != 'tensorflow':
        raise AttributeError('Only Tensorflow backend is currently supported, '
                             'as other backends do not support depthwise convolution.')

    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as ImageNet with `include_top`'
                         ' as true, `classes` should be 1001')

    if weights == 'imagenet':
        assert depth_multiplier == 1, "If imagenet weights are being loaded, depth multiplier must be 1"

        assert alpha in [0.25, 0.50, 0.75, 1.0], "If imagenet weights are being loaded, alpha can be one of" \
                                                 "`0.25`, `0.50`, `0.75` or `1.0` only."

        rows, cols = (0, 1) if K.image_data_format() == 'channels_last' else (1, 2)

        rows = int(input_shape[rows])
        cols = int(input_shape[cols])

        assert rows == cols and rows in [128, 160, 192, 224], "If imagenet weights are being loaded," \
                                                              "image must have a square shape (one of " \
                                                              "(128,128), (160,160), (192,192), or (224, 224))." \
                                                              "Image shape provided = (%d, %d)" % (rows, cols)

    if K.image_data_format() != 'channels_last':
        warnings.warn('The MobileNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height). '
                      'You should set `image_data_format="channels_last"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    # Determine proper input shape. Note, include_top is False by default, as
    # input shape can be anything larger than 32x32 and the same number of parameters
    # will be used.
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=32,
                                      data_format=K.image_data_format(),
                                      include_top=False)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = __create_mobilenet(classes, img_input, include_top, alpha, depth_multiplier, dropout, pooling)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='mobilenet')

    # load weights
    if weights == 'imagenet':
        if K.image_data_format() == 'channels_first':
            raise AttributeError('Weights for Channels Last format are not available')

        if alpha == 1.0:
            alpha_text = "1_0"
        elif alpha == 0.75:
            alpha_text = "7_5"
        elif alpha == 0.50:
            alpha_text = "5_0"
        else:
            alpha_text = "2_5"

        if include_top:
            model_name = "mobilenet_%s_%d_tf.h5" % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name,
                                    weigh_path,
                                    cache_subdir='models')
        else:
            model_name = "mobilenet_%s_%d_tf_no_top.h5" % (alpha_text, rows)
            weigh_path = BASE_WEIGHT_PATH + model_name
            weights_path = get_file(model_name,
                                    weigh_path,
                                    cache_subdir='models')

        model.load_weights(weights_path)

    if old_data_format:
        K.set_image_data_format(old_data_format)

    return model


def __conv_block(input, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    ''' Adds an initail convolution layer (with batch normalization and relu6)
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)

    x = Convolution2D(filters, kernel, padding='same', use_bias=False, strides=strides,
                      name='conv1')(input)
    x = BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    x = Activation(lambda x: relu(x, max_value=6), name='conv1_relu')(x)

    return x


def __depthwise_conv_block(input, pointwise_conv_filters, alpha,
                           depth_multiplier=1, strides=(1, 1), id=1):
    ''' Adds a depthwise convolution block (depthwise conv, batch normalization, relu6,
        pointwise convolution, batch normalization and relu6)
    '''
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    x = DepthwiseConvolution2D(kernel_size=(3, 3), padding='same', depth_multiplier=depth_multiplier,
                               strides=strides, use_bias=False, name='conv_dw_%d' % id)(input)
    x = BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % id)(x)
    x = Activation(lambda x: relu(x, max_value=6), name='conv_dw_%d_relu' % id)(x)

    x = Convolution2D(pointwise_conv_filters, (1, 1), padding='same', use_bias=False, strides=(1, 1),
                      name='conv_pw_%d' % id)(x)
    x = BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % id)(x)
    x = Activation(lambda x: relu(x, max_value=6), name='conv_pw_%d_relu' % id)(x)

    return x


def __create_mobilenet(classes, img_input, include_top, alpha, depth_multiplier, dropout, pooling):
    ''' Creates a MobileNet model with specified parameters
    Args:
        classes: Number of output classes
        img_input: Input tensor or layer
        include_top: Flag to include the last dense layer
        alpha: width multiplier of the MobileNet.
        depth_multiplier: depth multiplier for depthwise convolution
                          (also called the resolution multiplier)
        dropout: dropout rate
        pooling: Optional pooling mode for feature extraction
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
    Returns: a Keras Model
    '''

    x = __conv_block(img_input, 32, alpha, strides=(2, 2))
    x = __depthwise_conv_block(x, 64, alpha, depth_multiplier, id=1)

    x = __depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), id=2)
    x = __depthwise_conv_block(x, 128, alpha, depth_multiplier, id=3)

    x = __depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), id=4)
    x = __depthwise_conv_block(x, 256, alpha, depth_multiplier, id=5)

    x = __depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), id=6)
    x = __depthwise_conv_block(x, 512, alpha, depth_multiplier, id=7)
    x = __depthwise_conv_block(x, 512, alpha, depth_multiplier, id=8)
    x = __depthwise_conv_block(x, 512, alpha, depth_multiplier, id=9)
    x = __depthwise_conv_block(x, 512, alpha, depth_multiplier, id=10)
    x = __depthwise_conv_block(x, 512, alpha, depth_multiplier, id=11)

    x = __depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), id=12)
    x = __depthwise_conv_block(x, 1024, alpha, depth_multiplier, id=13)

    if include_top:
        if K.image_data_format() == 'channels_first':
            shape = (int(1024 * alpha), 1, 1)
        else:
            shape = (1, 1, int(1024 * alpha))

        x = GlobalAveragePooling2D()(x)
        x = Reshape(shape, name='reshape_1')(x)
        x = Dropout(dropout, name='dropout')(x)
        x = Convolution2D(classes, (1, 1), padding='same', name='conv_preds')(x)
        x = Activation('softmax', name='act_softmax')(x)
        x = Reshape((classes,), name='reshape_2')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    return x
