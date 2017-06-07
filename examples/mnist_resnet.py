'''This example demonstrates Deep Residual Networks for image classification

Based on Kaiming He et al's paper:

1. Deep Residual Learning for Image Recognition. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.(https://arxiv.org/abs/1512.03385)
2. Identify Mappings in Deep Residual Networks. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.(https://arxiv.org/abs/1603.05027)

Performance on MNIST:
ResNet-18 gets to 98.58% test accuracy after 12 epochs.(1010s per epoch on average on a GRID K520 GPU)
ResNet-50 gets to 98.24% test accuracy after 12 epochs.(3556s per epoch on average on a GRID K520 GPU)

Based on Raghavendra Kotikalapudi's keras-resnet repository(https://github.com/raghakot/keras-resnet)

'''

from __future__ import print_function
np.random.seed(1337)  # for reproducibility

from keras.models import Model, Input, Activation, merge, Dense, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.datasets import mnist
from keras import backend as K

# Build a Conv -> BN -> ReLU block
# This is the original block proposed in https://arxiv.org/abs/1512.03385
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col,
                             subsample=subsample,
                             init="he_normal",
                             border_mode="same")(input)
        norm = BatchNormalization(mode=0, axis=1)(conv)
        return Activation("relu")(norm)
    return f


# Build a BN -> ReLU -> Conv block
# This is an improved block proposed in http://arxiv.org/abs/1603.05027
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col,
                             subsample=subsample,
                             init="he_normal",
                             border_mode="same")(activation)
    return f


# Merge the input and the residual by sum them up
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = int(round(input._keras_shape[2] * 1.0 / residual._keras_shape[2]))
    stride_height = int(round(input._keras_shape[3] * 1.0 / residual._keras_shape[3]))
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 subsample=(stride_width, stride_height),
                                 init="he_normal",
                                 border_mode="valid")(input)
    return merge([shortcut, residual], mode="sum")


# Build a residual block with repeating basic_block or bottleneck blocks.
def _residual_block(block_function, nb_filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input
    return f


# Basic 3X3 Conv -> 3X3 Conv blocks as shown in https://arxiv.org/abs/1512.03385
# Follow improved activations in http://arxiv.org/abs/1603.05027
# Used for resnet with layers <= 34
def basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv_3_3)
        return _shortcut(input, residual)
    return f


# Basic 1X1 Conv -> 3X3 Conv -> 1X1 Conv "bottleneck" blocks as shown in https://arxiv.org/abs/1512.03385
# Follow improved activations in http://arxiv.org/abs/1603.05027
# Used for resnet with layers <= 34
# Returns a final Conv layer of nb_filters * 4
def bottleneck_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1)(conv_3_3)
        return _shortcut(input, residual)
    return f


class ResNetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """
        Builds a custom Deep Residual Network Architecture
        :param input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)

        :param num_outputs: The number of outputs at final softmax layer

        :param block_fn: The block function to use. This is either :func:`basic_block` or :func:`bottleneck`.
        The original paper used basic_block for layers < 50

        :param repetitions: Number of repetitions of various block units.
        At each block unit, the number of filters are doubled and the input size is halved

        :return: The keras model.
        """
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(nb_filter=64, nb_row=7, nb_col=7, subsample=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode="same")(conv1)

        block = pool1
        nb_filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, nb_filters=nb_filters, repetitions=r, is_first_layer=i == 0)(block)
            nb_filters *= 2

        # Classifier block
        pool2 = AveragePooling2D(pool_size=(block._keras_shape[2], block._keras_shape[3]), strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(output_dim=num_outputs, init="he_normal", activation="softmax")(flatten1)

        model = Model(input=input, output=dense)
        return model

    # Build a ResNet-18, using the basic blocks
    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    # Build a ResNet-34, using the basic blocks
    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    # Build a ResNet-50, using the bottleneck blocks
    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, bottleneck_block, [3, 4, 6, 3])

    # Build a ResNet-101, using the bottleneck blocks
    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, bottleneck_block, [3, 4, 23, 3])

    # Build a ResNet-152, using the bottleneck blocks
    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResNetBuilder.build(input_shape, num_outputs, bottleneck_block, [3, 8, 36, 3])

batch_size = 128
nb_classes = 10
nb_epoch = 12

# Input image dimensions
img_rows, img_cols = 28, 28

# The data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = ResNetBuilder.build_resnet_18((1, img_rows, img_cols), nb_classes)

model.compile(loss="categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

model.fit(X_train, Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          verbose=1,
          validation_data=(X_test, Y_test))
