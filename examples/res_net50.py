# -*- coding: utf-8 -*-

"""
This script demonstrates how to build the resnet50 architecture
using the Keras functional API.

get_resne50 returns the resnet50 model, the names of the model follows the model given by Kaiming He

You may want to visit Kaiming He' github homepage:
https://github.com/KaimingHe
for more information and the visualizable model

The ralated paper is
"Deep Residual Learning for Image Recognition"
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
http://arxiv.org/abs/1512.03385

@author: BigMoyan, University of Electronic Science and Techonlogy of China

"""

from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras.utils.visualize_util import plot

# The names of layers in resnet50 are generated with the following format
# [type][stage][block]_branch[branch][layer]
# type: 'res' for conv layer, 'bn' and 'scale' for BN layer
# stage: from '2' to '5', current stage number
# block: 'a','b','c'... for different blocks in a stage
# branch: '1' for shortcut and '2' for main path
# layer: 'a','b','c'... for different layers in a block

def identity_block(x, nb_filter, stage, block, kernel_size=3):
    """
    the identity_block indicate the block that has no conv layer at shortcut
    params:
        x: input tensor
        nb_filter: list of integers, the nb_filters of 3 conv layer at main path
        stage: integet, current stage number
        block: str like 'a','b'.., curretn block
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
    """
    k1, k2, k3 = nb_filter

    out = Convolution2D(k1, 1, 1, name='res'+str(stage)+block+'_branch2a')(x)
    out = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch2a')(out)
    out = Activation('relu')(out)

    out = out = Convolution2D(k2, kernel_size, kernel_size, border_mode='same',
                              name='res'+str(stage)+block+'_branch2b')(out)
    out = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3, 1, 1, name='res'+str(stage)+block+'_branch2c')(out)
    out = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch2c')(out)

    out = merge([out, x], mode='sum')
    out = Activation('relu')(out)
    return out

def conv_block(x, nb_filter, stage, block, kernel_size=3):
    """
    conv_block indicate the block that has a conv layer at shortcut
    params:
        x: input tensor
        nb_filter: list of integers, the nb_filters of 3 conv layer at main path
        stage: integet, current stage number
        block: str like 'a','b'.., curretn block
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
    """
    k1, k2, k3 = nb_filter

    out = Convolution2D(k1, 1, 1, name='res'+str(stage)+block+'_branch2a')(x)
    out = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch2a')(out)
    out = Activation('relu')(out)

    out = out = Convolution2D(k2, kernel_size, kernel_size, border_mode='same',
                              name='res'+str(stage)+block+'_branch2b')(out)
    out = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3, 1, 1, name='res'+str(stage)+block+'_branch2c')(out)
    out = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch2c')(out)

    x = Convolution2D(k3, 1, 1, name='res'+str(stage)+block+'_branch1')(x)
    x = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch1')(x)

    out = merge([out, x], mode='sum')
    out = Activation('relu')(out)
    return out

# we build resnet 50 with the blocks defined above
def get_resnet50():
    """
    returns the resnet50 model
    Note that if you want to load weights from caffemodel,
    the order of channels of input image should be 'BGR'
    """

    inp = Input(shape=(3, 224, 224))
    out = ZeroPadding2D((3, 3))(inp)
    out = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(out)
    out = BatchNormalization(axis=1, name='bn_conv1')(out)
    out = Activation('relu')(out)
    out = MaxPooling2D((3, 3), strides=(2, 2))(out)

    # stage 2
    out = conv_block(out, [64, 64, 256], 2, 'a')
    out = identity_block(out, [64, 64, 256], 2, 'b')
    out = identity_block(out, [64, 64, 256], 2, 'c')

    # stage 3
    out = conv_block(out, [128, 128, 512], 3, 'a')
    out = identity_block(out, [128, 128, 512], 3, 'b')
    out = identity_block(out, [128, 128, 512], 3, 'c')
    out = identity_block(out, [128, 128, 512], 3, 'd')

    # stage 4
    out = conv_block(out, [256, 256, 1024], 4, 'a')
    out = identity_block(out, [256, 256, 1024], 4, 'b')
    out = identity_block(out, [256, 256, 1024], 4, 'c')
    out = identity_block(out, [256, 256, 1024], 4, 'd')
    out = identity_block(out, [256, 256, 1024], 4, 'e')
    out = identity_block(out, [256, 256, 1024], 4, 'f')

    # stage 5
    out = conv_block(out, [512, 512, 2048], 5, 'a')
    out = identity_block(out, [512, 512, 2048], 5, 'b')
    out = identity_block(out, [512, 512, 2048], 5, 'c')

    out = AveragePooling2D((7, 7))(out) # note that this is an AveragePooling
    out = Flatten()(out)
    out = Dense(1000, activation='softmax', name='fc1000')(out)

    model = Model(inp, out)

    return model


if __name__ == '__main__':
    resnet50 = get_resnet50()
    plot(resnet50)
    