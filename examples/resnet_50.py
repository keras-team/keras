'''This script demonstrates how to build the deep residual network
using the Keras functional API.

get_resne50 returns the deep residual network model (50 layers)

Pleas visit Kaiming He' github homepage:
https://github.com/KaimingHe
for more information.

The related paper is
"Deep Residual Learning for Image Recognition"
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
http://arxiv.org/abs/1512.03385

Pretrained weights were converted from Kaiming He's caffemodel directly.

For now we provide weights for tensorflow backend only, please donwload them at:
http://pan.baidu.com/s/1i4Qp6CH (For China)
https://drive.google.com/open?id=0B4ChsjFJvew3NVQ2U041Q0xHRHM (For Other countries)

@author: BigMoyan, University of Electronic Science and Techonlogy of China
'''
from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras.preprocessing.image import load_img, img_to_array
import keras.backend as K
import numpy as np


# The names of layers in resnet50 are generated with the following format
# [type][stage][block]_branch[branch][layer]
# type: 'res' for conv layer, 'bn' and 'scale' for BN layer
# stage: from '2' to '5', current stage number
# block: 'a','b','c'... for different blocks in a stage
# branch: '1' for shortcut and '2' for main path
# layer: 'a','b','c'... for different layers in a block

def identity_block(input_tensor, nb_filter, stage, block, kernel_size=3):
    """
    the identity_block is the block that has no conv layer at shortcut
    params:
        input_tensor: input tensor
        nb_filter: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage number
        block: str like 'a','b'.., curretn block
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
    """
    nb_filter1, nb_filter2, nb_filter3 = nb_filter

    out = Convolution2D(nb_filter1, 1, 1, name='res'+str(stage)+block+'_branch2a')(input_tensor)
    out = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch2a')(out)
    out = Activation('relu')(out)

    out = out = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                              name='res'+str(stage)+block+'_branch2b')(out)
    out = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter3, 1, 1, name='res'+str(stage)+block+'_branch2c')(out)
    out = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch2c')(out)

    out = merge([out, input_tensor], mode='sum')
    out = Activation('relu')(out)
    return out


def conv_block(input_tensor, nb_filter, stage, block, kernel_size=3):
    """
    conv_block is the block that has a conv layer at shortcut
    params:
        input_tensor: input tensor
        nb_filter: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage number
        block: str like 'a','b'.., current block
        kernel_size: defualt 3, the kernel size of middle conv layer at main path

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should has subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = nb_filter
    if stage == 2:
        out = Convolution2D(nb_filter1, 1, 1, name='res'+str(stage)+block+'_branch2a')(input_tensor)
    else:
        out = Convolution2D(nb_filter1, 1, 1, subsample=(2, 2), name='res'+str(stage)+block+'_branch2a')(input_tensor)

    out = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch2a')(out)
    out = Activation('relu')(out)

    out = out = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                              name='res'+str(stage)+block+'_branch2b')(out)
    out = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter3, 1, 1, name='res'+str(stage)+block+'_branch2c')(out)
    out = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch2c')(out)

    if stage == 2:
        shortcut = Convolution2D(nb_filter3, 1, 1, name='res'+str(stage)+block+'_branch1')(input_tensor)
    else:
        shortcut = Convolution2D(nb_filter3, 1, 1, subsample=(2, 2), name='res'+str(stage)+block+'_branch1')(input_tensor)
    shortcut = BatchNormalization(axis=1, name='bn'+str(stage)+block+'_branch1')(shortcut)

    out = merge([out, shortcut], mode='sum')
    out = Activation('relu')(out)
    return out


def read_img(img_path):
    """
    this function returns preprocessed image
    """
    mean = (103.939, 116.779, 123.68)
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img, dim_ordering='tf')

    # decenterize
    img[0, :, :] -= mean[0]
    img[1, :, :] -= mean[1]
    img[2, :, :] -= mean[2]

    # 'RGB'->'BGR'
    img = img[::-1, :, :]

    # expand dim for test
    img = np.expand_dims(img, axis=0)
    return img


def get_resnet50():
    """
    this function returns the 50-layer residual network model
    you should load pretrained weights if you want to use it directly.
    Note that since the pretrained weights is converted from caffemodel
    the order of channels for input image should be 'BGR' (the channel order of caffe)
    """
    K.set_image_dim_ordering('tf')
    inp = Input(shape=(3, 224, 224))
    out = ZeroPadding2D((3, 3))(inp)
    out = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(out)
    out = BatchNormalization(axis=1, name='bn_conv1')(out)
    out = Activation('relu')(out)
    out = MaxPooling2D((3, 3), strides=(2, 2))(out)

    out = conv_block(out, [64, 64, 256], 2, 'a')
    out = identity_block(out, [64, 64, 256], 2, 'b')
    out = identity_block(out, [64, 64, 256], 2, 'c')

    out = conv_block(out, [128, 128, 512], 3, 'a')
    out = identity_block(out, [128, 128, 512], 3, 'b')
    out = identity_block(out, [128, 128, 512], 3, 'c')
    out = identity_block(out, [128, 128, 512], 3, 'd')

    out = conv_block(out, [256, 256, 1024], 4, 'a')
    out = identity_block(out, [256, 256, 1024], 4, 'b')
    out = identity_block(out, [256, 256, 1024], 4, 'c')
    out = identity_block(out, [256, 256, 1024], 4, 'd')
    out = identity_block(out, [256, 256, 1024], 4, 'e')
    out = identity_block(out, [256, 256, 1024], 4, 'f')

    out = conv_block(out, [512, 512, 2048], 5, 'a')
    out = identity_block(out, [512, 512, 2048], 5, 'b')
    out = identity_block(out, [512, 512, 2048], 5, 'c')

    out = AveragePooling2D((7, 7))(out)
    out = Flatten()(out)
    out = Dense(1000, activation='softmax', name='fc1000')(out)

    model = Model(inp, out)

    return model


if __name__ == '__main__':
    resnet_model = get_resnet50()
    resnet_model.load_weights('th_resnet50.h5')
    test_img1 = read_img('cat.jpg')
    test_img2 = read_img('airplane.jpg')
    # you may download synset_words from address given at the begining of this file
    class_table = open('synset_words', 'r')
    lines = class_table.readlines()
    print "result for test 1 is"
    print lines[np.argmax(resnet_model.predict(test_img1)[0])]
    print "result for test 2 is"
    print lines[np.argmax(resnet_model.predict(test_img2)[0])]
    class_table.close()
