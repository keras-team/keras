'''This script demonstrates how to build a deep residual network
using the Keras functional API.

get_resnet50() returns the deep residual network model (50 layers)

Please visit Kaiming He's GitHub homepage:
https://github.com/KaimingHe
for more information.

The related paper is
'Deep Residual Learning for Image Recognition'
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
http://arxiv.org/abs/1512.03385

Pretrained weights were converted from Kaiming He's caffe model directly.

For now we provide weights for the tensorflow backend only,
thus use 'tf' dim_ordering (e.g. input_shape=(224, 224, 3) for 224*224 color image)
would accelerate the computation, but we also provide weights for 'th' dim_ordering for compatibility.
You can set your default dim ordering in your Keras config file at ~/.keras/keras.json

please donwload them at:
http://pan.baidu.com/s/1o8pO2q2 ('th' dim ordering, for China)
http://pan.baidu.com/s/1pLanuTt ('tf' dim ordering, for China)

https://drive.google.com/open?id=0B4ChsjFJvew3NVQ2U041Q0xHRHM ('th' dim ordering, for other countries)
https://drive.google.com/open?id=0B4ChsjFJvew3NWN5THdxcTdSWmc ('tf' dim ordering, for other countries)

@author: BigMoyan, University of Electronic Science and Technology of China
'''
from __future__ import print_function
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


def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    dim_ordering = K.image_dim_ordering()
    nb_filter1, nb_filter2, nb_filter3 = filters
    if dim_ordering == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    out = Convolution2D(nb_filter1, 1, 1, dim_ordering=dim_ordering, name=conv_name_base + '2a')(input_tensor)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                        dim_ordering=dim_ordering, name=conv_name_base + '2b')(out)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter3, 1, 1, dim_ordering=dim_ordering, name=conv_name_base + '2c')(out)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(out)

    out = merge([out, input_tensor], mode='sum')
    out = Activation('relu')(out)
    return out


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should has subsample=(2,2) as well
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    out = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                        dim_ordering=dim_ordering, name=conv_name_base + '2a')(input_tensor)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                        dim_ordering=dim_ordering, name=conv_name_base + '2b')(out)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(out)
    out = Activation('relu')(out)

    out = Convolution2D(nb_filter3, 1, 1, dim_ordering=dim_ordering, name=conv_name_base + '2c')(out)
    out = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(out)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             dim_ordering=dim_ordering, name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    out = merge([out, shortcut], mode='sum')
    out = Activation('relu')(out)
    return out


def read_img(img_path):
    '''This function returns a preprocessed image
    '''
    dim_ordering = K.image_dim_ordering()
    mean = (103.939, 116.779, 123.68)
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img, dim_ordering=dim_ordering)

    if dim_ordering == 'th':
        img[0, :, :] -= mean[0]
        img[1, :, :] -= mean[1]
        img[2, :, :] -= mean[2]
        # 'RGB'->'BGR'
        img = img[::-1, :, :]
    else:
        img[:, :, 0] -= mean[0]
        img[:, :, 1] -= mean[1]
        img[:, :, 2] -= mean[2]
        img = img[:, :, ::-1]

    img = np.expand_dims(img, axis=0)
    return img


def get_resnet50():
    '''This function returns the 50-layer residual network model
    you should load pretrained weights if you want to use it directly.
    Note that since the pretrained weights is converted from caffemodel
    the order of channels for input image should be 'BGR' (the channel order of caffe)
    '''
    if K.image_dim_ordering() == 'tf':
        inp = Input(shape=(224, 224, 3))
        bn_axis = 3
    else:
        inp = Input(shape=(3, 224, 224))
        bn_axis = 1

    dim_ordering = K.image_dim_ordering()
    out = ZeroPadding2D((3, 3), dim_ordering=dim_ordering)(inp)
    out = Convolution2D(64, 7, 7, subsample=(2, 2), dim_ordering=dim_ordering, name='conv1')(out)
    out = BatchNormalization(axis=bn_axis, name='bn_conv1')(out)
    out = Activation('relu')(out)
    out = MaxPooling2D((3, 3), strides=(2, 2), dim_ordering=dim_ordering)(out)

    out = conv_block(out, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    out = identity_block(out, 3, [64, 64, 256], stage=2, block='b')
    out = identity_block(out, 3, [64, 64, 256], stage=2, block='c')

    out = conv_block(out, 3, [128, 128, 512], stage=3, block='a')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='b')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='c')
    out = identity_block(out, 3, [128, 128, 512], stage=3, block='d')

    out = conv_block(out, 3, [256, 256, 1024], stage=4, block='a')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='b')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='c')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='d')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='e')
    out = identity_block(out, 3, [256, 256, 1024], stage=4, block='f')

    out = conv_block(out, 3, [512, 512, 2048], stage=5, block='a')
    out = identity_block(out, 3, [512, 512, 2048], stage=5, block='b')
    out = identity_block(out, 3, [512, 512, 2048], stage=5, block='c')

    out = AveragePooling2D((7, 7), dim_ordering=dim_ordering)(out)
    out = Flatten()(out)
    out = Dense(1000, activation='softmax', name='fc1000')(out)

    model = Model(inp, out)

    return model


if __name__ == '__main__':
    weights_file = K.image_dim_ordering() + '_dim_ordering_resnet50.h5'
    resnet_model = get_resnet50()
    resnet_model.load_weights(weights_file)

    # you may download synset_words from the address given at the begining of this file
    class_table = open('synset_words.txt', 'r')
    lines = class_table.readlines()

    test_img1 = read_img('cat.jpg')
    print('Result for test 1 is:')
    print(lines[np.argmax(resnet_model.predict(test_img1)[0])])

    test_img2 = read_img('elephant.jpg')
    print('Result for test 2 is:')
    print(lines[np.argmax(resnet_model.predict(test_img2)[0])])
    class_table.close()
