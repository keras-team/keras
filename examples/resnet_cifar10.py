'''
Pai Peng (pengpai_sh@163.com)

An re-implementation of Residual Network (ResNet) on CIFAR-10 dataset in paper "Deep Residual Learning for
Image Recognition". The Reisdual Blocks refers to
 https://github.com/keunwoochoi/residual_block_keras/blob/master/residual_blocks.py
(Keunwoo Choi).
Note that his implementation has already used "pre-activation" proposed in the new version of ResNet
paper "Identity Mappings in Deep Residual Networks".

1. Set n = 3 to obtain a 20-layer ResNet, testing accuracy is 89.4% (loss: 0.31668),
   training with 36 hours (75 epochs).
2. Set n = 5 to obtain a 32-layer ResNet, testing accuracy is 90.8% (loss: 0.28545),
   training with 58 hours (75 epochs).
3. Set n = 8 to obtain a 50-layer ResNet, testing accuracy is 90.89% (loss: 0.28211),
   training with 94 hours (75 epochs).

'''

import sys
sys.setrecursionlimit(99999)
import numpy as np
np.random.seed(2016)  # for reproducibility
import keras
import keras.backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.optimizers import RMSprop, SGD
from sklearn.cross_validation import train_test_split
import tensorflow as tf


# Allocate full of GPU resource to me
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def building_residual_block(input_shape, n_feature_maps, kernel_sizes=None, n_skip=2, is_subsample=False, subsample=None):
    '''
    [1] Building block of layers for residual learning.
        Code based on https://github.com/ndronen/modeling/blob/master/modeling/residual.py
        , but modification of (perhaps) incorrect relu(f)+x thing and it's for conv layer
    [2] MaxPooling is used instead of strided convolution to make it easier
        to set size(output of short-cut) == size(output of conv-layers).
        If you want to remove MaxPooling,
           i) change (border_mode in Convolution2D in shortcut), 'same'-->'valid'
           ii) uncomment ZeroPadding2D in conv layers.
               (Then the following Conv2D is not the first layer of this container anymore,
                so you can remove the input_shape in the line 101, the line with comment #'OPTION' )
    [3] It can be used for both cases whether it subsamples or not.
    [4] In the short-cut connection, I used 1x1 convolution to increase #channel.
        It occurs when is_expand_channels == True
    input_shape = (None, num_channel, height, width)
    n_feature_maps: number of feature maps. In ResidualNet it increases whenever image is downsampled.
    kernel_sizes : list or tuple, (3,3) or [3,3] for example
    n_skip       : number of layers to skip
    is_subsample : If it is True, the layers subsamples by *subsample* to reduce the size.
    subsample    : tuple, (2,2) or (1,2) for example. Used only if is_subsample==True
    '''
    # ***** VERBOSE_PART *****
    print ('   - New residual block with')
    print ('      input shape:', input_shape)
    print ('      kernel size:', kernel_sizes)
    # is_expand_channels == True when num_channels increases.
    #    E.g. the very first residual block (e.g. 1->64, 3->128, 128->256, ...)
    is_expand_channels = not (input_shape[0] == n_feature_maps)
    if is_expand_channels:
        print ('      - Input channels: %d ---> num feature maps on out: %d' % (input_shape[0],
               n_feature_maps))
    if is_subsample:
        print ('      - with subsample:', subsample)
    kernel_row, kernel_col = kernel_sizes
    # set input
    x = Input(shape=(input_shape))
    # ***** SHORTCUT PATH *****
    if is_subsample: # subsample (+ channel expansion if needed)
        shortcut_y = Convolution2D(n_feature_maps, kernel_row, kernel_col,
                                   subsample=subsample, W_regularizer=l2(0.0001),
                                   border_mode='valid')(x)
    else: # channel expansion only (e.g. the very first layer of the whole networks)
        if is_expand_channels:
            shortcut_y = Convolution2D(n_feature_maps, 1, 1,
                                       W_regularizer=l2(0.0001), border_mode='same')(x)
        else:
            # if no subsample and no channel expension, there's nothing to add on the shortcut.
            shortcut_y = x
    # ***** CONVOLUTION_PATH *****
    conv_y = x
    for i in range(n_skip):
        conv_y = BatchNormalization(axis=1)(conv_y)
        conv_y = Activation('relu')(conv_y)
        if i==0 and is_subsample: # [Subsample at layer 0 if needed]
            conv_y = Convolution2D(n_feature_maps, kernel_row, kernel_col,subsample=subsample,
                                   W_regularizer=l2(0.0001),border_mode='valid')(conv_y)
        else:
            conv_y = Convolution2D(n_feature_maps, kernel_row, kernel_col,
                                   W_regularizer=l2(0.0001), border_mode='same')(conv_y)
    # output
    y = merge([shortcut_y, conv_y], mode='sum')
    block = Model(input=x, output=y)
    print ('        -- model was built.')
    return block


def residual_model(n):
    '''
        The 6n+2 network architecture for CIFAR-10 in [1].
    '''
    model = Sequential() # it's a CONTAINER, not MODEL

    # 1st layer is a 3x3 conv, output shape: (16, 32, 32)
    model.add(Convolution2D(16, 3, 3, subsample = (1, 1), border_mode='same', input_shape=(3, 32, 32)))

    # 1st 2n layer, 16 filters, output shape: (16, 32, 32)
    for i in range(2 * n):
        model.add(building_residual_block(input_shape = (16, 32, 32), n_feature_maps = 16,
                  kernel_sizes = (3, 3), n_skip = 2, is_subsample = False, subsample = None))

    # 2nd 2n layer, 32 filters, output shape: (32, 16, 16)
    for i in range(2 * n):
        # expand dimensions and half the outpu shape size
        if i == 0:
            model.add(building_residual_block(input_shape = (16, 32, 32), n_feature_maps = 32,
                      kernel_sizes = (3, 3), n_skip = 2, is_subsample = True, subsample = (2, 2)))
        else:
            model.add(building_residual_block(input_shape = (32, 16, 16), n_feature_maps = 32,
                      kernel_sizes = (3, 3), n_skip = 2, is_subsample = False, subsample = None))

    # 3rd 2n layer, 64 filters, output shape: (64, 8, 8)
    for i in range(2 * n):
        # expand dimensions and half the outpu shape size
        if i == 0:
            model.add(building_residual_block(input_shape = (32, 16, 16), n_feature_maps = 64,
                      kernel_sizes = (3, 3), n_skip = 2, is_subsample = True, subsample = (2, 2)))
        else:
            model.add(building_residual_block(input_shape = (64, 8, 8), n_feature_maps = 64,
                      kernel_sizes = (3, 3), n_skip = 2, is_subsample = False, subsample = None))

    # global averaging layer
    model.add(AveragePooling2D(pool_size = (7, 7)))

    model.add(BatchNormalization(axis=1))
    model.add(Activation('relu'))

    # Classifier
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model


if __name__ =='__main__':

    learning_rate = 0.1
    batch_size = 128
    nb_epoch = 25
    data_augment = True

    #  X_train, X_test: (nb_samples, 3, 32, 32)
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # perpixel mean substracted
    X_train = (X_train - np.mean(X_train))/np.std(X_train)
    X_test = (X_test - np.mean(X_test))/np.std(X_test)

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
    #                             test_size =0.1, random_state = 2016)

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    # Y_val = np_utils.to_categorical(y_val, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    # set n = 3 means get a model with 3*6+2=20 layers, set n = 9 means 9*6+2=56 layers
    model = residual_model(n = 3)

    # optimizer = RMSprop(lr = learning_rate)
    optimizer = SGD(lr = learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
    model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

    # autosave best Model
    best_model_file = "./residual_cifar10_weights.h5"
    best_model = ModelCheckpoint(best_model_file, verbose = 1, save_best_only = True)

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center = False,  # set input mean to 0 over the dataset
        samplewise_center = False,  # set each sample mean to 0
        featurewise_std_normalization = False,  # divide inputs by std of the dataset
        samplewise_std_normalization = False,  # divide each input by its std
        zca_whitening = False,  # apply ZCA whitening
        rotation_range = 10,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range = 0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range = 0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip = False)  # randomly flip images

    datagen.fit(X_train)

    for i in range(3):
        if i != 0:
            # devide the learning rate by 10 for two times
            lr_old = K.get_value(optimizer.lr)
            K.set_value(optimizer.lr, 0.1 * lr_old)
            print('Changing learning rate from %f to %f' % (lr_old, K.get_value(optimizer.lr)))

        if data_augment:
            # fit the model on the batches generated by datagen.flow()
            model.fit_generator(datagen.flow(X_train, Y_train,
                                batch_size = batch_size),
                                samples_per_epoch = X_train.shape[0],
                                nb_epoch = nb_epoch,
                                validation_data = (X_test, Y_test),
                                callbacks = [best_model])
        else:
            model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = nb_epoch,
                      verbose = 1, validation_data = (X_test, Y_test), callbacks = [best_model])

    print('loading best model...')
    model.load_weights(best_model_file)
    score = model.evaluate(X_test, Y_test, batch_size = batch_size, verbose = 1)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
