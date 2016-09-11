'''Trains a simple resnet on the CIFAR 10 dataset using the StepLearningScheduler.
'''
from __future__ import print_function
import os
import sys
import math
import time
import string
import random
from sys import stdout
from time import sleep
from keras.models import Model
from keras.layers import Input, Activation, merge, Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2, activity_l2
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as B, metrics
from keras.datasets import mnist, cifar10, cifar100
from keras.utils import np_utils
import scipy as sc
from PIL import Image
from keras.preprocessing import image
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils.visualize_util import plot
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
from keras.callbacks import StepLearningRateScheduler
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

np.random.seed(1337)  # for reproducibility

# for the larger networks (n>=9), we need to adjust pythons recursion limit
sys.setrecursionlimit(100000)

batch_size = 128 # was 128
nb_classes = 10
max_epoch = 200  # 182 # This is equivalent to 64K caffe iterations with 45K/5K train/val split

# early stopping patience
patience = 200

# input image dimensions
img_rows, img_cols = 32, 32
# size of pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
# image channel
img_channels = 3
# block type
#block_fn = 'bottleneck'
block_fn = 'basic block'

# number of repeated cells for blocks
n = 2 # Try total layers is 6*n + 2
# learning rate and regularization related
lr = 0.1 # was 0.1
reg_fac = 0.0005

# weights initialization method
w_init = "he_normal"  # "he_normal" "he_uniform"

# width of the feature maps for wide resnets
K = 1 # 4 for n=3

# Number of residual blocks
NR = 3

filters = [16*K, 32*K, 64*K, 128*K]
pool = [32, 16, 8, 4]
first_layer = [True, False, False, False]

# Data Augmentation stuff
data_augmentation = True
M = 1 # multiplier for data augmentation

epochs = [60, 120, 160, 200]

# this will do preprocessing and realtime data augmentation
datagen = ImageDataGenerator(
    # rescale=1./255
    # featurewise_center=True,  # set input mean to 0 over the dataset
    # samplewise_center=False,  # set each sample mean to 0
    # featurewise_std_normalization=True,  # divide inputs by std of the dataset
    # samplewise_std_normalization=False,  # divide each input by its std
    # zca_whitening=False,  # apply ZCA whitening
    # rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    # zoom_range=0.1,
    # shear_range=0.1,
    width_shift_range=0.25,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.25,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images

def error(y_true, y_pred):
    return (1.0 - metrics.categorical_accuracy(y_true, y_pred))*100

def print_summary(layers, line_length):
    total_params = 0
    for i in range(len(layers)):
        total_params += layers[i].count_params()

    print('=' * line_length)
    print('Total params: %0.2f M' % (total_params/1e6))
    print('Total params: %d ' % (total_params))
    print('Total number of layers %d' % (n*6 + 2))
    print('=' * line_length)

def summary(model, line_length=100):
    if hasattr(model, 'flattened_layers'):
        flattened_layers = model.flattened_layers
    else:
        flattened_layers = model.layers
    print_summary(flattened_layers, line_length)

def get_cifar_data(dataset='cifar10'):
    # if dataset == 'cifar10':
    print('Loading CIFAR10')
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # else:
    #    print('Loading CIFAR100')
    #    (X_train, y_train), (X_test, y_test) = cifar100.load_data()

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    # Center by subtracting mean image
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    #print('X_mean shape: ', X_mean.shape)
    X_train -= X_mean
    X_test -= X_mean
    X_train /= X_std
    X_test /= X_std
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)
    return X_train, X_test, Y_train, Y_test

# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        x = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col,
                          subsample=subsample, W_regularizer=l2(reg_fac),
                          b_regularizer=l2(reg_fac),
                          # activity_regularizer=activity_l2(reg_fac),
                          init=w_init, border_mode="same")(input)
        x = BatchNormalization(mode=0, axis=1)(x)
        return Activation("relu")(x)
    return f

# Helper to build a conv -> BN -> relu block
def _conv_bn(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        x = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col,
                          subsample=subsample, W_regularizer=l2(reg_fac),
                          b_regularizer=l2(reg_fac),
                          # activity_regularizer=activity_l2(reg_fac),
                          init=w_init, border_mode="same")(input)
        x = BatchNormalization(mode=0, axis=1)(x)
        return x
    return f

# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation("relu")(norm)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col,
                             subsample=subsample, W_regularizer=l2(reg_fac),
                             b_regularizer=l2(reg_fac),
                             # activity_regularizer=activity_l2(reg_fac),
                             init=w_init, border_mode="same")(activation)
    return f

# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    stride_width = input._keras_shape[2] / residual._keras_shape[2]
    stride_height = input._keras_shape[3] / residual._keras_shape[3]
    equal_channels = residual._keras_shape[1] == input._keras_shape[1]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                                 W_regularizer=l2(reg_fac),
                                 b_regularizer=l2(reg_fac),
                                 # activity_regularizer=activity_l2(reg_fac),
                                 subsample=(stride_width, stride_height),
                                 init=w_init, border_mode="valid")(input)
    return merge([shortcut, residual], mode="sum")

def _first_shortcut(input, residual):
    shortcut = Convolution2D(nb_filter=residual._keras_shape[1], nb_row=1, nb_col=1,
                             W_regularizer=l2(reg_fac),
                             b_regularizer=l2(reg_fac),
                             # activity_regularizer=activity_l2(reg_fac),
                             subsample=(1, 1),
                             init=w_init, border_mode="same")(input)
    return merge([shortcut, residual], mode="sum")

def _basic_first_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _conv_bn_relu(nb_filters, 3, 3, subsample=init_subsample)(input)
        #conv1 = Dropout(0.3)(conv1)
        residual = Convolution2D(nb_filter=nb_filters, nb_row=3, nb_col=3,
                                 subsample=init_subsample, W_regularizer=l2(reg_fac),
                                 b_regularizer=l2(reg_fac),
                                 # activity_regularizer=activity_l2(reg_fac),
                                 init=w_init, border_mode="same")(conv1)
        return _first_shortcut(input, residual)
    return f

# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def _basic_block(nb_filters, init_subsample=(1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, subsample=init_subsample)(input)
        #conv1 = Dropout(0.3)(conv1)
        residual = _bn_relu_conv(nb_filters, 3, 3)(conv1)
        return _shortcut(input, residual)
    return f

# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def _bottleneck(nb_filters, init_subsample=(1, 1)):
    def f(input):
        residual = _bottleneck_plain(nb_filters, init_subsample=init_subsample)(input)
        return _shortcut(input, residual)

    return f

# Builds a plain block with repeating dual convolutional blocks without any shortcut connection.
def _block(block_function, nb_filters, n, is_first_layer=False):
    def f(input):
        for i in range(n):
            init_subsample = (1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
            return input
    return f

# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_fun, nb_filters, n, is_first_layer=False):
    block_function = _basic_block
    if block_fun == 'bottleneck':
        block_function = _bottleneck
    return _block(block_function, nb_filters, n, is_first_layer)

def _output_block_1fc(nb_classes):
    def f(input):
        x = Dense(nb_classes, init=w_init, W_regularizer=l2(reg_fac),
                  b_regularizer=l2(reg_fac),
                  # activity_regularizer=activity_l2(reg_fac)
                  )(input)
        return Activation('softmax')(x)
    return f

def compile_model(model, loss_func='categorical_crossentropy'):
    opt = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    #opt = Adam(lr=lr)
    model.compile(loss=loss_func,
                  optimizer=opt,
                  metrics=['accuracy', error])
    plot(model, to_file='model.png', show_shapes=True)
    summary(model)
    return model

def get_conv_branch(resnet=True, mao=False, pool_sz=8):
    inputs = Input(shape=(img_channels, img_rows, img_cols))
    x = _conv_bn_relu(filters[0], 3, 3)(inputs)
    x = _basic_first_block(filters[0])(x)
    block_function = _residual_block
    for i in range(NR):
        if i == 0:
            x = block_function(block_fn, nb_filters=filters[i], n=n-1,
                               is_first_layer=first_layer[i])(x)
        else:
            x = block_function(block_fn, nb_filters=filters[i], n=n,
                               is_first_layer=first_layer[i])(x)
    x = BatchNormalization(mode=0, axis=1)(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling2D()(x)
    return inputs, x

def get_resnet_cifar10_model():
    inputs, x = get_conv_branch()
    pred = _output_block_1fc(nb_classes=nb_classes)(x)
    model = Model(input=inputs, output=pred)
    return compile_model(model)

X_train, X_test, y_train, y_test = get_cifar_data('cifar10')
model = get_resnet_cifar10_model()

print(X_train.shape)
print(y_train.shape)
model_name = 'resnet'
model_file = 'data/' + model_name + '.hdf5'

model_checkpoint = ModelCheckpoint(model_file, monitor='val_loss', save_best_only=True)
stop_early = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')
lr_scheduler = StepLearningRateScheduler(monitor='val_loss', epochs=10, verbose=2)
#lr_scheduler = StepLearningRateScheduler(epochs=epochs, verbose=2)

if not data_augmentation:
    print('Not using data augmentation.')
    hist = model.fit(X_train, y_train,
                     batch_size=batch_size,
                     nb_epoch=max_epoch,
                     verbose=1,
                     callbacks=[stop_early, model_checkpoint, lr_scheduler],
                     #validation_data=(X_test, y_test),
                     validation_split=0.1)
else:
    print('Using real-time data augmentation.')
    # fit the model on the batches generated by datagen.flow()
    hist = model.fit_generator(datagen.flow(X_train, y_train,
                                            # save_to_dir='data/aug_data'
                                            batch_size=batch_size),
                               samples_per_epoch=X_train.shape[0]*M,
                               nb_epoch=max_epoch,
                               callbacks=[stop_early, model_checkpoint, lr_scheduler],
                               validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test scores and losses:', score)
print(hist.history)
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']

# save the lossses here by generating a pandas data frame from them and to csv
losses = pd.DataFrame({'train-loss': train_loss, 'val-loss': val_loss})

losses.to_csv('data/losses_' + model_name + '.csv')
