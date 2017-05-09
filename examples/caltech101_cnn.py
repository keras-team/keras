from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.datasets import caltech101
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import LRN2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils
from six.moves import range

import shutil
import os
from PIL import Image
from resizeimage import resizeimage

'''
    Train a (fairly simple) deep CNN on the Caltech101 images dataset.
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python caltech101_cnn.py
'''


def resize_imgs(fpaths, shapex, shapey, mode='contain', quality=90, verbose=0):
    resized_fpaths = np.array([])

    tmpdir = os.path.expanduser(os.path.join('~', '.keras', 'datasets', 'tmp'))
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)

    os.makedirs(tmpdir)

    try:
        for i, f in enumerate(fpaths):
            img = Image.open(f)
            if mode is 'contain':
                img = resizeimage.resize_contain(img, [shapex, shapey])
            elif mode is 'crop':
                img = resizeimage.resize_crop(img, [shapex, shapey])
            elif mode is 'cover':
                img = resizeimage.resize_crop(img, [shapex, shapey])
            elif mode is 'thumbnail':
                img = resizeimage.resize_thumbnail(img, [shapex, shapey])
            elif mode is 'height':
                img = resizeimage.resize_height(img, shapey)

            _, extension = os.path.splitext(f)
            out_file = os.path.join(tmpdir, 'resized_img_%05d%s' % (i, extension))
            resized_fpaths = np.append(resized_fpaths, out_file)
            img.save(out_file, img.format, quality=quality)
            if verbose > 0:
                print("Resizing file : %s" % (f))
            img.close()
    except Exception, e:
        print("Error resize file : %s" % (f))

    return resized_fpaths


def load_data(X_path, resize=True, shapex=240, shapey=180, mode='contain', quality=90, verbose=0):
    if resize:
        X_path = resize_imgs(X_path, shapex, shapey, mode=mode, quality=quality, verbose=verbose)

    data = np.zeros((X_path.shape[0], 3, shapey, shapex), dtype="uint8")

    for i, f in enumerate(X_path):
        img = Image.open(f)
        r, g, b = img.split()
        data[i, 0, :, :] = np.array(r)
        data[i, 1, :, :] = np.array(g)
        data[i, 2, :, :] = np.array(b)
        img.close()

    return data

# parameters
batch_size = 4
nb_classes = 102
nb_epoch = 10
data_augmentation = False

shuffle_data = True

# shape of the image (SHAPE x SHAPE)
shapex, shapey = 240, 180

# the caltech101 images are RGB
image_dimensions = 3

# load the data, shuffled and split between train and test sets
print("Loading data...")
(X_train_path, y_train), (X_test_path, y_test) = caltech101.load_paths(train_imgs_per_category=15,
                                                                       test_imgs_per_category=3,
                                                                       shuffle=shuffle_data)
X_train = load_data(X_train_path, shapex=shapex, shapey=shapey, mode='contain', verbose=1)
X_test = load_data(X_test_path, shapex=shapex, shapey=shapey, mode='contain', verbose=1)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# cnn architecture from the CNN-S of http://arxiv.org/abs/1405.3531
model = Sequential()

model.add(Convolution2D(96, 7, 7, subsample=(2, 2), input_shape=(image_dimensions, shapex, shapey)))
model.add(Activation('relu'))
model.add(LRN2D(alpha=0.0005, beta=0.75, k=2, n=5))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))

model.add(Convolution2D(256, 5, 5))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, 3, 3))
model.add(Activation('relu'))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, 3, 3))
model.add(Activation('relu'))

model.add(ZeroPadding2D(padding=(1, 1)))
model.add(Convolution2D(512, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), stride=(2, 2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

print('Compiling model...')
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255

if not data_augmentation:
    print("Not using data augmentation or normalization")
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)

else:
    print("Using real time data augmentation")

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    for e in range(nb_epoch):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in datagen.flow(X_train, Y_train):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("train loss", loss)])

        print("Testing...")
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in datagen.flow(X_test, Y_test):
            score = model.test_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("test loss", score)])
