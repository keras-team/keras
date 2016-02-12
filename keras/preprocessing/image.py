from __future__ import absolute_import

import numpy as np
import re
from scipy import ndimage
from scipy import linalg

from os import listdir
from os.path import isfile, join
import random
import math
from six.moves import range
import threading

'''Fairly basic set of tools for realtime data augmentation on image data.
Can easily be extended to include new transformations, new preprocessing methods, etc...
'''


def random_rotation(x, rg, fill_mode="nearest", cval=0.):
    angle = random.uniform(-rg, rg)
    x = ndimage.interpolation.rotate(x, angle,
                                     axes=(1, 2),
                                     reshape=False,
                                     mode=fill_mode,
                                     cval=cval)
    return x


def random_shift(x, wrg, hrg, fill_mode="nearest", cval=0.):
    shift_x = shift_y = 0
    
    if wrg:
        shift_x = random.uniform(-wrg, wrg) * x.shape[2]
    if hrg:
        shift_y = random.uniform(-hrg, hrg) * x.shape[1]
        
    x = ndimage.interpolation.shift(x, (0, shift_y, shift_x),
                                    order=0,
                                    mode=fill_mode,
                                    cval=cval)
    return x


def horizontal_flip(x):
    for i in range(x.shape[0]):
        x[i] = np.fliplr(x[i])
    return x


def vertical_flip(x):
    for i in range(x.shape[0]):
        x[i] = np.flipud(x[i])
    return x


def random_barrel_transform(x, intensity):
    # TODO
    pass


def random_shear(x, intensity, fill_mode="nearest", cval=0.):
    shear = random.uniform(-intensity, intensity)
    shear_matrix = np.array([[1.0, -math.sin(shear), 0.0],
                            [0.0, math.cos(shear), 0.0],
                            [0.0, 0.0, 1.0]])
    x = ndimage.interpolation.affine_transform(x, shear_matrix,
                                               mode=fill_mode,
                                               order=3,
                                               cval=cval)
    return x


def random_channel_shift(x, rg):
    # TODO
    pass


def random_zoom(x, rg, fill_mode="nearest", cval=0.):
    zoom_w = random.uniform(1.-rg, 1.)
    zoom_h = random.uniform(1.-rg, 1.)
    x = ndimage.interpolation.zoom(x, zoom=(1., zoom_w, zoom_h),
                                   mode=fill_mode,
                                   cval=cval)
    return x  # shape of result will be different from shape of input!


def array_to_img(x, scale=True):
    from PIL import Image
    x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype("uint8"), "RGB")
    else:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype("uint8"), "L")


def img_to_array(img):
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        # RGB: height, width, channel -> channel, height, width
        x = x.transpose(2, 0, 1)
    else:
        # grayscale: height, width -> channel, height, width
        x = x.reshape((1, x.shape[0], x.shape[1]))
    return x


def load_img(path, grayscale=False):
    from PIL import Image
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    return img


def list_pictures(directory, ext='jpg|jpeg|bmp|png'):
    return [join(directory, f) for f in listdir(directory)
            if isfile(join(directory, f)) and re.match('([\w]+\.(?:' + ext + '))', f)]


class ImageDataGenerator(object):
    '''Generate minibatches with
    realtime data augmentation.
    '''
    def __init__(self,
                 featurewise_center=True,  # set input mean to 0 over the dataset
                 samplewise_center=False,  # set each sample mean to 0
                 featurewise_std_normalization=True,  # divide inputs by std of the dataset
                 samplewise_std_normalization=False,  # divide each input by its std
                 zca_whitening=False,  # apply ZCA whitening
                 rotation_range=0.,  # degrees (0 to 180)
                 width_shift_range=0.,  # fraction of total width
                 height_shift_range=0.,  # fraction of total height
                 shear_range=0.,  # shear intensity (shear angle in radians)
                 horizontal_flip=False,
                 vertical_flip=False):

        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.lock = threading.Lock()

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        b = 0
        total_b = 0
        while 1:
            if b == 0:
                if seed is not None:
                    np.random.seed(seed + total_b)

                if shuffle:
                    index_array = np.random.permutation(N)
                else:
                    index_array = np.arange(N)

            current_index = (b * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
            else:
                current_batch_size = N - current_index

            if current_batch_size == batch_size:
                b += 1
            else:
                b = 0
            total_b += 1
            yield index_array[current_index: current_index + current_batch_size], current_index, current_batch_size

    def flow(self, X, y, batch_size=32, shuffle=False, seed=None,
             save_to_dir=None, save_prefix="", save_format="jpeg"):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.flow_generator = self._flow_index(X.shape[0], batch_size, shuffle, seed)
        return self

    def __iter__(self):
        # needed if we want to do something like for x,y in data_gen.flow(...):
        return self

    def next(self):
        # for python 2.x
        # Keep under lock only the mechainsem which advance the indexing of each batch
        # see # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.flow_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        bX = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            x = self.random_transform(x.astype("float32"))
            x = self.standardize(x)
            bX[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(bX[i], scale=True)
                img.save(self.save_to_dir + "/" + self.save_prefix + "_" + str(current_index + i) + "." + self.save_format)
        bY = self.y[index_array]
        return bX, bY

    def __next__(self):
        # for python 3.x
        return self.next()

    def standardize(self, x):
        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= self.std

        if self.zca_whitening:
            flatx = np.reshape(x, (x.shape[0]*x.shape[1]*x.shape[2]))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        if self.samplewise_center:
            x -= np.mean(x)
        if self.samplewise_std_normalization:
            x /= np.std(x)

        return x

    def random_transform(self, x):
        if self.rotation_range:
            x = random_rotation(x, self.rotation_range)
        if self.width_shift_range or self.height_shift_range:
            x = random_shift(x, self.width_shift_range, self.height_shift_range)
        if self.horizontal_flip:
            if random.random() < 0.5:
                x = horizontal_flip(x)
        if self.vertical_flip:
            if random.random() < 0.5:
                x = vertical_flip(x)
        if self.shear_range:
            x = random_shear(x,self.shear_range)
        # TODO:
        # zoom
        # barrel/fisheye
        # shearing
        # channel shifting
        return x

    def fit(self, X,
            augment=False,  # fit on randomly augmented samples
            rounds=1,  # if augment, how many augmentation passes over the data do we use
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization and zca_whitening.
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds*X.shape[0]]+list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    img = array_to_img(X[i])
                    img = self.random_transform(img)
                    aX[i+r*X.shape[0]] = img_to_array(img)
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean
        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= self.std

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]*X.shape[3]))
            fudge = 10e-6
            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + fudge))), U.T)


class GraphImageDataGenerator(ImageDataGenerator):
    '''Example of how to build a generator for a Graph model
    '''

    def next(self):
        bX, bY = super(GraphImageDataGenerator, self).next()
        return {'input': bX, 'output': bY}
