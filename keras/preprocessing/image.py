'''Fairly basic set of tools for realtime data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
'''
from __future__ import absolute_import

import numpy as np
from scipy import ndimage
from scipy import linalg
from skimage import transform
from six.moves import range
import threading


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def random_barrel_transform(x, intensity):
    # TODO
    pass


def random_channel_shift(x, rg):
    # TODO
    pass


def array_to_img(x, scale=True):
    from PIL import Image
    x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    else:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')


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


class ImageDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.

    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom: amount of zoom. if scalar z, zoom will be randomly picked in
                the range [1-z, 1+z]. A sequence of two can be passed instead
                to select this range.          
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
    '''
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 dim_ordering='th'):
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.lock = threading.Lock()
        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and \
            column) or "th" (channel before row and column). Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == "th":
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == "tf":
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        if np.isscalar(zoom):
            self.zoom_range = [1 - zoom, 1 + zoom]
        else:
            self.zoom_range = [zoom[0], zoom[1]]

        self.batch_index = 0
        self.total_batches_seen = 0

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        while 1:
            if self.batch_index == 0:
                if seed is not None:
                    np.random.seed(seed + self.total_batches_seen)

                if shuffle:
                    index_array = np.random.permutation(N)
                else:
                    index_array = np.arange(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
            else:
                current_batch_size = N - current_index

            if current_batch_size == batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def flow(self, X, y, batch_size=32, shuffle=False, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        assert len(X) == len(y)
        self.X = X
        self.y = y
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.flow_generator = self._flow_index(X.shape[0], batch_size,
                                               shuffle, seed)
        return self

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.flow_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        bX = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            x = self.random_transform(x.astype('float32'))
            x = self.standardize(x)
            bX[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(bX[i], scale=True)
                img.save(self.save_to_dir + '/' + self.save_prefix + '_' + str(current_index + i) + '.' + self.save_format)
        bY = self.y[index_array]
        return bX, bY

    def __next__(self):
        # for python 3.x.
        return self.next()

    def standardize(self, x):
        if self.samplewise_center:
            x -= np.mean(x, axis=self.channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=self.channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        return x

    def random_transform(self, x):
        # x is a single image, so it doesn't have image number at index 0
        img_col_index = self.col_index - 1
        img_row_index = self.row_index - 1
        img_channel_index = self.channel_index - 1


        # find inverse permutation
        inverse_transpose = [0,0,0]
        inverse_transpose[img_row_index], inverse_transpose[img_col_index], inverse_transpose[img_channel_index] = 0,1,2
        orig_dtype = x.dtype

        # for compatability with skimage
        x = x.transpose((img_row_index, img_col_index, img_channel_index)).astype('float64')
        img_min, img_max = np.min(x), np.max(x)
        x = (x - img_min)/(img_max - img_min)


        if self.rotation_range:
            theta = np.pi/180*np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[0]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[1]
        else:
            ty = 0
        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range != [1., 1.]:
            zx = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
            zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        else:
            zx, zy = 1, 1
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        # recentering origin to the centre of image
        o_x = float(x.shape[0])/2 + 0.5
        o_y = float(x.shape[1])/2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, transform_matrix), reset_matrix)

        projective_transform = transform.ProjectiveTransform(transform_matrix)
        x = transform.warp(x, projective_transform, mode='nearest', order=0)

        #revert back to original dim ordering and scale
        x = x.transpose(inverse_transpose)
        x = (x*(img_max-img_min) + img_min).astype(orig_dtype, copy=False)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
        
        # TODO:
        # barrel/fisheye
        # channel shifting
        return x

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.

        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    img = array_to_img(X[i])
                    img = self.random_transform(img)
                    aX[i + r * X.shape[0]] = img_to_array(img)
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean
        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)


class GraphImageDataGenerator(ImageDataGenerator):
    '''Example of how to build a generator for a Graph model
    '''

    def next(self):
        bX, bY = super(GraphImageDataGenerator, self).next()
        return {'input': bX, 'output': bY}
