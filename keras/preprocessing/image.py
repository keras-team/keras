"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import backend
from ..utils import generic_utils

from keras_preprocessing import image
from keras_preprocessing.image import random_rotation
from keras_preprocessing.image import random_shift
from keras_preprocessing.image import random_shear
from keras_preprocessing.image import random_zoom
from keras_preprocessing.image import apply_channel_shift
from keras_preprocessing.image import random_channel_shift
from keras_preprocessing.image import apply_brightness_shift
from keras_preprocessing.image import random_brightness
from keras_preprocessing.image import apply_affine_transform
from keras_preprocessing.image import load_img
from keras_preprocessing.image import ImageDataGenerator


def array_to_img(x, data_format=None, scale=True, dtype=None):
    if data_format is None:
        data_format = backend.image_data_format()
    if 'dtype' in generic_utils.getargspec(image.array_to_img).args:
        if dtype is None:
            dtype = backend.floatx()
        return image.array_to_img(x,
                                  data_format=data_format,
                                  scale=scale,
                                  dtype=dtype)
    return image.array_to_img(x,
                              data_format=data_format,
                              scale=scale)


def img_to_array(img, data_format=None, dtype=None):
    if data_format is None:
        data_format = backend.image_data_format()
    if 'dtype' in generic_utils.getargspec(image.img_to_array).args:
        if dtype is None:
            dtype = backend.floatx()
        return image.img_to_array(img, data_format=data_format, dtype=dtype)
    return image.img_to_array(img, data_format=data_format)


def save_img(path,
             x,
             data_format=None,
             file_format=None,
             scale=True, **kwargs):
    if data_format is None:
        data_format = backend.image_data_format()
    return image.save_img(path,
                          x,
                          data_format=data_format,
                          file_format=file_format,
                          scale=scale, **kwargs)


array_to_img.__doc__ = image.array_to_img.__doc__
img_to_array.__doc__ = image.img_to_array.__doc__
save_img.__doc__ = image.save_img.__doc__
