"""Utilities for real-time data augmentation on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import backend
from .. import utils
from keras_preprocessing import image

random_rotation = image.random_rotation
random_shift = image.random_shift
random_shear = image.random_shear
random_zoom = image.random_zoom
apply_channel_shift = image.apply_channel_shift
random_channel_shift = image.random_channel_shift
apply_brightness_shift = image.apply_brightness_shift
random_brightness = image.random_brightness
apply_affine_transform = image.apply_affine_transform
load_img = image.load_img


def array_to_img(x, data_format=None, scale=True, dtype=None):
    if data_format is None:
        data_format = backend.image_data_format()
    if dtype is None:
        dtype = backend.floatx()
    return image.array_to_img(x,
                              data_format=data_format,
                              scale=scale,
                              dtype=dtype)


def img_to_array(img, data_format=None, dtype=None):
    if data_format is None:
        data_format = backend.image_data_format()
    if dtype is None:
        dtype = backend.floatx()
    return image.img_to_array(img, data_format=data_format, dtype=dtype)


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


class Iterator(image.Iterator, utils.Sequence):
    pass


class DirectoryIterator(image.DirectoryIterator, Iterator):

    def __init__(self, directory, image_data_generator,
                 target_size=(256, 256),
                 color_mode='rgb',
                 classes=None,
                 class_mode='categorical',
                 batch_size=32,
                 shuffle=True,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 follow_links=False,
                 subset=None,
                 interpolation='nearest',
                 dtype=None):
        if data_format is None:
            data_format = backend.image_data_format()
        if dtype is None:
            dtype = backend.floatx()
        super(DirectoryIterator, self).__init__(
            directory, image_data_generator,
            target_size=target_size,
            color_mode=color_mode,
            classes=classes,
            class_mode=class_mode,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation,
            dtype=dtype)


class NumpyArrayIterator(image.NumpyArrayIterator, Iterator):

    def __init__(self, x, y, image_data_generator,
                 batch_size=32,
                 shuffle=False,
                 sample_weight=None,
                 seed=None,
                 data_format=None,
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 subset=None,
                 dtype=None):
        if data_format is None:
            data_format = backend.image_data_format()
        if dtype is None:
            dtype = backend.floatx()
        super(NumpyArrayIterator, self).__init__(
            x, y, image_data_generator,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            data_format=data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset,
            dtype=dtype)


class ImageDataGenerator(image.ImageDataGenerator):

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 validation_split=0.0,
                 dtype=None):
        if data_format is None:
            data_format = backend.image_data_format()
        if dtype is None:
            dtype = backend.floatx()
        super(ImageDataGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            validation_split=validation_split,
            dtype=dtype)


array_to_img.__doc__ = image.array_to_img.__doc__
img_to_array.__doc__ = image.img_to_array.__doc__
save_img.__doc__ = image.save_img.__doc__

# Iterator.__doc__ = image.Iterator.__doc__
# DirectoryIterator.__doc__ = image.DirectoryIterator
# NumpyArrayIterator.__doc__ = image.NumpyArrayIterator
# ImageDataGenerator.__doc__ = image.ImageDataGenerator
