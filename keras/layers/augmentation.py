# -*- coding: utf-8 -*-
from __future__ import absolute_import

from ..engine import Layer
from .. import backend as K


class Augment2D(Layer):
    """Apply additive augmentation on 2D data.

    This is useful to expand possiblity of input data and avoid overfitting
    As it is a regularization layer, it can be used not only as the first layer but also among neural layers.

    Augmentation makes it possible to augment data using GPU or native APIs from backend,
    which is flexible and much faster than using ImageDataGenerator

    # Arguments
        rotate: float, maximum rotation angle (-rotate, rotate) in degrees, 0 <= rotate < 180
        horizontal_flip: randomly flipping inputs horizontally
        vertical_flip: randomly flipping inputs vertically

    # Input shape
        must be of 2D shape. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        to specify the 2D shape of input data.

    # Output shape
        Same shape as input.

    # Example
      >> from keras.layers.augmentation import Augment2D
      >> model = Sequential()
      >> model.add(Augment2D(rotate=0.1, input_shape=(28, 28, 1)))

    """

    def __init__(self, rotate=0,
                 horizontal_flip=False,
                 vertical_flip=False,
                 **kwargs):
        super(Augment2D, self).__init__(**kwargs)
        self.supports_masking = True
        self.rotate = rotate
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def call(self, inputs, training=None):
        return K.augment2d(inputs,
                           rotate=self.rotate,
                           horizontal_flip=self.horizontal_flip,
                           vertical_flip=self.vertical_flip)

    def get_config(self):
        config = {'rotate': self.rotate,
                  'horizontal_flip': self.horizontal_flip,
                  'vertical_flip': self.vertical_flip}
        base_config = super(Augment2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
