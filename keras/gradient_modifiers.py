from __future__ import absolute_import
import six
import copy
import numpy as np

from . import backend as K
from .utils.generic_utils import serialize_keras_object
from .utils.generic_utils import deserialize_keras_object


if K.backend() == 'tensorflow':
    import tensorflow as tf


def _clip_norm(g, c, n):
    if c <= 0:  # if clipnorm == 0 no need to add ops to the graph
        return g

    # tf require using a special op to multiply IndexedSliced by scalar
    if K.backend() == 'tensorflow':
        condition = n >= c
        then_expression = tf.scalar_mul(c / n, g)
        else_expression = g

        # saving the shape to avoid converting sparse tensor to dense
        if isinstance(then_expression, tf.Tensor):
            g_shape = copy.copy(then_expression.get_shape())
        elif isinstance(then_expression, tf.IndexedSlices):
            g_shape = copy.copy(then_expression.dense_shape)
        if condition.dtype != tf.bool:
            condition = tf.cast(condition, 'bool')
        g = tf.cond(condition,
                    lambda: then_expression,
                    lambda: else_expression)
        if isinstance(then_expression, tf.Tensor):
            g.set_shape(g_shape)
        elif isinstance(then_expression, tf.IndexedSlices):
            g._dense_shape = g_shape
    else:
        g = K.switch(K.greater_equal(n, c), g * c / n, g)
    return g


def _norm(gradients, ord):
    if np.isinf(ord):
        return K.max(K.stack([K.max(g) for g in gradients]))
    elif ord == 1:
        return sum([K.sum(K.abs(g) for g in gradients)])
    elif ord == 2:
        return K.sqrt(sum([K.sum(K.square(g)) for g in gradients]))
    elif ord >= 3:
        return K.pow(sum([K.sum(K.pow(K.abs(g), ord)) for g in gradients]), 1. / ord)
    else:
        raise ValueError("ord '{}' is not supported".format(ord))


class GradientModifier(object):

    def __call__(self, gradients):
        return gradients

    def get_config(self):
        return {}


class ClipNorm(GradientModifier):

    def __init__(self, norm_value=1., p=2):
        self.norm_value = norm_value
        self.p = p

    def __call__(self, gradients):
        norm = _norm(gradients, self.p)
        return [_clip_norm(g, self.norm_value, norm) for g in gradients]

    def get_config(self):
        return {'norm_value': self.norm_value,
                'p': self.p}


class ClipValue(GradientModifier):

    def __init__(self, clip_value=1.):
        self.clip_value = clip_value

    def __call__(self, gradients):
        return [K.clip(g, -self.clip_value, self.clip_value) for g in gradients]

    def get_config(self):
        return {'clip_value': self.clip_value}


class Normalize(GradientModifier):

    def __init__(self, p=2):
        self.p = p

    def __call__(self, gradients):
        norm = K.maximum(_norm(gradients, self.p), K.epsilon())
        return [g / norm for g in gradients]

    def get_config(self):
        return {'p': self.p}


class CompositeModifier(GradientModifier):

    def __init__(self, modifiers):
        self.modifiers = []
        for modifier in modifiers:
            self.modifiers.append(get(modifier))

    def __call__(self, gradients):
        for modifier in self.modifiers:
            gradients = modifier(gradients)
        return gradients

    def get_config(self):
        return {'modifiers': [serialize(modifier) for modifier in self.modifiers]}


def serialize(gradient_modifier):
    return serialize_keras_object(gradient_modifier)


def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='gradient_modifier')


# Aliases.
l2_normalize = Normalize
clip_value = ClipValue
clip_norm = ClipNorm


def get(identifier):
    if identifier is None:
        return None
    if isinstance(identifier, dict):
        return deserialize(identifier)
    elif isinstance(identifier, six.string_types):
        config = {'class_name': str(identifier), 'config': {}}
        return deserialize(config)
    elif callable(identifier):
        return identifier
    else:
        raise ValueError('Could not interpret `gradient_modifier` identifier:', identifier)
