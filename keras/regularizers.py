from __future__ import absolute_import
import six
from . import backend as K
from .utils.generic_utils import serialize_keras_object
from .utils.generic_utils import deserialize_keras_object


class Regularizer(object):
    """Regularizer base class.
    """

    def __call__(self, x):
        return 0.

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class BiRegularizer(object):
    """BiRegularizer base class.
    """

    def __call__(self, x,w):
        return 0.

    @classmethod
    def from_config(cls, config):
            return cls(**config)


class L1L2(Regularizer):
    """ Regularizer for L1 and L2 regularization.
    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
    """

    def __init__(self, l1=0., l2=0.):
        self.l1 = K.cast_to_floatx(l1)

        self.l2 = K.cast_to_floatx(l2)


    def __call__(self, x):
        regularization = 0.
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(x))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(x))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2)}

class HybridL1L2(BiRegularizer):
    """BiRegularizer for L1 and L2 regularization.

    # Arguments
        l1: Float; L1 regularization factor.
        l2: Float; L2 regularization factor.
        sx : minimum of the input before comparing with the weights
        sw : minimum of the weights before comparing with the input
        sl : power of inputs when compared to weigths
        level = 0 , we keep the regularizer as a log,
        level != 0 , we exponentiate the log , so more agressivity for the extrem values
    """

    def __init__(self, l1=0., l2=0. , sx=0.000001, sw=0.000001,sl=1.0):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.sx = K.cast_to_floatx(sx)
        self.sw = K.cast_to_floatx(sw)
        self.sl = K.cast_to_floatx(sl)



    def __call__(self, x, weights):
        regularization = 0.
        kernel = weights[0]
        OutputDim = K.int_shape(kernel)[-1]
        x_effectif = K.maximum(K.abs(x), self.sx)
        weights_effectif = K.maximum(K.abs(kernel), self.sw)

        if self.l1:
            regularization += K.sum(self.l1 * K.dot(K.exp(-1.0 *  self.sl * K.log(x_effectif)), weights_effectif))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(K.dot(K.exp(-1.0 *  self.sl * K.log(x_effectif)), weights_effectif)))


        return regularization

    def get_config(self):
        return {'l1': float(self.l1),
                'l2': float(self.l2),
                'sx': float(self.sx),
                'sw': float(self.sw),
                'sl': float(self.sl)}


# Aliases.


def l1(l=0.01):
    return L1L2(l1=l)


def l2(l=0.01):
    return L1L2(l2=l)

def ll1(l,l1=0.01):
    return HybridL1L2(l1=l1,sl=l)

def ll2(l,l2=0.01):
    return HybridL1L2(l2=l2,sl=l)


def l1_l2(l1=0.01, l2=0.01):
    return L1L2(l1=l1, l2=l2)


def serialize(regularizer):
    return serialize_keras_object(regularizer)



def deserialize(config, custom_objects=None):
    return deserialize_keras_object(config,
                                    module_objects=globals(),
                                    custom_objects=custom_objects,
                                    printable_module_name='regularizer')



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
        raise ValueError('Could not interpret regularizer identifier:',
                         identifier)
