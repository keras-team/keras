from __future__ import absolute_import

from ..utils.generic_utils import deserialize_keras_object
from ..engine.base_layer import Layer
from ..engine import Input
from ..engine import InputLayer
from ..engine.base_layer import InputSpec

from .merge import Add
from .merge import Subtract
from .merge import Multiply
from .merge import Average
from .merge import Maximum
from .merge import Minimum
from .merge import Concatenate
from .merge import Dot
from .merge import add
from .merge import subtract
from .merge import multiply
from .merge import average
from .merge import maximum
from .merge import minimum
from .merge import concatenate
from .merge import dot

from .core import Dense
from .core import Activation
from .core import Dropout
from .core import Flatten
from .core import Reshape
from .core import Permute
from .core import RepeatVector
from .core import Lambda
from .core import ActivityRegularization
from .core import Masking
from .core import SpatialDropout1D
from .core import SpatialDropout2D
from .core import SpatialDropout3D

from .convolutional import Conv1D
from .convolutional import Conv2D
from .convolutional import SeparableConv1D
from .convolutional import SeparableConv2D
from .convolutional import DepthwiseConv2D
from .convolutional import Conv2DTranspose
from .convolutional import Conv3D
from .convolutional import Conv3DTranspose
from .convolutional import Cropping1D
from .convolutional import Cropping2D
from .convolutional import Cropping3D
from .convolutional import UpSampling1D
from .convolutional import UpSampling2D
from .convolutional import UpSampling3D
from .convolutional import ZeroPadding1D
from .convolutional import ZeroPadding2D
from .convolutional import ZeroPadding3D

# Aliases (not in the docs)
from .convolutional import Convolution1D
from .convolutional import Convolution2D
from .convolutional import Convolution3D
from .convolutional import Deconvolution2D
from .convolutional import Deconvolution3D

from .pooling import MaxPooling1D
from .pooling import MaxPooling2D
from .pooling import MaxPooling3D
from .pooling import AveragePooling1D
from .pooling import AveragePooling2D
from .pooling import AveragePooling3D
from .pooling import GlobalMaxPooling1D
from .pooling import GlobalMaxPooling2D
from .pooling import GlobalMaxPooling3D
from .pooling import GlobalAveragePooling2D
from .pooling import GlobalAveragePooling1D
from .pooling import GlobalAveragePooling3D

# Aliases (not in the docs)
from .pooling import MaxPool1D
from .pooling import MaxPool2D
from .pooling import MaxPool3D
from .pooling import AvgPool1D
from .pooling import AvgPool2D
from .pooling import AvgPool3D
from .pooling import GlobalMaxPool1D
from .pooling import GlobalMaxPool2D
from .pooling import GlobalMaxPool3D
from .pooling import GlobalAvgPool1D
from .pooling import GlobalAvgPool2D
from .pooling import GlobalAvgPool3D

from .local import LocallyConnected1D
from .local import LocallyConnected2D

from .recurrent import RNN
from .recurrent import SimpleRNN
from .recurrent import GRU
from .recurrent import LSTM
from .recurrent import SimpleRNNCell
from .recurrent import GRUCell
from .recurrent import LSTMCell
from .recurrent import StackedRNNCells

from .cudnn_recurrent import CuDNNGRU
from .cudnn_recurrent import CuDNNLSTM

from .normalization import BatchNormalization

from .embeddings import Embedding

from .noise import GaussianNoise
from .noise import GaussianDropout
from .noise import AlphaDropout

from .advanced_activations import LeakyReLU
from .advanced_activations import PReLU
from .advanced_activations import ELU
from .advanced_activations import ThresholdedReLU
from .advanced_activations import Softmax
from .advanced_activations import ReLU

from .wrappers import Bidirectional
from .wrappers import TimeDistributed

from .convolutional_recurrent import ConvLSTM2D
from .convolutional_recurrent import ConvLSTM2DCell

# Legacy imports
from ..legacy.layers import MaxoutDense
from ..legacy.layers import Highway
from ..legacy.layers import AtrousConvolution1D
from ..legacy.layers import AtrousConvolution2D
from ..legacy.layers import Recurrent
from ..legacy.layers import ConvRecurrent2D


def serialize(layer):
    """Serialize a layer.

    # Arguments
        layer: a Layer object.

    # Returns
        dictionary with config.
    """
    return {'class_name': layer.__class__.__name__,
            'config': layer.get_config()}


def deserialize(config, custom_objects=None):
    """Instantiate a layer from a config dictionary.

    # Arguments
        config: dict of the form {'class_name': str, 'config': dict}
        custom_objects: dict mapping class names (or function names)
            of custom (non-Keras) objects to class/functions

    # Returns
        Layer instance (may be Model, Sequential, Layer...)
    """
    from .. import models
    globs = globals()  # All layers.
    globs['Model'] = models.Model
    globs['Sequential'] = models.Sequential
    return deserialize_keras_object(config,
                                    module_objects=globs,
                                    custom_objects=custom_objects,
                                    printable_module_name='layer')
