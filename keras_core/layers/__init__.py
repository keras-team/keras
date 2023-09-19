from keras_core.api_export import keras_core_export
from keras_core.layers.activations.activation import Activation
from keras_core.layers.activations.elu import ELU
from keras_core.layers.activations.leaky_relu import LeakyReLU
from keras_core.layers.activations.prelu import PReLU
from keras_core.layers.activations.relu import ReLU
from keras_core.layers.activations.softmax import Softmax
from keras_core.layers.attention.additive_attention import AdditiveAttention
from keras_core.layers.attention.attention import Attention
from keras_core.layers.attention.multi_head_attention import MultiHeadAttention
from keras_core.layers.convolutional.conv1d import Conv1D
from keras_core.layers.convolutional.conv1d_transpose import Conv1DTranspose
from keras_core.layers.convolutional.conv2d import Conv2D
from keras_core.layers.convolutional.conv2d_transpose import Conv2DTranspose
from keras_core.layers.convolutional.conv3d import Conv3D
from keras_core.layers.convolutional.conv3d_transpose import Conv3DTranspose
from keras_core.layers.convolutional.depthwise_conv1d import DepthwiseConv1D
from keras_core.layers.convolutional.depthwise_conv2d import DepthwiseConv2D
from keras_core.layers.convolutional.separable_conv1d import SeparableConv1D
from keras_core.layers.convolutional.separable_conv2d import SeparableConv2D
from keras_core.layers.core.dense import Dense
from keras_core.layers.core.einsum_dense import EinsumDense
from keras_core.layers.core.embedding import Embedding
from keras_core.layers.core.identity import Identity
from keras_core.layers.core.input_layer import Input
from keras_core.layers.core.input_layer import InputLayer
from keras_core.layers.core.lambda_layer import Lambda
from keras_core.layers.core.masking import Masking
from keras_core.layers.core.wrapper import Wrapper
from keras_core.layers.layer import Layer
from keras_core.layers.merging.add import Add
from keras_core.layers.merging.add import add
from keras_core.layers.merging.average import Average
from keras_core.layers.merging.average import average
from keras_core.layers.merging.concatenate import Concatenate
from keras_core.layers.merging.concatenate import concatenate
from keras_core.layers.merging.dot import Dot
from keras_core.layers.merging.dot import dot
from keras_core.layers.merging.maximum import Maximum
from keras_core.layers.merging.maximum import maximum
from keras_core.layers.merging.minimum import Minimum
from keras_core.layers.merging.minimum import minimum
from keras_core.layers.merging.multiply import Multiply
from keras_core.layers.merging.multiply import multiply
from keras_core.layers.merging.subtract import Subtract
from keras_core.layers.merging.subtract import subtract
from keras_core.layers.normalization.batch_normalization import (
    BatchNormalization,
)
from keras_core.layers.normalization.group_normalization import (
    GroupNormalization,
)
from keras_core.layers.normalization.layer_normalization import (
    LayerNormalization,
)
from keras_core.layers.normalization.spectral_normalization import (
    SpectralNormalization,
)
from keras_core.layers.normalization.unit_normalization import UnitNormalization
from keras_core.layers.pooling.average_pooling1d import AveragePooling1D
from keras_core.layers.pooling.average_pooling2d import AveragePooling2D
from keras_core.layers.pooling.average_pooling3d import AveragePooling3D
from keras_core.layers.pooling.global_average_pooling1d import (
    GlobalAveragePooling1D,
)
from keras_core.layers.pooling.global_average_pooling2d import (
    GlobalAveragePooling2D,
)
from keras_core.layers.pooling.global_average_pooling3d import (
    GlobalAveragePooling3D,
)
from keras_core.layers.pooling.global_max_pooling1d import GlobalMaxPooling1D
from keras_core.layers.pooling.global_max_pooling2d import GlobalMaxPooling2D
from keras_core.layers.pooling.global_max_pooling3d import GlobalMaxPooling3D
from keras_core.layers.pooling.max_pooling1d import MaxPooling1D
from keras_core.layers.pooling.max_pooling2d import MaxPooling2D
from keras_core.layers.pooling.max_pooling3d import MaxPooling3D
from keras_core.layers.preprocessing.category_encoding import CategoryEncoding
from keras_core.layers.preprocessing.center_crop import CenterCrop
from keras_core.layers.preprocessing.discretization import Discretization
from keras_core.layers.preprocessing.hashed_crossing import HashedCrossing
from keras_core.layers.preprocessing.hashing import Hashing
from keras_core.layers.preprocessing.index_lookup import IndexLookup
from keras_core.layers.preprocessing.integer_lookup import IntegerLookup
from keras_core.layers.preprocessing.normalization import Normalization
from keras_core.layers.preprocessing.random_brightness import RandomBrightness
from keras_core.layers.preprocessing.random_contrast import RandomContrast
from keras_core.layers.preprocessing.random_crop import RandomCrop
from keras_core.layers.preprocessing.random_flip import RandomFlip
from keras_core.layers.preprocessing.random_rotation import RandomRotation
from keras_core.layers.preprocessing.random_translation import RandomTranslation
from keras_core.layers.preprocessing.random_zoom import RandomZoom
from keras_core.layers.preprocessing.rescaling import Rescaling
from keras_core.layers.preprocessing.resizing import Resizing
from keras_core.layers.preprocessing.string_lookup import StringLookup
from keras_core.layers.preprocessing.text_vectorization import TextVectorization
from keras_core.layers.regularization.activity_regularization import (
    ActivityRegularization,
)
from keras_core.layers.regularization.dropout import Dropout
from keras_core.layers.regularization.gaussian_dropout import GaussianDropout
from keras_core.layers.regularization.gaussian_noise import GaussianNoise
from keras_core.layers.regularization.spatial_dropout import SpatialDropout1D
from keras_core.layers.regularization.spatial_dropout import SpatialDropout2D
from keras_core.layers.regularization.spatial_dropout import SpatialDropout3D
from keras_core.layers.reshaping.cropping1d import Cropping1D
from keras_core.layers.reshaping.cropping2d import Cropping2D
from keras_core.layers.reshaping.cropping3d import Cropping3D
from keras_core.layers.reshaping.flatten import Flatten
from keras_core.layers.reshaping.permute import Permute
from keras_core.layers.reshaping.repeat_vector import RepeatVector
from keras_core.layers.reshaping.reshape import Reshape
from keras_core.layers.reshaping.up_sampling1d import UpSampling1D
from keras_core.layers.reshaping.up_sampling2d import UpSampling2D
from keras_core.layers.reshaping.up_sampling3d import UpSampling3D
from keras_core.layers.reshaping.zero_padding1d import ZeroPadding1D
from keras_core.layers.reshaping.zero_padding2d import ZeroPadding2D
from keras_core.layers.reshaping.zero_padding3d import ZeroPadding3D
from keras_core.layers.rnn.bidirectional import Bidirectional
from keras_core.layers.rnn.conv_lstm1d import ConvLSTM1D
from keras_core.layers.rnn.conv_lstm2d import ConvLSTM2D
from keras_core.layers.rnn.conv_lstm3d import ConvLSTM3D
from keras_core.layers.rnn.gru import GRU
from keras_core.layers.rnn.gru import GRUCell
from keras_core.layers.rnn.lstm import LSTM
from keras_core.layers.rnn.lstm import LSTMCell
from keras_core.layers.rnn.rnn import RNN
from keras_core.layers.rnn.simple_rnn import SimpleRNN
from keras_core.layers.rnn.simple_rnn import SimpleRNNCell
from keras_core.layers.rnn.stacked_rnn_cells import StackedRNNCells
from keras_core.layers.rnn.time_distributed import TimeDistributed
from keras_core.saving import serialization_lib


@keras_core_export("keras_core.layers.serialize")
def serialize(layer):
    """Returns the layer configuration as a Python dict.

    Args:
        layer: A `keras.layers.Layer` instance to serialize.

    Returns:
        Python dict which contains the configuration of the layer.
    """
    return serialization_lib.serialize_keras_object(layer)


@keras_core_export("keras_core.layers.deserialize")
def deserialize(config, custom_objects=None):
    """Returns a Keras layer object via its configuration.

    Args:
        config: A python dict containing a serialized layer configuration.
        custom_objects: Optional dictionary mapping names (strings) to custom
            objects (classes and functions) to be considered during
            deserialization.

    Returns:
        A Keras layer instance.
    """
    obj = serialization_lib.deserialize_keras_object(
        config,
        custom_objects=custom_objects,
    )
    if not isinstance(obj, Layer):
        raise ValueError(
            "`keras.layers.deserialize` was passed a `config` object that is "
            f"not a `keras.layers.Layer`. Received: {config}"
        )
    return obj
