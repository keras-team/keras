from keras.api_export import keras_export
from keras.layers.activations.activation import Activation
from keras.layers.activations.elu import ELU
from keras.layers.activations.leaky_relu import LeakyReLU
from keras.layers.activations.prelu import PReLU
from keras.layers.activations.relu import ReLU
from keras.layers.activations.softmax import Softmax
from keras.layers.attention.additive_attention import AdditiveAttention
from keras.layers.attention.attention import Attention
from keras.layers.attention.grouped_query_attention import GroupedQueryAttention
from keras.layers.attention.multi_head_attention import MultiHeadAttention
from keras.layers.convolutional.conv1d import Conv1D
from keras.layers.convolutional.conv1d_transpose import Conv1DTranspose
from keras.layers.convolutional.conv2d import Conv2D
from keras.layers.convolutional.conv2d_transpose import Conv2DTranspose
from keras.layers.convolutional.conv3d import Conv3D
from keras.layers.convolutional.conv3d_transpose import Conv3DTranspose
from keras.layers.convolutional.depthwise_conv1d import DepthwiseConv1D
from keras.layers.convolutional.depthwise_conv2d import DepthwiseConv2D
from keras.layers.convolutional.separable_conv1d import SeparableConv1D
from keras.layers.convolutional.separable_conv2d import SeparableConv2D
from keras.layers.core.dense import Dense
from keras.layers.core.einsum_dense import EinsumDense
from keras.layers.core.embedding import Embedding
from keras.layers.core.identity import Identity
from keras.layers.core.input_layer import Input
from keras.layers.core.input_layer import InputLayer
from keras.layers.core.lambda_layer import Lambda
from keras.layers.core.masking import Masking
from keras.layers.core.wrapper import Wrapper
from keras.layers.layer import Layer
from keras.layers.merging.add import Add
from keras.layers.merging.add import add
from keras.layers.merging.average import Average
from keras.layers.merging.average import average
from keras.layers.merging.concatenate import Concatenate
from keras.layers.merging.concatenate import concatenate
from keras.layers.merging.dot import Dot
from keras.layers.merging.dot import dot
from keras.layers.merging.maximum import Maximum
from keras.layers.merging.maximum import maximum
from keras.layers.merging.minimum import Minimum
from keras.layers.merging.minimum import minimum
from keras.layers.merging.multiply import Multiply
from keras.layers.merging.multiply import multiply
from keras.layers.merging.subtract import Subtract
from keras.layers.merging.subtract import subtract
from keras.layers.normalization.batch_normalization import BatchNormalization
from keras.layers.normalization.group_normalization import GroupNormalization
from keras.layers.normalization.layer_normalization import LayerNormalization
from keras.layers.normalization.spectral_normalization import (
    SpectralNormalization,
)
from keras.layers.normalization.unit_normalization import UnitNormalization
from keras.layers.pooling.average_pooling1d import AveragePooling1D
from keras.layers.pooling.average_pooling2d import AveragePooling2D
from keras.layers.pooling.average_pooling3d import AveragePooling3D
from keras.layers.pooling.global_average_pooling1d import GlobalAveragePooling1D
from keras.layers.pooling.global_average_pooling2d import GlobalAveragePooling2D
from keras.layers.pooling.global_average_pooling3d import GlobalAveragePooling3D
from keras.layers.pooling.global_max_pooling1d import GlobalMaxPooling1D
from keras.layers.pooling.global_max_pooling2d import GlobalMaxPooling2D
from keras.layers.pooling.global_max_pooling3d import GlobalMaxPooling3D
from keras.layers.pooling.max_pooling1d import MaxPooling1D
from keras.layers.pooling.max_pooling2d import MaxPooling2D
from keras.layers.pooling.max_pooling3d import MaxPooling3D
from keras.layers.preprocessing.category_encoding import CategoryEncoding
from keras.layers.preprocessing.center_crop import CenterCrop
from keras.layers.preprocessing.discretization import Discretization
from keras.layers.preprocessing.hashed_crossing import HashedCrossing
from keras.layers.preprocessing.hashing import Hashing
from keras.layers.preprocessing.index_lookup import IndexLookup
from keras.layers.preprocessing.integer_lookup import IntegerLookup
from keras.layers.preprocessing.normalization import Normalization
from keras.layers.preprocessing.random_brightness import RandomBrightness
from keras.layers.preprocessing.random_contrast import RandomContrast
from keras.layers.preprocessing.random_crop import RandomCrop
from keras.layers.preprocessing.random_flip import RandomFlip
from keras.layers.preprocessing.random_rotation import RandomRotation
from keras.layers.preprocessing.random_translation import RandomTranslation
from keras.layers.preprocessing.random_zoom import RandomZoom
from keras.layers.preprocessing.rescaling import Rescaling
from keras.layers.preprocessing.resizing import Resizing
from keras.layers.preprocessing.string_lookup import StringLookup
from keras.layers.preprocessing.text_vectorization import TextVectorization
from keras.layers.regularization.activity_regularization import (
    ActivityRegularization,
)
from keras.layers.regularization.alpha_dropout import AlphaDropout
from keras.layers.regularization.dropout import Dropout
from keras.layers.regularization.gaussian_dropout import GaussianDropout
from keras.layers.regularization.gaussian_noise import GaussianNoise
from keras.layers.regularization.spatial_dropout import SpatialDropout1D
from keras.layers.regularization.spatial_dropout import SpatialDropout2D
from keras.layers.regularization.spatial_dropout import SpatialDropout3D
from keras.layers.reshaping.cropping1d import Cropping1D
from keras.layers.reshaping.cropping2d import Cropping2D
from keras.layers.reshaping.cropping3d import Cropping3D
from keras.layers.reshaping.flatten import Flatten
from keras.layers.reshaping.permute import Permute
from keras.layers.reshaping.repeat_vector import RepeatVector
from keras.layers.reshaping.reshape import Reshape
from keras.layers.reshaping.up_sampling1d import UpSampling1D
from keras.layers.reshaping.up_sampling2d import UpSampling2D
from keras.layers.reshaping.up_sampling3d import UpSampling3D
from keras.layers.reshaping.zero_padding1d import ZeroPadding1D
from keras.layers.reshaping.zero_padding2d import ZeroPadding2D
from keras.layers.reshaping.zero_padding3d import ZeroPadding3D
from keras.layers.rnn.bidirectional import Bidirectional
from keras.layers.rnn.conv_lstm1d import ConvLSTM1D
from keras.layers.rnn.conv_lstm2d import ConvLSTM2D
from keras.layers.rnn.conv_lstm3d import ConvLSTM3D
from keras.layers.rnn.gru import GRU
from keras.layers.rnn.gru import GRUCell
from keras.layers.rnn.lstm import LSTM
from keras.layers.rnn.lstm import LSTMCell
from keras.layers.rnn.rnn import RNN
from keras.layers.rnn.simple_rnn import SimpleRNN
from keras.layers.rnn.simple_rnn import SimpleRNNCell
from keras.layers.rnn.stacked_rnn_cells import StackedRNNCells
from keras.layers.rnn.time_distributed import TimeDistributed
from keras.saving import serialization_lib


@keras_export("keras.layers.serialize")
def serialize(layer):
    """Returns the layer configuration as a Python dict.

    Args:
        layer: A `keras.layers.Layer` instance to serialize.

    Returns:
        Python dict which contains the configuration of the layer.
    """
    return serialization_lib.serialize_keras_object(layer)


@keras_export("keras.layers.deserialize")
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
