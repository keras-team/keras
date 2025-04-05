# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras layers API."""

# isort: off
import tensorflow.compat.v2 as tf

from tf_keras.src.engine.base_layer import Layer
from tf_keras.src.engine.base_preprocessing_layer import PreprocessingLayer

# Generic layers.
from tf_keras.src.engine.input_layer import Input
from tf_keras.src.engine.input_layer import InputLayer
from tf_keras.src.engine.input_spec import InputSpec
from tf_keras.src.layers.activation.elu import ELU
from tf_keras.src.layers.activation.leaky_relu import LeakyReLU
from tf_keras.src.layers.activation.prelu import PReLU

# Activations layers.
from tf_keras.src.layers.activation.relu import ReLU
from tf_keras.src.layers.activation.softmax import Softmax
from tf_keras.src.layers.activation.thresholded_relu import ThresholdedReLU
from tf_keras.src.layers.attention.additive_attention import AdditiveAttention
from tf_keras.src.layers.attention.attention import Attention

# Attention layers.
from tf_keras.src.layers.attention.multi_head_attention import MultiHeadAttention

# Convolution layer aliases.
# Convolution layers.
from tf_keras.src.layers.convolutional.conv1d import Conv1D
from tf_keras.src.layers.convolutional.conv1d import Convolution1D
from tf_keras.src.layers.convolutional.conv1d_transpose import Conv1DTranspose
from tf_keras.src.layers.convolutional.conv1d_transpose import (
    Convolution1DTranspose,
)
from tf_keras.src.layers.convolutional.conv2d import Conv2D
from tf_keras.src.layers.convolutional.conv2d import Convolution2D
from tf_keras.src.layers.convolutional.conv2d_transpose import Conv2DTranspose
from tf_keras.src.layers.convolutional.conv2d_transpose import (
    Convolution2DTranspose,
)
from tf_keras.src.layers.convolutional.conv3d import Conv3D
from tf_keras.src.layers.convolutional.conv3d import Convolution3D
from tf_keras.src.layers.convolutional.conv3d_transpose import Conv3DTranspose
from tf_keras.src.layers.convolutional.conv3d_transpose import (
    Convolution3DTranspose,
)
from tf_keras.src.layers.convolutional.depthwise_conv1d import DepthwiseConv1D
from tf_keras.src.layers.convolutional.depthwise_conv2d import DepthwiseConv2D
from tf_keras.src.layers.convolutional.separable_conv1d import SeparableConv1D
from tf_keras.src.layers.convolutional.separable_conv1d import (
    SeparableConvolution1D,
)
from tf_keras.src.layers.convolutional.separable_conv2d import SeparableConv2D
from tf_keras.src.layers.convolutional.separable_conv2d import (
    SeparableConvolution2D,
)

# Core layers.
from tf_keras.src.layers.core.activation import Activation
from tf_keras.src.layers.core.dense import Dense
from tf_keras.src.layers.core.einsum_dense import EinsumDense
from tf_keras.src.layers.core.embedding import Embedding
from tf_keras.src.layers.core.identity import Identity
from tf_keras.src.layers.core.lambda_layer import Lambda
from tf_keras.src.layers.core.masking import Masking
from tf_keras.src.layers.core.tf_op_layer import ClassMethod
from tf_keras.src.layers.core.tf_op_layer import InstanceMethod
from tf_keras.src.layers.core.tf_op_layer import InstanceProperty
from tf_keras.src.layers.core.tf_op_layer import SlicingOpLambda
from tf_keras.src.layers.core.tf_op_layer import TFOpLambda

# Locally-connected layers.
from tf_keras.src.layers.locally_connected.locally_connected1d import (
    LocallyConnected1D,
)
from tf_keras.src.layers.locally_connected.locally_connected2d import (
    LocallyConnected2D,
)

# Merging functions.
# Merging layers.
from tf_keras.src.layers.merging.add import Add
from tf_keras.src.layers.merging.add import add
from tf_keras.src.layers.merging.average import Average
from tf_keras.src.layers.merging.average import average
from tf_keras.src.layers.merging.concatenate import Concatenate
from tf_keras.src.layers.merging.concatenate import concatenate
from tf_keras.src.layers.merging.dot import Dot
from tf_keras.src.layers.merging.dot import dot
from tf_keras.src.layers.merging.maximum import Maximum
from tf_keras.src.layers.merging.maximum import maximum
from tf_keras.src.layers.merging.minimum import Minimum
from tf_keras.src.layers.merging.minimum import minimum
from tf_keras.src.layers.merging.multiply import Multiply
from tf_keras.src.layers.merging.multiply import multiply
from tf_keras.src.layers.merging.subtract import Subtract
from tf_keras.src.layers.merging.subtract import subtract
from tf_keras.src.layers.normalization.batch_normalization import (
    SyncBatchNormalization,
)

# Normalization layers.
from tf_keras.src.layers.normalization.group_normalization import GroupNormalization
from tf_keras.src.layers.normalization.layer_normalization import LayerNormalization
from tf_keras.src.layers.normalization.unit_normalization import UnitNormalization
from tf_keras.src.layers.normalization.spectral_normalization import (
    SpectralNormalization,
)  # noqa: E501

# Preprocessing layers.
from tf_keras.src.layers.preprocessing.category_encoding import CategoryEncoding
from tf_keras.src.layers.preprocessing.discretization import Discretization
from tf_keras.src.layers.preprocessing.hashed_crossing import HashedCrossing
from tf_keras.src.layers.preprocessing.hashing import Hashing

# Image preprocessing layers.
from tf_keras.src.layers.preprocessing.image_preprocessing import CenterCrop
from tf_keras.src.layers.preprocessing.image_preprocessing import RandomBrightness
from tf_keras.src.layers.preprocessing.image_preprocessing import RandomContrast
from tf_keras.src.layers.preprocessing.image_preprocessing import RandomCrop
from tf_keras.src.layers.preprocessing.image_preprocessing import RandomFlip
from tf_keras.src.layers.preprocessing.image_preprocessing import RandomHeight
from tf_keras.src.layers.preprocessing.image_preprocessing import RandomRotation
from tf_keras.src.layers.preprocessing.image_preprocessing import RandomTranslation
from tf_keras.src.layers.preprocessing.image_preprocessing import RandomWidth
from tf_keras.src.layers.preprocessing.image_preprocessing import RandomZoom
from tf_keras.src.layers.preprocessing.image_preprocessing import Rescaling
from tf_keras.src.layers.preprocessing.image_preprocessing import Resizing
from tf_keras.src.layers.preprocessing.integer_lookup import IntegerLookup
from tf_keras.src.layers.preprocessing.normalization import Normalization
from tf_keras.src.layers.preprocessing.string_lookup import StringLookup
from tf_keras.src.layers.preprocessing.text_vectorization import TextVectorization
from tf_keras.src.layers.regularization.activity_regularization import (
    ActivityRegularization,
)
from tf_keras.src.layers.regularization.alpha_dropout import AlphaDropout

# Regularization layers.
from tf_keras.src.layers.regularization.dropout import Dropout
from tf_keras.src.layers.regularization.gaussian_dropout import GaussianDropout
from tf_keras.src.layers.regularization.gaussian_noise import GaussianNoise
from tf_keras.src.layers.regularization.spatial_dropout1d import SpatialDropout1D
from tf_keras.src.layers.regularization.spatial_dropout2d import SpatialDropout2D
from tf_keras.src.layers.regularization.spatial_dropout3d import SpatialDropout3D

# Reshaping layers.
from tf_keras.src.layers.reshaping.cropping1d import Cropping1D
from tf_keras.src.layers.reshaping.cropping2d import Cropping2D
from tf_keras.src.layers.reshaping.cropping3d import Cropping3D
from tf_keras.src.layers.reshaping.flatten import Flatten
from tf_keras.src.layers.reshaping.permute import Permute
from tf_keras.src.layers.reshaping.repeat_vector import RepeatVector
from tf_keras.src.layers.reshaping.reshape import Reshape
from tf_keras.src.layers.reshaping.up_sampling1d import UpSampling1D
from tf_keras.src.layers.reshaping.up_sampling2d import UpSampling2D
from tf_keras.src.layers.reshaping.up_sampling3d import UpSampling3D
from tf_keras.src.layers.reshaping.zero_padding1d import ZeroPadding1D
from tf_keras.src.layers.reshaping.zero_padding2d import ZeroPadding2D
from tf_keras.src.layers.reshaping.zero_padding3d import ZeroPadding3D

if tf.__internal__.tf2.enabled():
    from tf_keras.src.layers.normalization.batch_normalization import (
        BatchNormalization,
    )
    from tf_keras.src.layers.normalization.batch_normalization_v1 import (
        BatchNormalization as BatchNormalizationV1,
    )

    BatchNormalizationV2 = BatchNormalization
else:
    from tf_keras.src.layers.normalization.batch_normalization import (
        BatchNormalization as BatchNormalizationV2,
    )
    from tf_keras.src.layers.normalization.batch_normalization_v1 import (
        BatchNormalization,
    )

    BatchNormalizationV1 = BatchNormalization

# Kernelized layers.
from tf_keras.src.layers.kernelized import RandomFourierFeatures

# Pooling layer aliases.
# Pooling layers.
from tf_keras.src.layers.pooling.average_pooling1d import AveragePooling1D
from tf_keras.src.layers.pooling.average_pooling1d import AvgPool1D
from tf_keras.src.layers.pooling.average_pooling2d import AveragePooling2D
from tf_keras.src.layers.pooling.average_pooling2d import AvgPool2D
from tf_keras.src.layers.pooling.average_pooling3d import AveragePooling3D
from tf_keras.src.layers.pooling.average_pooling3d import AvgPool3D
from tf_keras.src.layers.pooling.global_average_pooling1d import (
    GlobalAveragePooling1D,
)
from tf_keras.src.layers.pooling.global_average_pooling1d import GlobalAvgPool1D
from tf_keras.src.layers.pooling.global_average_pooling2d import (
    GlobalAveragePooling2D,
)
from tf_keras.src.layers.pooling.global_average_pooling2d import GlobalAvgPool2D
from tf_keras.src.layers.pooling.global_average_pooling3d import (
    GlobalAveragePooling3D,
)
from tf_keras.src.layers.pooling.global_average_pooling3d import GlobalAvgPool3D
from tf_keras.src.layers.pooling.global_max_pooling1d import GlobalMaxPool1D
from tf_keras.src.layers.pooling.global_max_pooling1d import GlobalMaxPooling1D
from tf_keras.src.layers.pooling.global_max_pooling2d import GlobalMaxPool2D
from tf_keras.src.layers.pooling.global_max_pooling2d import GlobalMaxPooling2D
from tf_keras.src.layers.pooling.global_max_pooling3d import GlobalMaxPool3D
from tf_keras.src.layers.pooling.global_max_pooling3d import GlobalMaxPooling3D
from tf_keras.src.layers.pooling.max_pooling1d import MaxPool1D
from tf_keras.src.layers.pooling.max_pooling1d import MaxPooling1D
from tf_keras.src.layers.pooling.max_pooling2d import MaxPool2D
from tf_keras.src.layers.pooling.max_pooling2d import MaxPooling2D
from tf_keras.src.layers.pooling.max_pooling3d import MaxPool3D
from tf_keras.src.layers.pooling.max_pooling3d import MaxPooling3D
from tf_keras.src.layers.rnn.abstract_rnn_cell import AbstractRNNCell

# Recurrent layers.
from tf_keras.src.layers.rnn.base_rnn import RNN
from tf_keras.src.layers.rnn.simple_rnn import SimpleRNN
from tf_keras.src.layers.rnn.simple_rnn import SimpleRNNCell
from tf_keras.src.layers.rnn.stacked_rnn_cells import StackedRNNCells

if tf.__internal__.tf2.enabled():
    from tf_keras.src.layers.rnn.gru import GRU
    from tf_keras.src.layers.rnn.gru import GRUCell
    from tf_keras.src.layers.rnn.gru_v1 import GRU as GRUV1
    from tf_keras.src.layers.rnn.gru_v1 import GRUCell as GRUCellV1
    from tf_keras.src.layers.rnn.lstm import LSTM
    from tf_keras.src.layers.rnn.lstm import LSTMCell
    from tf_keras.src.layers.rnn.lstm_v1 import LSTM as LSTMV1
    from tf_keras.src.layers.rnn.lstm_v1 import LSTMCell as LSTMCellV1

    GRUV2 = GRU
    GRUCellV2 = GRUCell
    LSTMV2 = LSTM
    LSTMCellV2 = LSTMCell
else:
    from tf_keras.src.layers.rnn.gru import GRU as GRUV2
    from tf_keras.src.layers.rnn.gru import GRUCell as GRUCellV2
    from tf_keras.src.layers.rnn.gru_v1 import GRU
    from tf_keras.src.layers.rnn.gru_v1 import GRUCell
    from tf_keras.src.layers.rnn.lstm import LSTM as LSTMV2
    from tf_keras.src.layers.rnn.lstm import LSTMCell as LSTMCellV2
    from tf_keras.src.layers.rnn.lstm_v1 import LSTM
    from tf_keras.src.layers.rnn.lstm_v1 import LSTMCell

    GRUV1 = GRU
    GRUCellV1 = GRUCell
    LSTMV1 = LSTM
    LSTMCellV1 = LSTMCell

# Serialization functions.
from tf_keras.src.layers import serialization

# Wrapper functions.
from tf_keras.src.layers.rnn.base_wrapper import Wrapper
from tf_keras.src.layers.rnn.bidirectional import Bidirectional

# RNN Cell wrappers.
from tf_keras.src.layers.rnn.cell_wrappers import DeviceWrapper
from tf_keras.src.layers.rnn.cell_wrappers import DropoutWrapper
from tf_keras.src.layers.rnn.cell_wrappers import ResidualWrapper

# Convolutional-recurrent layers.
from tf_keras.src.layers.rnn.conv_lstm1d import ConvLSTM1D
from tf_keras.src.layers.rnn.conv_lstm2d import ConvLSTM2D
from tf_keras.src.layers.rnn.conv_lstm3d import ConvLSTM3D
from tf_keras.src.layers.rnn.cudnn_gru import CuDNNGRU

# cuDNN recurrent layers.
from tf_keras.src.layers.rnn.cudnn_lstm import CuDNNLSTM
from tf_keras.src.layers.rnn.time_distributed import TimeDistributed
from tf_keras.src.layers.serialization import deserialize
from tf_keras.src.layers.serialization import deserialize_from_json
from tf_keras.src.layers.serialization import get_builtin_layer
from tf_keras.src.layers.serialization import serialize


class VersionAwareLayers:
    """Utility to be used internally to access layers in a V1/V2-aware fashion.

    When using layers within the TF-Keras codebase, under the constraint that
    e.g. `layers.BatchNormalization` should be the `BatchNormalization` version
    corresponding to the current runtime (TF1 or TF2), do not simply access
    `layers.BatchNormalization` since it would ignore e.g. an early
    `compat.v2.disable_v2_behavior()` call. Instead, use an instance
    of `VersionAwareLayers` (which you can use just like the `layers` module).
    """

    def __getattr__(self, name):
        serialization.populate_deserializable_objects()
        if name in serialization.LOCAL.ALL_OBJECTS:
            return serialization.LOCAL.ALL_OBJECTS[name]
        return super().__getattr__(name)

