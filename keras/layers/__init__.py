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

import tensorflow.compat.v2 as tf

# pylint: disable=g-bad-import-order,g-direct-tensorflow-import,disable=g-import-not-at-top
from tensorflow.python import tf2

# Generic layers.
from keras.engine.input_layer import Input
from keras.engine.input_layer import InputLayer
from keras.engine.input_spec import InputSpec
from keras.engine.base_layer import Layer
from keras.engine.base_preprocessing_layer import PreprocessingLayer

# Image preprocessing layers.
from keras.layers.preprocessing.image_preprocessing import CenterCrop
from keras.layers.preprocessing.image_preprocessing import RandomCrop
from keras.layers.preprocessing.image_preprocessing import RandomFlip
from keras.layers.preprocessing.image_preprocessing import RandomContrast
from keras.layers.preprocessing.image_preprocessing import RandomHeight
from keras.layers.preprocessing.image_preprocessing import RandomRotation
from keras.layers.preprocessing.image_preprocessing import RandomTranslation
from keras.layers.preprocessing.image_preprocessing import RandomWidth
from keras.layers.preprocessing.image_preprocessing import RandomZoom
from keras.layers.preprocessing.image_preprocessing import Resizing
from keras.layers.preprocessing.image_preprocessing import Rescaling

# Preprocessing layers.
from keras.layers.preprocessing.category_encoding import CategoryEncoding
from keras.layers.preprocessing.discretization import Discretization
from keras.layers.preprocessing.hashing import Hashing
from keras.layers.preprocessing.hashed_crossing import HashedCrossing
from keras.layers.preprocessing.integer_lookup import IntegerLookup
from keras.layers.preprocessing.normalization import Normalization
from keras.layers.preprocessing.string_lookup import StringLookup
from keras.layers.preprocessing.text_vectorization import TextVectorization

# Activations layers.
from keras.layers.activation.relu import ReLU
from keras.layers.activation.softmax import Softmax
from keras.layers.activation.leaky_relu import LeakyReLU
from keras.layers.activation.prelu import PReLU
from keras.layers.activation.elu import ELU
from keras.layers.activation.thresholded_relu import ThresholdedReLU

# Attention layers.
from keras.layers.attention.multi_head_attention import MultiHeadAttention
from keras.layers.attention.attention import Attention
from keras.layers.attention.additive_attention import AdditiveAttention

# Convolution layers.
from keras.layers.convolutional.conv1d import Conv1D
from keras.layers.convolutional.conv2d import Conv2D
from keras.layers.convolutional.conv3d import Conv3D
from keras.layers.convolutional.conv1d_transpose import Conv1DTranspose
from keras.layers.convolutional.conv2d_transpose import Conv2DTranspose
from keras.layers.convolutional.conv3d_transpose import Conv3DTranspose
from keras.layers.convolutional.depthwise_conv1d import DepthwiseConv1D
from keras.layers.convolutional.depthwise_conv2d import DepthwiseConv2D
from keras.layers.convolutional.separable_conv1d import SeparableConv1D
from keras.layers.convolutional.separable_conv2d import SeparableConv2D

# Convolution layer aliases.
from keras.layers.convolutional.conv1d import Convolution1D
from keras.layers.convolutional.conv2d import Convolution2D
from keras.layers.convolutional.conv3d import Convolution3D
from keras.layers.convolutional.conv1d_transpose import Convolution1DTranspose
from keras.layers.convolutional.conv2d_transpose import Convolution2DTranspose
from keras.layers.convolutional.conv3d_transpose import Convolution3DTranspose
from keras.layers.convolutional.separable_conv1d import SeparableConvolution1D
from keras.layers.convolutional.separable_conv2d import SeparableConvolution2D

# Regularization layers.
from keras.layers.regularization.dropout import Dropout
from keras.layers.regularization.spatial_dropout1d import SpatialDropout1D
from keras.layers.regularization.spatial_dropout2d import SpatialDropout2D
from keras.layers.regularization.spatial_dropout3d import SpatialDropout3D
from keras.layers.regularization.gaussian_dropout import GaussianDropout
from keras.layers.regularization.gaussian_noise import GaussianNoise
from keras.layers.regularization.activity_regularization import ActivityRegularization
from keras.layers.regularization.alpha_dropout import AlphaDropout

# Reshaping layers.
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

# Core layers.
from keras.layers.core.activation import Activation
from keras.layers.core.dense import Dense
from keras.layers.core.einsum_dense import EinsumDense
from keras.layers.core.embedding import Embedding
from keras.layers.core.lambda_layer import Lambda
from keras.layers.core.masking import Masking
from keras.layers.core.tf_op_layer import ClassMethod
from keras.layers.core.tf_op_layer import InstanceMethod
from keras.layers.core.tf_op_layer import InstanceProperty
from keras.layers.core.tf_op_layer import SlicingOpLambda
from keras.layers.core.tf_op_layer import TFOpLambda

# Locally-connected layers.
from keras.layers.locally_connected.locally_connected1d import LocallyConnected1D
from keras.layers.locally_connected.locally_connected2d import LocallyConnected2D

# Merging layers.
from keras.layers.merging.add import Add
from keras.layers.merging.subtract import Subtract
from keras.layers.merging.multiply import Multiply
from keras.layers.merging.average import Average
from keras.layers.merging.maximum import Maximum
from keras.layers.merging.minimum import Minimum
from keras.layers.merging.concatenate import Concatenate
from keras.layers.merging.dot import Dot

# Merging functions.
from keras.layers.merging.add import add
from keras.layers.merging.subtract import subtract
from keras.layers.merging.multiply import multiply
from keras.layers.merging.average import average
from keras.layers.merging.maximum import maximum
from keras.layers.merging.minimum import minimum
from keras.layers.merging.concatenate import concatenate
from keras.layers.merging.dot import dot

# Normalization layers.
from keras.layers.normalization.layer_normalization import LayerNormalization
from keras.layers.normalization.batch_normalization import SyncBatchNormalization
from keras.layers.normalization.unit_normalization import UnitNormalization

if tf.__internal__.tf2.enabled():
  from keras.layers.normalization.batch_normalization import BatchNormalization
  from keras.layers.normalization.batch_normalization_v1 import BatchNormalization as BatchNormalizationV1
  BatchNormalizationV2 = BatchNormalization
else:
  from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
  from keras.layers.normalization.batch_normalization import BatchNormalization as BatchNormalizationV2
  BatchNormalizationV1 = BatchNormalization

# Kernelized layers.
from keras.layers.kernelized import RandomFourierFeatures

# Pooling layers.
from keras.layers.pooling.average_pooling1d import AveragePooling1D
from keras.layers.pooling.average_pooling2d import AveragePooling2D
from keras.layers.pooling.average_pooling3d import AveragePooling3D
from keras.layers.pooling.max_pooling1d import MaxPooling1D
from keras.layers.pooling.max_pooling2d import MaxPooling2D
from keras.layers.pooling.max_pooling3d import MaxPooling3D
from keras.layers.pooling.global_average_pooling1d import GlobalAveragePooling1D
from keras.layers.pooling.global_average_pooling2d import GlobalAveragePooling2D
from keras.layers.pooling.global_average_pooling3d import GlobalAveragePooling3D
from keras.layers.pooling.global_max_pooling1d import GlobalMaxPooling1D
from keras.layers.pooling.global_max_pooling2d import GlobalMaxPooling2D
from keras.layers.pooling.global_max_pooling3d import GlobalMaxPooling3D

# Pooling layer aliases.
from keras.layers.pooling.average_pooling1d import AvgPool1D
from keras.layers.pooling.average_pooling2d import AvgPool2D
from keras.layers.pooling.average_pooling3d import AvgPool3D
from keras.layers.pooling.max_pooling1d import MaxPool1D
from keras.layers.pooling.max_pooling2d import MaxPool2D
from keras.layers.pooling.max_pooling3d import MaxPool3D
from keras.layers.pooling.global_average_pooling1d import GlobalAvgPool1D
from keras.layers.pooling.global_average_pooling2d import GlobalAvgPool2D
from keras.layers.pooling.global_average_pooling3d import GlobalAvgPool3D
from keras.layers.pooling.global_max_pooling1d import GlobalMaxPool1D
from keras.layers.pooling.global_max_pooling2d import GlobalMaxPool2D
from keras.layers.pooling.global_max_pooling3d import GlobalMaxPool3D

# Recurrent layers.
from keras.layers.rnn.base_rnn import RNN
from keras.layers.rnn.abstract_rnn_cell import AbstractRNNCell
from keras.layers.rnn.stacked_rnn_cells import StackedRNNCells
from keras.layers.rnn.simple_rnn import SimpleRNNCell
from keras.layers.rnn.simple_rnn import SimpleRNN

if tf.__internal__.tf2.enabled():
  from keras.layers.rnn.gru import GRU
  from keras.layers.rnn.gru import GRUCell
  from keras.layers.rnn.lstm import LSTM
  from keras.layers.rnn.lstm import LSTMCell
  from keras.layers.rnn.gru_v1 import GRU as GRUV1
  from keras.layers.rnn.gru_v1 import GRUCell as GRUCellV1
  from keras.layers.rnn.lstm_v1 import LSTM as LSTMV1
  from keras.layers.rnn.lstm_v1 import LSTMCell as LSTMCellV1
  GRUV2 = GRU
  GRUCellV2 = GRUCell
  LSTMV2 = LSTM
  LSTMCellV2 = LSTMCell
else:
  from keras.layers.rnn.gru_v1 import GRU
  from keras.layers.rnn.gru_v1 import GRUCell
  from keras.layers.rnn.lstm_v1 import LSTM
  from keras.layers.rnn.lstm_v1 import LSTMCell
  from keras.layers.rnn.gru import GRU as GRUV2
  from keras.layers.rnn.gru import GRUCell as GRUCellV2
  from keras.layers.rnn.lstm import LSTM as LSTMV2
  from keras.layers.rnn.lstm import LSTMCell as LSTMCellV2
  GRUV1 = GRU
  GRUCellV1 = GRUCell
  LSTMV1 = LSTM
  LSTMCellV1 = LSTMCell

# Convolutional-recurrent layers.
from keras.layers.rnn.conv_lstm1d import ConvLSTM1D
from keras.layers.rnn.conv_lstm2d import ConvLSTM2D
from keras.layers.rnn.conv_lstm3d import ConvLSTM3D

# cuDNN recurrent layers.
from keras.layers.rnn.cudnn_lstm import CuDNNLSTM
from keras.layers.rnn.cudnn_gru import CuDNNGRU

# Wrapper functions.
from keras.layers.rnn.base_wrapper import Wrapper
from keras.layers.rnn.bidirectional import Bidirectional
from keras.layers.rnn.time_distributed import TimeDistributed

# RNN Cell wrappers.
from keras.layers.rnn.cell_wrappers import DeviceWrapper
from keras.layers.rnn.cell_wrappers import DropoutWrapper
from keras.layers.rnn.cell_wrappers import ResidualWrapper

# Serialization functions.
from keras.layers import serialization
from keras.layers.serialization import deserialize
from keras.layers.serialization import deserialize_from_json
from keras.layers.serialization import serialize
from keras.layers.serialization import get_builtin_layer


class VersionAwareLayers:
  """Utility to be used internally to access layers in a V1/V2-aware fashion.

  When using layers within the Keras codebase, under the constraint that
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
    return super(VersionAwareLayers, self).__getattr__(name)
