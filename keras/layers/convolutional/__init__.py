# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Keras convolution layers."""


# Convolution layer aliases.
# Convolution layers.
from keras.layers.convolutional.conv1d import Conv1D
from keras.layers.convolutional.conv1d import Convolution1D
from keras.layers.convolutional.conv1d_transpose import Conv1DTranspose
from keras.layers.convolutional.conv1d_transpose import Convolution1DTranspose
from keras.layers.convolutional.conv2d import Conv2D
from keras.layers.convolutional.conv2d import Convolution2D
from keras.layers.convolutional.conv2d_transpose import Conv2DTranspose
from keras.layers.convolutional.conv2d_transpose import Convolution2DTranspose
from keras.layers.convolutional.conv3d import Conv3D
from keras.layers.convolutional.conv3d import Convolution3D
from keras.layers.convolutional.conv3d_transpose import Conv3DTranspose
from keras.layers.convolutional.conv3d_transpose import Convolution3DTranspose
from keras.layers.convolutional.depthwise_conv1d import DepthwiseConv1D
from keras.layers.convolutional.depthwise_conv2d import DepthwiseConv2D
from keras.layers.convolutional.separable_conv1d import SeparableConv1D
from keras.layers.convolutional.separable_conv1d import SeparableConvolution1D
from keras.layers.convolutional.separable_conv2d import SeparableConv2D
from keras.layers.convolutional.separable_conv2d import SeparableConvolution2D

# Pooling layers imported for backwards namespace compatibility.
from keras.layers.pooling.average_pooling1d import AveragePooling1D
from keras.layers.pooling.average_pooling2d import AveragePooling2D
from keras.layers.pooling.average_pooling3d import AveragePooling3D
from keras.layers.pooling.max_pooling1d import MaxPooling1D
from keras.layers.pooling.max_pooling2d import MaxPooling2D
from keras.layers.pooling.max_pooling3d import MaxPooling3D

# Reshaping layers imported for backwards namespace compatibility
from keras.layers.reshaping.cropping1d import Cropping1D
from keras.layers.reshaping.cropping2d import Cropping2D
from keras.layers.reshaping.cropping3d import Cropping3D
from keras.layers.reshaping.up_sampling1d import UpSampling1D
from keras.layers.reshaping.up_sampling2d import UpSampling2D
from keras.layers.reshaping.up_sampling3d import UpSampling3D
from keras.layers.reshaping.zero_padding1d import ZeroPadding1D
from keras.layers.reshaping.zero_padding2d import ZeroPadding2D
from keras.layers.reshaping.zero_padding3d import ZeroPadding3D
