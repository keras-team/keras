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
"""Keras regularization layers."""


from tf_keras.src.layers.regularization.activity_regularization import (
    ActivityRegularization,
)
from tf_keras.src.layers.regularization.alpha_dropout import AlphaDropout
from tf_keras.src.layers.regularization.dropout import Dropout
from tf_keras.src.layers.regularization.gaussian_dropout import GaussianDropout
from tf_keras.src.layers.regularization.gaussian_noise import GaussianNoise
from tf_keras.src.layers.regularization.spatial_dropout1d import SpatialDropout1D
from tf_keras.src.layers.regularization.spatial_dropout2d import SpatialDropout2D
from tf_keras.src.layers.regularization.spatial_dropout3d import SpatialDropout3D

