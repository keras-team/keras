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
"""Core TF-Keras layers."""

from tf_keras.src.layers.core.activation import Activation
from tf_keras.src.layers.core.dense import Dense
from tf_keras.src.layers.core.einsum_dense import EinsumDense
from tf_keras.src.layers.core.embedding import Embedding
from tf_keras.src.layers.core.identity import Identity
from tf_keras.src.layers.core.lambda_layer import Lambda
from tf_keras.src.layers.core.masking import Masking

# Required by third_party/py/tensorflow_gnn/tf_keras/keras_tensors.py
from tf_keras.src.layers.core.tf_op_layer import ClassMethod
from tf_keras.src.layers.core.tf_op_layer import InstanceMethod
from tf_keras.src.layers.core.tf_op_layer import InstanceProperty
from tf_keras.src.layers.core.tf_op_layer import SlicingOpLambda
from tf_keras.src.layers.core.tf_op_layer import TFOpLambda
from tf_keras.src.layers.core.tf_op_layer import _delegate_method
from tf_keras.src.layers.core.tf_op_layer import _delegate_property

# Regularization layers imported for backwards namespace compatibility
from tf_keras.src.layers.regularization.activity_regularization import (
    ActivityRegularization,
)
from tf_keras.src.layers.regularization.dropout import Dropout
from tf_keras.src.layers.regularization.spatial_dropout1d import SpatialDropout1D
from tf_keras.src.layers.regularization.spatial_dropout2d import SpatialDropout2D
from tf_keras.src.layers.regularization.spatial_dropout3d import SpatialDropout3D

# Reshaping layers imported for backwards namespace compatibility
from tf_keras.src.layers.reshaping.flatten import Flatten
from tf_keras.src.layers.reshaping.permute import Permute
from tf_keras.src.layers.reshaping.repeat_vector import RepeatVector
from tf_keras.src.layers.reshaping.reshape import Reshape

