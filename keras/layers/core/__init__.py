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
"""Core Keras layers."""

from keras.layers.core.activation import Activation
from keras.layers.core.activity_regularization import ActivityRegularization
from keras.layers.core.dense import Dense
from keras.layers.core.dropout import Dropout
from keras.layers.core.flatten import Flatten
from keras.layers.core.lambda_layer import Lambda
from keras.layers.core.masking import Masking
from keras.layers.core.permute import Permute
from keras.layers.core.repeat_vector import RepeatVector
from keras.layers.core.reshape import Reshape
from keras.layers.core.spatial_dropout import SpatialDropout1D
from keras.layers.core.spatial_dropout import SpatialDropout2D
from keras.layers.core.spatial_dropout import SpatialDropout3D
# Required by third_party/py/tensorflow_gnn/graph/keras/keras_tensors.py
from keras.layers.core.tf_op_layer import _delegate_method
from keras.layers.core.tf_op_layer import _delegate_property
from keras.layers.core.tf_op_layer import ClassMethod
from keras.layers.core.tf_op_layer import InstanceMethod
from keras.layers.core.tf_op_layer import InstanceProperty

from keras.layers.core.tf_op_layer import SlicingOpLambda
from keras.layers.core.tf_op_layer import TFOpLambda

