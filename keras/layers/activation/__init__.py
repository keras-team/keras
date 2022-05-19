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
"""Layers that act as activation functions."""
# pylint: disable=g-bad-import-order

from keras.layers.activation.relu import ReLU
from keras.layers.activation.softmax import Softmax
from keras.layers.activation.leaky_relu import LeakyReLU
from keras.layers.activation.prelu import PReLU
from keras.layers.activation.elu import ELU
from keras.layers.activation.thresholded_relu import ThresholdedReLU
