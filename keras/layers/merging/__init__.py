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
"""Keras merging layers."""
# pylint: disable=g-bad-import-order

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
