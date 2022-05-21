# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Legacy Adadelta optimizer implementation."""

from keras.optimizers.optimizer_v2 import adadelta

from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.optimizers.legacy.Adadelta")
class Adadelta(adadelta.Adadelta):
    pass
