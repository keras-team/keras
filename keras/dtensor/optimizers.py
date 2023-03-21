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
"""DTensor specific Keras optimizers."""

from keras.optimizers import adadelta
from keras.optimizers import adagrad
from keras.optimizers import adam
from keras.optimizers import adamw
from keras.optimizers import rmsprop
from keras.optimizers import sgd

Adadelta = adadelta.Adadelta
Adagrad = adagrad.Adagrad
Adam = adam.Adam
AdamW = adamw.AdamW
RMSprop = rmsprop.RMSprop
SGD = sgd.SGD
