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
"""Utilities to preprocess data before training.

Deprecated: `tf.keras.preprocessing` APIs do not operate on tensors and are
not recommended for new code. Prefer loading data with either
`tf.keras.utils.text_dataset_from_directory` or
`tf.keras.utils.image_dataset_from_directory`, and then transforming the output
`tf.data.Dataset` with preprocessing layers. These approaches will offer
better performance and intergration with the broader Tensorflow ecosystem. For
more information, see the tutorials for [loading text](
https://www.tensorflow.org/tutorials/load_data/text), [loading images](
https://www.tensorflow.org/tutorials/load_data/images), and [augmenting images](
https://www.tensorflow.org/tutorials/images/data_augmentation), as well as the
[preprocessing layer guide](
https://www.tensorflow.org/guide/keras/preprocessing_layers).
"""
from tf_keras.src import backend
from tf_keras.src.preprocessing import image
from tf_keras.src.preprocessing import sequence
from tf_keras.src.preprocessing import text

