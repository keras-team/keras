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
"""Keras audio dataset loading utilities."""

import tensorflow.compat.v2 as tf #type: ignore

# pylint: disable=g-classes-have-attributes

import numpy as np

from keras.utils import dataset_utils
from tensorflow.python.util.tf_export import keras_export



def split_dataset(dataset, left_size=None, right_size=None, shuffle=False, seed=None):
    """Split a dataset into a left half and a right half (e.g. training / validation).
    
    Args:
        dataset: A `tf.data.Dataset` object or a list/tuple of arrays with the same length.
        left_size: If float, it should be in range `[0, 1]` range and signifies the fraction of the
            data to pack in the left dataset. If integer, it signifies the number of samples
            to pack in the left dataset. If `None`, it defaults to the complement to `right_size`.
        right_size: If float, it should be in range `[0, 1]` range and signifies the fraction of the
            data to pack in the right dataset. If integer, it signifies the number of samples
            to pack in the right dataset. If `None`, it defaults to the complement to `left_size`.
        shuffle: Boolean, whether to shuffle the data before splitting it.
        seed: A random seed for shuffling.

    Returns:
        A tuple of two `tf.data.Dataset` objects: the left and right splits.
    """

    pass