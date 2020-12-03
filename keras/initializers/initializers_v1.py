# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Keras initializers for TF 1.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.util.tf_export import keras_export

keras_export(v1=['keras.initializers.Zeros', 'keras.initializers.zeros'], allow_multiple_exports=True)(
    tf.compat.v1.zeros_initializer)
keras_export(v1=['keras.initializers.Ones', 'keras.initializers.ones'], allow_multiple_exports=True)(
    tf.compat.v1.ones_initializer)
keras_export(v1=['keras.initializers.Constant', 'keras.initializers.constant'], allow_multiple_exports=True)(
    tf.compat.v1.constant_initializer)
keras_export(v1=['keras.initializers.VarianceScaling'], allow_multiple_exports=True)(
    tf.compat.v1.variance_scaling_initializer)
keras_export(v1=['keras.initializers.Orthogonal',
                 'keras.initializers.orthogonal'], allow_multiple_exports=True)(tf.compat.v1.orthogonal_initializer)
keras_export(v1=['keras.initializers.Identity',
                 'keras.initializers.identity'], allow_multiple_exports=True)(tf.compat.v1.initializers.identity)
keras_export(v1=['keras.initializers.glorot_uniform'], allow_multiple_exports=True)(tf.compat.v1.glorot_uniform_initializer)
keras_export(v1=['keras.initializers.glorot_normal'], allow_multiple_exports=True)(tf.compat.v1.glorot_normal_initializer)


@keras_export(v1=['keras.initializers.RandomNormal',
                  'keras.initializers.random_normal',
                  'keras.initializers.normal'])
class RandomNormal(tf.compat.v1.random_normal_initializer):

  def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=tf.float32):
    super(RandomNormal, self).__init__(
        mean=mean, stddev=stddev, seed=seed, dtype=dtype)


@keras_export(v1=['keras.initializers.RandomUniform',
                  'keras.initializers.random_uniform',
                  'keras.initializers.uniform'])
class RandomUniform(tf.compat.v1.random_uniform_initializer):

  def __init__(self, minval=-0.05, maxval=0.05, seed=None,
               dtype=tf.float32):
    super(RandomUniform, self).__init__(
        minval=minval, maxval=maxval, seed=seed, dtype=dtype)


@keras_export(v1=['keras.initializers.TruncatedNormal',
                  'keras.initializers.truncated_normal'])
class TruncatedNormal(tf.compat.v1.truncated_normal_initializer):

  def __init__(self, mean=0.0, stddev=0.05, seed=None, dtype=tf.float32):
    super(TruncatedNormal, self).__init__(
        mean=mean, stddev=stddev, seed=seed, dtype=dtype)


@keras_export(v1=['keras.initializers.lecun_normal'])
class LecunNormal(tf.compat.v1.variance_scaling_initializer):

  def __init__(self, seed=None):
    super(LecunNormal, self).__init__(
        scale=1., mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


@keras_export(v1=['keras.initializers.lecun_uniform'])
class LecunUniform(tf.compat.v1.variance_scaling_initializer):

  def __init__(self, seed=None):
    super(LecunUniform, self).__init__(
        scale=1., mode='fan_in', distribution='uniform', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


@keras_export(v1=['keras.initializers.he_normal'])
class HeNormal(tf.compat.v1.variance_scaling_initializer):

  def __init__(self, seed=None):
    super(HeNormal, self).__init__(
        scale=2., mode='fan_in', distribution='truncated_normal', seed=seed)

  def get_config(self):
    return {'seed': self.seed}


@keras_export(v1=['keras.initializers.he_uniform'])
class HeUniform(tf.compat.v1.variance_scaling_initializer):

  def __init__(self, seed=None):
    super(HeUniform, self).__init__(
        scale=2., mode='fan_in', distribution='uniform', seed=seed)

  def get_config(self):
    return {'seed': self.seed}
