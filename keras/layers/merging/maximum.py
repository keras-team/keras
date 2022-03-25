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
"""Layer that computes the maximum (element-wise) of several inputs."""
# pylint: disable=g-direct-tensorflow-import

from keras.layers.merging.base_merge import _Merge
import tensorflow.compat.v2 as tf

from tensorflow.python.util.tf_export import keras_export


@keras_export('keras.layers.Maximum')
class Maximum(_Merge):
  """Layer that computes the maximum (element-wise) a list of inputs.

  It takes as input a list of tensors, all of the same shape, and returns
  a single tensor (also of the same shape).

  >>> tf.keras.layers.Maximum()([np.arange(5).reshape(5, 1),
  ...                            np.arange(5, 10).reshape(5, 1)])
  <tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[5],
       [6],
       [7],
       [8],
       [9]])>

  >>> x1 = tf.keras.layers.Dense(8)(np.arange(10).reshape(5, 2))
  >>> x2 = tf.keras.layers.Dense(8)(np.arange(10, 20).reshape(5, 2))
  >>> maxed = tf.keras.layers.Maximum()([x1, x2])
  >>> maxed.shape
  TensorShape([5, 8])
  """

  def _merge_function(self, inputs):
    output = inputs[0]
    for i in range(1, len(inputs)):
      output = tf.maximum(output, inputs[i])
    return output


@keras_export('keras.layers.maximum')
def maximum(inputs, **kwargs):
  """Functional interface to compute maximum (element-wise) list of `inputs`.

  This is equivalent to the `tf.keras.layers.Maximum` layer.

  For example:

  ```python
  input1 = tf.keras.layers.Input(shape=(16,))
  x1 = tf.keras.layers.Dense(8, activation='relu')(input1) #shape=(None, 8)
  input2 = tf.keras.layers.Input(shape=(32,))
  x2 = tf.keras.layers.Dense(8, activation='relu')(input2) #shape=(None, 8)
  max_inp=tf.keras.layers.maximum([x1,x2]) #shape=(None, 8)
  out = tf.keras.layers.Dense(4)(max_inp)
  model = tf.keras.models.Model(inputs=[input1, input2], outputs=out)
  ```

  Args:
      inputs: A list of input tensors of same shape.
      **kwargs: Standard layer keyword arguments.

  Returns:
      A tensor (of same shape as input tensor) with the element-wise
      maximum of the inputs.

  Raises:
      ValueError: If input tensors are of different shape.
  """
  return Maximum(**kwargs)(inputs)
