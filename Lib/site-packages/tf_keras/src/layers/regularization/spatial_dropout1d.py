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
"""Contains the SpatialDropout1D layer."""


import tensorflow.compat.v2 as tf

from tf_keras.src.engine.input_spec import InputSpec
from tf_keras.src.layers.regularization.dropout import Dropout

# isort: off
from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.SpatialDropout1D")
class SpatialDropout1D(Dropout):
    """Spatial 1D version of Dropout.

    This version performs the same function as Dropout, however, it drops
    entire 1D feature maps instead of individual elements. If adjacent frames
    within feature maps are strongly correlated (as is normally the case in
    early convolution layers) then regular dropout will not regularize the
    activations and will otherwise just result in an effective learning rate
    decrease. In this case, SpatialDropout1D will help promote independence
    between feature maps and should be used instead.

    Args:
      rate: Float between 0 and 1. Fraction of the input units to drop.
    Call arguments:
      inputs: A 3D tensor.
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    Input shape:
      3D tensor with shape: `(samples, timesteps, channels)`
    Output shape: Same as input.
    References: - [Efficient Object Localization Using Convolutional
        Networks](https://arxiv.org/abs/1411.4280)
    """

    def __init__(self, rate, **kwargs):
        super().__init__(rate, **kwargs)
        self.input_spec = InputSpec(ndim=3)

    def _get_noise_shape(self, inputs):
        input_shape = tf.shape(inputs)
        noise_shape = (input_shape[0], 1, input_shape[2])
        return noise_shape

