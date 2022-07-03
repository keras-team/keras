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
"""Private base class for generalized pooling 1D layers."""


from keras import backend
from keras.engine.base_layer import Layer
from keras.engine.input_spec import InputSpec
from keras.utils import conv_utils


class GeneralizedPooling3D(Layer):
    """Abstract class for different generalized mean pooling 3D layers."""

    def __init__(
        self,
        power,
        pool_size,
        strides,
        padding="valid",
        data_format="channels_last",
        name=None,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)

        if power <= 0:
            raise ValueError(
                "The value of `power` in GeneralizedMeanPooling must "
                f"be positive number. Got: {power}"
            )

        if data_format is None:
            data_format = backend.image_data_format()

        if strides is None:
            strides = pool_size

        self.power = power
        self.pool_size = conv_utils.normalize_tuple(pool_size, 3, "pool_size")
        self.strides = conv_utils.normalize_tuple(
            strides, 3, "strides", allow_zero=True
        )
        self.padding = conv_utils.normalize_padding(padding).upper()
        self.data_format = conv_utils.convert_data_format(self.data_format, 5)
        self.input_spec = InputSpec(ndim=5)

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = {
            "power": self.power,
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
            "data_format": self.data_format,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
