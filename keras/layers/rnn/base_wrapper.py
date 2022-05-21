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
"""Base class for wrapper layers.

Wrappers are layers that augment the functionality of another layer.
"""
# pylint: disable=g-classes-have-attributes,g-direct-tensorflow-import

import copy

from keras.engine.base_layer import Layer
from keras.utils import generic_utils

from tensorflow.python.util.tf_export import keras_export


@keras_export("keras.layers.Wrapper")
class Wrapper(Layer):
    """Abstract wrapper base class.

    Wrappers take another layer and augment it in various ways.
    Do not use this class as a layer, it is only an abstract base class.
    Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

    Args:
      layer: The layer to be wrapped.
    """

    def __init__(self, layer, **kwargs):
        assert isinstance(layer, Layer)
        self.layer = layer
        super().__init__(**kwargs)

    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.built = True

    @property
    def activity_regularizer(self):
        if hasattr(self.layer, "activity_regularizer"):
            return self.layer.activity_regularizer
        else:
            return None

    def get_config(self):
        config = {"layer": generic_utils.serialize_keras_object(self.layer)}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        from keras.layers import (
            deserialize as deserialize_layer,
        )  # pylint: disable=g-import-not-at-top

        # Avoid mutating the input dict
        config = copy.deepcopy(config)
        layer = deserialize_layer(
            config.pop("layer"), custom_objects=custom_objects
        )
        return cls(layer, **config)
