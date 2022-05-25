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
"""Tests for utils."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import layers
from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import test_util
from keras.dtensor import utils


class UtilsTest(test_util.DTensorBaseTest):
    def setUp(self):
        super().setUp()
        global_ids = test_util.create_device_ids_array((2, 2))
        local_device_ids = np.ravel(global_ids).tolist()
        mesh_dict = {
            "CPU": dtensor.Mesh(
                ["X", "Y"],
                global_ids,
                local_device_ids,
                test_util.create_device_list((2, 2), "CPU"),
            )
        }
        self.mesh = self.configTestMesh(mesh_dict)
        self.layout = dtensor.Layout.replicated(self.mesh, rank=1)

    @parameterized.named_parameters(
        ("Dense", layers.Dense, {"units": 4}, ["kernel_layout", "bias_layout"]),
        (
            "Conv2D",
            layers.Conv2D,
            {"filters": 2, "kernel_size": 3},
            ["kernel_layout", "bias_layout"],
        ),
        (
            "BatchNorm",
            layers.BatchNormalization,
            {},
            [
                "beta_layout",
                "gamma_layout",
                "moving_mean_layout",
                "moving_variance_layout",
            ],
        ),
        (
            "Embedding",
            layers.Embedding,
            {"input_dim": 100, "output_dim": 20},
            ["embeddings_layout"],
        ),
        (" PReLU", layers.PReLU, {}, ["alpha_layout"]),
        (
            "SeparableConv2D",
            layers.SeparableConv2D,
            {"filters": 2, "kernel_size": 3},
            ["depthwise_layout", "pointwise_layout", "bias_layout"],
        ),
        # TODO(scottzhu): Probably add more coverage for all the layers.
    )
    def test_all_layout_decorator(self, layer_cls, init_args, layout_args):

        layer_cls.__init__ = utils.allow_initializer_layout(layer_cls.__init__)

        # Make sure we don't set the layout attribute if the init kwargs is not
        # provided.
        layer = layer_cls(**init_args)
        for layout_arg in layout_args:
            self.assertFalse(hasattr(layer, layout_arg))

        layout_kwargs = {k: self.layout for k in layout_args}
        init_args.update(layout_kwargs)
        layer = layer_cls(**init_args)

        for layout_arg in layout_args:
            self.assertEqual(getattr(layer, layout_arg), self.layout)


if __name__ == "__main__":
    tf.test.main()
