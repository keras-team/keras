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
"""Tests for layers."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import backend
from keras import layers
from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import test_util
from keras.utils import tf_utils


class LayersTest(test_util.DTensorBaseTest):
    def setUp(self):
        super().setUp()
        backend.enable_tf_random_generator()
        tf_utils.set_random_seed(1337)
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

    @parameterized.named_parameters(
        (
            "dense",
            layers.Dense,
            {"units": 4},
            {"kernel": 2, "bias": 1},
            [10, 8],
        ),
        # TODO(b/224861663): Enable this test.
        # ('embedding', layers.Embedding, {'input_dim': 100, 'output_dim': 32},
        #  {'embeddings': 2}, [10,], np.int32),
        (
            "conv1d",
            layers.Conv1D,
            {"filters": 4, "kernel_size": 3},
            {"kernel": 3, "bias": 1},
            [10, 28, 3],
        ),
        (
            "conv1d_transpose",
            layers.Conv1DTranspose,
            {"filters": 4, "kernel_size": 3},
            {"kernel": 3, "bias": 1},
            [10, 28, 3],
        ),
        (
            "conv2d",
            layers.Conv2D,
            {"filters": 4, "kernel_size": (3, 3)},
            {"kernel": 4, "bias": 1},
            [10, 28, 28, 3],
        ),
        (
            "conv2d_transpose",
            layers.Conv2DTranspose,
            {"filters": 4, "kernel_size": (3, 3)},
            {"kernel": 4, "bias": 1},
            [10, 28, 28, 3],
        ),
        (
            "conv3d",
            layers.Conv3D,
            {"filters": 4, "kernel_size": (3, 3, 3)},
            {"kernel": 5, "bias": 1},
            [10, 28, 28, 28, 3],
        ),
        # TODO(b/224862394): Add support for tf.Conv3DBackpropInputV2
        # ('conv3dtranspose', layers.Conv3DTranspose,
        #  {'filters': 4, 'kernel_size': (3, 3, 3)},
        #  {'kernel': 5, 'bias': 1}, [10, 28, 28, 28, 3]),
        (
            "batch_norm",
            layers.BatchNormalization,
            {"fused": False},
            {"beta": 1, "gamma": 1, "moving_mean": 1, "moving_variance": 1},
            [10, 28, 28, 3],
        ),
        (
            "layer_norm",
            layers.LayerNormalization,
            {"dtype": tf.float64},
            {"beta": 1, "gamma": 1},
            [10, 28, 28, 3],
        ),
    )
    def test_layer(
        self,
        layer_cls,
        init_args,
        variable_settings,
        input_shape,
        input_dtype=np.float32,
    ):
        args_with_layout = init_args.copy()
        for variable_name, variable_rank in variable_settings.items():
            args_with_layout[
                variable_name + "_layout"
            ] = dtensor.Layout.replicated(self.mesh, variable_rank)

        layer = layer_cls(**args_with_layout)
        # inputs = np.random.random(input_shape)
        inputs = np.random.randn(*input_shape).astype(input_dtype)
        d_inputs = dtensor.copy_to_mesh(
            inputs, dtensor.Layout.replicated(self.mesh, len(input_shape))
        )
        d_output = layer(d_inputs)

        for variable_name, variable_rank in variable_settings.items():
            self.assertIsInstance(
                getattr(layer, variable_name), dtensor.DVariable
            )

        expected_layout = dtensor.Layout.replicated(
            self.mesh, d_output.shape.rank
        )
        self.assertEqual(dtensor.fetch_layout(d_output), expected_layout)

        # Make sure to produce same output when layout is not used
        tf_utils.set_random_seed(1337)
        layer_2 = layer_cls(**init_args)
        output = layer_2(inputs)
        self.assertAllClose(d_output, output)

        for variable_name, variable_rank in variable_settings.items():
            self.assertNotIsInstance(
                getattr(layer_2, variable_name), dtensor.DVariable
            )


if __name__ == "__main__":
    tf.test.main()
