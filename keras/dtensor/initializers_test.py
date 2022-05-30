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
"""Tests for initializers."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import backend
from keras import initializers
from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import test_util
from keras.utils import tf_utils


class InitializersTest(test_util.DTensorBaseTest):
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

    @parameterized.named_parameters(
        ("Zeros", initializers.Zeros, {}),
        ("Ones", initializers.Ones, {}),
        ("Constant", initializers.Constant, {"value": 3.0}),
        # TODO(b/222160686): Add Identity after after we have SPMD support for
        # tf.MatrixDiagV3
        # ('Identity', initializers.Identity, {}),
    )
    def test_static_value_initializer(self, initializer_cls, init_args):
        layout = dtensor.Layout(
            [dtensor.UNSHARDED, dtensor.UNSHARDED], self.mesh
        )
        shape = (4, 4)
        initializer = initializer_cls(**init_args)
        value = initializer(shape=shape, layout=layout)
        normal_tensor_value = initializer(shape=shape)

        self.assertEqual(value.shape, shape)
        fetched_layout = dtensor.fetch_layout(value)
        self.assertEqual(layout, fetched_layout)

        self.assertAllClose(value, normal_tensor_value)

    @parameterized.named_parameters(
        ("RandomUniform", initializers.RandomUniform, {}),
        ("RandomUniform_seeded", initializers.RandomUniform, {"seed": 1}),
        ("RandomNormal", initializers.RandomNormal, {}),
        ("RandomNormal_seeded", initializers.RandomNormal, {"seed": 1}),
        ("TruncatedNormal", initializers.TruncatedNormal, {}),
        ("TruncatedNormal_seeded", initializers.TruncatedNormal, {"seed": 1}),
        ("Orthogonal", initializers.Orthogonal, {}),
        ("Orthogonal_seeded", initializers.Orthogonal, {"seed": 1}),
        ("VarianceScaling", initializers.VarianceScaling, {}),
        ("VarianceScaling_seeded", initializers.VarianceScaling, {"seed": 1}),
        ("GlorotUniform", initializers.GlorotUniform, {}),
        ("GlorotUniform_seeded", initializers.GlorotUniform, {"seed": 1}),
        ("GlorotNormal", initializers.GlorotNormal, {}),
        ("GlorotNormal_seeded", initializers.GlorotNormal, {"seed": 1}),
        ("LecunNormal", initializers.LecunNormal, {}),
        ("LecunNormal_seeded", initializers.LecunNormal, {"seed": 1}),
        ("LecunUniform", initializers.LecunUniform, {}),
        ("LecunUniform_seeded", initializers.LecunUniform, {"seed": 1}),
        ("HeNormal", initializers.HeNormal, {}),
        ("HeNormal_seeded", initializers.HeNormal, {"seed": 1}),
        ("HeUniform", initializers.HeUniform, {}),
        ("HeUniform_seeded", initializers.HeUniform, {"seed": 1}),
    )
    def test_random_value_initializer(self, initializer_cls, init_args):
        layout = dtensor.Layout(
            [dtensor.UNSHARDED, dtensor.UNSHARDED], self.mesh
        )
        shape = (4, 4)
        initializer = initializer_cls(**init_args)
        # Make sure to raise error when keras global seed is not set.
        with self.assertRaisesRegex(ValueError, "set the global seed"):
            initializer(shape=shape, layout=layout)

        try:
            tf_utils.set_random_seed(1337)
            value = initializer(shape=shape, layout=layout)
            self.assertEqual(value.shape, shape)
            fetched_layout = dtensor.fetch_layout(value)
            self.assertEqual(layout, fetched_layout)

            # Make sure when same seed is set again, the new initializer should
            # generate same result
            tf_utils.set_random_seed(1337)
            initializer = initializer_cls(**init_args)
            new_value = initializer(shape=shape, layout=layout)
            self.assertAllClose(value, new_value)
        finally:
            # Unset the keras global generator so that it doesn't affect other
            # tests that need to verify the existence of global generator.
            backend._SEED_GENERATOR.generator = None

    @parameterized.named_parameters(
        ("zeros", "zeros", initializers.Zeros),
        ("Zeros", "Zeros", initializers.Zeros),
        ("ones", "ones", initializers.Ones),
        ("Ones", "Ones", initializers.Ones),
        ("constant", "constant", initializers.Constant),
        ("Constant", "Constant", initializers.Constant),
        ("random_uniform", "random_uniform", initializers.RandomUniform),
        ("RandomUniform", "RandomUniform", initializers.RandomUniform),
        ("random_normal", "random_normal", initializers.RandomNormal),
        ("RandomNormal", "RandomNormal", initializers.RandomNormal),
        ("truncated_normal", "truncated_normal", initializers.TruncatedNormal),
        ("TruncatedNormal", "TruncatedNormal", initializers.TruncatedNormal),
        ("Identity", "Identity", initializers.Identity),
        ("identity", "identity", initializers.Identity),
        ("Orthogonal", "Orthogonal", initializers.Orthogonal),
        ("orthogonal", "orthogonal", initializers.Orthogonal),
        ("variance_scaling", "variance_scaling", initializers.VarianceScaling),
        ("VarianceScaling", "VarianceScaling", initializers.VarianceScaling),
        ("glorot_uniform", "glorot_uniform", initializers.GlorotUniform),
        ("GlorotUniform", "GlorotUniform", initializers.GlorotUniform),
        ("glorot_normal", "glorot_normal", initializers.GlorotNormal),
        ("GlorotNormal", "GlorotNormal", initializers.GlorotNormal),
        ("lecun_normal", "lecun_normal", initializers.LecunNormal),
        ("LecunNormal", "LecunNormal", initializers.LecunNormal),
        ("lecun_uniform", "lecun_uniform", initializers.LecunUniform),
        ("LecunUniform", "LecunUniform", initializers.LecunUniform),
        ("he_normal", "he_normal", initializers.HeNormal),
        ("HeNormal", "HeNormal", initializers.HeNormal),
        ("he_uniform", "he_uniform", initializers.HeUniform),
        ("HeUniform", "HeUniform", initializers.HeUniform),
    )
    def test_serialization_deserialization(self, cls_name, expected_cls):
        initializer = initializers.get(cls_name)
        self.assertIsInstance(initializer, expected_cls)

        config = initializers.serialize(initializer)
        recreated = initializers.deserialize(config)

        self.assertIsInstance(recreated, expected_cls)
        self.assertEqual(config, initializers.serialize(recreated))


if __name__ == "__main__":
    tf.test.main()
