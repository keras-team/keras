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
"""Tests for DTensor based strategy training."""

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import backend
from keras import mixed_precision
from keras.dtensor import integration_test_utils
from keras.optimizers import adam
from keras.utils import tf_utils

# isort: off
# Import the MirroredStrategy that is backed by DTensor
# It is not a public API yet, so we do a private symbol import for now.
from tensorflow.python.distribute.experimental import (
    mirrored_strategy as dtensor_mirrored_strategy,
)
from tensorflow.dtensor.python.tests import test_util


class TrainingTest(test_util.DTensorBaseTest):
    def setUp(self):
        super().setUp()
        backend.enable_tf_random_generator()
        tf_utils.set_random_seed(1337)
        global_ids = test_util.create_device_ids_array((2,))
        local_device_ids = np.ravel(global_ids).tolist()
        mesh_dict = {
            device: tf.experimental.dtensor.Mesh(
                ["batch"],
                global_ids,
                local_device_ids,
                test_util.create_device_list((2,), device),
            )
            for device in ("CPU", "GPU", "TPU")
        }
        self.mesh = self.configTestMesh(mesh_dict)

    def tearDown(self):
        super().tearDown()
        # clean up the mixed precision setting if any.
        mixed_precision.set_global_policy("float32")

    @parameterized.product(
        run_eagerly=[True, False],
        jit_compile=[True, False],
        optimizer_creator=[lambda: adam.Adam(), lambda: "adam"],
        enable_mixed_precision=[True, False],
    )
    def test_model_fit(
        self,
        run_eagerly,
        jit_compile,
        optimizer_creator,
        enable_mixed_precision,
    ):
        if run_eagerly and jit_compile:
            self.skipTest("run_eagerly can't run with jit_compile")
        if enable_mixed_precision and self.mesh.device_type() != "GPU":
            self.skipTest("Only run mixed_precision on GPU for performance")

        if enable_mixed_precision:
            mixed_precision.set_global_policy("mixed_float16")
        dtensor_strategy = dtensor_mirrored_strategy.MirroredStrategy(
            mesh=self.mesh
        )
        # Make fake MNIST-like image data.
        batch_size = 64
        dataset = tf.data.Dataset.from_tensor_slices(
            (
                np.random.uniform(size=(batch_size, 28, 28, 1)).astype(
                    np.float32
                ),
                np.random.randint(0, 10, size=(batch_size,)),
            )
        )
        dataset = dataset.shuffle(64).repeat().batch(64, drop_remainder=True)

        with dtensor_strategy.scope():
            model = integration_test_utils.get_model()
            optimizer = optimizer_creator()

        model.compile(
            loss="SparseCategoricalCrossentropy",
            optimizer=optimizer,
            metrics="acc",
            run_eagerly=run_eagerly,
            jit_compile=jit_compile,
        )
        model.fit(dataset, steps_per_epoch=10)

        prediction = model.predict(
            np.random.uniform(size=(batch_size, 28, 28, 1)).astype(np.float32)
        )
        self.assertEqual(prediction.shape, (batch_size, 10))
        if enable_mixed_precision:
            self.assertEqual(prediction.dtype, tf.float16)
        else:
            self.assertEqual(prediction.dtype, tf.float32)


if __name__ == "__main__":
    tf.test.main()
