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

from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import optimizers
from keras.dtensor import test_util


class OptimizersTest(test_util.DTensorBaseTest):
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

    def test_add_variable_from_reference(self):
        optimizer = optimizers.Adam(mesh=self.mesh)
        variable_init_value = tf.ones([4, 4], dtype=tf.float32)
        variable_init_value = dtensor.copy_to_mesh(
            variable_init_value,
            layout=dtensor.Layout.replicated(self.mesh, rank=2),
        )
        model_variable = dtensor.DVariable(
            variable_init_value, trainable=True, name="tmp"
        )
        state_variable = optimizer.add_variable_from_reference(
            model_variable, "test"
        )
        self.assertEqual(state_variable._shared_name, "test/tmp")
        self.assertAllClose(self.evaluate(state_variable), tf.zeros([4, 4]))
        # Make sure the variable contains the correct layout info
        self.assertEqual(state_variable.layout, model_variable.layout)

    def test_build_index_dict(self):
        optimizer = optimizers.Adam(mesh=self.mesh)
        variable_init_value = tf.ones(shape=(), dtype=tf.float32)
        variable_init_value = dtensor.copy_to_mesh(
            variable_init_value,
            layout=dtensor.Layout.replicated(self.mesh, rank=0),
        )
        var_list = [
            dtensor.DVariable(variable_init_value, name=f"var{i}")
            for i in range(10)
        ]
        optimizer._build_index_dict(var_list)
        self.assertEqual(
            optimizer._index_dict[optimizer._var_key(var_list[7])], 7
        )

    @parameterized.named_parameters(
        (
            "Adadelta",
            optimizers.Adadelta,
            {},
            [
                "Adadelta/accumulated_grad/Variable",
                "Adadelta/accumulated_delta_var/Variable",
            ],
        ),
        (
            "Adam",
            optimizers.Adam,
            {"amsgrad": True},
            ["Adam/m/Variable", "Adam/v/Variable", "Adam/vhat/Variable"],
        ),
        (
            "AdamW",
            optimizers.AdamW,
            {"amsgrad": True},
            ["AdamW/m/Variable", "AdamW/v/Variable", "AdamW/vhat/Variable"],
        ),
        ("Adagrad", optimizers.Adagrad, {}, ["Adagrad/accumulator/Variable"]),
        (
            "RMSprop",
            optimizers.RMSprop,
            {"momentum": 0.1, "centered": True},
            [
                "RMSprop/velocity/Variable",
                "RMSprop/momentum/Variable",
                "RMSprop/average_gradient/Variable",
            ],
        ),
        ("SGD", optimizers.SGD, {"momentum": 0.1}, ["SGD/m/Variable"]),
    )
    def test_apply_gradients(
        self, optimizer_cls, init_args, expect_variable_names
    ):
        optimizer = optimizer_cls(mesh=self.mesh, **init_args)

        self.assertEqual(self.evaluate(optimizer.iterations), 0)
        self.assertEqual(
            optimizer.iterations.layout,
            dtensor.Layout.replicated(self.mesh, rank=0),
        )

        variable_init_value = tf.ones([4, 4], dtype=tf.float32)
        variable_init_value = dtensor.copy_to_mesh(
            variable_init_value,
            layout=dtensor.Layout.replicated(self.mesh, rank=2),
        )
        model_variable = dtensor.DVariable(variable_init_value, trainable=True)

        grads = tf.ones_like(variable_init_value)
        optimizer.apply_gradients(zip([grads], [model_variable]))
        optimizer_variables = optimizer.variables

        self.assertEqual(self.evaluate(optimizer.iterations), 1)

        all_names = [var._shared_name for var in optimizer_variables]
        expect_variable_names.extend(["iteration", "learning_rate"])
        self.assertCountEqual(all_names, expect_variable_names)


if __name__ == "__main__":
    tf.test.main()
