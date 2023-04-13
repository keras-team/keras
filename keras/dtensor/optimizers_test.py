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

import os

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

from keras import backend
from keras import layers
from keras import losses
from keras import models
from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import layout_map
from keras.dtensor import test_util
from keras.optimizers import adadelta
from keras.optimizers import adagrad
from keras.optimizers import adam
from keras.optimizers import adamw
from keras.optimizers import rmsprop
from keras.optimizers import sgd


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
        optimizer = adam.Adam(mesh=self.mesh)
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
        optimizer = adam.Adam(mesh=self.mesh)
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
            adadelta.Adadelta,
            {},
            [
                "Adadelta/accumulated_grad/Variable",
                "Adadelta/accumulated_delta_var/Variable",
                "iteration",
            ],
        ),
        (
            "Adam",
            adam.Adam,
            {"amsgrad": True},
            [
                "Adam/m/Variable",
                "Adam/v/Variable",
                "Adam/vhat/Variable",
                "iteration",
            ],
        ),
        (
            "AdamW",
            adamw.AdamW,
            {"amsgrad": True},
            [
                "AdamW/m/Variable",
                "AdamW/v/Variable",
                "AdamW/vhat/Variable",
                "iteration",
            ],
        ),
        (
            "Adagrad",
            adagrad.Adagrad,
            {},
            ["Adagrad/accumulator/Variable", "iteration"],
        ),
        (
            "RMSprop",
            rmsprop.RMSprop,
            {"momentum": 0.1, "centered": True},
            [
                "RMSprop/velocity/Variable",
                "RMSprop/momentum/Variable",
                "RMSprop/average_gradient/Variable",
                "iteration",
            ],
        ),
        (
            "SGD",
            sgd.SGD,
            {"momentum": 0.1},
            ["SGD/m/Variable", "iteration"],
        ),
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
        self.assertCountEqual(all_names, expect_variable_names)

    def test_embedding_lookup_backward_path(self):
        # See b/265441685 for more context.
        backend.enable_tf_random_generator()
        os.environ[
            "DTENSOR_ENABLE_REPLICATED_SPMD_AS_DEFAULT_TF.RESOURCESCATTERADD"
        ] = "1"
        # Build a small functional model with embedding layer, it contains
        # tf.gather ops which will trigger the _deduplicate_sparse_grad() code
        # path. tf.unique op will have a shape mismatch issue for dtensor.
        batch_size = 16
        seq_length = 10
        vocab_size = 100
        output_size = 8

        def produce_data():
            inputs = tf.random.uniform(
                maxval=vocab_size,
                shape=(batch_size, seq_length),
                dtype=tf.int32,
            )
            label = tf.random.uniform(
                maxval=output_size, shape=(batch_size,), dtype=tf.int32
            )
            inputs = dtensor.copy_to_mesh(
                inputs, layout=dtensor.Layout.replicated(self.mesh, rank=2)
            )
            inputs = dtensor.relayout(
                inputs, dtensor.Layout.batch_sharded(self.mesh, "X", 2)
            )
            label = dtensor.copy_to_mesh(
                label, layout=dtensor.Layout.replicated(self.mesh, rank=1)
            )
            label = dtensor.relayout(
                label, dtensor.Layout.batch_sharded(self.mesh, "X", 1)
            )
            return inputs, label

        with layout_map.LayoutMap(self.mesh).scope():
            inputs = layers.Input(shape=(seq_length,))
            x = layers.Embedding(vocab_size, 64)(inputs)
            x = layers.GlobalAveragePooling1D()(x)
            preds = layers.Dense(output_size, activation="softmax")(x)
            model = models.Model(inputs, preds)

        optimizer = adam.Adam(mesh=self.mesh)

        @tf.function
        def train_func(model, inputs, label, optimizer):
            with tf.GradientTape() as tape:
                output = model(inputs)
                loss = losses.sparse_categorical_crossentropy(label, output)
            optimizer.minimize(loss, model.variables, tape)
            return loss

        # The error only happens across the batch, where the value of
        # tf.unique are different.
        input1, label1 = produce_data()
        train_func(model, input1, label1, optimizer)
        input2, label2 = produce_data()
        train_func(model, input2, label2, optimizer)
        # Assert nothing here, and expect the train_func can run properly with
        # different inputs.


if __name__ == "__main__":
    tf.test.main()
