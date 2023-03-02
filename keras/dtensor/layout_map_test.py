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
"""Tests for layout_map."""

import os
import shutil

import numpy as np
import tensorflow.compat.v2 as tf

from keras import backend
from keras import layers
from keras import models
from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import layout_map as layout_map_lib
from keras.dtensor import test_util
from keras.utils import tf_utils


class LayoutMapTest(test_util.DTensorBaseTest):
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
        self.layout_2d = dtensor.Layout.replicated(self.mesh, rank=2)
        self.layout_1d = dtensor.Layout.replicated(self.mesh, rank=1)

        self.sharded_2d = dtensor.Layout.batch_sharded(self.mesh, "X", rank=2)
        self.sharded_1d = dtensor.Layout.batch_sharded(self.mesh, "X", rank=1)

    def test_add(self):
        layout_map = layout_map_lib.LayoutMap()

        layout_map["dense/kernel"] = self.layout_2d
        layout_map["dense/bias"] = self.layout_1d

        # Make there are two items in the map, and we access them via the
        # underlying container at layout_map._layout_map
        self.assertLen(layout_map._layout_map, 2)
        self.assertEqual(layout_map._layout_map["dense/kernel"], self.layout_2d)
        self.assertEqual(layout_map._layout_map["dense/bias"], self.layout_1d)

        with self.assertRaisesRegex(ValueError, "dense/kernel already exist"):
            layout_map["dense/kernel"] = self.layout_1d

        with self.assertRaisesRegex(ValueError, "should be a dtensor.Layout"):
            layout_map["conv.kernel"] = [1, 2, 3]

    def test_get(self):
        layout_map = layout_map_lib.LayoutMap()

        layout_map["dense/kernel"] = self.sharded_2d
        layout_map["dense/bias"] = self.sharded_1d

        layout_map["dense.*kernel"] = self.layout_2d
        layout_map["dense.*bias"] = self.layout_1d

        layout_map[".*bias"] = self.sharded_1d

        self.assertEqual(layout_map["dense/kernel"], self.sharded_2d)
        self.assertEqual(layout_map["dense/bias"], self.sharded_1d)

        # Map against the wildcard bias rule for dense, and based on the order
        # of insertion, it will not use .*bias.
        self.assertEqual(layout_map["dense_2/kernel"], self.layout_2d)
        self.assertEqual(layout_map["dense_2/bias"], self.layout_1d)

        self.assertIsNone(layout_map["conv2d/kernel"])
        self.assertEqual(layout_map["conv2d/bias"], self.sharded_1d)

    def test_delete(self):
        layout_map = layout_map_lib.LayoutMap()

        layout_map["dense/kernel"] = self.layout_2d
        layout_map["dense/bias"] = self.layout_1d

        self.assertEqual(layout_map.pop("dense/kernel"), self.layout_2d)
        # Make sure to match against the exact string, not the regex
        with self.assertRaises(KeyError):
            layout_map.pop(".*bias")

        # Make sure del also works
        del layout_map["dense/bias"]

        self.assertEmpty(layout_map._layout_map)

    def test_len(self):
        layout_map = layout_map_lib.LayoutMap()
        self.assertEmpty(layout_map)

        layout_map["dense/kernel"] = self.layout_2d
        layout_map["dense/bias"] = self.layout_1d

        self.assertLen(layout_map, 2)

    def test_iter(self):
        layout_map = layout_map_lib.LayoutMap()

        layout_map["dense/kernel"] = self.layout_2d
        layout_map["dense/bias"] = self.layout_1d

        # Make sure the items are ordered based on the insertion order.
        self.assertEqual(
            list(layout_map.keys()), ["dense/kernel", "dense/bias"]
        )

        keys = []
        values = []
        for k, v in layout_map.items():
            keys.append(k)
            values.append(v)

        self.assertEqual(keys, ["dense/kernel", "dense/bias"])
        self.assertEqual(values, [self.layout_2d, self.layout_1d])


# Class used for testing.
class SubclassModel(models.Model):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.d1 = layers.Dense(1000)
        self.d2 = layers.Dense(1000)
        self.dropout = layers.Dropout(0.1)

    def call(self, inputs, training=None):
        x = self.d1(inputs)
        x = self.dropout(x, training=training)
        return self.d2(x)


class SubclassLayer(layers.Layer):
    def __init__(self, unit):
        super().__init__()
        self.unit = unit

    def build(self, input_shape):
        weight_shape = (input_shape[-1], self.unit)
        # Note that the variable name is "kernel", but assigned to "_weight"
        # This will cause the checkpoint to record 2 dependencies.
        self._weight = self.add_weight(shape=weight_shape, name="kernel")

    def call(self, inputs):
        return tf.matmul(inputs, self._weight)


class ObjectPathMappingTest(test_util.DTensorBaseTest):
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
        self.layout_2d = dtensor.Layout.replicated(self.mesh, rank=2)
        self.layout_1d = dtensor.Layout.replicated(self.mesh, rank=1)

        self.sharded_2d = dtensor.Layout.batch_sharded(self.mesh, "X", rank=2)
        self.sharded_1d = dtensor.Layout.batch_sharded(self.mesh, "X", rank=1)

    def test_init_subclass_model_variable_with_layout(self):
        layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
        layout_map["d1.kernel"] = self.layout_2d
        layout_map["d1.bias"] = self.layout_1d
        layout_map["d2.kernel"] = self.layout_2d
        layout_map["d2.bias"] = self.layout_1d

        with layout_map.scope():
            model = SubclassModel(name="model")

        # Init the model with eager tensor, make sure the model weights have
        # correct layout, as well as produce correct result.
        inputs = tf.zeros((10, 10))
        inputs = dtensor.copy_to_mesh(inputs, layout=self.layout_2d)
        result = model(inputs)
        self.assertAllClose(result, tf.zeros((10, 1000)))
        d1 = model.d1
        d2 = model.d2
        self.assertEqual(d1.kernel.layout, self.layout_2d)
        self.assertEqual(d1.bias.layout, self.layout_1d)
        self.assertEqual(d2.kernel.layout, self.layout_2d)
        self.assertEqual(d2.bias.layout, self.layout_1d)

        # Also make sure we repopulate the cached attributes like
        # layer._trainable_weights
        self.assertIs(d1.kernel, d1._trainable_weights[0])
        self.assertIs(d1.bias, d1._trainable_weights[1])
        self.assertIs(d2.kernel, d2._trainable_weights[0])
        self.assertIs(d2.bias, d2._trainable_weights[1])

        result = model(inputs, training=True)
        self.assertAllClose(
            result,
            tf.experimental.dtensor.copy_to_mesh(
                tf.zeros((10, 1000)), self.layout_2d
            ),
        )

    def test_init_functional_model_variable_with_layout(self):
        # Note that the functional model is using layers name + attribute name
        # the layer name are unique among the functional model, and when the
        # layer doesn't have a name, keras will give it a unique name based on
        # the layer class.
        layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
        layout_map["d1.kernel"] = self.layout_2d
        layout_map["d1.bias"] = self.layout_1d
        layout_map["d2.kernel"] = self.layout_2d
        layout_map["d2.bias"] = self.layout_1d

        with layout_map.scope():
            inputs = layers.Input((10,), batch_size=10)
            x = layers.Dense(20, name="d1")(inputs)
            x = layers.Dropout(0.1)(x)
            output = layers.Dense(30, name="d2")(x)

            model = models.Model(inputs, output)

        # It includes input layer as well.
        self.assertLen(model.layers, 4)
        d1 = model.layers[1]
        d2 = model.layers[3]

        self.assertEqual(d1.kernel.layout, self.layout_2d)
        self.assertEqual(d1.bias.layout, self.layout_1d)
        self.assertEqual(d2.kernel.layout, self.layout_2d)
        self.assertEqual(d2.bias.layout, self.layout_1d)

        # Also make sure we repopulate the cached attributes like
        # layer._trainable_weights
        self.assertIs(d1.kernel, d1._trainable_weights[0])
        self.assertIs(d1.bias, d1._trainable_weights[1])
        self.assertIs(d2.kernel, d2._trainable_weights[0])
        self.assertIs(d2.bias, d2._trainable_weights[1])

        inputs = tf.zeros((10, 10))
        inputs = dtensor.copy_to_mesh(inputs, layout=self.layout_2d)
        result = model(inputs, training=True)
        expected_result = tf.zeros((10, 30))
        expected_result = dtensor.copy_to_mesh(
            expected_result, layout=self.layout_2d
        )
        self.assertAllClose(result, expected_result)

    def test_init_sequential_model_variable_with_layout(self):
        # Note that the sequential model is using layers name + attribute name
        # the layer name are unique among the functional model, and when the
        # layer doesn't have a name, keras will give it a unique name based on
        # the layer class.
        layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
        layout_map["d1.kernel"] = self.layout_2d
        layout_map["d1.bias"] = self.layout_1d
        layout_map["d2.kernel"] = self.layout_2d
        layout_map["d2.bias"] = self.layout_1d

        with layout_map.scope():
            model = models.Sequential(
                [
                    layers.Dense(20, name="d1", input_shape=(10,)),
                    layers.Dropout(0.1),
                    layers.Dense(30, name="d2"),
                ]
            )

        self.assertLen(model.layers, 3)
        d1 = model.layers[0]
        d2 = model.layers[2]

        self.assertEqual(d1.kernel.layout, self.layout_2d)
        self.assertEqual(d1.bias.layout, self.layout_1d)
        self.assertEqual(d2.kernel.layout, self.layout_2d)
        self.assertEqual(d2.bias.layout, self.layout_1d)

        # Also make sure we repopulate the cached attributes like
        # layer._trainable_weights
        self.assertIs(d1.kernel, d1._trainable_weights[0])
        self.assertIs(d1.bias, d1._trainable_weights[1])
        self.assertIs(d2.kernel, d2._trainable_weights[0])
        self.assertIs(d2.bias, d2._trainable_weights[1])

        inputs = tf.zeros((10, 10))
        inputs = dtensor.copy_to_mesh(inputs, layout=self.layout_2d)
        result = model(inputs, training=True)
        expected_result = tf.zeros((10, 30))
        expected_result = dtensor.copy_to_mesh(
            expected_result, layout=self.layout_2d
        )
        self.assertAllClose(result, expected_result)

    def test_init_model_with_empty_layout_map(self):
        # Create empty layout map, which means all the weights just default to
        # all replicated.
        layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
        with layout_map.scope():
            model = models.Sequential(
                [
                    layers.Dense(20, name="d1", input_shape=(10,)),
                    layers.Dropout(0.1),
                    layers.Dense(30, name="d2"),
                ]
            )

        self.assertLen(model.layers, 3)
        d1 = model.layers[0]
        d2 = model.layers[2]

        self.assertEqual(d1.kernel.layout, self.layout_2d)
        self.assertEqual(d1.bias.layout, self.layout_1d)
        self.assertEqual(d2.kernel.layout, self.layout_2d)
        self.assertEqual(d2.bias.layout, self.layout_1d)

    def test_weight_regularization(self):
        layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
        with layout_map_lib.layout_map_scope(layout_map):
            model = models.Sequential(
                [
                    layers.Dense(
                        20,
                        name="d1",
                        input_shape=(10,),
                        kernel_initializer="ones",
                        kernel_regularizer="l2",
                    ),
                    layers.Dropout(0.1),
                    layers.Dense(
                        30,
                        name="d2",
                        kernel_initializer="ones",
                        kernel_regularizer="l2",
                    ),
                ]
            )

        self.assertLen(model.losses, 2)
        # kernel shape [10, 20] with all "1", timed by 0.01 from l2
        self.assertAllClose(model.losses[0], 2.0)
        # kernel shape [20, 30] with all "1", timed by 0.01 from l2
        self.assertAllClose(model.losses[1], 6.0)

    def test_dvariable_name(self):
        layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
        with layout_map.scope():
            model = models.Sequential(
                [
                    layers.Dense(20, name="d1", input_shape=(10,)),
                    layers.Dropout(0.1),
                    layers.Dense(30, name="d2"),
                ]
            )

        self.assertLen(model.layers, 3)
        self.assertEqual(model.layers[0].kernel.name, "d1/kernel:0")
        self.assertEqual(model.layers[0].bias.name, "d1/bias:0")

    @tf.compat.v1.test.mock.patch.dict(
        "os.environ", {"DTENSOR_ENABLE_CHECKPOINT_V2": "True"}
    )
    def test_checkpoint(self):
        layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
        with layout_map.scope():
            model = models.Sequential(
                [
                    layers.Dense(20, name="d1", input_shape=(10,)),
                    SubclassLayer(10),
                ]
            )
        cpt = tf.train.Checkpoint(root=model)
        options = tf.train.CheckpointOptions(
            experimental_io_device=dtensor.device_name()
        )
        tmpdir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, tmpdir, ignore_errors=True)

        saved_path = cpt.save(
            os.path.join(tmpdir, "checkpoint"),
            options=options,
        )

        cpt.restore(saved_path, options=options)


if __name__ == "__main__":
    tf.test.main()
