# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for layer_utils."""

import collections
import contextlib
import io
import multiprocessing.dummy
import os
import pickle
import shutil
import sys
import tempfile
import time
import timeit

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras import backend
from keras import layers
from keras.dtensor import dtensor_api as dtensor
from keras.dtensor import layout_map as layout_map_lib
from keras.dtensor import test_util
from keras.testing_infra import test_utils
from keras.utils import io_utils
from keras.utils import layer_utils
from keras.utils import tf_utils

_PICKLEABLE_CALL_COUNT = collections.Counter()


class MyPickleableObject(tf.__internal__.tracking.AutoTrackable):
    """Needed for InterfaceTests.test_property_cache_serialization.

    This class must be at the top level. This is a constraint of pickle,
    unrelated to `cached_per_instance`.
    """

    @property
    @layer_utils.cached_per_instance
    def my_id(self):
        _PICKLEABLE_CALL_COUNT[self] += 1
        return id(self)


class LayerUtilsTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        # Reset the UID so that all the layer/model ID will always start with 1.
        # This will help remove the undetermined IDs from the model.summary()
        backend.reset_uids()

    def test_print_summary(self):
        model = keras.Sequential()
        model.add(
            keras.layers.Conv2D(
                filters=2,
                kernel_size=(2, 3),
                input_shape=(3, 5, 5),
                name="conv",
            )
        )
        model.add(keras.layers.Flatten(name="flat"))
        model.add(keras.layers.Dense(5, name="dense"))

        file_name = "model_1.txt"
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        fpath = os.path.join(temp_dir, file_name)
        writer = open(fpath, "w")

        def print_to_file(text):
            print(text, file=writer)

        try:
            layer_utils.print_summary(model, print_fn=print_to_file)
            self.assertTrue(tf.io.gfile.exists(fpath))
            writer.close()
            with open(fpath, "r") as reader:
                lines = reader.readlines()
            self.assertEqual(len(lines), 15)
        except ImportError:
            pass

    def test_print_summary_without_print_fn(self):
        model = keras.Sequential(
            [keras.layers.Dense(5, input_shape=(10,), name="dense")]
        )
        io_utils.enable_interactive_logging()
        with self.captureWritesToStream(sys.stdout) as printed:
            layer_utils.print_summary(model)
        self.assertIn("dense (Dense)", printed.contents())

    def test_print_summary_format_long_names(self):
        shape = (8, 8, 3)

        model = keras.Sequential(
            [
                keras.Input(shape),
                keras.layers.Conv2D(4, 3, name="Really-Long-name-test"),
                keras.layers.Conv2D(4, 3, name="Another-long-name-test"),
                keras.layers.Flatten(),
                keras.layers.Dense(2, name="long-name-test-output"),
            ]
        )
        file_name = "sequential.txt"
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        fpath = os.path.join(temp_dir, file_name)
        writer = open(fpath, "w")

        def print_to_file(text):
            print(text, file=writer)

        layer_utils.print_summary(model, print_fn=print_to_file)
        self.assertTrue(tf.io.gfile.exists(fpath))
        writer.close()
        reader = open(fpath, "r")
        lines = reader.readlines()
        reader.close()
        check_str = (
            'Model: "sequential"\n'
            "_________________________________________________________________\n"  # noqa: E501
            " Layer (type)                Output Shape              Param #   \n"  # noqa: E501
            "=================================================================\n"  # noqa: E501
            " Really-Long-name-test (Con  (None, 6, 6, 4)           112       \n"  # noqa: E501
            " v2D)                                                            \n"  # noqa: E501
            "                                                                 \n"  # noqa: E501
            " Another-long-name-test (Co  (None, 4, 4, 4)           148       \n"  # noqa: E501
            " nv2D)                                                           \n"  # noqa: E501
            "                                                                 \n"  # noqa: E501
            " flatten (Flatten)           (None, 64)                0         \n"  # noqa: E501
            "                                                                 \n"  # noqa: E501
            " long-name-test-output (Den  (None, 2)                 130       \n"  # noqa: E501
            " se)                                                             \n"  # noqa: E501
            "                                                                 \n"  # noqa: E501
            "=================================================================\n"  # noqa: E501
            "Total params: 390 (1.52 KB)\n"
            "Trainable params: 390 (1.52 KB)\n"
            "Non-trainable params: 0 (0.00 Byte)\n"
            "_________________________________________________________________\n"  # noqa: E501
        )
        fin_str = "".join(lines)
        self.assertIn(fin_str, check_str)
        self.assertEqual(len(lines), 20)

    def test_print_summary_expand_nested(self):
        shape = (None, None, 3)

        def make_model():
            x = inputs = keras.Input(shape)
            x = keras.layers.Conv2D(3, 1)(x)
            x = keras.layers.BatchNormalization()(x)
            return keras.Model(inputs, x)

        x = inner_inputs = keras.Input(shape)
        x = make_model()(x)
        inner_model = keras.Model(inner_inputs, x)

        inputs = keras.Input(shape)
        model = keras.Model(inputs, inner_model(inputs))

        file_name = "model_2.txt"
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        fpath = os.path.join(temp_dir, file_name)
        writer = open(fpath, "w")

        def print_to_file(text):
            print(text, file=writer)

        try:
            layer_utils.print_summary(
                model, print_fn=print_to_file, expand_nested=True
            )
            self.assertTrue(tf.io.gfile.exists(fpath))
            writer.close()
            reader = open(fpath, "r")
            lines = reader.readlines()
            reader.close()
            check_str = (
                'Model: "model_2"\n'
                "_________________________________________________________________\n"  # noqa: E501
                " Layer (type)                Output Shape              Param #   \n"  # noqa: E501
                "=================================================================\n"  # noqa: E501
                " input_3 (InputLayer)        [(None, None, None, 3)]   0         \n"  # noqa: E501
                "                                                                 \n"  # noqa: E501
                " model_1 (Functional)        (None, None, None, 3)     24        \n"  # noqa: E501
                "|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|\n"  # noqa: E501
                "| input_1 (InputLayer)       [(None, None, None, 3)]   0        |\n"  # noqa: E501
                "|                                                               |\n"  # noqa: E501
                "| model (Functional)         (None, None, None, 3)     24       |\n"  # noqa: E501
                "||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||\n"  # noqa: E501
                "|| input_2 (InputLayer)      [(None, None, None, 3)]   0       ||\n"  # noqa: E501
                "||                                                             ||\n"  # noqa: E501
                "|| conv2d (Conv2D)           (None, None, None, 3)     12      ||\n"  # noqa: E501
                "||                                                             ||\n"  # noqa: E501
                "|| batch_normalization (Bat  (None, None, None, 3)     12      ||\n"  # noqa: E501
                "|| chNormalization)                                            ||\n"  # noqa: E501
                "|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|\n"  # noqa: E501
                "¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\n"  # noqa: E501
                "=================================================================\n"  # noqa: E501
                "Total params: 24 (96.00 Byte)\n"
                "Trainable params: 18 (72.00 Byte)\n"
                "Non-trainable params: 6 (24.00 Byte)\n"
                "_________________________________________________________________\n"  # noqa: E501
            )

            fin_str = "".join(lines)

            self.assertIn(fin_str, check_str)
            self.assertEqual(len(lines), 25)
        except ImportError:
            pass

    def test_summary_subclass_model_expand_nested(self):
        class Sequential(keras.Model):
            def __init__(self, *args):
                super().__init__()
                self.module_list = list(args) if args else []

            def call(self, x):
                for module in self.module_list:
                    x = module(x)
                return x

        class Block(keras.Model):
            def __init__(self):
                super().__init__()
                self.module = Sequential(
                    keras.layers.Dense(10),
                    keras.layers.Dense(10),
                )

            def call(self, input_tensor):
                x = self.module(input_tensor)
                return x

        class Base(keras.Model):
            def __init__(self):
                super().__init__()
                self.module = Sequential(Block(), Block())

            def call(self, input_tensor):
                x = self.module(input_tensor)
                y = self.module(x)
                return x, y

        class Network(keras.Model):
            def __init__(self):
                super().__init__()
                self.child = Base()

            def call(self, inputs):
                return self.child(inputs)

        net = Network()
        inputs = keras.Input(shape=(10,))
        outputs = net(inputs)
        model = keras.models.Model(inputs=inputs, outputs=outputs)

        file_name = "model_3.txt"
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        fpath = os.path.join(temp_dir, file_name)
        writer = open(fpath, "w")

        def print_to_file(text):
            print(text, file=writer)

        try:
            layer_utils.print_summary(
                model,
                line_length=120,
                print_fn=print_to_file,
                expand_nested=True,
            )
            self.assertTrue(tf.io.gfile.exists(fpath))
            writer.close()
            with open(fpath, "r") as reader:
                lines = reader.readlines()
            # The output content are slightly different for the input shapes
            # between v1 and v2.
            if tf.__internal__.tf2.enabled():
                self.assertEqual(len(lines), 39)
            else:
                self.assertEqual(len(lines), 40)
        except ImportError:
            pass

    def test_print_summary_show_trainable(self):
        model = keras.Sequential(name="trainable")
        untrained = keras.layers.Conv2D(
            filters=2, kernel_size=(2, 3), input_shape=(3, 5, 5), name="conv"
        )
        model.add(untrained)
        model.add(keras.layers.Flatten(name="flat"))
        model.add(keras.layers.Dense(5, name="dense"))

        untrained.trainable = False

        file_name = "model_4.txt"
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        fpath = os.path.join(temp_dir, file_name)
        writer = open(fpath, "w")

        def print_to_file(text):
            print(text, file=writer)

        try:
            layer_utils.print_summary(
                model, print_fn=print_to_file, show_trainable=True
            )
            self.assertTrue(tf.io.gfile.exists(fpath))
            writer.close()
            with open(fpath, "r") as reader:
                lines = reader.readlines()
            check_str = (
                'Model: "trainable"\n'
                "____________________________________________________________________________\n"  # noqa: E501
                " Layer (type)                Output Shape              Param #   Trainable  \n"  # noqa: E501
                "============================================================================\n"  # noqa: E501
                " conv (Conv2D)               (None, 2, 3, 2)           62        N          \n"  # noqa: E501
                "                                                                            \n"  # noqa: E501
                " flat (Flatten)              (None, 12)                0         Y          \n"  # noqa: E501
                "                                                                            \n"  # noqa: E501
                " dense (Dense)               (None, 5)                 65        Y          \n"  # noqa: E501
                "                                                                            \n"  # noqa: E501
                "============================================================================\n"  # noqa: E501
                "Total params: 127 (508.00 Byte)\n"
                "Trainable params: 65 (260.00 Byte)\n"
                "Non-trainable params: 62 (248.00 Byte)\n"
                "____________________________________________________________________________\n"  # noqa: E501
                "____________________________________________________________________________\n"  # noqa: E501
            )

            fin_str = "".join(lines)

            self.assertIn(fin_str, check_str)
            self.assertEqual(len(lines), 15)
        except ImportError:
            pass

    def test_print_summary_expand_nested_show_trainable(self):
        shape = (None, None, 3)

        def make_model():
            x = inputs = keras.Input(shape, name="input2")
            untrainable = keras.layers.Conv2D(3, 1)
            untrainable.trainable = False
            x = untrainable(x)
            x = keras.layers.BatchNormalization()(x)
            return keras.Model(inputs, x)

        x = inner_inputs = keras.Input(shape, name="input1")
        x = make_model()(x)
        inner_model = keras.Model(inner_inputs, x)

        inputs = keras.Input(shape, name="input3")
        model = keras.Model(inputs, inner_model(inputs))

        file_name = "model_6.txt"
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        fpath = os.path.join(temp_dir, file_name)
        writer = open(fpath, "w")

        def print_to_file(text):
            print(text, file=writer)

        try:
            layer_utils.print_summary(
                model,
                print_fn=print_to_file,
                expand_nested=True,
                show_trainable=True,
            )
            self.assertTrue(tf.io.gfile.exists(fpath))
            writer.close()
            with open(fpath, "r") as reader:
                lines = reader.readlines()
            check_str = (
                'Model: "model_2"\n'
                "____________________________________________________________________________\n"  # noqa: E501
                " Layer (type)                Output Shape              Param #   Trainable  \n"  # noqa: E501
                "============================================================================\n"  # noqa: E501
                " input3 (InputLayer)         [(None, None, None, 3)]   0         Y          \n"  # noqa: E501
                "                                                                            \n"  # noqa: E501
                " model_1 (Functional)        (None, None, None, 3)     24        Y          \n"  # noqa: E501
                "|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|\n"  # noqa: E501
                "| input1 (InputLayer)        [(None, None, None, 3)]   0         Y         |\n"  # noqa: E501
                "|                                                                          |\n"  # noqa: E501
                "| model (Functional)         (None, None, None, 3)     24        Y         |\n"  # noqa: E501
                "||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||\n"  # noqa: E501
                "|| input2 (InputLayer)       [(None, None, None, 3)]   0         Y        ||\n"  # noqa: E501
                "||                                                                        ||\n"  # noqa: E501
                "|| conv2d (Conv2D)           (None, None, None, 3)     12        N        ||\n"  # noqa: E501
                "||                                                                        ||\n"  # noqa: E501
                "|| batch_normalization (Bat  (None, None, None, 3)     12        Y        ||\n"  # noqa: E501
                "|| chNormalization)                                                       ||\n"  # noqa: E501
                "|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|\n"  # noqa: E501
                "¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\n"  # noqa: E501
                "============================================================================\n"  # noqa: E501
                "Total params: 24 (96.00 Byte)\n"
                "Trainable params: 6 (24.00 Byte)\n"
                "Non-trainable params: 18 (72.00 Byte)\n"
                "____________________________________________________________________________\n"  # noqa: E501
            )

            fin_str = "".join(lines)

            self.assertIn(fin_str, check_str)
            self.assertEqual(len(lines), 25)
        except ImportError:
            pass

    def test_print_summary_layer_range(self):
        model = keras.Sequential()
        model.add(
            keras.layers.Conv2D(
                filters=2,
                kernel_size=(2, 3),
                input_shape=(3, 5, 5),
                name="conv",
            )
        )
        model.add(keras.layers.Flatten(name="flat"))
        model.add(keras.layers.Dense(5, name="dense"))

        file_name = "model_7.txt"
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        fpath = os.path.join(temp_dir, file_name)
        writer = open(fpath, "w")

        def print_to_file(text):
            print(text, file=writer)

        try:
            layer_utils.print_summary(
                model, print_fn=print_to_file, layer_range=["conv", "flat"]
            )
            self.assertTrue(tf.io.gfile.exists(fpath))
            writer.close()
            with open(fpath, "r") as reader:
                lines = reader.readlines()
            # The expected lenght with no layer filter is 15
            # we filtered out 2 lines by excluding the layer 'dense'
            self.assertEqual(len(lines), 15 - 2)
        except ImportError:
            pass

    def test_print_summary_layer_range_with_expand_nested(self):
        shape = (None, None, 3)

        def make_model():
            x = inputs = keras.Input(shape, name="input_2")
            x = keras.layers.Conv2D(3, 1)(x)
            x = keras.layers.BatchNormalization()(x)
            return keras.Model(inputs, x, name="2nd_inner")

        x = inner_inputs = keras.Input(shape, name="input_1")
        x = make_model()(x)
        inner_model = keras.Model(inner_inputs, x, name="1st_inner")

        inputs = keras.Input(shape, name="input_3")
        model = keras.Model(inputs, inner_model(inputs))

        file_name = "model_8.txt"
        temp_dir = self.get_temp_dir()
        self.addCleanup(shutil.rmtree, temp_dir, ignore_errors=True)
        fpath = os.path.join(temp_dir, file_name)
        writer = open(fpath, "w")

        def print_to_file(text):
            print(text, file=writer)

        try:
            layer_utils.print_summary(
                model,
                print_fn=print_to_file,
                expand_nested=True,
                layer_range=["1st_inner", "1st_inner"],
            )
            layer_utils.print_summary(
                model,
                expand_nested=True,
                layer_range=["1st_inner", "1st_inner"],
            )
            self.assertTrue(tf.io.gfile.exists(fpath))
            writer.close()
            with open(fpath, "r") as reader:
                lines = reader.readlines()
            check_str = (
                'Model: "model"\n'
                "_________________________________________________________________\n"  # noqa: E501
                " Layer (type)                Output Shape              Param #   \n"  # noqa: E501
                "=================================================================\n"  # noqa: E501
                " 1st_inner (Functional)      (None, None, None, 3)     24        \n"  # noqa: E501
                "|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|\n"  # noqa: E501
                "| input_1 (InputLayer)       [(None, None, None, 3)]   0        |\n"  # noqa: E501
                "|                                                               |\n"  # noqa: E501
                "| 2nd_inner (Functional)     (None, None, None, 3)     24       |\n"  # noqa: E501
                "||¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯||\n"  # noqa: E501
                "|| input_2 (InputLayer)      [(None, None, None, 3)]   0       ||\n"  # noqa: E501
                "||                                                             ||\n"  # noqa: E501
                "|| conv2d (Conv2D)           (None, None, None, 3)     12      ||\n"  # noqa: E501
                "||                                                             ||\n"  # noqa: E501
                "|| batch_normalization (Bat  (None, None, None, 3)     12      ||\n"  # noqa: E501
                "|| chNormalization)                                            ||\n"  # noqa: E501
                "|¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯|\n"  # noqa: E501
                "¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯¯\n"  # noqa: E501
                "=================================================================\n"  # noqa: E501
                "Total params: 24 (96.00 Byte)\n"
                "Trainable params: 18 (72.00 Byte)\n"
                "Non-trainable params: 6 (24.00 Byte)\n"
                "_________________________________________________________________\n"  # noqa: E501
            )

            check_lines = check_str.split("\n")[
                :-1
            ]  # Removing final empty string which is not a line

            fin_str = "".join(lines)
            self.assertIn(fin_str, check_str)
            self.assertEqual(len(lines), len(check_lines))
        except ImportError:
            pass

    def test_weight_memory_size(self):
        v1 = tf.Variable(tf.zeros(shape=(1, 2), dtype=tf.float32))
        v2 = tf.Variable(tf.zeros(shape=(2, 3), dtype=tf.float64))
        v3 = tf.Variable(tf.zeros(shape=(4, 5), dtype=tf.int16))
        v4 = tf.Variable(tf.zeros(shape=(6,), dtype=tf.uint8))

        weights = [v1, v1, v2, v3, v4]
        weight_memory_size = layer_utils.weight_memory_size(weights)
        expected_memory_size = 1 * 2 * 4 + 2 * 3 * 8 + 4 * 5 * 2 + 6 * 1
        self.assertEqual(weight_memory_size, expected_memory_size)

    @parameterized.parameters(
        (0, "0.00 Byte"),
        (1000, "1000.00 Byte"),
        (1024, "1.00 KB"),
        (1024 * 2 - 1, "2.00 KB"),
        (1024 * 2 + 1, "2.00 KB"),
        (1024**2 + 1, "1.00 MB"),
        (1024**3 - 1, "1024.00 MB"),
        (1024**3, "1.00 GB"),
        (1024**4, "1.00 TB"),
        (1024**5, "1.00 PB"),
        (1024**5 * 1.41415, "1.41 PB"),
    )
    def test_readable_weight_memory_size(self, size, expected_result):
        result = layer_utils.readable_memory_size(size)
        self.assertEqual(result, expected_result)

    def test_property_cache(self):
        test_counter = collections.Counter()

        class MyObject(tf.__internal__.tracking.AutoTrackable):
            def __init__(self):
                super().__init__()
                self._frozen = True

            def __setattr__(self, key, value):
                """Enforce that cache does not set attribute on MyObject."""
                if getattr(self, "_frozen", False):
                    raise ValueError("Cannot mutate when frozen.")
                return super().__setattr__(key, value)

            @property
            @layer_utils.cached_per_instance
            def test_property(self):
                test_counter[id(self)] += 1
                return id(self)

        first_object = MyObject()
        second_object = MyObject()

        # Make sure the objects return the correct values
        self.assertEqual(first_object.test_property, id(first_object))
        self.assertEqual(second_object.test_property, id(second_object))

        # Make sure the cache does not share across objects
        self.assertNotEqual(
            first_object.test_property, second_object.test_property
        )

        # Check again (Now the values should be cached.)
        self.assertEqual(first_object.test_property, id(first_object))
        self.assertEqual(second_object.test_property, id(second_object))

        # Count the function calls to make sure the cache is actually being
        # used.
        self.assertAllEqual(tuple(test_counter.values()), (1, 1))

    def test_property_cache_threaded(self):
        call_count = collections.Counter()

        class MyObject(tf.__internal__.tracking.AutoTrackable):
            @property
            @layer_utils.cached_per_instance
            def test_property(self):
                # Random sleeps to ensure that the execution thread changes
                # mid-computation.
                call_count["test_property"] += 1
                time.sleep(np.random.random() + 1.0)

                # Use a RandomState which is seeded off the instance's id (the
                # mod is because numpy limits the range of seeds) to ensure that
                # an instance returns the same value in different threads, but
                # different instances return different values.
                return int(
                    np.random.RandomState(id(self) % (2**31)).randint(2**16)
                )

            def get_test_property(self, _):
                """Function provided to .map for threading test."""
                return self.test_property

        # Test that multiple threads return the same value. This requires that
        # the underlying function is repeatable, as cached_property makes no
        # attempt to prioritize the first call.
        test_obj = MyObject()
        with contextlib.closing(multiprocessing.dummy.Pool(32)) as pool:
            # Intentionally make a large pool (even when there are only a small
            # number of cpus) to ensure that the runtime switches threads.
            results = pool.map(test_obj.get_test_property, range(64))
        self.assertEqual(len(set(results)), 1)

        # Make sure we actually are testing threaded behavior.
        self.assertGreater(call_count["test_property"], 1)

        # Make sure new threads still cache hit.
        with contextlib.closing(multiprocessing.dummy.Pool(2)) as pool:
            start_time = (
                timeit.default_timer()
            )  # Don't time pool instantiation.
            results = pool.map(test_obj.get_test_property, range(4))
        total_time = timeit.default_timer() - start_time

        # Note(taylorrobie): The reason that it is safe to time a unit test is
        # that a cache hit will be << 1 second, and a cache miss is guaranteed
        # to be >= 1 second. Empirically confirmed by 100,000 runs with no
        # flakes.
        self.assertLess(total_time, 0.95)

    def test_property_cache_serialization(self):
        # Reset call count. .keys() must be wrapped in a list, because otherwise
        # we would mutate the iterator while iterating.
        for k in list(_PICKLEABLE_CALL_COUNT.keys()):
            _PICKLEABLE_CALL_COUNT.pop(k)

        first_instance = MyPickleableObject()
        self.assertEqual(id(first_instance), first_instance.my_id)

        # Test that we can pickle and un-pickle
        second_instance = pickle.loads(pickle.dumps(first_instance))

        self.assertEqual(id(second_instance), second_instance.my_id)
        self.assertNotEqual(first_instance.my_id, second_instance.my_id)

        # Make sure de-serialized object uses the cache.
        self.assertEqual(_PICKLEABLE_CALL_COUNT[second_instance], 1)

        # Make sure the decorator cache is not being serialized with the object.
        expected_size = len(pickle.dumps(second_instance))
        for _ in range(5):
            # Add some more entries to the cache.
            _ = MyPickleableObject().my_id
        self.assertEqual(len(_PICKLEABLE_CALL_COUNT), 7)
        size_check_instance = MyPickleableObject()
        _ = size_check_instance.my_id
        self.assertEqual(expected_size, len(pickle.dumps(size_check_instance)))

    def test_warmstart_embedding_matrix_with_list(self):
        vocab_base = ["unk", "a", "b", "c"]
        vocab_new = ["unk", "unk", "a", "b", "c", "d", "e"]
        vectorized_vocab_base = np.random.rand(len(vocab_base), 3)
        vectorized_vocab_new = np.random.rand(len(vocab_new), 3)
        warmstarted_embedding_matrix = layer_utils.warmstart_embedding_matrix(
            base_vocabulary=vocab_base,
            new_vocabulary=vocab_new,
            base_embeddings=vectorized_vocab_base,
            new_embeddings_initializer=keras.initializers.Constant(
                vectorized_vocab_new
            ),
        )
        self.assertAllEqual(
            warmstarted_embedding_matrix[2],
            vectorized_vocab_base[1],
        )

    def test_warmstart_embedding_matrix_with_nparray(self):
        vocab_base = np.array(["unk", "a", "b", "c"])
        vocab_new = np.array(["unk", "unk", "a", "b", "c", "d", "e"])
        vectorized_vocab_base = np.random.rand(len(vocab_base), 3)
        vectorized_vocab_new = np.random.rand(len(vocab_new), 3)
        warmstarted_embedding_matrix = layer_utils.warmstart_embedding_matrix(
            base_vocabulary=vocab_base,
            new_vocabulary=vocab_new,
            base_embeddings=vectorized_vocab_base,
            new_embeddings_initializer=keras.initializers.Constant(
                vectorized_vocab_new
            ),
        )
        self.assertAllEqual(
            warmstarted_embedding_matrix[2],
            vectorized_vocab_base[1],
        )

    @test_utils.run_v2_only
    def test_warmstart_embedding_matrix_with_tensor(self):
        vocab_base = tf.convert_to_tensor(["unk", "a", "b", "c"])
        vocab_new = tf.convert_to_tensor(
            ["unk", "unk", "a", "b", "c", "d", "e"]
        )
        vectorized_vocab_base = np.random.rand(vocab_base.shape[0], 3)
        vectorized_vocab_new = np.random.rand(vocab_new.shape[0], 3)
        warmstarted_embedding_matrix = layer_utils.warmstart_embedding_matrix(
            base_vocabulary=vocab_base,
            new_vocabulary=vocab_new,
            base_embeddings=vectorized_vocab_base,
            new_embeddings_initializer=keras.initializers.Constant(
                vectorized_vocab_new
            ),
        )
        self.assertAllEqual(
            warmstarted_embedding_matrix[2],
            vectorized_vocab_base[1],
        )

    def test_warmstart_embedding_matrix_with_file_name(self):
        def _write_list_to_file(filename, content_list):
            with tf.io.gfile.GFile(filename, "w") as output_file:
                for line in content_list:
                    output_file.write(line + "\n")

        vocab_base = ["UNK", "a", "b", "c"]
        vocab_base_file = tempfile.mktemp(".tsv")
        _write_list_to_file(vocab_base_file, vocab_base)
        vocab_new = ["UNK", "UNK", "a", "b", "c", "d", "e"]
        vocab_new_file = tempfile.mktemp(".tsv")
        vectorized_vocab_base = np.random.rand(len(vocab_base), 3)
        vectorized_vocab_new = np.random.rand(len(vocab_new), 3)
        _write_list_to_file(vocab_new_file, vocab_new)
        warmstarted_embedding_matrix = layer_utils.warmstart_embedding_matrix(
            base_vocabulary=vocab_base_file,
            new_vocabulary=vocab_new_file,
            base_embeddings=vectorized_vocab_base,
            new_embeddings_initializer=keras.initializers.Constant(
                vectorized_vocab_new
            ),
        )
        self.assertAllEqual(
            warmstarted_embedding_matrix[3],
            vectorized_vocab_base[2],
        )

    def test_warmstart_default_initialization(self):
        def _write_list_to_file(filename, content_list):
            with tf.io.gfile.GFile(filename, "w") as output_file:
                for line in content_list:
                    output_file.write(line + "\n")

        vocab_base = ["UNK", "a", "b", "c"]
        vocab_base_file = tempfile.mktemp(".tsv")
        _write_list_to_file(vocab_base_file, vocab_base)
        vocab_new = ["UNK", "UNK", "a", "b", "c", "d", "e"]
        vocab_new_file = tempfile.mktemp(".tsv")
        vectorized_vocab_base = np.random.rand(len(vocab_base), 3)
        _write_list_to_file(vocab_new_file, vocab_new)
        warmstarted_embedding_matrix = layer_utils.warmstart_embedding_matrix(
            base_vocabulary=vocab_base_file,
            new_vocabulary=vocab_new_file,
            base_embeddings=vectorized_vocab_base,
        )
        self.assertAllEqual(
            warmstarted_embedding_matrix[3],
            vectorized_vocab_base[2],
        )

    def test_warmstart_default_value(self):
        vocab_base = np.array(["unk", "a", "b", "c"])
        vocab_new = np.array(["unk", "unk", "a", "b", "c", "d", "e"])
        vectorized_vocab_base = np.random.rand(len(vocab_base), 3)
        warmstarted_embedding_matrix = layer_utils.warmstart_embedding_matrix(
            base_vocabulary=vocab_base,
            new_vocabulary=vocab_new,
            base_embeddings=vectorized_vocab_base,
        )
        self.assertAllEqual(
            warmstarted_embedding_matrix[2],
            vectorized_vocab_base[1],
        )

    def test_warmstart_with_randomuniform_initializer(self):
        vocab_base = np.array(["unk", "a", "b", "c"])
        vocab_new = np.array(["unk", "unk", "a", "b", "c", "d", "e"])
        vectorized_vocab_base = np.random.rand(len(vocab_base), 3)
        warmstarted_embedding_matrix = layer_utils.warmstart_embedding_matrix(
            base_vocabulary=vocab_base,
            new_vocabulary=vocab_new,
            base_embeddings=vectorized_vocab_base,
            new_embeddings_initializer="RandomUniform",
        )
        self.assertAllEqual(
            warmstarted_embedding_matrix[2],
            vectorized_vocab_base[1],
        )

    def test_warmstart_with_nothing_in_common(self):
        vocab_base = np.array(["unk", "a", "b", "c"])
        vocab_new = np.array(["d", "e", "f", "g", "h"])
        vectorized_vocab_base = np.random.rand(len(vocab_base), 3)
        vectorized_vocab_new = np.random.rand(len(vocab_new), 3)
        warmstarted_embedding_matrix = layer_utils.warmstart_embedding_matrix(
            base_vocabulary=vocab_base,
            new_vocabulary=vocab_new,
            base_embeddings=vectorized_vocab_base,
            new_embeddings_initializer=keras.initializers.Constant(
                vectorized_vocab_new
            ),
        )
        self.assertAllEqual(
            warmstarted_embedding_matrix,
            vectorized_vocab_new,
        )

    def test_warmstart_with_new_vocab_smaller(self):
        vocab_base = np.array(["unk", "a", "b", "c"])
        vocab_new = np.array(["d", "e", "f", "a"])
        vectorized_vocab_base = np.random.rand(len(vocab_base), 3)
        warmstarted_embedding_matrix = layer_utils.warmstart_embedding_matrix(
            base_vocabulary=vocab_base,
            new_vocabulary=vocab_new,
            base_embeddings=vectorized_vocab_base,
            new_embeddings_initializer="uniform",
        )
        self.assertAllEqual(
            warmstarted_embedding_matrix[3],
            vectorized_vocab_base[1],
        )


@test_utils.run_v2_only
class DTensorVariableSummaryTest(test_util.DTensorBaseTest):
    def setUp(self):
        super().setUp()
        backend.reset_uids()
        backend.enable_tf_random_generator()
        tf_utils.set_random_seed(1337)
        global_ids = test_util.create_device_ids_array((2, 2))
        local_device_ids = np.ravel(global_ids).tolist()
        mesh_dict = {
            "CPU": dtensor.Mesh(
                ["batch", "model"],
                global_ids,
                local_device_ids,
                test_util.create_device_list((2, 2), "CPU"),
            )
        }
        self.mesh = self.configTestMesh(mesh_dict)
        self.replicated_2d = dtensor.Layout.replicated(self.mesh, rank=2)
        self.replicated_1d = dtensor.Layout.replicated(self.mesh, rank=1)
        self.sharded_2d = dtensor.Layout(["model", "batch"], self.mesh)
        self.sharded_1d = dtensor.Layout(["model"], self.mesh)

    def test_model_summary(self):
        layout_map = layout_map_lib.LayoutMap(mesh=self.mesh)
        layout_map["d1.kernel"] = self.replicated_2d
        layout_map["d1.bias"] = self.replicated_1d
        layout_map["d2.kernel"] = self.sharded_2d
        layout_map["d2.bias"] = self.sharded_1d

        with layout_map.scope():
            inputs = layers.Input((10,), batch_size=10)
            x = layers.Dense(20, name="d1")(inputs)
            x = layers.Dropout(0.1)(x)
            output = layers.Dense(30, name="d2")(x)

            model = keras.Model(inputs, output)

        # For dtype = float32, following value are expected from memory stats
        expected_result = {}
        replicated_var_count = 10 * 20 + 20  # For d1 kernel and bias
        model_batch_shard_var_count = 30 * 20  # For d2 kernel
        model_shard_var_count = 30  # For d2 bias
        expected_result[()] = (replicated_var_count, replicated_var_count * 4)
        expected_result[("batch", "model")] = (
            model_batch_shard_var_count,
            model_batch_shard_var_count * 4,
        )
        expected_result[("model",)] = (
            model_shard_var_count,
            model_shard_var_count * 4,
        )

        expected_total_weight_count = (
            replicated_var_count
            + model_batch_shard_var_count
            + model_shard_var_count
        )
        expected_total_memory_size = expected_total_weight_count * 4

        (
            total_weight_count,
            total_memory_size,
            per_sharing_spec_result,
        ) = layer_utils.dtensor_variable_summary(model.weights)

        self.assertEqual(total_weight_count, expected_total_weight_count)
        self.assertEqual(total_memory_size, expected_total_memory_size)
        self.assertDictEqual(per_sharing_spec_result, expected_result)

        output_buffer = io.StringIO()

        def print_to_buffer(content):
            output_buffer.write(content)

        model.summary(print_fn=print_to_buffer)

        self.assertRegex(
            output_buffer.getvalue(),
            f"{replicated_var_count} / {expected_total_weight_count} params "
            ".* are fully replicated",
        )
        self.assertRegex(
            output_buffer.getvalue(),
            f"{model_batch_shard_var_count} / {expected_total_weight_count} "
            r"params .* are sharded based on spec .*batch.*model"
            r".* across 4 devices",
        )
        self.assertRegex(
            output_buffer.getvalue(),
            f"{model_shard_var_count} / {expected_total_weight_count} "
            r"params .* are sharded based on spec .*model"
            r".* across 2 devices",
        )
        self.assertIn(
            "Overall per device memory usage: 1.50 KB", output_buffer.getvalue()
        )
        self.assertIn("Overall sharding factor: 2.21", output_buffer.getvalue())


if __name__ == "__main__":
    tf.test.main()
