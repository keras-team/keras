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
"""Tests for Keras python-based idempotent saving functions (experimental)."""
import os
import sys

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized

import keras
from keras import backend
from keras.saving.experimental import saving_lib
from keras.saving.saved_model import json_utils
from keras.utils import generic_utils
from keras.utils import io_utils

train_step_message = "This is my training step"


@keras.utils.generic_utils.register_keras_serializable(
    package="my_custom_package"
)
class MyDense(keras.layers.Dense):
    def two(self):
        return 2


@keras.utils.generic_utils.register_keras_serializable(
    package="my_custom_package"
)
class CustomModelX(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = MyDense(1)

    def call(self, inputs):
        return self.dense1(inputs)

    def train_step(self, data):
        tf.print(train_step_message)
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x)
            loss = self.compiled_loss(y, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {}

    def one(self):
        return 1


@keras.utils.generic_utils.register_keras_serializable(
    package="my_custom_package"
)
def my_mean_squared_error(y_true, y_pred):
    """Identical to built-in `mean_squared_error`, added here as a custom
    func."""
    return backend.mean(tf.math.squared_difference(y_pred, y_true), axis=-1)


module_my_mean_squared_error = my_mean_squared_error


class NewSavingTest(tf.test.TestCase, parameterized.TestCase):
    def setUp(self):
        super().setUp()
        saving_lib._ENABLED = True

    def tearDown(self):
        super().tearDown()
        saving_lib._ENABLED = False

    def _get_subclassed_model(self):
        subclassed_model = CustomModelX()
        subclassed_model.compile(
            optimizer="adam",
            loss=[
                "mse",
                keras.losses.mean_squared_error,
                keras.losses.MeanSquaredError(),
                my_mean_squared_error,
            ],
        )
        return subclassed_model

    def test_saving_after_compile_but_before_fit(self):
        temp_dir = os.path.join(self.get_temp_dir(), "my_model")
        subclassed_model = self._get_subclassed_model()
        subclassed_model._save_new(temp_dir)

        # This is so that we can register another function with the same custom
        # object key, and make sure the newly registered function is used while
        # loading.
        del generic_utils._GLOBAL_CUSTOM_OBJECTS[
            "my_custom_package>my_mean_squared_error"
        ]

        @keras.utils.generic_utils.register_keras_serializable(
            package="my_custom_package"
        )
        def my_mean_squared_error(y_true, y_pred):
            """Function-local `mean_squared_error`."""
            return backend.mean(
                tf.math.squared_difference(y_pred, y_true), axis=-1
            )

        loaded_model = saving_lib.load(temp_dir)

        # Everything should be the same class or function for the original model
        # and the loaded model.
        for model in [subclassed_model, loaded_model]:
            self.assertIs(
                model.optimizer.__class__,
                keras.optimizers.optimizer_v2.adam.Adam,
            )
            self.assertIs(
                model.compiled_loss.__class__,
                keras.engine.compile_utils.LossesContainer,
            )
            self.assertEqual(model.compiled_loss._losses[0], "mse")
            self.assertIs(
                model.compiled_loss._losses[1], keras.losses.mean_squared_error
            )
            self.assertIs(
                model.compiled_loss._losses[2].__class__,
                keras.losses.MeanSquaredError,
            )
            self.assertIs(
                model.compiled_loss._total_loss_mean.__class__,
                keras.metrics.base_metric.Mean,
            )

        # Except for a custom function used because the loaded model is supposed
        # to be using the newly registered custom function.
        self.assertIs(
            subclassed_model.compiled_loss._losses[3],
            module_my_mean_squared_error,
        )
        self.assertIs(
            loaded_model.compiled_loss._losses[3], my_mean_squared_error
        )
        self.assertIsNot(module_my_mean_squared_error, my_mean_squared_error)

    def test_saving_after_fit(self):
        temp_dir = os.path.join(self.get_temp_dir(), "my_model")
        subclassed_model = self._get_subclassed_model()

        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        subclassed_model.fit(x, y, epochs=1)
        subclassed_model._save_new(temp_dir)
        loaded_model = saving_lib.load(temp_dir)

        io_utils.enable_interactive_logging()
        # `tf.print` writes to stderr. This is to make sure the custom training
        # step is used.
        with self.captureWritesToStream(sys.stderr) as printed:
            loaded_model.fit(x, y, epochs=1)
            self.assertRegex(printed.contents(), train_step_message)

        # Check that the custom classes do get used.
        self.assertIsInstance(loaded_model, CustomModelX)
        self.assertIsInstance(loaded_model.dense1, MyDense)
        # Check that the custom method is available.
        self.assertEqual(loaded_model.one(), 1)
        self.assertEqual(loaded_model.dense1.two(), 2)

        # Everything should be the same class or function for the original model
        # and the loaded model.
        for model in [subclassed_model, loaded_model]:
            self.assertIs(
                model.optimizer.__class__,
                keras.optimizers.optimizer_v2.adam.Adam,
            )
            self.assertIs(
                model.compiled_loss.__class__,
                keras.engine.compile_utils.LossesContainer,
            )
            self.assertIs(
                model.compiled_loss._losses[0].__class__,
                keras.losses.LossFunctionWrapper,
            )
            self.assertIs(
                model.compiled_loss._losses[1].__class__,
                keras.losses.LossFunctionWrapper,
            )
            self.assertIs(
                model.compiled_loss._losses[2].__class__,
                keras.losses.MeanSquaredError,
            )
            self.assertIs(
                model.compiled_loss._losses[3].__class__,
                keras.losses.LossFunctionWrapper,
            )
            self.assertIs(
                model.compiled_loss._total_loss_mean.__class__,
                keras.metrics.base_metric.Mean,
            )

    def test_saving_preserve_unbuilt_state(self):
        temp_dir = os.path.join(self.get_temp_dir(), "my_model")
        subclassed_model = CustomModelX()
        subclassed_model._save_new(temp_dir)
        loaded_model = saving_lib.load(temp_dir)
        self.assertFalse(subclassed_model.built)
        self.assertFalse(loaded_model.built)

    def test_saving_preserve_built_state(self):
        temp_dir = os.path.join(self.get_temp_dir(), "my_model")
        subclassed_model = self._get_subclassed_model()
        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        subclassed_model.fit(x, y, epochs=1)
        subclassed_model._save_new(temp_dir)
        loaded_model = saving_lib.load(temp_dir)
        self.assertTrue(subclassed_model.built)
        self.assertTrue(loaded_model.built)
        self.assertEqual(
            subclassed_model._build_input_shape, loaded_model._build_input_shape
        )
        self.assertEqual(
            tf.TensorShape([None, 32]), loaded_model._build_input_shape
        )

    def test_saved_module_paths_and_class_names(self):
        temp_dir = os.path.join(self.get_temp_dir(), "my_model")
        subclassed_model = self._get_subclassed_model()
        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        subclassed_model.fit(x, y, epochs=1)
        subclassed_model._save_new(temp_dir)

        file_path = os.path.join(temp_dir, saving_lib._CONFIG_FILE)
        with tf.io.gfile.GFile(file_path, "r") as f:
            config_json = f.read()
        config_dict = json_utils.decode(config_json)
        self.assertEqual(
            config_dict["registered_name"], "my_custom_package>CustomModelX"
        )
        self.assertIsNone(config_dict["config"]["optimizer"]["module"])
        self.assertEqual(
            config_dict["config"]["optimizer"]["class_name"],
            "keras.optimizers.Adam",
        )
        self.assertEqual(
            config_dict["config"]["loss"]["module"],
            "keras.engine.compile_utils",
        )
        self.assertEqual(
            config_dict["config"]["loss"]["class_name"], "LossesContainer"
        )

    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(
            layer=["tf_op_lambda", "lambda"],
        )
    )
    def test_functional_model_with_tf_op_lambda_layer(self, layer):
        class ToString:
            def __init__(self):
                self.contents = ""

            def __call__(self, msg):
                self.contents += msg + "\n"

        temp_dir = os.path.join(self.get_temp_dir(), "my_model")

        if layer == "lambda":
            func = tf.function(lambda x: tf.math.cos(x) + tf.math.sin(x))
            inputs = keras.layers.Input(shape=(32,))
            outputs = keras.layers.Dense(1)(inputs)
            outputs = keras.layers.Lambda(func._python_function)(outputs)

        elif layer == "tf_op_lambda":
            inputs = keras.layers.Input(shape=(32,))
            outputs = keras.layers.Dense(1)(inputs)
            outputs = outputs + inputs

        functional_model = keras.Model(inputs, outputs)
        functional_to_string = ToString()
        functional_model.summary(print_fn=functional_to_string)
        functional_model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        x = np.random.random((1000, 32))
        y = np.random.random((1000, 1))
        functional_model.fit(x, y, epochs=3)
        functional_model._save_new(temp_dir)
        loaded_model = saving_lib.load(temp_dir)
        loaded_model.fit(x, y, epochs=3)
        loaded_to_string = ToString()
        loaded_model.summary(print_fn=loaded_to_string)

        # Confirming the original and saved/loaded model have same structure.
        self.assertEqual(
            functional_to_string.contents, loaded_to_string.contents
        )


if __name__ == "__main__":
    if tf.__internal__.tf2.enabled():
        tf.test.main()
