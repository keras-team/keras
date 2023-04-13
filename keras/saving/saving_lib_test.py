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
"""Tests for Keras python-based idempotent saving functions."""
import os
import sys
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import tensorflow.compat.v2 as tf
from absl.testing import parameterized
from tensorflow.python.platform import tf_logging as logging

import keras
from keras import backend
from keras.optimizers import adam
from keras.saving import object_registration
from keras.saving import saving_lib
from keras.saving.legacy.saved_model import json_utils
from keras.testing_infra import test_utils
from keras.utils import io_utils

train_step_message = "This is my training step"
assets_data = "These are my assets"
variables_data = np.random.random((10,))


@keras.utils.register_keras_serializable(package="my_custom_package")
class MyDense(keras.layers.Dense):
    def build(self, input_shape):
        self.additional_weights = [
            self.add_weight(
                "my_additional_weight",
                initializer="ones",
                trainable=True,
            ),
            self.add_weight(
                "my_additional_weight_2",
                initializer="ones",
                trainable=True,
            ),
        ]
        self.weights_in_dict = {
            "my_weight": self.add_weight(
                "my_dict_weight",
                initializer="ones",
                trainable=True,
            ),
        }
        self.nested_layer = keras.layers.Dense(1)
        return super().build(input_shape)

    def call(self, inputs):
        call_result = super().call(inputs)
        return self.nested_layer(call_result)

    def two(self):
        return 2


@keras.utils.register_keras_serializable(package="my_custom_package")
class LayerWithCustomSaving(MyDense):
    def build(self, input_shape):
        self.assets = assets_data
        self.stored_variables = variables_data
        return super().build(input_shape)

    def save_assets(self, inner_path):
        with open(os.path.join(inner_path, "assets.txt"), "w") as f:
            f.write(self.assets)

    def save_own_variables(self, store):
        store["variables"] = self.stored_variables

    def load_assets(self, inner_path):
        with open(os.path.join(inner_path, "assets.txt"), "r") as f:
            text = f.read()
        self.assets = text

    def load_own_variables(self, store):
        self.stored_variables = np.array(store["variables"])


@keras.utils.register_keras_serializable(package="my_custom_package")
class CustomModelX(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = MyDense(1)
        self.dense2 = MyDense(1)

    def call(self, inputs):
        out = self.dense1(inputs)
        return self.dense2(out)

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


@keras.utils.register_keras_serializable(package="my_custom_package")
class ModelWithCustomSaving(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_dense = LayerWithCustomSaving(1)

    def call(self, inputs):
        return self.custom_dense(inputs)


@keras.utils.register_keras_serializable(package="my_custom_package")
class CompileOverridingModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = MyDense(1)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)

    def call(self, inputs):
        return self.dense1(inputs)


@keras.utils.register_keras_serializable(package="my_custom_package")
class CompileOverridingSequential(keras.Sequential):
    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)


@keras.utils.register_keras_serializable(package="my_custom_package")
def my_mean_squared_error(y_true, y_pred):
    """Identical to built-in `mean_squared_error`, added here as a custom

    func.
    """
    return backend.mean(tf.math.squared_difference(y_pred, y_true), axis=-1)


module_my_mean_squared_error = my_mean_squared_error


@test_utils.run_v2_only
class SavingV3Test(tf.test.TestCase, parameterized.TestCase):
    def _get_subclassed_model(self):
        subclassed_model = CustomModelX()
        subclassed_model.compile(
            optimizer=adam.Adam(),
            loss=[
                "mse",
                keras.losses.mean_squared_error,
                keras.losses.MeanSquaredError(),
                my_mean_squared_error,
            ],
        )
        return subclassed_model

    def _get_sequential_model(self):
        sequential_model = keras.Sequential([MyDense(1), MyDense(1)])
        sequential_model.compile(
            optimizer="adam", loss=["mse", keras.losses.mean_squared_error]
        )
        return sequential_model

    def _get_functional_model(self):
        inputs = keras.Input(shape=(32,))
        x = MyDense(1, name="first_dense")(inputs)
        outputs = MyDense(1, name="second_dense")(x)
        functional_model = keras.Model(inputs, outputs)
        functional_model.compile(
            optimizer="adam", loss=["mse", keras.losses.mean_squared_error]
        )
        return functional_model

    def test_saving_after_compile_but_before_fit(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        subclassed_model = self._get_subclassed_model()
        subclassed_model._save_experimental(temp_filepath)

        # This is so that we can register another function with the same custom
        # object key, and make sure the newly registered function is used while
        # loading.
        del object_registration._GLOBAL_CUSTOM_OBJECTS[
            "my_custom_package>my_mean_squared_error"
        ]

        @keras.utils.register_keras_serializable(package="my_custom_package")
        def my_mean_squared_error(y_true, y_pred):
            """Function-local `mean_squared_error`."""
            return backend.mean(
                tf.math.squared_difference(y_pred, y_true), axis=-1
            )

        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(
            subclassed_model._is_compiled, loaded_model._is_compiled
        )

        # Everything should be the same class or function for the original model
        # and the loaded model.
        for model in [subclassed_model, loaded_model]:
            self.assertIs(
                model.optimizer.__class__,
                adam.Adam,
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
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        subclassed_model = self._get_subclassed_model()

        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        subclassed_model.fit(x, y, epochs=1)
        subclassed_model._save_experimental(temp_filepath)
        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(
            subclassed_model._is_compiled, loaded_model._is_compiled
        )

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
                adam.Adam,
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
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        subclassed_model = CustomModelX()
        subclassed_model._save_experimental(temp_filepath)
        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(
            subclassed_model._is_compiled, loaded_model._is_compiled
        )
        self.assertFalse(subclassed_model.built)
        self.assertFalse(loaded_model.built)

    def test_saving_preserve_built_state(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        model = self._get_subclassed_model()
        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        model.fit(x, y, epochs=1)
        model._save_experimental(temp_filepath)
        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(model._is_compiled, loaded_model._is_compiled)
        self.assertTrue(model.built)
        self.assertTrue(loaded_model.built)
        self.assertEqual(
            model._build_input_shape, loaded_model._build_input_shape
        )
        self.assertEqual(
            tf.TensorShape([None, 32]), loaded_model._build_input_shape
        )

    def test_saved_module_paths_and_class_names(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        subclassed_model = self._get_subclassed_model()
        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        subclassed_model.fit(x, y, epochs=1)
        subclassed_model._save_experimental(temp_filepath)

        with zipfile.ZipFile(temp_filepath, "r") as z:
            with z.open(saving_lib._CONFIG_FILENAME, "r") as c:
                config_json = c.read()
        config_dict = json_utils.decode(config_json)
        self.assertEqual(
            config_dict["registered_name"], "my_custom_package>CustomModelX"
        )
        self.assertEqual(
            config_dict["compile_config"]["optimizer"]["config"][
                "is_legacy_optimizer"
            ],
            False,
        )
        self.assertEqual(
            config_dict["compile_config"]["optimizer"]["class_name"],
            "Adam",
        )
        self.assertLen(config_dict["compile_config"]["loss"], 4)
        self.assertEqual(
            config_dict["compile_config"]["loss"][0],
            "mse",
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

        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")

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
        functional_model._save_experimental(temp_filepath)
        loaded_model = saving_lib.load_model(temp_filepath, safe_mode=False)
        self.assertEqual(
            functional_model._is_compiled, loaded_model._is_compiled
        )

        loaded_model.fit(x, y, epochs=3)
        loaded_to_string = ToString()
        loaded_model.summary(print_fn=loaded_to_string)

        # Confirming the original and saved/loaded model have same structure.
        self.assertEqual(
            functional_to_string.contents, loaded_to_string.contents
        )

    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(
            model_type=["sequential", "functional", "subclassed"],
        )
    )
    def test_saving_model_state(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        model = getattr(self, f"_get_{model_type}_model")()
        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        model.fit(x, y, epochs=1)

        # Assert that the archive has not been saved.
        self.assertFalse(os.path.exists(temp_filepath))

        # Mutate the `Dense` layer custom weights to ensure that list and
        # dict-contained weights get restored.
        model.layers[1].additional_weights[0].assign(2)
        model.layers[1].weights_in_dict["my_weight"].assign(2)
        model.layers[1].nested_layer.kernel.assign([[1]])

        model._save_experimental(temp_filepath)

        # Assert that the archive has been saved.
        self.assertTrue(os.path.exists(temp_filepath))
        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(model._is_compiled, loaded_model._is_compiled)

        # The weights are supposed to be the same (between original and loaded
        # models).
        for original_weights, loaded_weights in zip(
            model.get_weights(), loaded_model.get_weights()
        ):
            np.testing.assert_allclose(original_weights, loaded_weights)

        # The optimizer variables are supposed to be the same (between original
        # and loaded models).
        for original_weights, loaded_weights in zip(
            model.optimizer.variables, loaded_model.optimizer.variables
        ):
            np.testing.assert_allclose(original_weights, loaded_weights)

    def test_saving_custom_assets_and_variables(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        model = ModelWithCustomSaving()
        model.compile(
            optimizer=adam.Adam(),
            loss=[
                "mse",
                keras.losses.mean_squared_error,
                keras.losses.MeanSquaredError(),
                my_mean_squared_error,
            ],
        )
        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        model.fit(x, y, epochs=1)

        # Assert that the archive has not been saved.
        self.assertFalse(os.path.exists(temp_filepath))

        model._save_experimental(temp_filepath)

        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(loaded_model.custom_dense.assets, assets_data)
        self.assertEqual(
            loaded_model.custom_dense.stored_variables.tolist(),
            variables_data.tolist(),
        )

    @tf.__internal__.distribute.combinations.generate(
        tf.__internal__.test.combinations.combine(
            model_type=["subclassed", "sequential"],
        )
    )
    def test_compile_overridden_model_raises_if_no_from_config_overridden(
        self, model_type
    ):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        model = (
            CompileOverridingModel()
            if model_type == "subclassed"
            else CompileOverridingSequential(
                [keras.layers.Embedding(4, 1), MyDense(1), MyDense(1)]
            )
        )
        model.compile("rmsprop", "mse")
        model._save_experimental(temp_filepath)

        with mock.patch.object(logging, "warning") as mock_warn:
            saving_lib.load_model(temp_filepath)
        if not mock_warn.call_args_list:
            raise AssertionError("Did not warn.")
        self.assertIn(
            "`compile()` was not called as part of model loading "
            "because the model's `compile()` method is custom. ",
            mock_warn.call_args_list[0][0][0],
        )

    def test_metadata(self):
        temp_filepath = Path(
            os.path.join(self.get_temp_dir(), "my_model.keras")
        )
        model = CompileOverridingModel()
        model._save_experimental(temp_filepath)
        with zipfile.ZipFile(temp_filepath, "r") as z:
            with z.open(saving_lib._METADATA_FILENAME, "r") as c:
                metadata_json = c.read()
        metadata = json_utils.decode(metadata_json)
        self.assertIn("keras_version", metadata)
        self.assertIn("date_saved", metadata)

    def test_gfile_copy_local_called(self):
        temp_filepath = Path(
            os.path.join(self.get_temp_dir(), "my_model.keras")
        )
        model = CompileOverridingModel()
        with mock.patch("re.match", autospec=True) as mock_re_match, mock.patch(
            "tensorflow.compat.v2.io.gfile.copy", autospec=True
        ) as mock_copy:
            # Mock Remote Path check to true to test gfile copy logic
            mock_re_match.return_value = True
            model._save_experimental(temp_filepath)
            mock_re_match.assert_called()
            mock_copy.assert_called()
            self.assertIn(str(temp_filepath), mock_re_match.call_args.args)
            self.assertIn(str(temp_filepath), mock_copy.call_args.args)

    def test_load_model_api_endpoint(self):
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "mymodel.keras"))
        model = self._get_functional_model()
        ref_input = np.random.random((10, 32))
        ref_output = model.predict(ref_input)
        model.save(temp_filepath, save_format="keras_v3")
        model = keras.models.load_model(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    def test_save_load_weights_only(self):
        temp_filepath = Path(
            os.path.join(self.get_temp_dir(), "mymodel.weights.h5")
        )
        model = self._get_functional_model()
        ref_input = np.random.random((10, 32))
        ref_output = model.predict(ref_input)
        saving_lib.save_weights_only(model, temp_filepath)
        model = self._get_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)
        # Test with Model method
        model = self._get_functional_model()
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    def test_load_weights_only_with_keras_file(self):
        # Test loading weights from whole saved model
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "mymodel.keras"))
        model = self._get_functional_model()
        ref_input = np.random.random((10, 32))
        ref_output = model.predict(ref_input)
        saving_lib.save_model(model, temp_filepath)
        model = self._get_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)
        # Test with Model method
        model = self._get_functional_model()
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    def test_compile_arg(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        model = self._get_functional_model()
        model.compile("rmsprop", "mse")
        model.fit(np.random.random((10, 32)), np.random.random((10, 1)))
        saving_lib.save_model(model, temp_filepath)

        model = saving_lib.load_model(temp_filepath)
        self.assertEqual(model._is_compiled, True)
        model = saving_lib.load_model(temp_filepath, compile=False)
        self.assertEqual(model._is_compiled, False)

    def test_overwrite(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        model = self._get_functional_model()
        model.save(temp_filepath, save_format="keras_v3")
        model.save(temp_filepath, save_format="keras_v3", overwrite=True)
        with self.assertRaises(EOFError):
            model.save(temp_filepath, save_format="keras_v3", overwrite=False)

        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.weights.h5")
        model = self._get_functional_model()
        model.save_weights(temp_filepath)
        model.save_weights(temp_filepath, overwrite=True)
        with self.assertRaises(EOFError):
            model.save_weights(temp_filepath, overwrite=False)

    def test_partial_load(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        original_model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Dense(4),
                keras.layers.Dense(5),
            ]
        )
        original_model.save(temp_filepath, save_format="keras_v3")

        # Test with a model that has a differently shaped layer
        new_model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Dense(4),
                keras.layers.Dense(6),
            ]
        )
        new_layer_kernel_value = new_model.layers[1].kernel.numpy()
        with self.assertRaisesRegex(ValueError, "Shape mismatch"):
            # Doesn't work by default
            new_model.load_weights(temp_filepath)
        # Now it works
        new_model.load_weights(temp_filepath, skip_mismatch=True)
        self.assertAllClose(
            original_model.layers[0].get_weights(),
            new_model.layers[0].get_weights(),
        )
        self.assertAllClose(
            new_model.layers[1].kernel.numpy(), new_layer_kernel_value
        )

        # Test with a model that has a new layer
        new_model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Dense(4),
                keras.layers.Dense(5),
                keras.layers.Dense(5),
            ]
        )
        new_layer_kernel_value = new_model.layers[2].kernel.numpy()
        with self.assertRaisesRegex(ValueError, "received 0 variables"):
            # Doesn't work by default
            new_model.load_weights(temp_filepath)
        # Now it works
        new_model.load_weights(temp_filepath, skip_mismatch=True)
        self.assertAllClose(
            original_model.layers[0].get_weights(),
            new_model.layers[0].get_weights(),
        )
        self.assertAllClose(
            original_model.layers[1].get_weights(),
            new_model.layers[1].get_weights(),
        )
        self.assertAllClose(
            new_model.layers[2].kernel.numpy(), new_layer_kernel_value
        )

    def test_api_errors(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.notkeras")
        model = self._get_functional_model()
        with self.assertRaisesRegex(ValueError, "Unknown `save_format`"):
            model.save(temp_filepath, save_format="invalid")
        with self.assertRaisesRegex(ValueError, "Invalid `filepath` argument"):
            model.save(temp_filepath, save_format="keras_v3")

        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        with self.assertRaisesRegex(ValueError, "not supported"):
            model.save(
                temp_filepath, include_optimizer=False, save_format="keras_v3"
            )

    def test_safe_mode(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "unsafe_model.keras")
        model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Lambda(lambda x: x * 2),
            ]
        )
        model.save(temp_filepath, save_format="keras_v3")
        with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
            model = saving_lib.load_model(temp_filepath)
        model = saving_lib.load_model(temp_filepath, safe_mode=False)

    def test_normalization_kpl(self):
        # With adapt
        temp_filepath = os.path.join(self.get_temp_dir(), "norm_model.keras")
        model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Normalization(),
            ]
        )
        data = np.random.random((3, 3))
        model.layers[0].adapt(data)
        ref_out = model(data)
        model.save(temp_filepath, save_format="keras_v3")
        model = saving_lib.load_model(temp_filepath)
        out = model(data)
        self.assertAllClose(ref_out, out, atol=1e-6)

        # Without adapt
        model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Normalization(
                    mean=np.random.random((3,)), variance=np.random.random((3,))
                ),
            ]
        )
        ref_out = model(data)
        model.save(temp_filepath, save_format="keras_v3")
        model = saving_lib.load_model(temp_filepath)
        out = model(data)
        self.assertAllClose(ref_out, out, atol=1e-6)


if __name__ == "__main__":
    tf.test.main()
