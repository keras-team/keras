"""Tests for Keras python-based idempotent saving functions."""
import json
import os
import warnings
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import keras_core
from keras_core import ops
from keras_core import testing
from keras_core.saving import saving_lib


@keras_core.saving.register_keras_serializable(package="my_custom_package")
class MyDense(keras_core.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.nested_layer = keras_core.layers.Dense(self.units, name="dense")

    def build(self, input_shape):
        self.additional_weights = [
            self.add_weight(
                shape=(),
                name="my_additional_weight",
                initializer="ones",
                trainable=True,
            ),
            self.add_weight(
                shape=(),
                name="my_additional_weight_2",
                initializer="ones",
                trainable=True,
            ),
        ]
        self.weights_in_dict = {
            "my_weight": self.add_weight(
                shape=(),
                name="my_dict_weight",
                initializer="ones",
                trainable=True,
            ),
        }
        self.nested_layer.build(input_shape)

    def call(self, inputs):
        return self.nested_layer(inputs)

    def two(self):
        return 2


ASSETS_DATA = "These are my assets"
VARIABLES_DATA = np.random.random((10,))


@keras_core.saving.register_keras_serializable(package="my_custom_package")
class LayerWithCustomSaving(MyDense):
    def build(self, input_shape):
        self.assets = ASSETS_DATA
        self.stored_variables = VARIABLES_DATA
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


@keras_core.saving.register_keras_serializable(package="my_custom_package")
class CustomModelX(keras_core.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = MyDense(1, name="my_dense_1")
        self.dense2 = MyDense(1, name="my_dense_2")

    def call(self, inputs):
        out = self.dense1(inputs)
        return self.dense2(out)

    def one(self):
        return 1


@keras_core.saving.register_keras_serializable(package="my_custom_package")
class ModelWithCustomSaving(keras_core.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_dense = LayerWithCustomSaving(1)

    def call(self, inputs):
        return self.custom_dense(inputs)


@keras_core.saving.register_keras_serializable(package="my_custom_package")
class CompileOverridingModel(keras_core.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = MyDense(1)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)

    def call(self, inputs):
        return self.dense1(inputs)


@keras_core.saving.register_keras_serializable(package="my_custom_package")
class CompileOverridingSequential(keras_core.Sequential):
    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)


@keras_core.saving.register_keras_serializable(package="my_custom_package")
def my_mean_squared_error(y_true, y_pred):
    """Identical to built-in `mean_squared_error`, but as a custom fn."""
    return ops.mean(ops.square(y_pred - y_true), axis=-1)


def _get_subclassed_model(compile=True):
    subclassed_model = CustomModelX(name="custom_model_x")
    if compile:
        subclassed_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras_core.metrics.Hinge(), "mse"],
        )
    return subclassed_model


def _get_custom_sequential_model(compile=True):
    sequential_model = keras_core.Sequential([MyDense(1), MyDense(1)])
    if compile:
        sequential_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras_core.metrics.Hinge(), "mse"],
        )
    return sequential_model


def _get_basic_sequential_model(compile=True):
    sequential_model = keras_core.Sequential(
        [
            keras_core.layers.Dense(1, name="dense_1"),
            keras_core.layers.Dense(1, name="dense_2"),
        ]
    )
    if compile:
        sequential_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras_core.metrics.Hinge(), "mse"],
        )
    return sequential_model


def _get_custom_functional_model(compile=True):
    inputs = keras_core.Input(shape=(4,), batch_size=2)
    x = MyDense(1, name="first_dense")(inputs)
    outputs = MyDense(1, name="second_dense")(x)
    functional_model = keras_core.Model(inputs, outputs)
    if compile:
        functional_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras_core.metrics.Hinge(), "mse"],
        )
    return functional_model


def _get_basic_functional_model(compile=True):
    inputs = keras_core.Input(shape=(4,), batch_size=2)
    x = keras_core.layers.Dense(1, name="first_dense")(inputs)
    outputs = keras_core.layers.Dense(1, name="second_dense")(x)
    functional_model = keras_core.Model(inputs, outputs)
    if compile:
        functional_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras_core.metrics.Hinge(), "mse"],
        )
    return functional_model


@pytest.mark.requires_trainable_backend
class SavingTest(testing.TestCase):
    def _test_inference_after_instantiation(self, model):
        x_ref = np.random.random((2, 4))
        y_ref = model(x_ref)
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        model.save(temp_filepath)

        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertFalse(model.compiled)
        for w_ref, w in zip(model.variables, loaded_model.variables):
            self.assertAllClose(w_ref, w)
        self.assertAllClose(y_ref, loaded_model(x_ref))

    def test_inference_after_instantiation_subclassed(self):
        model = _get_subclassed_model(compile=False)
        self._test_inference_after_instantiation(model)

    def test_inference_after_instantiation_basic_sequential(self):
        model = _get_basic_sequential_model(compile=False)
        self._test_inference_after_instantiation(model)

    def test_inference_after_instantiation_basic_functional(self):
        model = _get_basic_functional_model(compile=False)
        self._test_inference_after_instantiation(model)

    def test_inference_after_instantiation_custom_sequential(self):
        model = _get_custom_sequential_model(compile=False)
        self._test_inference_after_instantiation(model)

    def test_inference_after_instantiation_custom_functional(self):
        model = _get_custom_functional_model(compile=False)
        self._test_inference_after_instantiation(model)

    def _test_compile_preserved(self, model):
        x_ref = np.random.random((2, 4))
        y_ref = np.random.random((2, 1))

        model.fit(x_ref, y_ref)
        out_ref = model(x_ref)
        ref_metrics = model.evaluate(x_ref, y_ref)
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        model.save(temp_filepath)

        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertTrue(model.compiled)
        self.assertTrue(loaded_model.built)
        for w_ref, w in zip(model.variables, loaded_model.variables):
            self.assertAllClose(w_ref, w)
        self.assertAllClose(out_ref, loaded_model(x_ref))

        self.assertEqual(
            model.optimizer.__class__, loaded_model.optimizer.__class__
        )
        self.assertEqual(
            model.optimizer.get_config(), loaded_model.optimizer.get_config()
        )
        for w_ref, w in zip(
            model.optimizer.variables, loaded_model.optimizer.variables
        ):
            self.assertAllClose(w_ref, w)

        new_metrics = loaded_model.evaluate(x_ref, y_ref)
        for ref_m, m in zip(ref_metrics, new_metrics):
            self.assertAllClose(ref_m, m)

    def test_compile_preserved_subclassed(self):
        model = _get_subclassed_model(compile=True)
        self._test_compile_preserved(model)

    def test_compile_preserved_basic_sequential(self):
        model = _get_basic_sequential_model(compile=True)
        self._test_compile_preserved(model)

    def test_compile_preserved_custom_sequential(self):
        model = _get_custom_sequential_model(compile=True)
        self._test_compile_preserved(model)

    def test_compile_preserved_basic_functional(self):
        model = _get_basic_functional_model(compile=True)
        self._test_compile_preserved(model)

    def test_compile_preserved_custom_functional(self):
        model = _get_custom_functional_model(compile=True)
        self._test_compile_preserved(model)

    def test_saving_preserve_unbuilt_state(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        subclassed_model = CustomModelX()
        subclassed_model.save(temp_filepath)
        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(subclassed_model.compiled, loaded_model.compiled)
        self.assertFalse(subclassed_model.built)
        self.assertFalse(loaded_model.built)

    def test_saved_module_paths_and_class_names(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        subclassed_model = _get_subclassed_model()
        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        subclassed_model.fit(x, y, epochs=1)
        subclassed_model.save(temp_filepath)

        with zipfile.ZipFile(temp_filepath, "r") as z:
            with z.open(saving_lib._CONFIG_FILENAME, "r") as c:
                config_json = c.read()
        config_dict = json.loads(config_json)
        self.assertEqual(
            config_dict["registered_name"], "my_custom_package>CustomModelX"
        )
        self.assertEqual(
            config_dict["compile_config"]["optimizer"],
            "adam",
        )
        print(config_dict["compile_config"])
        self.assertEqual(
            config_dict["compile_config"]["loss"]["config"],
            "my_mean_squared_error",
        )

    def test_saving_custom_assets_and_variables(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        model = ModelWithCustomSaving()
        model.compile(
            optimizer="adam",
            loss="mse",
        )
        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        model.fit(x, y, epochs=1)

        # Assert that the archive has not been saved.
        self.assertFalse(os.path.exists(temp_filepath))

        model.save(temp_filepath)

        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(loaded_model.custom_dense.assets, ASSETS_DATA)
        self.assertEqual(
            loaded_model.custom_dense.stored_variables.tolist(),
            VARIABLES_DATA.tolist(),
        )

    def _test_compile_overridden_warnings(self, model_type):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        model = (
            CompileOverridingModel()
            if model_type == "subclassed"
            else CompileOverridingSequential(
                [keras_core.layers.Embedding(4, 1), MyDense(1), MyDense(1)]
            )
        )
        model.compile("sgd", "mse")
        model.save(temp_filepath)

        with mock.patch.object(warnings, "warn") as mock_warn:
            saving_lib.load_model(temp_filepath)
        if not mock_warn.call_args_list:
            raise AssertionError("Did not warn.")
        self.assertIn(
            "`compile()` was not called as part of model loading "
            "because the model's `compile()` method is custom. ",
            mock_warn.call_args_list[0][0][0],
        )

    def test_compile_overridden_warnings_sequential(self):
        self._test_compile_overridden_warnings("sequential")

    def test_compile_overridden_warnings_subclassed(self):
        self._test_compile_overridden_warnings("subclassed")

    def test_metadata(self):
        temp_filepath = Path(
            os.path.join(self.get_temp_dir(), "my_model.keras")
        )
        model = CompileOverridingModel()
        model.save(temp_filepath)
        with zipfile.ZipFile(temp_filepath, "r") as z:
            with z.open(saving_lib._METADATA_FILENAME, "r") as c:
                metadata_json = c.read()
        metadata = json.loads(metadata_json)
        self.assertIn("keras_version", metadata)
        self.assertIn("date_saved", metadata)

    # def test_gfile_copy_local_called(self):
    #     temp_filepath = Path(
    #         os.path.join(self.get_temp_dir(), "my_model.keras")
    #     )
    #     model = CompileOverridingModel()
    #     with mock.patch(
    #         "re.match", autospec=True
    #     ) as mock_re_match, mock.patch(
    #         "tensorflow.compat.v2.io.file_utils.copy", autospec=True
    #     ) as mock_copy:
    #         # Mock Remote Path check to true to test gfile copy logic
    #         mock_re_match.return_value = True
    #         model.save(temp_filepath)
    #         mock_re_match.assert_called()
    #         mock_copy.assert_called()
    #         self.assertIn(str(temp_filepath), mock_re_match.call_args.args)
    #         self.assertIn(str(temp_filepath), mock_copy.call_args.args)

    def test_load_model_api_endpoint(self):
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "mymodel.keras"))
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        model.save(temp_filepath)
        model = keras_core.saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    def test_save_load_weights_only(self):
        temp_filepath = Path(
            os.path.join(self.get_temp_dir(), "mymodel.weights.h5")
        )
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        saving_lib.save_weights_only(model, temp_filepath)
        model = _get_basic_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)
        # Test with Model method
        model = _get_basic_functional_model()
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    def test_load_weights_only_with_keras_file(self):
        # Test loading weights from whole saved model
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "mymodel.keras"))
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        saving_lib.save_model(model, temp_filepath)
        model = _get_basic_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)
        # Test with Model method
        model = _get_basic_functional_model()
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    def test_compile_arg(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        model = _get_basic_functional_model()
        model.compile("sgd", "mse")
        model.fit(np.random.random((2, 4)), np.random.random((2, 1)))
        saving_lib.save_model(model, temp_filepath)

        model = saving_lib.load_model(temp_filepath)
        self.assertEqual(model.compiled, True)
        model = saving_lib.load_model(temp_filepath, compile=False)
        self.assertEqual(model.compiled, False)

    # def test_overwrite(self):
    #     temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
    #     model = _get_basic_functional_model()
    #     model.save(temp_filepath)
    #     model.save(temp_filepath, overwrite=True)
    #     with self.assertRaises(EOFError):
    #         model.save(temp_filepath, overwrite=False)

    #     temp_filepath = os.path.join(
    #         self.get_temp_dir(), "mymodel.weights.h5"
    #     )
    #     model = _get_basic_functional_model()
    #     model.save_weights(temp_filepath)
    #     model.save_weights(temp_filepath, overwrite=True)
    #     with self.assertRaises(EOFError):
    #         model.save_weights(temp_filepath, overwrite=False)

    def test_partial_load(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        original_model = keras_core.Sequential(
            [
                keras_core.Input(shape=(3,), batch_size=2),
                keras_core.layers.Dense(4),
                keras_core.layers.Dense(5),
            ]
        )
        original_model.save(temp_filepath)

        # Test with a model that has a differently shaped layer
        new_model = keras_core.Sequential(
            [
                keras_core.Input(shape=(3,), batch_size=2),
                keras_core.layers.Dense(4),
                keras_core.layers.Dense(6),
            ]
        )
        new_layer_kernel_value = np.array(new_model.layers[1].kernel)
        with self.assertRaisesRegex(ValueError, "must match"):
            # Doesn't work by default
            new_model.load_weights(temp_filepath)
        # Now it works
        new_model.load_weights(temp_filepath, skip_mismatch=True)
        ref_weights = original_model.layers[0].get_weights()
        new_weights = new_model.layers[0].get_weights()
        self.assertEqual(len(ref_weights), len(new_weights))
        for ref_w, w in zip(ref_weights, new_weights):
            self.assertAllClose(ref_w, w)
        self.assertAllClose(
            np.array(new_model.layers[1].kernel), new_layer_kernel_value
        )

        # Test with a model that has a new layer at the end
        new_model = keras_core.Sequential(
            [
                keras_core.Input(shape=(3,), batch_size=2),
                keras_core.layers.Dense(4),
                keras_core.layers.Dense(5),
                keras_core.layers.Dense(5),
            ]
        )
        new_layer_kernel_value = np.array(new_model.layers[2].kernel)
        with self.assertRaisesRegex(ValueError, "received 0 variables"):
            # Doesn't work by default
            new_model.load_weights(temp_filepath)
        # Now it works
        new_model.load_weights(temp_filepath, skip_mismatch=True)
        for layer_index in [0, 1]:
            ref_weights = original_model.layers[layer_index].get_weights()
            new_weights = new_model.layers[layer_index].get_weights()
            self.assertEqual(len(ref_weights), len(new_weights))
            for ref_w, w in zip(ref_weights, new_weights):
                self.assertAllClose(ref_w, w)
        self.assertAllClose(
            np.array(new_model.layers[2].kernel), new_layer_kernel_value
        )

    def test_api_errors(self):
        from keras_core.saving import saving_api

        model = _get_basic_functional_model()

        # Saving API errors
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel")
        with self.assertRaisesRegex(ValueError, "argument is deprecated"):
            saving_api.save_model(model, temp_filepath, save_format="keras")

        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.notkeras")
        with self.assertRaisesRegex(ValueError, "Invalid filepath extension"):
            saving_api.save_model(model, temp_filepath)

        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        with self.assertRaisesRegex(ValueError, "are not supported"):
            saving_api.save_model(model, temp_filepath, invalid_arg="hello")

        # Loading API errors
        temp_filepath = os.path.join(self.get_temp_dir(), "non_existent.keras")
        with self.assertRaisesRegex(
            ValueError, "Please ensure the file is an accessible"
        ):
            _ = saving_api.load_model(temp_filepath)

        temp_filepath = os.path.join(self.get_temp_dir(), "my_saved_model")
        with self.assertRaisesRegex(ValueError, "File format not supported"):
            _ = saving_api.load_model(temp_filepath)


# def test_safe_mode(self):
#     temp_filepath = os.path.join(self.get_temp_dir(), "unsafe_model.keras")
#     model = keras_core.Sequential(
#         [
#             keras_core.Input(shape=(3,)),
#             keras_core.layers.Dense(2, activation=lambda x: x * 2),
#         ]
#     )
#     model.save(temp_filepath)
#     with self.assertRaisesRegex(ValueError, "arbitrary code execution"):
#         model = saving_lib.load_model(temp_filepath)
#     model = saving_lib.load_model(temp_filepath, safe_mode=False)

#     def test_normalization_kpl(self):
#         # With adapt
#         temp_filepath = os.path.join(self.get_temp_dir(), "norm_model.keras")
#         model = keras_core.Sequential(
#             [
#                 keras_core.Input(shape=(3,)),
#                 keras_core.layers.Normalization(),
#             ]
#         )
#         data = np.random.random((3, 3))
#         model.layers[0].adapt(data)
#         ref_out = model(data)
#         model.save(temp_filepath)
#         model = saving_lib.load_model(temp_filepath)
#         out = model(data)
#         self.assertAllClose(ref_out, out, atol=1e-6)

#         # Without adapt
#         model = keras_core.Sequential(
#             [
#                 keras_core.Input(shape=(3,)),
#                 keras_core.layers.Normalization(
#                     mean=np.random.random((3,)),
#                     variance=np.random.random((3,)),
#                 ),
#             ]
#         )
#         ref_out = model(data)
#         model.save(temp_filepath)
#         model = saving_lib.load_model(temp_filepath)
#         out = model(data)
#         self.assertAllClose(ref_out, out, atol=1e-6)


# # This custom class lacks custom object registration.
# class CustomRNN(keras_core.layers.Layer):
#     def __init__(self, units):
#         super(CustomRNN, self).__init__()
#         self.units = units
#         self.projection_1 = keras_core.layers.Dense(
#             units=units, activation="tanh"
#         )
#         self.projection_2 = keras_core.layers.Dense(
#             units=units, activation="tanh"
#         )
#         self.classifier = keras_core.layers.Dense(1)

#     def call(self, inputs):
#         outputs = []
#         state = ops.zeros(shape=(inputs.shape[0], self.units))
#         for t in range(inputs.shape[1]):
#             x = inputs[:, t, :]
#             h = self.projection_1(x)
#             y = h + self.projection_2(state)
#             state = y
#             outputs.append(y)
#         features = ops.stack(outputs, axis=1)
#         return self.classifier(features)


# # This class is properly registered with a `get_config()` method.
# # However, since it does not subclass keras_core.layers.Layer, it lacks
# # `from_config()` for deserialization.
# @keras_core.saving.register_keras_serializable()
# class GrowthFactor:
#     def __init__(self, factor):
#         self.factor = factor

#     def __call__(self, inputs):
#         return inputs * self.factor

#     def get_config(self):
#         return {"factor": self.factor}


# @keras_core.saving.register_keras_serializable(package="Complex")
# class FactorLayer(keras_core.layers.Layer):
#     def __init__(self, factor):
#         super().__init__()
#         self.factor = factor

#     def call(self, x):
#         return x * self.factor

#     def get_config(self):
#         return {"factor": self.factor}


# # This custom model does not explicitly deserialize the layers it includes
# # in its `get_config`. Explicit deserialization in a `from_config` override
# # or `__init__` is needed here, or an error will be thrown at loading time.
# @keras_core.saving.register_keras_serializable(package="Complex")
# class ComplexModel(keras_core.layers.Layer):
#     def __init__(self, first_layer, second_layer=None, **kwargs):
#         super().__init__(**kwargs)
#         self.first_layer = first_layer
#         if second_layer is not None:
#             self.second_layer = second_layer
#         else:
#             self.second_layer = keras_core.layers.Dense(8)

#     def get_config(self):
#         config = super().get_config()
#         config.update(
#             {
#                 "first_layer": self.first_layer,
#                 "second_layer": self.second_layer,
#             }
#         )
#         return config

#     def call(self, inputs):
#         return self.first_layer(self.second_layer(inputs))


# class SavingBattleTest(testing.TestCase):
#     def test_custom_model_without_registration_error(self):
#         temp_filepath = os.path.join(
#             self.get_temp_dir(), "my_custom_model.keras"
#         )
#         timesteps = 10
#         input_dim = 5
#         batch_size = 16

#         inputs = keras_core.Input(
#             batch_shape=(batch_size, timesteps, input_dim)
#         )
#         x = keras_core.layers.Conv1D(32, 3)(inputs)
#         outputs = CustomRNN(32)(x)

#         model = keras_core.Model(inputs, outputs)

#         with self.assertRaisesRegex(
#             TypeError, "is a custom class, please register it"
#         ):
#             model.save(temp_filepath)
#             _ = keras_core.models.load_model(temp_filepath)

#     def test_custom_object_without_from_config(self):
#         temp_filepath = os.path.join(
#             self.get_temp_dir(), "custom_fn_model.keras"
#         )

#         inputs = keras_core.Input(shape=(4, 4))
#         outputs = keras_core.layers.Dense(
#             1, activation=GrowthFactor(0.5)
#         )(inputs)
#         model = keras_core.Model(inputs, outputs)

#         model.save(temp_filepath)

#         with self.assertRaisesRegex(
#             TypeError, "Unable to reconstruct an instance"
#         ):
#             _ = keras_core.models.load_model(temp_filepath)

#     def test_complex_model_without_explicit_deserialization(self):
#         temp_filepath = os.path.join(
#             self.get_temp_dir(), "complex_model.keras"
#         )

#         inputs = keras_core.Input((32,))
#         outputs = ComplexModel(first_layer=FactorLayer(0.5))(inputs)
#         model = keras_core.Model(inputs, outputs)

#         model.save(temp_filepath)

#         with self.assertRaisesRegex(TypeError, "are explicitly deserialized"):
#             _ = keras_core.models.load_model(temp_filepath)
