"""Tests for Keras python-based idempotent saving functions."""

import json
import os
import warnings
import zipfile
from io import BytesIO
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

import keras
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.saving import saving_lib


@keras.saving.register_keras_serializable(package="my_custom_package")
class MyDense(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.nested_layer = keras.layers.Dense(self.units, name="dense")

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


@keras.saving.register_keras_serializable(package="my_custom_package")
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


@keras.saving.register_keras_serializable(package="my_custom_package")
class CustomModelX(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = MyDense(1, name="my_dense_1")
        self.dense2 = MyDense(1, name="my_dense_2")

    def call(self, inputs):
        out = self.dense1(inputs)
        return self.dense2(out)

    def one(self):
        return 1


@keras.saving.register_keras_serializable(package="my_custom_package")
class ModelWithCustomSaving(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_dense = LayerWithCustomSaving(1)

    def call(self, inputs):
        return self.custom_dense(inputs)


@keras.saving.register_keras_serializable(package="my_custom_package")
class CompileOverridingModel(keras.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dense1 = MyDense(1)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)

    def call(self, inputs):
        return self.dense1(inputs)


@keras.saving.register_keras_serializable(package="my_custom_package")
class CompileOverridingSequential(keras.Sequential):
    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)


@keras.saving.register_keras_serializable(package="my_custom_package")
class SubclassFunctional(keras.Model):
    """Subclassed functional identical to `_get_basic_functional_model`."""

    def __init__(self, **kwargs):
        inputs = keras.Input(shape=(4,), batch_size=2)
        dense = keras.layers.Dense(1, name="first_dense")
        x = dense(inputs)
        outputs = keras.layers.Dense(1, name="second_dense")(x)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        # Attrs for layers in the functional graph should not affect saving
        self.layer_attr = dense

    @property
    def layer_property(self):
        # Properties for layers in the functional graph should not affect saving
        return self.layer_attr

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@keras.saving.register_keras_serializable(package="my_custom_package")
def my_mean_squared_error(y_true, y_pred):
    """Identical to built-in `mean_squared_error`, but as a custom fn."""
    return ops.mean(ops.square(y_pred - y_true), axis=-1)


def _get_subclassed_model(compile=True):
    subclassed_model = CustomModelX(name="custom_model_x")
    if compile:
        subclassed_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return subclassed_model


def _get_custom_sequential_model(compile=True):
    sequential_model = keras.Sequential(
        [MyDense(1), MyDense(1)], name="sequential"
    )
    if compile:
        sequential_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return sequential_model


def _get_basic_sequential_model(compile=True):
    sequential_model = keras.Sequential(
        [
            keras.layers.Dense(1, name="dense_1"),
            keras.layers.Dense(1, name="dense_2"),
        ],
        name="sequential",
    )
    if compile:
        sequential_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return sequential_model


def _get_custom_functional_model(compile=True):
    inputs = keras.Input(shape=(4,), batch_size=2)
    x = MyDense(1, name="first_dense")(inputs)
    outputs = MyDense(1, name="second_dense")(x)
    functional_model = keras.Model(inputs, outputs)
    if compile:
        functional_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return functional_model


def _get_basic_functional_model(compile=True):
    inputs = keras.Input(shape=(4,), batch_size=2)
    x = keras.layers.Dense(1, name="first_dense")(inputs)
    outputs = keras.layers.Dense(1, name="second_dense")(x)
    functional_model = keras.Model(inputs, outputs)
    if compile:
        functional_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return functional_model


def _get_subclassed_functional_model(compile=True):
    functional_model = SubclassFunctional()
    if compile:
        functional_model.compile(
            optimizer="adam",
            loss=my_mean_squared_error,
            metrics=[keras.metrics.Hinge(), "mse"],
        )
    return functional_model


# We need a global function for `Pool.apply_async`
def _load_model_fn(filepath):
    saving_lib.load_model(filepath)


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

    def test_inference_after_instantiation_subclassed_functional(self):
        model = _get_subclassed_functional_model(compile=False)
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

    @pytest.mark.requires_trainable_backend
    def test_compile_preserved_subclassed(self):
        model = _get_subclassed_model(compile=True)
        self._test_compile_preserved(model)

    @pytest.mark.requires_trainable_backend
    def test_compile_preserved_basic_sequential(self):
        model = _get_basic_sequential_model(compile=True)
        self._test_compile_preserved(model)

    @pytest.mark.requires_trainable_backend
    def test_compile_preserved_custom_sequential(self):
        model = _get_custom_sequential_model(compile=True)
        self._test_compile_preserved(model)

    @pytest.mark.requires_trainable_backend
    def test_compile_preserved_basic_functional(self):
        model = _get_basic_functional_model(compile=True)
        self._test_compile_preserved(model)

    @pytest.mark.requires_trainable_backend
    def test_compile_preserved_custom_functional(self):
        model = _get_custom_functional_model(compile=True)
        self._test_compile_preserved(model)

    @pytest.mark.requires_trainable_backend
    def test_compile_preserved_subclassed_functional(self):
        model = _get_subclassed_functional_model(compile=True)
        self._test_compile_preserved(model)

    def test_saving_preserve_unbuilt_state(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "my_model.keras")
        subclassed_model = CustomModelX()
        subclassed_model.save(temp_filepath)
        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(subclassed_model.compiled, loaded_model.compiled)
        self.assertFalse(subclassed_model.built)
        self.assertFalse(loaded_model.built)

    @pytest.mark.requires_trainable_backend
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
            keras.src.saving.serialize_keras_object(
                keras.src.optimizers.get("adam")
            ),
        )
        self.assertEqual(
            config_dict["compile_config"]["loss"]["config"],
            "my_mean_squared_error",
        )

    @pytest.mark.requires_trainable_backend
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
                [keras.layers.Embedding(4, 1), MyDense(1), MyDense(1)]
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

    def test_save_weights_subclassed_functional(self):
        # The subclassed and basic functional model should have the same
        # weights structure.
        temp_filepath = Path(
            os.path.join(self.get_temp_dir(), "mymodel.weights.h5")
        )
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        # Test saving basic, loading subclassed.
        saving_lib.save_weights_only(model, temp_filepath)
        model = _get_subclassed_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)
        # Test saving subclassed, loading basic.
        saving_lib.save_weights_only(model, temp_filepath)
        model = _get_basic_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    @pytest.mark.requires_trainable_backend
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
        original_model = keras.Sequential(
            [
                keras.Input(shape=(3,), batch_size=2),
                keras.layers.Dense(4),
                keras.layers.Dense(5),
            ]
        )
        original_model.save(temp_filepath)

        # Test with a model that has a differently shaped layer
        new_model = keras.Sequential(
            [
                keras.Input(shape=(3,), batch_size=2),
                keras.layers.Dense(4),
                keras.layers.Dense(6),
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
        new_model = keras.Sequential(
            [
                keras.Input(shape=(3,), batch_size=2),
                keras.layers.Dense(4),
                keras.layers.Dense(5),
                keras.layers.Dense(5),
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

    @pytest.mark.requires_trainable_backend
    def test_save_to_fileobj(self) -> None:
        model = keras.Sequential(
            [keras.layers.Dense(1, input_shape=(1,)), keras.layers.Dense(1)]
        )
        model.compile(optimizer="adam", loss="mse")

        out = BytesIO()
        saving_lib.save_model(model, out)
        out.seek(0)
        model = saving_lib.load_model(out)

        model.fit(np.array([1, 2]), np.array([1, 2]))
        pred1 = model.predict(np.array([1, 2]))

        out = BytesIO()
        saving_lib.save_model(model, out)
        out.seek(0)
        new_model = saving_lib.load_model(out)

        pred2 = new_model.predict(np.array([1, 2]))

        self.assertAllClose(pred1, pred2, atol=1e-5)

    def test_save_model_exception_raised(self):
        # Assume we have an error in `save_own_variables`.
        class RaiseErrorLayer(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def call(self, inputs):
                return inputs

            def save_own_variables(self, store):
                raise ValueError

        model = keras.Sequential([keras.Input([1]), RaiseErrorLayer()])
        filepath = f"{self.get_temp_dir()}/model.keras"
        with self.assertRaises(ValueError):
            saving_lib.save_model(model, filepath)

        # Ensure we don't have a bad "model.weights.h5" inside the zip file.
        self.assertTrue(Path(filepath).exists())
        with zipfile.ZipFile(filepath) as zf:
            all_filenames = zf.namelist()
            self.assertNotIn("model.weights.h5", all_filenames)

        # Ensure we don't have any temporary files left.
        self.assertLen(os.listdir(Path(filepath).parent), 1)
        self.assertIn("model.keras", os.listdir(Path(filepath).parent))

    def test_load_model_exception_raised(self):
        # Assume we have an error in `load_own_variables`.
        class RaiseErrorLayer(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)

            def call(self, inputs):
                return inputs

            def load_own_variables(self, store):
                raise ValueError

        model = keras.Sequential([keras.Input([1]), RaiseErrorLayer()])
        filepath = f"{self.get_temp_dir()}/model.keras"
        saving_lib.save_model(model, filepath)
        with self.assertRaises(ValueError):
            saving_lib.load_model(
                filepath, custom_objects={"RaiseErrorLayer": RaiseErrorLayer}
            )

        # Ensure we don't have any temporary files left.
        self.assertLen(os.listdir(Path(filepath).parent), 1)
        self.assertIn("model.keras", os.listdir(Path(filepath).parent))

    def test_load_model_read_only_system(self):
        model = keras.Sequential([keras.Input([1]), keras.layers.Dense(2)])
        filepath = f"{self.get_temp_dir()}/model.keras"
        saving_lib.save_model(model, filepath)

        # Load the model correctly, regardless of whether an OSError occurs.
        original_mode = os.stat(Path(filepath).parent).st_mode
        os.chmod(Path(filepath).parent, mode=0o555)
        model = saving_lib.load_model(filepath)
        os.chmod(Path(filepath).parent, mode=original_mode)

        # Ensure we don't have any temporary files left.
        self.assertLen(os.listdir(Path(filepath).parent), 1)
        self.assertIn("model.keras", os.listdir(Path(filepath).parent))

    @pytest.mark.skipif(
        backend.backend() == "jax",
        reason="JAX backend doesn't support Python's multiprocessing",
    )
    @pytest.mark.skipif(
        testing.tensorflow_uses_gpu() or testing.torch_uses_gpu(),
        reason="This test doesn't support GPU",
    )
    def test_load_model_concurrently(self):
        import multiprocessing as mp

        model = keras.Sequential([keras.Input([1]), keras.layers.Dense(2)])
        filepath = f"{self.get_temp_dir()}/model.keras"
        saving_lib.save_model(model, filepath)

        # Load the model concurrently.
        results = []
        with mp.Pool(4) as pool:
            for i in range(4):
                results.append(pool.apply_async(_load_model_fn, (filepath,)))
            pool.close()
            pool.join()
        [r.get() for r in results]  # No error occurs here


@pytest.mark.requires_trainable_backend
class SavingAPITest(testing.TestCase):
    def test_saving_api_errors(self):
        from keras.src.saving import saving_api

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

    def test_model_api_endpoint(self):
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "mymodel.keras"))
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        model.save(temp_filepath)
        model = keras.saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    def test_model_api_endpoint_h5(self):
        temp_filepath = Path(os.path.join(self.get_temp_dir(), "mymodel.h5"))
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        model.save(temp_filepath)
        model = keras.saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-6)

    def test_model_api_errors(self):
        model = _get_basic_functional_model()

        # Saving API errors
        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel")
        with self.assertRaisesRegex(ValueError, "argument is deprecated"):
            model.save(temp_filepath, save_format="keras")

        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.notkeras")
        with self.assertRaisesRegex(ValueError, "Invalid filepath extension"):
            model.save(temp_filepath)

        temp_filepath = os.path.join(self.get_temp_dir(), "mymodel.keras")
        with self.assertRaisesRegex(ValueError, "are not supported"):
            model.save(temp_filepath, invalid_arg="hello")

    def test_safe_mode(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "unsafe_model.keras")
        model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Lambda(lambda x: x * 2),
            ]
        )
        model.save(temp_filepath)
        with self.assertRaisesRegex(ValueError, "Deserializing it is unsafe"):
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
        model.save(temp_filepath)
        model = saving_lib.load_model(temp_filepath)
        out = model(data)
        self.assertAllClose(ref_out, out, atol=1e-6)

        # Without adapt
        model = keras.Sequential(
            [
                keras.Input(shape=(3,)),
                keras.layers.Normalization(
                    mean=np.random.random((3,)),
                    variance=np.random.random((3,)),
                ),
            ]
        )
        ref_out = model(data)
        model.save(temp_filepath)
        model = saving_lib.load_model(temp_filepath)
        out = model(data)
        self.assertAllClose(ref_out, out, atol=1e-6)


# This class is properly registered with a `get_config()` method.
# However, since it does not subclass keras.layers.Layer, it lacks
# `from_config()` for deserialization.
@keras.saving.register_keras_serializable()
class GrowthFactor:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, inputs):
        return inputs * self.factor

    def get_config(self):
        return {"factor": self.factor}


@keras.saving.register_keras_serializable(package="Complex")
class FactorLayer(keras.layers.Layer):
    def __init__(self, factor, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return x * self.factor

    def get_config(self):
        return {"factor": self.factor}


# This custom model does not explicitly deserialize the layers it includes
# in its `get_config`. Explicit deserialization in a `from_config` override
# or `__init__` is needed here, or an error will be thrown at loading time.
@keras.saving.register_keras_serializable(package="Complex")
class ComplexModel(keras.layers.Layer):
    def __init__(self, first_layer, second_layer=None, **kwargs):
        super().__init__(**kwargs)
        self.first_layer = first_layer
        if second_layer is not None:
            self.second_layer = second_layer
        else:
            self.second_layer = keras.layers.Dense(8)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "first_layer": self.first_layer,
                "second_layer": self.second_layer,
            }
        )
        return config

    def call(self, inputs):
        return self.first_layer(self.second_layer(inputs))


class SavingBattleTest(testing.TestCase):
    def test_custom_object_without_from_config(self):
        temp_filepath = os.path.join(
            self.get_temp_dir(), "custom_fn_model.keras"
        )

        inputs = keras.Input(shape=(4, 4))
        outputs = keras.layers.Dense(1, activation=GrowthFactor(0.5))(inputs)
        model = keras.Model(inputs, outputs)

        model.save(temp_filepath)

        with self.assertRaisesRegex(
            TypeError, "Unable to reconstruct an instance"
        ):
            _ = saving_lib.load_model(temp_filepath)

    def test_complex_model_without_explicit_deserialization(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "complex_model.keras")

        inputs = keras.Input((32,))
        outputs = ComplexModel(first_layer=FactorLayer(0.5))(inputs)
        model = keras.Model(inputs, outputs)

        model.save(temp_filepath)

        with self.assertRaisesRegex(TypeError, "are explicitly deserialized"):
            _ = saving_lib.load_model(temp_filepath)

    def test_redefinition_of_trackable(self):
        """Test that a trackable can be aliased under a new name."""

        class NormalModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.dense = keras.layers.Dense(3)

            def call(self, x):
                return self.dense(x)

        class WeirdModel(keras.Model):
            def __init__(self):
                super().__init__()
                # This property will be traversed first,
                # but "_dense" isn't in the saved file
                # generated by NormalModel.
                self.a_dense = keras.layers.Dense(3)

            @property
            def dense(self):
                return self.a_dense

            def call(self, x):
                return self.dense(x)

        temp_filepath = os.path.join(
            self.get_temp_dir(), "normal_model.weights.h5"
        )
        model_a = NormalModel()
        model_a(np.random.random((2, 2)))
        model_a.save_weights(temp_filepath)
        model_b = WeirdModel()
        model_b(np.random.random((2, 2)))
        model_b.load_weights(temp_filepath)
        self.assertAllClose(
            model_a.dense.kernel.numpy(), model_b.dense.kernel.numpy()
        )

    def test_legacy_h5_format(self):
        temp_filepath = os.path.join(self.get_temp_dir(), "custom_model.h5")

        inputs = keras.Input((32,))
        x = MyDense(2)(inputs)
        outputs = CustomModelX()(x)
        model = keras.Model(inputs, outputs)

        x = np.random.random((1, 32))
        ref_out = model(x)

        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        out = new_model(x)
        self.assertAllClose(ref_out, out, atol=1e-6)

    def test_nested_functional_model_saving(self):
        def func(in_size=4, out_size=2, name=None):
            inputs = keras.layers.Input(shape=(in_size,))
            outputs = keras.layers.Dense(out_size)((inputs))
            return keras.Model(inputs, outputs=outputs, name=name)

        input_a, input_b = keras.Input((4,)), keras.Input((4,))
        out_a = func(out_size=2, name="func_a")(input_a)
        out_b = func(out_size=3, name="func_b")(input_b)
        model = keras.Model([input_a, input_b], outputs=[out_a, out_b])

        temp_filepath = os.path.join(self.get_temp_dir(), "nested_func.keras")
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        x = [np.random.random((2, 4))], np.random.random((2, 4))
        ref_out = model(x)
        out = new_model(x)
        self.assertAllClose(ref_out[0], out[0])
        self.assertAllClose(ref_out[1], out[1])

    def test_nested_shared_functional_model_saving(self):
        def func(in_size=4, out_size=2, name=None):
            inputs = keras.layers.Input(shape=(in_size,))
            outputs = keras.layers.Dense(out_size)((inputs))
            return keras.Model(inputs, outputs=outputs, name=name)

        inputs = [keras.Input((4,)), keras.Input((4,))]
        func_shared = func(out_size=4, name="func_shared")
        shared_a = func_shared(inputs[0])
        shared_b = func_shared(inputs[1])
        out_a = keras.layers.Dense(2)(shared_a)
        out_b = keras.layers.Dense(2)(shared_b)
        model = keras.Model(inputs, outputs=[out_a, out_b])

        temp_filepath = os.path.join(
            self.get_temp_dir(), "nested_shared_func.keras"
        )
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        x = [np.random.random((2, 4))], np.random.random((2, 4))
        ref_out = model(x)
        out = new_model(x)
        self.assertAllClose(ref_out[0], out[0])
        self.assertAllClose(ref_out[1], out[1])

    def test_bidirectional_lstm_saving(self):
        inputs = keras.Input((3, 2))
        outputs = keras.layers.Bidirectional(keras.layers.LSTM(64))(inputs)
        model = keras.Model(inputs, outputs)
        temp_filepath = os.path.join(self.get_temp_dir(), "bidir_lstm.keras")
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        x = np.random.random((1, 3, 2))
        ref_out = model(x)
        out = new_model(x)
        self.assertAllClose(ref_out, out)
