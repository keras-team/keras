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
from absl.testing import parameterized
import keras
from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.saving import saving_lib

@keras.saving.register_keras_serializable(package='my_custom_package')
class MyDense(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        """Execute __init__ with input (units)."""
        super().__init__(**kwargs)
        self.units = units
        self.nested_layer = keras.layers.Dense(self.units, name='dense')

    def build(self, input_shape):
        """Execute build with input (input_shape)."""
        self.additional_weights = [self.add_weight(shape=(), name='my_additional_weight', initializer='ones', trainable=True), self.add_weight(shape=(), name='my_additional_weight_2', initializer='ones', trainable=True)]
        self.weights_in_dict = {'my_weight': self.add_weight(shape=(), name='my_dict_weight', initializer='ones', trainable=True)}
        self.nested_layer.build(input_shape)

    def call(self, inputs):
        """Calculate and return the output of call based on inputs."""
        return self.nested_layer(inputs)

    def two(self):
        """Execute two with input (no arguments)."""
        return 2
ASSETS_DATA = 'These are my assets'
VARIABLES_DATA = np.random.random((10,))

@keras.saving.register_keras_serializable(package='my_custom_package')
class LayerWithCustomSaving(MyDense):

    def build(self, input_shape):
        """Return the result of build using parameters: input_shape."""
        self.assets = ASSETS_DATA
        self.stored_variables = VARIABLES_DATA
        return super().build(input_shape)

    def save_assets(self, inner_path):
        """Execute save_assets with input (inner_path)."""
        with open(os.path.join(inner_path, 'assets.txt'), 'w') as f:
            f.write(self.assets)

    def save_own_variables(self, store):
        """Implement the save_own_variables operation on provided input: store."""
        store['variables'] = self.stored_variables

    def load_assets(self, inner_path):
        """Compute load_assets given inner_path."""
        with open(os.path.join(inner_path, 'assets.txt'), 'r') as f:
            text = f.read()
        self.assets = text

    def load_own_variables(self, store):
        """Implement the load_own_variables operation on provided input: store."""
        self.stored_variables = np.array(store['variables'])

@keras.saving.register_keras_serializable(package='my_custom_package')
class CustomModelX(keras.Model):

    def __init__(self, *args, **kwargs):
        """Implement the __init__ operation on provided input: no arguments."""
        super().__init__(*args, **kwargs)
        self.dense1 = MyDense(1, name='my_dense_1')
        self.dense2 = MyDense(1, name='my_dense_2')

    def call(self, inputs):
        """Return the result of call using parameters: inputs."""
        out = self.dense1(inputs)
        return self.dense2(out)

    def one(self):
        """Compute one given no arguments."""
        return 1

@keras.saving.register_keras_serializable(package='my_custom_package')
class ModelWithCustomSaving(keras.Model):

    def __init__(self, *args, **kwargs):
        """Calculate and return the output of __init__ based on no arguments."""
        super().__init__(*args, **kwargs)
        self.custom_dense = LayerWithCustomSaving(1)

    def call(self, inputs):
        """Execute call with input (inputs)."""
        return self.custom_dense(inputs)

@keras.saving.register_keras_serializable(package='my_custom_package')
class CompileOverridingModel(keras.Model):

    def __init__(self, *args, **kwargs):
        """Process data using __init__ with arguments no arguments."""
        super().__init__(*args, **kwargs)
        self.dense1 = MyDense(1)

    def compile(self, *args, **kwargs):
        """Implement the compile operation on provided input: no arguments."""
        super().compile(*args, **kwargs)

    def call(self, inputs):
        """Execute call with input (inputs)."""
        return self.dense1(inputs)

@keras.saving.register_keras_serializable(package='my_custom_package')
class CompileOverridingSequential(keras.Sequential):

    def compile(self, *args, **kwargs):
        """Compute compile given no arguments."""
        super().compile(*args, **kwargs)

@keras.saving.register_keras_serializable(package='my_custom_package')
class SubclassFunctional(keras.Model):
    """Subclassed functional identical to `_get_basic_functional_model`."""

    def __init__(self, **kwargs):
        """Implement the __init__ operation on provided input: no arguments."""
        inputs = keras.Input(shape=(4,), batch_size=2)
        dense = keras.layers.Dense(1, name='first_dense')
        x = dense(inputs)
        outputs = keras.layers.Dense(1, name='second_dense')(x)
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.layer_attr = dense

    @property
    def layer_property(self):
        """Calculate and return the output of layer_property based on no arguments."""
        return self.layer_attr

    def get_config(self):
        """Compute get_config given no arguments."""
        return {}

    @classmethod
    def from_config(cls, config):
        """Execute from_config with input (cls, config)."""
        return cls(**config)

@keras.saving.register_keras_serializable(package='my_custom_package')
def my_mean_squared_error(y_true, y_pred):
    """Identical to built-in `mean_squared_error`, but as a custom fn."""
    return ops.mean(ops.square(y_pred - y_true), axis=-1)

def _get_subclassed_model(compile=True):
    """Process data using _get_subclassed_model with arguments compile."""
    subclassed_model = CustomModelX(name='custom_model_x')
    if compile:
        subclassed_model.compile(optimizer='adam', loss=my_mean_squared_error, metrics=[keras.metrics.Hinge(), 'mse'])
    return subclassed_model

def _get_custom_sequential_model(compile=True):
    """Calculate and return the output of _get_custom_sequential_model based on compile."""
    sequential_model = keras.Sequential([MyDense(1), MyDense(1)], name='sequential')
    if compile:
        sequential_model.compile(optimizer='adam', loss=my_mean_squared_error, metrics=[keras.metrics.Hinge(), 'mse'])
    return sequential_model

def _get_basic_sequential_model(compile=True):
    """Calculate and return the output of _get_basic_sequential_model based on compile."""
    sequential_model = keras.Sequential([keras.layers.Dense(1, name='dense_1'), keras.layers.Dense(1, name='dense_2')], name='sequential')
    if compile:
        sequential_model.compile(optimizer='adam', loss=my_mean_squared_error, metrics=[keras.metrics.Hinge(), 'mse'])
    return sequential_model

def _get_custom_functional_model(compile=True):
    """Calculate and return the output of _get_custom_functional_model based on compile."""
    inputs = keras.Input(shape=(4,), batch_size=2)
    x = MyDense(1, name='first_dense')(inputs)
    outputs = MyDense(1, name='second_dense')(x)
    functional_model = keras.Model(inputs, outputs)
    if compile:
        functional_model.compile(optimizer='adam', loss=my_mean_squared_error, metrics=[keras.metrics.Hinge(), 'mse'])
    return functional_model

def _get_basic_functional_model(compile=True):
    """Calculate and return the output of _get_basic_functional_model based on compile."""
    inputs = keras.Input(shape=(4,), batch_size=2)
    x = keras.layers.Dense(1, name='first_dense')(inputs)
    outputs = keras.layers.Dense(1, name='second_dense')(x)
    functional_model = keras.Model(inputs, outputs)
    if compile:
        functional_model.compile(optimizer='adam', loss=my_mean_squared_error, metrics=[keras.metrics.Hinge(), 'mse'])
    return functional_model

def _get_subclassed_functional_model(compile=True):
    """Return the result of _get_subclassed_functional_model using parameters: compile."""
    functional_model = SubclassFunctional()
    if compile:
        functional_model.compile(optimizer='adam', loss=my_mean_squared_error, metrics=[keras.metrics.Hinge(), 'mse'])
    return functional_model

def _load_model_fn(filepath):
    """Compute _load_model_fn given filepath."""
    saving_lib.load_model(filepath)

class SavingTest(testing.TestCase):

    def setUp(self):
        """Return the result of setUp using parameters: no arguments."""
        self.original_value = saving_lib._MEMORY_UPPER_BOUND
        saving_lib._MEMORY_UPPER_BOUND = 0
        return super().setUp()

    def tearDown(self):
        """Execute tearDown with input (no arguments)."""
        saving_lib._MEMORY_UPPER_BOUND = self.original_value
        return super().tearDown()

    def _test_inference_after_instantiation(self, model):
        """Calculate and return the output of _test_inference_after_instantiation based on model."""
        x_ref = np.random.random((2, 4))
        y_ref = model(x_ref)
        temp_filepath = os.path.join(self.get_temp_dir(), 'my_model.keras')
        model.save(temp_filepath)
        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertFalse(model.compiled)
        for w_ref, w in zip(model.variables, loaded_model.variables):
            self.assertAllClose(w_ref, w)
        self.assertAllClose(y_ref, loaded_model(x_ref))

    @parameterized.named_parameters(('subclassed', _get_subclassed_model), ('basic_sequential', _get_basic_sequential_model), ('basic_functional', _get_basic_functional_model), ('custom_sequential', _get_custom_sequential_model), ('custom_functional', _get_custom_functional_model), ('subclassed_functional', _get_subclassed_functional_model))
    def test_inference_after_instantiation(self, model_fn):
        """Calculate and return the output of test_inference_after_instantiation based on model_fn."""
        model = model_fn(compile=False)
        self._test_inference_after_instantiation(model)
        saving_lib._MEMORY_UPPER_BOUND = 1.0
        self._test_inference_after_instantiation(model)

    def _test_compile_preserved(self, model):
        """Execute _test_compile_preserved with input (model)."""
        x_ref = np.random.random((2, 4))
        y_ref = np.random.random((2, 1))
        model.fit(x_ref, y_ref)
        out_ref = model(x_ref)
        ref_metrics = model.evaluate(x_ref, y_ref)
        temp_filepath = os.path.join(self.get_temp_dir(), 'my_model.keras')
        model.save(temp_filepath)
        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertTrue(model.compiled)
        self.assertTrue(loaded_model.built)
        for w_ref, w in zip(model.variables, loaded_model.variables):
            self.assertAllClose(w_ref, w)
        self.assertAllClose(out_ref, loaded_model(x_ref))
        self.assertEqual(model.optimizer.__class__, loaded_model.optimizer.__class__)
        self.assertEqual(model.optimizer.get_config(), loaded_model.optimizer.get_config())
        for w_ref, w in zip(model.optimizer.variables, loaded_model.optimizer.variables):
            self.assertAllClose(w_ref, w)
        new_metrics = loaded_model.evaluate(x_ref, y_ref)
        for ref_m, m in zip(ref_metrics, new_metrics):
            self.assertAllClose(ref_m, m)

    @parameterized.named_parameters(('subclassed', _get_subclassed_model), ('basic_sequential', _get_basic_sequential_model), ('basic_functional', _get_basic_functional_model), ('custom_sequential', _get_custom_sequential_model), ('custom_functional', _get_custom_functional_model), ('subclassed_functional', _get_subclassed_functional_model))
    @pytest.mark.requires_trainable_backend
    def test_compile_preserved(self, model_fn):
        """Calculate and return the output of test_compile_preserved based on model_fn."""
        model = model_fn(compile=True)
        self._test_compile_preserved(model)
        saving_lib._MEMORY_UPPER_BOUND = 1.0
        self._test_compile_preserved(model)

    def test_saving_preserve_unbuilt_state(self):
        """Execute test_saving_preserve_unbuilt_state with input (no arguments)."""
        temp_filepath = os.path.join(self.get_temp_dir(), 'my_model.keras')
        subclassed_model = CustomModelX()
        subclassed_model.save(temp_filepath)
        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(subclassed_model.compiled, loaded_model.compiled)
        self.assertFalse(subclassed_model.built)
        self.assertFalse(loaded_model.built)

    @pytest.mark.requires_trainable_backend
    def test_saved_module_paths_and_class_names(self):
        """Process data using test_saved_module_paths_and_class_names with arguments no arguments."""
        temp_filepath = os.path.join(self.get_temp_dir(), 'my_model.keras')
        subclassed_model = _get_subclassed_model()
        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        subclassed_model.fit(x, y, epochs=1)
        subclassed_model.save(temp_filepath)
        with zipfile.ZipFile(temp_filepath, 'r') as z:
            with z.open(saving_lib._CONFIG_FILENAME, 'r') as c:
                config_json = c.read()
        config_dict = json.loads(config_json)
        self.assertEqual(config_dict['registered_name'], 'my_custom_package>CustomModelX')
        self.assertEqual(config_dict['compile_config']['optimizer'], keras.src.saving.serialize_keras_object(keras.src.optimizers.get('adam')))
        self.assertEqual(config_dict['compile_config']['loss']['config'], 'my_custom_package>my_mean_squared_error')

    @pytest.mark.requires_trainable_backend
    def test_saving_custom_assets_and_variables(self):
        """Return the result of test_saving_custom_assets_and_variables using parameters: no arguments."""
        temp_filepath = os.path.join(self.get_temp_dir(), 'my_model.keras')
        model = ModelWithCustomSaving()
        model.compile(optimizer='adam', loss='mse')
        x = np.random.random((100, 32))
        y = np.random.random((100, 1))
        model.fit(x, y, epochs=1)
        self.assertFalse(os.path.exists(temp_filepath))
        model.save(temp_filepath)
        loaded_model = saving_lib.load_model(temp_filepath)
        self.assertEqual(loaded_model.custom_dense.assets, ASSETS_DATA)
        self.assertEqual(loaded_model.custom_dense.stored_variables.tolist(), VARIABLES_DATA.tolist())

    def _test_compile_overridden_warnings(self, model_type):
        """Return the result of _test_compile_overridden_warnings using parameters: model_type."""
        temp_filepath = os.path.join(self.get_temp_dir(), 'my_model.keras')
        model = CompileOverridingModel() if model_type == 'subclassed' else CompileOverridingSequential([keras.layers.Embedding(4, 1), MyDense(1), MyDense(1)])
        model.compile('sgd', 'mse')
        model.save(temp_filepath)
        with mock.patch.object(warnings, 'warn') as mock_warn:
            saving_lib.load_model(temp_filepath)
        if not mock_warn.call_args_list:
            raise AssertionError('Did not warn.')
        self.assertIn("`compile()` was not called as part of model loading because the model's `compile()` method is custom. ", mock_warn.call_args_list[0][0][0])

    def test_compile_overridden_warnings_sequential(self):
        """Compute test_compile_overridden_warnings_sequential given no arguments."""
        self._test_compile_overridden_warnings('sequential')

    def test_compile_overridden_warnings_subclassed(self):
        """Return the result of test_compile_overridden_warnings_subclassed using parameters: no arguments."""
        self._test_compile_overridden_warnings('subclassed')

    def test_metadata(self):
        """Calculate and return the output of test_metadata based on no arguments."""
        temp_filepath = Path(os.path.join(self.get_temp_dir(), 'my_model.keras'))
        model = CompileOverridingModel()
        model.save(temp_filepath)
        with zipfile.ZipFile(temp_filepath, 'r') as z:
            with z.open(saving_lib._METADATA_FILENAME, 'r') as c:
                metadata_json = c.read()
        metadata = json.loads(metadata_json)
        self.assertIn('keras_version', metadata)
        self.assertIn('date_saved', metadata)

    def test_save_load_weights_only(self):
        """Process data using test_save_load_weights_only with arguments no arguments."""
        temp_filepath = Path(os.path.join(self.get_temp_dir(), 'mymodel.weights.h5'))
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        saving_lib.save_weights_only(model, temp_filepath)
        model = _get_basic_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-06)
        model = _get_basic_functional_model()
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-06)

    def test_save_weights_only_with_unbuilt_model(self):
        """Process data using test_save_weights_only_with_unbuilt_model with arguments no arguments."""
        temp_filepath = Path(os.path.join(self.get_temp_dir(), 'mymodel.weights.h5'))
        model = _get_subclassed_model()
        with self.assertRaisesRegex(ValueError, 'You are saving a model that has not yet been built.'):
            saving_lib.save_weights_only(model, temp_filepath)

    def test_load_weights_only_with_unbuilt_model(self):
        """Calculate and return the output of test_load_weights_only_with_unbuilt_model based on no arguments."""
        temp_filepath = Path(os.path.join(self.get_temp_dir(), 'mymodel.weights.h5'))
        model = _get_subclassed_model()
        x = np.random.random((100, 32))
        _ = model.predict(x)
        saving_lib.save_weights_only(model, temp_filepath)
        saving_lib.load_weights_only(model, temp_filepath)
        new_model = _get_subclassed_model()
        with self.assertRaisesRegex(ValueError, 'You are loading weights into a model that has not yet been built.'):
            saving_lib.load_weights_only(new_model, temp_filepath)

    def test_load_weights_only_with_keras_file(self):
        """Execute test_load_weights_only_with_keras_file with input (no arguments)."""
        temp_filepath = Path(os.path.join(self.get_temp_dir(), 'mymodel.keras'))
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        saving_lib.save_model(model, temp_filepath)
        model = _get_basic_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-06)
        model = _get_basic_functional_model()
        model.load_weights(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-06)

    def test_save_weights_subclassed_functional(self):
        """Return the result of test_save_weights_subclassed_functional using parameters: no arguments."""
        temp_filepath = Path(os.path.join(self.get_temp_dir(), 'mymodel.weights.h5'))
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        saving_lib.save_weights_only(model, temp_filepath)
        model = _get_subclassed_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-06)
        saving_lib.save_weights_only(model, temp_filepath)
        model = _get_basic_functional_model()
        saving_lib.load_weights_only(model, temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-06)

    @pytest.mark.requires_trainable_backend
    def test_compile_arg(self):
        """Calculate and return the output of test_compile_arg based on no arguments."""
        temp_filepath = os.path.join(self.get_temp_dir(), 'mymodel.keras')
        model = _get_basic_functional_model()
        model.compile('sgd', 'mse')
        model.fit(np.random.random((2, 4)), np.random.random((2, 1)))
        saving_lib.save_model(model, temp_filepath)
        model = saving_lib.load_model(temp_filepath)
        self.assertEqual(model.compiled, True)
        model = saving_lib.load_model(temp_filepath, compile=False)
        self.assertEqual(model.compiled, False)

    def test_partial_load(self):
        """Calculate and return the output of test_partial_load based on no arguments."""
        temp_filepath = os.path.join(self.get_temp_dir(), 'mymodel.keras')
        original_model = keras.Sequential([keras.Input(shape=(3,), batch_size=2), keras.layers.Dense(4), keras.layers.Dense(5)])
        original_model.save(temp_filepath)
        new_model = keras.Sequential([keras.Input(shape=(3,), batch_size=2), keras.layers.Dense(4), keras.layers.Dense(6)])
        new_layer_kernel_value = np.array(new_model.layers[1].kernel)
        with self.assertRaisesRegex(ValueError, 'must match'):
            new_model.load_weights(temp_filepath)
        new_model.load_weights(temp_filepath, skip_mismatch=True)
        ref_weights = original_model.layers[0].get_weights()
        new_weights = new_model.layers[0].get_weights()
        self.assertEqual(len(ref_weights), len(new_weights))
        for ref_w, w in zip(ref_weights, new_weights):
            self.assertAllClose(ref_w, w)
        self.assertAllClose(np.array(new_model.layers[1].kernel), new_layer_kernel_value)
        new_model = keras.Sequential([keras.Input(shape=(3,), batch_size=2), keras.layers.Dense(4), keras.layers.Dense(5), keras.layers.Dense(5)])
        new_layer_kernel_value = np.array(new_model.layers[2].kernel)
        with self.assertRaisesRegex(ValueError, 'received 0 variables'):
            new_model.load_weights(temp_filepath)
        new_model.load_weights(temp_filepath, skip_mismatch=True)
        for layer_index in [0, 1]:
            ref_weights = original_model.layers[layer_index].get_weights()
            new_weights = new_model.layers[layer_index].get_weights()
            self.assertEqual(len(ref_weights), len(new_weights))
            for ref_w, w in zip(ref_weights, new_weights):
                self.assertAllClose(ref_w, w)
        self.assertAllClose(np.array(new_model.layers[2].kernel), new_layer_kernel_value)

    @pytest.mark.requires_trainable_backend
    def test_save_to_fileobj(self):
        """Execute test_save_to_fileobj with input (no arguments)."""
        model = keras.Sequential([keras.layers.Dense(1, input_shape=(1,)), keras.layers.Dense(1)])
        model.compile(optimizer='adam', loss='mse')
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
        self.assertAllClose(pred1, pred2, atol=1e-05)

    @parameterized.named_parameters(('high_memory_config', True), ('low_memory_config', False))
    def test_save_model_exception_raised(self, is_memory_sufficient):
        """Implement the test_save_model_exception_raised operation on provided input: is_memory_sufficient."""
        if is_memory_sufficient:
            saving_lib._MEMORY_UPPER_BOUND = 0.5

        class RaiseErrorLayer(keras.layers.Layer):

            def __init__(self, units, **kwargs):
                """Implement the __init__ operation on provided input: units."""
                super().__init__(**kwargs)
                self.dense = keras.layers.Dense(units)

            def call(self, inputs):
                """Process data using call with arguments inputs."""
                return self.dense(inputs)

            def save_own_variables(self, store):
                """Implement the save_own_variables operation on provided input: store."""
                raise ValueError
        model = keras.Sequential([keras.Input([1]), RaiseErrorLayer(1)])
        filepath = f'{self.get_temp_dir()}/model.keras'
        with self.assertRaises(ValueError):
            saving_lib.save_model(model, filepath)
        self.assertTrue(Path(filepath).exists())
        with zipfile.ZipFile(filepath) as zf:
            all_filenames = zf.namelist()
            self.assertNotIn('model.weights.h5', all_filenames)
        self.assertLen(os.listdir(Path(filepath).parent), 1)
        self.assertIn('model.keras', os.listdir(Path(filepath).parent))

    @parameterized.named_parameters(('high_memory_config', True), ('low_memory_config', False))
    def test_load_model_exception_raised(self, is_memory_sufficient):
        """Compute test_load_model_exception_raised given is_memory_sufficient."""
        if is_memory_sufficient:
            saving_lib._MEMORY_UPPER_BOUND = 0.5

        class RaiseErrorLayer(keras.layers.Layer):

            def __init__(self, units, **kwargs):
                """Implement the __init__ operation on provided input: units."""
                super().__init__(**kwargs)
                self.dense = keras.layers.Dense(units)

            def call(self, inputs):
                """Implement the call operation on provided input: inputs."""
                return self.dense(inputs)

            def load_own_variables(self, store):
                """Calculate and return the output of load_own_variables based on store."""
                raise ValueError
        model = keras.Sequential([keras.Input([1]), RaiseErrorLayer(1)])
        filepath = f'{self.get_temp_dir()}/model.keras'
        saving_lib.save_model(model, filepath)
        with self.assertRaises(ValueError):
            saving_lib.load_model(filepath, custom_objects={'RaiseErrorLayer': RaiseErrorLayer})
        self.assertLen(os.listdir(Path(filepath).parent), 1)
        self.assertIn('model.keras', os.listdir(Path(filepath).parent))

    def test_load_model_read_only_system(self):
        """Return the result of test_load_model_read_only_system using parameters: no arguments."""
        model = keras.Sequential([keras.Input([1]), keras.layers.Dense(32)])
        filepath = f'{self.get_temp_dir()}/model.keras'
        saving_lib.save_model(model, filepath)
        original_mode = os.stat(Path(filepath).parent).st_mode
        os.chmod(Path(filepath).parent, mode=365)
        model = saving_lib.load_model(filepath)
        os.chmod(Path(filepath).parent, mode=original_mode)
        self.assertLen(os.listdir(Path(filepath).parent), 1)
        self.assertIn('model.keras', os.listdir(Path(filepath).parent))

    @pytest.mark.skipif(backend.backend() == 'jax', reason="JAX backend doesn't support Python's multiprocessing")
    @pytest.mark.skipif(testing.tensorflow_uses_gpu() or testing.torch_uses_gpu(), reason="This test doesn't support GPU")
    def test_load_model_concurrently(self):
        """Calculate and return the output of test_load_model_concurrently based on no arguments."""
        import multiprocessing as mp
        model = keras.Sequential([keras.Input([1]), keras.layers.Dense(2)])
        filepath = f'{self.get_temp_dir()}/model.keras'
        saving_lib.save_model(model, filepath)
        results = []
        with mp.Pool(4) as pool:
            for i in range(4):
                results.append(pool.apply_async(_load_model_fn, (filepath,)))
            pool.close()
            pool.join()
        [r.get() for r in results]

    def test_load_model_containing_reused_layer(self):
        """Calculate and return the output of test_load_model_containing_reused_layer based on no arguments."""
        inputs = keras.Input((4,))
        reused_layer = keras.layers.Dense(4)
        x = reused_layer(inputs)
        x = keras.layers.Dense(4)(x)
        outputs = reused_layer(x)
        model = keras.Model(inputs, outputs)
        self.assertLen(model.layers, 3)
        self._test_inference_after_instantiation(model)

@pytest.mark.requires_trainable_backend
class SavingAPITest(testing.TestCase):

    def test_saving_api_errors(self):
        """Implement the test_saving_api_errors operation on provided input: no arguments."""
        from keras.src.saving import saving_api
        model = _get_basic_functional_model()
        temp_filepath = os.path.join(self.get_temp_dir(), 'mymodel')
        with self.assertRaisesRegex(ValueError, 'argument is deprecated'):
            saving_api.save_model(model, temp_filepath, save_format='keras')
        temp_filepath = os.path.join(self.get_temp_dir(), 'mymodel.notkeras')
        with self.assertRaisesRegex(ValueError, 'Invalid filepath extension'):
            saving_api.save_model(model, temp_filepath)
        temp_filepath = os.path.join(self.get_temp_dir(), 'mymodel.keras')
        with self.assertRaisesRegex(ValueError, 'are not supported'):
            saving_api.save_model(model, temp_filepath, invalid_arg='hello')
        temp_filepath = os.path.join(self.get_temp_dir(), 'non_existent.keras')
        with self.assertRaisesRegex(ValueError, 'Please ensure the file is an accessible'):
            _ = saving_api.load_model(temp_filepath)
        temp_filepath = os.path.join(self.get_temp_dir(), 'my_saved_model')
        with self.assertRaisesRegex(ValueError, 'File format not supported'):
            _ = saving_api.load_model(temp_filepath)

    def test_model_api_endpoint(self):
        """Process data using test_model_api_endpoint with arguments no arguments."""
        temp_filepath = Path(os.path.join(self.get_temp_dir(), 'mymodel.keras'))
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        model.save(temp_filepath)
        model = keras.saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-06)

    def test_model_api_endpoint_h5(self):
        """Calculate and return the output of test_model_api_endpoint_h5 based on no arguments."""
        temp_filepath = Path(os.path.join(self.get_temp_dir(), 'mymodel.h5'))
        model = _get_basic_functional_model()
        ref_input = np.random.random((2, 4))
        ref_output = model.predict(ref_input)
        model.save(temp_filepath)
        model = keras.saving.load_model(temp_filepath)
        self.assertAllClose(model.predict(ref_input), ref_output, atol=1e-06)

    def test_model_api_errors(self):
        """Return the result of test_model_api_errors using parameters: no arguments."""
        model = _get_basic_functional_model()
        temp_filepath = os.path.join(self.get_temp_dir(), 'mymodel')
        with self.assertRaisesRegex(ValueError, 'argument is deprecated'):
            model.save(temp_filepath, save_format='keras')
        temp_filepath = os.path.join(self.get_temp_dir(), 'mymodel.notkeras')
        with self.assertRaisesRegex(ValueError, 'Invalid filepath extension'):
            model.save(temp_filepath)
        temp_filepath = os.path.join(self.get_temp_dir(), 'mymodel.keras')
        with self.assertRaisesRegex(ValueError, 'are not supported'):
            model.save(temp_filepath, invalid_arg='hello')

    def test_safe_mode(self):
        """Process data using test_safe_mode with arguments no arguments."""
        temp_filepath = os.path.join(self.get_temp_dir(), 'unsafe_model.keras')
        model = keras.Sequential([keras.Input(shape=(3,)), keras.layers.Lambda(lambda x: x * 2)])
        model.save(temp_filepath)
        with self.assertRaisesRegex(ValueError, 'Deserializing it is unsafe'):
            model = saving_lib.load_model(temp_filepath)
        model = saving_lib.load_model(temp_filepath, safe_mode=False)

    def test_normalization_kpl(self):
        """Calculate and return the output of test_normalization_kpl based on no arguments."""
        temp_filepath = os.path.join(self.get_temp_dir(), 'norm_model.keras')
        model = keras.Sequential([keras.Input(shape=(3,)), keras.layers.Normalization()])
        data = np.random.random((3, 3))
        model.layers[0].adapt(data)
        ref_out = model(data)
        model.save(temp_filepath)
        model = saving_lib.load_model(temp_filepath)
        out = model(data)
        self.assertAllClose(ref_out, out, atol=1e-06)
        model = keras.Sequential([keras.Input(shape=(3,)), keras.layers.Normalization(mean=np.random.random((3,)), variance=np.random.random((3,)))])
        ref_out = model(data)
        model.save(temp_filepath)
        model = saving_lib.load_model(temp_filepath)
        out = model(data)
        self.assertAllClose(ref_out, out, atol=1e-06)

@keras.saving.register_keras_serializable()
class GrowthFactor:

    def __init__(self, factor):
        """Process data using __init__ with arguments factor."""
        self.factor = factor

    def __call__(self, inputs):
        """Compute __call__ given inputs."""
        return inputs * self.factor

    def get_config(self):
        """Implement the get_config operation on provided input: no arguments."""
        return {'factor': self.factor}

@keras.saving.register_keras_serializable(package='Complex')
class FactorLayer(keras.layers.Layer):

    def __init__(self, factor, **kwargs):
        """Compute __init__ given factor."""
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        """Process data using call with arguments x."""
        return x * self.factor

    def get_config(self):
        """Implement the get_config operation on provided input: no arguments."""
        return {'factor': self.factor}

@keras.saving.register_keras_serializable(package='Complex')
class ComplexModel(keras.layers.Layer):

    def __init__(self, first_layer, second_layer=None, **kwargs):
        """Process data using __init__ with arguments first_layer, second_layer."""
        super().__init__(**kwargs)
        self.first_layer = first_layer
        if second_layer is not None:
            self.second_layer = second_layer
        else:
            self.second_layer = keras.layers.Dense(8)

    def get_config(self):
        """Calculate and return the output of get_config based on no arguments."""
        config = super().get_config()
        config.update({'first_layer': self.first_layer, 'second_layer': self.second_layer})
        return config

    def call(self, inputs):
        """Calculate and return the output of call based on inputs."""
        return self.first_layer(self.second_layer(inputs))

class SavingBattleTest(testing.TestCase):

    def test_custom_object_without_from_config(self):
        """Implement the test_custom_object_without_from_config operation on provided input: no arguments."""
        temp_filepath = os.path.join(self.get_temp_dir(), 'custom_fn_model.keras')
        inputs = keras.Input(shape=(4, 4))
        outputs = keras.layers.Dense(1, activation=GrowthFactor(0.5))(inputs)
        model = keras.Model(inputs, outputs)
        model.save(temp_filepath)
        with self.assertRaisesRegex(TypeError, 'Unable to reconstruct an instance'):
            _ = saving_lib.load_model(temp_filepath)

    def test_complex_model_without_explicit_deserialization(self):
        """Implement the test_complex_model_without_explicit_deserialization operation on provided input: no arguments."""
        temp_filepath = os.path.join(self.get_temp_dir(), 'complex_model.keras')
        inputs = keras.Input((32,))
        outputs = ComplexModel(first_layer=FactorLayer(0.5))(inputs)
        model = keras.Model(inputs, outputs)
        model.save(temp_filepath)
        with self.assertRaisesRegex(TypeError, 'are explicitly deserialized'):
            _ = saving_lib.load_model(temp_filepath)

    def test_redefinition_of_trackable(self):
        """Test that a trackable can be aliased under a new name."""

        class NormalModel(keras.Model):

            def __init__(self):
                """Process data using __init__ with arguments no arguments."""
                super().__init__()
                self.dense = keras.layers.Dense(3)

            def call(self, x):
                """Implement the call operation on provided input: x."""
                return self.dense(x)

        class WeirdModel(keras.Model):

            def __init__(self):
                """Execute __init__ with input (no arguments)."""
                super().__init__()
                self.a_dense = keras.layers.Dense(3)

            @property
            def dense(self):
                """Process data using dense with arguments no arguments."""
                return self.a_dense

            def call(self, x):
                """Compute call given x."""
                return self.dense(x)
        temp_filepath = os.path.join(self.get_temp_dir(), 'normal_model.weights.h5')
        model_a = NormalModel()
        model_a(np.random.random((2, 2)))
        model_a.save_weights(temp_filepath)
        model_b = WeirdModel()
        model_b(np.random.random((2, 2)))
        model_b.load_weights(temp_filepath)
        self.assertAllClose(model_a.dense.kernel.numpy(), model_b.dense.kernel.numpy())

    def test_normalization_legacy_h5_format(self):
        """Process data using test_normalization_legacy_h5_format with arguments no arguments."""
        temp_filepath = os.path.join(self.get_temp_dir(), 'custom_model.h5')
        inputs = keras.Input((32,))
        normalization = keras.layers.Normalization()
        outputs = normalization(inputs)
        model = keras.Model(inputs, outputs)
        x = np.random.random((1, 32))
        normalization.adapt(x)
        ref_out = model(x)
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        out = new_model(x)
        self.assertAllClose(ref_out, out, atol=1e-06)

    def test_legacy_h5_format(self):
        """Compute test_legacy_h5_format given no arguments."""
        temp_filepath = os.path.join(self.get_temp_dir(), 'custom_model.h5')
        inputs = keras.Input((32,))
        x = MyDense(2)(inputs)
        outputs = CustomModelX()(x)
        model = keras.Model(inputs, outputs)
        x = np.random.random((1, 32))
        ref_out = model(x)
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        out = new_model(x)
        self.assertAllClose(ref_out, out, atol=1e-06)

    def test_nested_functional_model_saving(self):
        """Compute test_nested_functional_model_saving given no arguments."""

        def func(in_size=4, out_size=2, name=None):
            """Calculate and return the output of func based on in_size, out_size, name."""
            inputs = keras.layers.Input(shape=(in_size,))
            outputs = keras.layers.Dense(out_size)(inputs)
            return keras.Model(inputs, outputs=outputs, name=name)
        input_a, input_b = (keras.Input((4,)), keras.Input((4,)))
        out_a = func(out_size=2, name='func_a')(input_a)
        out_b = func(out_size=3, name='func_b')(input_b)
        model = keras.Model([input_a, input_b], outputs=[out_a, out_b])
        temp_filepath = os.path.join(self.get_temp_dir(), 'nested_func.keras')
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        x = ([np.random.random((2, 4))], np.random.random((2, 4)))
        ref_out = model(x)
        out = new_model(x)
        self.assertAllClose(ref_out[0], out[0])
        self.assertAllClose(ref_out[1], out[1])

    def test_nested_shared_functional_model_saving(self):
        """Calculate and return the output of test_nested_shared_functional_model_saving based on no arguments."""

        def func(in_size=4, out_size=2, name=None):
            """Process data using func with arguments in_size, out_size, name."""
            inputs = keras.layers.Input(shape=(in_size,))
            outputs = keras.layers.Dense(out_size)(inputs)
            return keras.Model(inputs, outputs=outputs, name=name)
        inputs = [keras.Input((4,)), keras.Input((4,))]
        func_shared = func(out_size=4, name='func_shared')
        shared_a = func_shared(inputs[0])
        shared_b = func_shared(inputs[1])
        out_a = keras.layers.Dense(2)(shared_a)
        out_b = keras.layers.Dense(2)(shared_b)
        model = keras.Model(inputs, outputs=[out_a, out_b])
        temp_filepath = os.path.join(self.get_temp_dir(), 'nested_shared_func.keras')
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        x = ([np.random.random((2, 4))], np.random.random((2, 4)))
        ref_out = model(x)
        out = new_model(x)
        self.assertAllClose(ref_out[0], out[0])
        self.assertAllClose(ref_out[1], out[1])

    def test_bidirectional_lstm_saving(self):
        """Process data using test_bidirectional_lstm_saving with arguments no arguments."""
        inputs = keras.Input((3, 2))
        outputs = keras.layers.Bidirectional(keras.layers.LSTM(64))(inputs)
        model = keras.Model(inputs, outputs)
        temp_filepath = os.path.join(self.get_temp_dir(), 'bidir_lstm.keras')
        model.save(temp_filepath)
        new_model = keras.saving.load_model(temp_filepath)
        x = np.random.random((1, 3, 2))
        ref_out = model(x)
        out = new_model(x)
        self.assertAllClose(ref_out, out)

    def test_remove_weights_only_saving_and_loading(self):
        """Implement the test_remove_weights_only_saving_and_loading operation on provided input: no arguments."""

        def is_remote_path(path):
            """Return the result of is_remote_path using parameters: path."""
            return True
        temp_filepath = os.path.join(self.get_temp_dir(), 'model.weights.h5')
        with mock.patch('keras.src.utils.file_utils.is_remote_path', is_remote_path):
            model = _get_basic_functional_model()
            model.save_weights(temp_filepath)
            model.load_weights(temp_filepath)