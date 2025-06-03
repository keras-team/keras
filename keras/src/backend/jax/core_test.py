import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

import keras
from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.backend.jax.core import JaxVariable
from keras.src.backend.jax.core import NnxVariable
from keras.src.backend.jax.layer import JaxLayer
from keras.src.backend.jax.layer import NnxLayer


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX backend specific test for core Variable integration with NNX.",
)
@pytest.mark.skipif(
    not keras.config.is_nnx_backend_enabled(),
    reason="Test requires NNX backend to be enabled by default for setup.",
)
class JaxCoreVariableTest(testing.TestCase):
    def setup(self):
        super().setup()

        class NNXModel(nnx.Module):
            def __init__(self, rngs):
                self.linear = nnx.Linear(2, 3, rngs=rngs)
                # Use NnxVariable directly as KerasJaxVariable
                # might be JaxVariable if NNX is disabled globally.
                self.custom_variable = NnxVariable(jnp.ones((1, 3)))

            def __call__(self, x):
                return self.linear(x) + self.custom_variable

        self.nnx_model = NNXModel(rngs=nnx.Rngs(0))
        self.keras_nnx_model = keras.Sequential(
            [keras.layers.Dense(units=1, input_shape=(10,))]
        )
        self.single_dummy_input = np.random.rand(1, 10)

    def test_variable_in_nnx_module(self):
        self.assertTrue(hasattr(self.nnx_model.custom_variable, "_trace_state"))
        self.assertIsNotNone(self.nnx_model.custom_variable._trace_state)
        self.assertAllEqual(self.nnx_model.custom_variable.value, [[1, 1, 1]])
        self.assertTrue(
            isinstance(self.nnx_model.custom_variable, nnx.Variable)
        )

    def test_model_saving(self):
        path = os.path.join(self.get_temp_dir(), "model.keras")
        original_outputs = self.keras_nnx_model(self.single_dummy_input)
        self.keras_nnx_model.save(path, save_format="keras_v3")
        restored_model = keras.models.load_model(path)
        restored_outputs = restored_model(self.single_dummy_input)
        self.assertAllEqual(original_outputs, restored_outputs)

    def test_keras_variable_nnx_split_merge_sync(self):
        variable1 = keras.Variable(jnp.array(1.0))
        graphdef, state = nnx.split(variable1)
        state = jax.tree.map(lambda x: x + 1, state)
        variable2 = nnx.merge(graphdef, state)
        self.assertEqual(variable2._value, variable2.value)


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX backend specific test for NNX flag.",
)
class JaxNnxFlagTest(testing.TestCase):
    def tearDown(self):
        # Reset the flag to its default (False) after each test
        keras.config.disable_nnx_backend()
        super().tearDown()

    def test_variable_selection_based_on_nnx_flag(self):
        # Test with NNX backend enabled
        keras.config.enable_nnx_backend()
        self.assertTrue(keras.config.is_nnx_backend_enabled())
        var_nnx_enabled = backend.Variable(1.0)
        self.assertIsInstance(var_nnx_enabled, NnxVariable)
        self.assertNotIsInstance(var_nnx_enabled, JaxVariable)

        # Test with NNX backend disabled
        keras.config.disable_nnx_backend()
        self.assertFalse(keras.config.is_nnx_backend_enabled())
        var_nnx_disabled = backend.Variable(1.0)
        self.assertIsInstance(var_nnx_disabled, JaxVariable)
        self.assertNotIsInstance(var_nnx_disabled, NnxVariable)

    def test_layer_backend_selection_based_on_nnx_flag(self):
        # Test with NNX backend enabled
        keras.config.enable_nnx_backend()
        self.assertTrue(keras.config.is_nnx_backend_enabled())

        class MyLayerNnxEnabled(layers.Layer):
            pass

        layer_nnx_enabled = MyLayerNnxEnabled()
        self.assertIsInstance(layer_nnx_enabled, NnxLayer)
        self.assertNotIsInstance(layer_nnx_enabled, JaxLayer)

        # Test with NNX backend disabled
        # Must clear global state to re-evaluate Layer's base class
        keras.src.backend.common.global_state.clear_session()
        keras.config.disable_nnx_backend()
        self.assertFalse(keras.config.is_nnx_backend_enabled())

        class MyLayerNnxDisabled(layers.Layer):
            pass

        layer_nnx_disabled = MyLayerNnxDisabled()
        self.assertIsInstance(layer_nnx_disabled, JaxLayer)
        self.assertNotIsInstance(layer_nnx_disabled, NnxLayer)
