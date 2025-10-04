"""Test for core.py."""

import os

os.environ["KERAS_BACKEND"] = "jax"
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import keras
from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend.config import is_nnx_enabled
from keras.src.backend.jax.core import JaxVariable
from keras.src.backend.jax.core import _ProtectedShardedArray

if is_nnx_enabled():
    from keras.src.backend.jax.core import NnxVariable
from keras.src.utils.variable_loading import load_variable_with_sharded_support

if is_nnx_enabled():
    from flax import nnx

    from keras.src.backend.jax.core import NnxVariable


class JaxCoreTest(testing.TestCase):
    def test_protected_sharded_array_deletion(self):
        """Test _ProtectedShardedArray prevents deletion of sharded arrays."""
        # Create a mock sharded array
        array = jax.numpy.ones((10, 10))
        sharded_array = jax.device_put(array, jax.devices()[0])
        sharded_array.addressable_shards = [
            jax.device_put(array, d) for d in jax.devices()
        ]

        protected = _ProtectedShardedArray(sharded_array)

        # Attempt deletion (should not delete sharded arrays)
        protected.delete()

        # Verify array is still accessible
        self.assertIs(protected._array, sharded_array)
        self.assertTrue(
            hasattr(protected, "_is_sharded") and protected._is_sharded
        )

    def test_jax_variable_strong_references_and_logging(self):
        """Test JaxVariable strong references and logging."""
        # Create a sharded variable
        var = JaxVariable(jax.numpy.ones((100, 100)))

        # Check strong references
        self.assertTrue(hasattr(var, "_shard_references"))
        self.assertGreater(len(var._shard_references), 0)

        # Access value multiple times to simulate inference
        for _ in range(5):
            value = var.value
            self.assertIsNotNone(
                value
            )  # Ensure no "Array has been deleted" error

        # Final check: Value should still be accessible
        self.assertIsNotNone(var.value)

    @pytest.mark.skipif(not is_nnx_enabled(), reason="NNX not enabled")
    def test_nnx_variable_strong_references_and_logging(self):
        """Test NnxVariable strong references and logging."""
        # Create NNX variable with sharding
        var = NnxVariable(jax.numpy.ones((50, 50)), layout=("model", None))

        # Check strong references
        self.assertTrue(hasattr(var, "_shard_references"))
        self.assertGreater(len(var._shard_references), 0)

        # Access value (simulates inference) and assert no deletion
        value = var.value
        self.assertIsNotNone(value)  # Ensure no "Array has been deleted" error

        # Additional accesses to simulate repeated inference
        for _ in range(5):
            value = var.value
            self.assertIsNotNone(value)

    def test_variable_loading_with_sharding(self):
        """Test variable loading with sharding support."""
        # Create a temporary file for the variable
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
            temp_file = f.name
            np.save(temp_file, jax.numpy.ones((10, 10)))

        try:
            # Create variable with sharding
            var = JaxVariable(jax.numpy.zeros((10, 10)))
            # Load data into it
            load_variable_with_sharded_support(var, np.load(temp_file))

            # Verify it's a JaxVariable with sharding
            self.assertIsInstance(var, JaxVariable)
            self.assertTrue(hasattr(var, "_shard_references"))
            self.assertGreater(len(var._shard_references), 0)

            # Access value to ensure no deletion
            self.assertIsNotNone(var.value)
        finally:
            os.unlink(temp_file)

    def test_inference_simulation_no_array_deletion(self):
        """Test inference simulation for no 'Array has been deleted' errors."""
        # Create a simple model with sharding
        inputs = layers.Input(shape=(10,))
        x = layers.Dense(50, name="dense")(inputs)
        model = models.Model(inputs, x)

        # Build and access weights (triggers sharding and protection)
        model.build((None, 10))
        for var in model.weights:
            value = var.value  # Access to trigger protection
            self.assertIsNotNone(value)  # Ensure initial access succeeds

        # Simulate inference (multiple accesses) and assert no deletion
        test_input = np.random.randn(1, 10)
        for _ in range(10):
            output = model(test_input)
            self.assertIsNotNone(
                output
            )  # Ensure inference succeeds without errors

        # Final check: Weights should still be accessible
        for var in model.weights:
            self.assertIsNotNone(var.value)


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX backend specific test for core Variable integration with NNX.",
)
@pytest.mark.skipif(
    not is_nnx_enabled(),
    reason="Test requires NNX backend to be enabled by default for setup.",
)
class NnxVariableTest(testing.TestCase):
    def setUp(self):
        super().setUp()

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
