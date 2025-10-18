import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import keras
from keras.src import backend
from keras.src import testing
from keras.src.backend.config import is_nnx_enabled
from keras.src.backend.jax.core import all_gather
from keras.src.backend.jax.core import all_reduce

if is_nnx_enabled():
    from flax import nnx

    from keras.src.backend.jax.core import NnxVariable


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX backend specific test for core Variable integration with NNX.",
)
@pytest.mark.skipif(
    not is_nnx_enabled(),
    reason="Test requires NNX backend to be enabled by default for setup.",
)
class NnxVariableTest(testing.TestCase):
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
    reason="JAX backend specific test for collective operations.",
)
@pytest.mark.skipif(
    jax.local_device_count() < 2,
    reason="Requires multiple local devices for testing.",
)
class JaxCollectiveOpsTest(testing.TestCase):
    def test_all_reduce_sum(self):
        """Tests the all_reduce operation with the 'sum' reduction."""
        num_devices = jax.local_device_count()
        local_value = 10.0

        local_inputs = jax.numpy.array([local_value] * num_devices)

        @functools.partial(
            jax.pmap, axis_name="all", devices=jax.devices("cpu")
        )
        def reduce_sum_fn(x):
            return all_reduce(x, op="sum", axis_name="all")

        result = reduce_sum_fn(local_inputs)
        expected_sum = local_value * num_devices

        self.assertTrue(np.allclose(result, expected_sum))
        self.assertEqual(result.shape, (num_devices,))

    def test_all_reduce_mean(self):
        """Tests the all_reduce operation with the 'mean' reduction."""
        num_devices = jax.local_device_count()
        local_value = 10.0

        local_inputs = jax.numpy.array([local_value] * num_devices)

        @functools.partial(
            jax.pmap, axis_name="all", devices=jax.devices("cpu")
        )
        def reduce_mean_fn(x):
            return all_reduce(x, op="mean", axis_name="all")

        result = reduce_mean_fn(local_inputs)
        expected_mean = local_value

        self.assertTrue(np.allclose(result, expected_mean))
        self.assertEqual(result.shape, (num_devices,))

    def test_all_gather(self):
        """Tests the all_gather operation."""
        num_devices = jax.local_device_count()
        local_data = np.arange(5)

        local_inputs = jax.numpy.stack(
            [local_data + (i * 5) for i in range(num_devices)]
        )

        @functools.partial(
            jax.pmap, axis_name="all", devices=jax.devices("cpu")
        )
        def gather_fn(x):
            return all_gather(x, axis=0, axis_name="all")

        result_array_on_devices = gather_fn(local_inputs)

        expected_shape = (num_devices, num_devices * local_data.shape[0])
        self.assertEqual(result_array_on_devices.shape, expected_shape)

        expected_gathered_data = np.arange(num_devices * local_data.shape[0])

        for i in range(num_devices):
            self.assertTrue(
                np.allclose(result_array_on_devices[i], expected_gathered_data)
            )
