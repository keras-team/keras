
import jax.numpy as jnp
import pytest
from flax import nnx

from keras.src import backend
from keras.src.backend.jax.core import Variable as KerasJaxVariable
from keras.src import testing


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX backend specific test for core Variable integration with NNX.",
)
class JaxCoreVariableTest(testing.TestCase):
    def test_variable_in_nnx_module(self):
        class Model(nnx.Module):
          def __init__(self, rngs):
            self.linear = nnx.Linear(2, 3, rngs=rngs)
            self.custom_variable = KerasJaxVariable(jnp.ones((1, 3)))
          def __call__(self, x):
            return self.linear(x) + self.custom_variable

        model = Model(rngs=nnx.Rngs(0))
        self.assertTrue(hasattr(model.custom_variable,"_trace_state"))
        self.assertIsNotNone(model.custom_variable._trace_state)
        self.assertAllEqual(model.custom_variable.value, [[1, 1, 1]])
        self.assertTrue(isinstance(model.custom_variable, nnx.Variable))
