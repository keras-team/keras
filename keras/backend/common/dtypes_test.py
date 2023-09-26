from absl.testing import parameterized

from keras import backend
from keras import ops
from keras.backend.common.variables import ALLOWED_DTYPES
from keras.backend.torch.core import to_torch_dtype
from keras.testing import test_case


class DtypesTest(test_case.TestCase, parameterized.TestCase):
    """Test the dtype to verify that the behavior matches JAX."""

    if backend.backend() == "torch":
        ALL_DTYPES = [
            str(to_torch_dtype(x)).split(".")[-1]
            for x in ALLOWED_DTYPES
            if x not in ["string", "uint64"]
        ] + [None]
    else:
        ALL_DTYPES = [x for x in ALLOWED_DTYPES if x != "string"] + [None]

    def canonicalize_tf(self, dtype):
        # TODO: canonicalize "int64" once the following issue resolved:
        # https://www.tensorflow.org/xla/known_issues#tfvariable_on_a_different_device
        return (
            "int32"
            if backend.backend() == "tensorflow" and dtype == "int64"
            else dtype
        )

    @parameterized.product(dtype1=ALL_DTYPES, dtype2=[bool, int, float])
    def test_result_dtype_with_python_scalar_types(self, dtype1, dtype2):
        import jax.numpy as jnp

        out = self.canonicalize_tf(backend.result_type(dtype1, dtype2))
        expected = jnp.result_type(dtype1, dtype2).name
        self.assertEqual(out, expected)

    @parameterized.product(dtype1=ALL_DTYPES, dtype2=ALL_DTYPES)
    def test_result_dtype_with_tensor(self, dtype1, dtype2):
        import jax.numpy as jnp

        x1 = ops.ones((1,), dtype=dtype1)
        x2 = ops.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)

        out = self.canonicalize_tf(backend.result_type(x1.dtype, x2.dtype))
        expected = jnp.result_type(x1_jax, x2_jax).name
        self.assertEqual(out, expected)

    def test_result_dtype_with_none(self):
        import jax.numpy as jnp

        self.assertEqual(backend.result_type(None), jnp.result_type(None).name)
