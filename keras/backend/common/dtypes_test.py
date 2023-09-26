from absl.testing import parameterized

from keras import backend
from keras import ops
from keras.backend.common.variables import ALLOWED_DTYPES
from keras.backend.torch.core import to_torch_dtype
from keras.testing import test_case


class DtypesTest(test_case.TestCase, parameterized.TestCase):
    """Dtypes test to verify that the result type matches `jnp.result_type`."""

    @parameterized.product(
        dtype1=[d for d in ALLOWED_DTYPES if d != "string"],
        dtype2=[bool, int, float],
    )
    def test_result_dtype_with_python_scalar_types(self, dtype1, dtype2):
        import jax.numpy as jnp

        out = backend.result_type(dtype1, dtype2)
        expected = jnp.result_type(dtype1, dtype2).name
        self.assertEqual(out, expected)

    @parameterized.product(
        # TODO: uint64, int64 and float64 are not supported by JAX by default
        dtype1=[d for d in ALLOWED_DTYPES if d != "string" and "64" not in d],
        dtype2=[d for d in ALLOWED_DTYPES if d != "string" and "64" not in d],
    )
    def test_result_dtype_with_tensor(self, dtype1, dtype2):
        # TODO: torch doesn't have `uint16` and `uint32` dtypes
        if backend.backend() == "torch":
            dtype1 = str(to_torch_dtype(dtype1)).split(".")[-1]
            dtype2 = str(to_torch_dtype(dtype2)).split(".")[-1]

        import jax.numpy as jnp

        x1 = ops.ones((1,), dtype=dtype1)
        x2 = ops.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)

        out = backend.result_type(x1.dtype, x2.dtype)
        expected = jnp.result_type(x1_jax, x2_jax).name
        self.assertEqual(out, expected)

    def test_result_dtype_with_none(self):
        import jax.numpy as jnp

        self.assertEqual(backend.result_type(None), jnp.result_type(None).name)
