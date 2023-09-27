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

    @parameterized.product(dtype1=ALL_DTYPES, dtype2=[bool, int, float])
    def test_result_type_with_python_scalar_types(self, dtype1, dtype2):
        import jax.numpy as jnp

        out = backend.result_type(dtype1, dtype2)
        expected = jnp.result_type(dtype1, dtype2).name
        self.assertEqual(out, expected)

    @parameterized.product(dtype1=ALL_DTYPES, dtype2=ALL_DTYPES)
    def test_result_type_with_tensor(self, dtype1, dtype2):
        import jax.numpy as jnp

        x1 = ops.ones((1,), dtype=dtype1)
        x2 = ops.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)

        out = backend.result_type(x1.dtype, x2.dtype)
        expected = jnp.result_type(x1_jax, x2_jax).name
        self.assertEqual(out, expected)

    def test_result_type_with_none(self):
        import jax.numpy as jnp

        self.assertEqual(backend.result_type(None), jnp.result_type(None).name)

    @parameterized.product(dtype1=ALL_DTYPES, dtype2=ALL_DTYPES)
    def test_result_type_precanonicalize(
        self, dtype1=ALL_DTYPES, dtype2=ALL_DTYPES
    ):
        import jax.numpy as jnp
        from jax._src.dtypes import _canonicalize_dtype

        out = backend.result_type(dtype1, dtype2)
        expected = jnp.result_type(
            _canonicalize_dtype(False, True, dtype1),
            _canonicalize_dtype(False, True, dtype2),
        ).name
        self.assertEqual(out, expected)

    def test_result_type_invalid_dtypes(self):
        with self.assertRaisesRegexp(
            ValueError, "Invalid `dtypes`. At least one dtype is required."
        ):
            backend.result_type()
