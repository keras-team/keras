from absl.testing import parameterized

from keras.backend.common import result_dtype
from keras.testing import test_case


class DtypesTest(test_case.TestCase, parameterized.TestCase):
    @parameterized.product(
        dtype=[
            "float16",
            "float32",
            "float64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "bfloat16",
            "bool",
            # "string",  # not supported
        ],
        python_scalar_type=[
            bool,
            int,
            float,
            # complex,  # not supported
        ],
    )
    def test_result_dtype_with_python_scalar_types(
        self, dtype, python_scalar_type
    ):
        import jax.numpy as jnp

        # match `jnp.result_dtype` result
        out = result_dtype(dtype, python_scalar_type)
        expected = jnp.result_type(dtype, python_scalar_type).name
        self.assertEqual(out, expected)
