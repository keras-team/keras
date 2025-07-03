import os

import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src.backend.common import dtypes
from keras.src.testing import test_case
from keras.src.testing.test_utils import named_product

os.environ["TPU_NAME"] = "harshith-tf-4"
os.environ["JAX_PLATFORMS"] = ""


@pytest.mark.requires_tpu
class DtypesTPUTest(test_case.TestCase):
    """Test the dtype to verify that the behavior matches
    JAX, with TPU support."""

    if backend.backend() == "torch":
        from keras.src.backend.torch.core import to_torch_dtype

        # TODO: torch doesn't support uint64.
        ALL_DTYPES = []
        for x in dtypes.ALLOWED_DTYPES:
            if x not in ["string", "uint64"]:
                x = str(to_torch_dtype(x)).split(".")[-1]
                if x not in ALL_DTYPES:  # skip duplicates created by remapping
                    ALL_DTYPES.append(x)
        ALL_DTYPES += [None]
    elif backend.backend() == "openvino":
        ALL_DTYPES = [
            x
            for x in dtypes.ALLOWED_DTYPES
            if x not in ["string", "complex64", "complex128"]
        ] + [None]
    else:
        ALL_DTYPES = [x for x in dtypes.ALLOWED_DTYPES if x != "string"] + [
            None
        ]
    # Remove float8 dtypes for the following tests
    ALL_DTYPES = [x for x in ALL_DTYPES if x not in dtypes.FLOAT8_TYPES]

    def setUp(self):
        from jax.experimental import enable_x64

        self.jax_enable_x64 = enable_x64()
        self.jax_enable_x64.__enter__()
        return super().setUp()

    def tearDown(self):
        self.jax_enable_x64.__exit__(None, None, None)
        return super().tearDown()

    @parameterized.named_parameters(
        named_product(dtype1=ALL_DTYPES, dtype2=ALL_DTYPES)
    )
    def test_result_type_with_tensor(self, dtype1, dtype2):
        import jax.numpy as jnp

        x1 = ops.ones((1,), dtype=dtype1)
        x2 = ops.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)

        out = backend.result_type(x1.dtype, x2.dtype)
        expected = jnp.result_type(x1_jax, x2_jax).name
        self.assertEqual(out, expected)

    @parameterized.named_parameters(
        named_product(dtype1=ALL_DTYPES, dtype2=ALL_DTYPES)
    )
    def test_result_type_with_tensor_on_tpu(self, dtype1, dtype2):
        """Test dtype result_type behavior specifically on TPU."""
        import jax.numpy as jnp

        def _test_on_tpu():
            x1 = ops.ones((1,), dtype=dtype1)
            x2 = ops.ones((1,), dtype=dtype2)

            result = ops.add(x1, x2)

            out = backend.result_type(x1.dtype, x2.dtype)
            return out, result.dtype

        with self.tpu_strategy.scope():
            out, result_dtype = _test_on_tpu()

        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected = jnp.result_type(x1_jax, x2_jax).name

        self.assertEqual(out, expected)
        self.assertEqual(result_dtype, expected)