from absl.testing import parameterized

from keras import backend
from keras import ops
from keras.backend.common import dtypes
from keras.backend.common.variables import ALLOWED_DTYPES
from keras.testing import test_case
from keras.testing.test_utils import named_product


class DtypesTest(test_case.TestCase, parameterized.TestCase):
    """Test the dtype to verify that the behavior matches JAX."""

    if backend.backend() == "torch":
        from keras.backend.torch.core import to_torch_dtype

        # TODO: torch doesn't support uint64.
        ALL_DTYPES = []
        for x in ALLOWED_DTYPES:
            if x not in ["string", "uint64"]:
                x = str(to_torch_dtype(x)).split(".")[-1]
                if x not in ALL_DTYPES:  # skip duplicates created by remapping
                    ALL_DTYPES.append(x)
        ALL_DTYPES += [None]
    else:
        ALL_DTYPES = [x for x in ALLOWED_DTYPES if x != "string"] + [None]

    def setUp(self):
        from jax.experimental import enable_x64

        self.jax_enable_x64 = enable_x64()
        self.jax_enable_x64.__enter__()
        return super().setUp()

    def tearDown(self) -> None:
        self.jax_enable_x64.__exit__(None, None, None)
        return super().tearDown()

    @parameterized.named_parameters(
        named_product(dtype1=ALL_DTYPES, dtype2=[bool, int, float])
    )
    def test_result_type_with_python_scalar_types(self, dtype1, dtype2):
        import jax.numpy as jnp

        out = backend.result_type(dtype1, dtype2)
        expected = jnp.result_type(dtype1, dtype2).name
        self.assertEqual(out, expected)

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

    def test_result_type_with_none(self):
        import jax.numpy as jnp

        self.assertEqual(backend.result_type(None), jnp.result_type(None).name)

    def test_result_type_empty_list(self):
        self.assertEqual(backend.result_type(), "float32")

    def test_respect_weak_type_for_bool(self):
        self.assertEqual(dtypes._respect_weak_type("bool", True), "bool")

    def test_respect_weak_type_for_int(self):
        self.assertEqual(dtypes._respect_weak_type("int32", True), "int")

    def test_respect_weak_type_for_float(self):
        self.assertEqual(dtypes._respect_weak_type("float32", True), "float")

    def test_resolve_weak_type_for_bfloat16(self):
        self.assertEqual(dtypes._resolve_weak_type("bfloat16"), "float32")

    def test_resolve_weak_type_for_bfloat16_with_precision(self):
        self.assertEqual(
            dtypes._resolve_weak_type("bfloat16", precision="64"), "float64"
        )

    def test_invalid_dtype_for_keras_promotion(self):
        with self.assertRaisesRegex(
            ValueError, "is not a valid dtype for Keras type promotion."
        ):
            dtypes._least_upper_bound("invalid_dtype")

    def test_resolve_weak_type_for_invalid_dtype(self):
        with self.assertRaisesRegex(
            ValueError, "Invalid value for argument `dtype`. Expected one of"
        ):
            dtypes._resolve_weak_type("invalid_dtype")

    def test_resolve_weak_type_for_invalid_precision(self):
        with self.assertRaisesRegex(
            ValueError,
            "Invalid value for argument `precision`. Expected one of",
        ):
            dtypes._resolve_weak_type("int32", precision="invalid_precision")

    def test_cycle_detection_in_make_lattice_upper_bounds(self):
        original_lattice_function = dtypes._type_promotion_lattice

        def mock_lattice():
            lattice = original_lattice_function()
            lattice["int32"].append("float32")
            lattice["float32"].append("int32")
            return lattice

        dtypes._type_promotion_lattice = mock_lattice

        with self.assertRaisesRegex(
            ValueError, "cycle detected in type promotion lattice for node"
        ):
            dtypes._make_lattice_upper_bounds()

        dtypes._type_promotion_lattice = original_lattice_function

    def test_respect_weak_type_for_invalid_dtype(self):
        with self.assertRaisesRegex(
            ValueError, "Invalid value for argument `dtype`. Expected one of"
        ):
            dtypes._respect_weak_type("invalid_dtype", True)
