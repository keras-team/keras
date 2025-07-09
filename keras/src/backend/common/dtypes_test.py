import tensorflow as tf # New import
import os
from unittest.mock import patch

from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src.backend.common import dtypes
from keras.src.testing import test_case
from keras.src.testing.test_utils import named_product

# Ensure the backend is set to TensorFlow if you intend to use TPU.
# This environment variable should ideally be set before Python starts for the process.

os.environ["KERAS_BACKEND"] = "tensorflow" # Moved to test_case module in Keras

# Set TPU_NAME if connecting to a specific TPU worker
os.environ["TPU_NAME"] = "harshith-tf-4"
# JAX_PLATFORMS is typically for JAX-specific environments, not directly for TF/Keras TPU.
os.environ["JAX_PLATFORMS"] = ""


# --- TPU-specific Dtype Definitions ---
# These must be defined at the module level for absl.testing.parameterized
# to find them when the class is being defined.

TPU_UNSUPPORTED_DTYPES = [
    "string",
    "complex64",
    "complex128",
    "float8_e4m3fn",
    "float8_e5m2",
    "float64", # Often problematic for general ops on TPU, or leads to performance issues
    "bool"     # Can cause issues with type promotion/XLA on TPU in some contexts
]

ALL_DTYPES_FOR_TPU_TESTS = [
    x for x in dtypes.ALLOWED_DTYPES if x not in TPU_UNSUPPORTED_DTYPES
] + [None]

# --- End TPU-specific Dtype Definitions ---


class DtypesTest(test_case.TestCase):
    """Test the dtype to verify that the behavior matches JAX, with optional TPU support."""

    # Original ALL_DTYPES logic (backend-dependent) remains for non-TPU tests
    if backend.backend() == "torch":
        from keras.src.backend.torch.core import to_torch_dtype

        ALL_DTYPES = []
        for x in dtypes.ALLOWED_DTYPES:
            if x not in ["string", "uint64"]:
                x = str(to_torch_dtype(x)).split(".")[-1]
                if x not in ALL_DTYPES:
                    ALL_DTYPES.append(x)
        ALL_DTYPES += [None]
    elif backend.backend() == "openvino":
        ALL_DTYPES = [
            x
            for x in dtypes.ALLOWED_DTYPES
            if x not in ["string", "complex64", "complex128"]
        ] + [None]
    else: # Default to TensorFlow or other backends
        ALL_DTYPES = [x for x in dtypes.ALLOWED_DTYPES if x != "string"] + [
            None
        ]
    # Remove float8 dtypes for the following tests (original logic)
    ALL_DTYPES = [x for x in ALL_DTYPES if x not in dtypes.FLOAT8_TYPES]

    # --- NEW: setUpClass for TPU initialization (no fixtures/markers) ---
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.tpu_available = False
        cls.tpu_strategy = None
        
        # Only attempt TPU initialization if the Keras backend is TensorFlow
        if backend.backend() == "tensorflow":
            print("\nAttempting TPU initialization from DtypesTest.setUpClass...")
            try:
                # Use empty string '' for auto-detection or 'grpc://<ip_address>:8470'
                # or your specific TPU_NAME from env var
                resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
                tf.config.experimental_connect_to_cluster(resolver)
                tf.tpu.experimental.initialize_tpu_system(resolver)
                cls.tpu_strategy = tf.distribute.TPUStrategy(resolver)
                cls.tpu_available = True
                print("✓ TPU initialization successful from DtypesTest.setUpClass!")
                print(f"Number of TPU devices: {cls.tpu_strategy.num_replicas_in_sync}")
                print(f"Logical TPU devices: {tf.config.list_logical_devices('TPU')}")
            except Exception as e:
                print(f"✗ TPU initialization failed from DtypesTest.setUpClass: {e}")
                print("Falling back to CPU/GPU testing for this class.")
                cls.tpu_available = False
        else:
            print(f"Skipping TPU initialization for backend: {backend.backend()}")

    @classmethod
    def tearDownClass(cls):
        # Optional: Shut down TPU system if it was initialized
        if cls.tpu_available:
            try:
                # This can sometimes cause issues if other processes are using it,
                # or if the context was already lost. Use with caution.
                # tf.tpu.experimental.shutdown_tpu_system()
                print("TPU system teardown (if applicable) completed.")
            except Exception as e:
                print(f"Error during TPU system teardown: {e}")
        super().tearDownClass()
    # --- END setUpClass for TPU ---

    def setUp(self):
        # The JAX x64 setup is for JAX backend tests, keep it.
        from jax.experimental import enable_x64
        self.jax_enable_x64 = enable_x64()
        self.jax_enable_x64.__enter__()
        # Clear Keras session for each test
        if backend.backend() == "tensorflow": # Only clear if TF backend is active
            tf.keras.backend.clear_session()
        return super().setUp()

    def tearDown(self):
        # JAX x64 teardown
        self.jax_enable_x64.__exit__(None, None, None)
        # Clear Keras session for each test
        if backend.backend() == "tensorflow": # Only clear if TF backend is active
            tf.keras.backend.clear_session()
        return super().tearDown()

    @parameterized.named_parameters(
        named_product(dtype1=ALL_DTYPES, dtype2=[bool, int, float])
    )
    def test_result_type_with_python_scalar_types(self, dtype1, dtype2):
        """Test dtype result_type behavior with Python scalar types (non-TPU)."""
        import jax.numpy as jnp

        out = backend.result_type(dtype1, dtype2)
        expected = jnp.result_type(dtype1, dtype2).name
        self.assertEqual(out, expected)

    @parameterized.named_parameters(
        named_product(dtype1=ALL_DTYPES, dtype2=ALL_DTYPES)
    )
    def test_result_type_with_tensor(self, dtype1, dtype2):
        """Test dtype result_type behavior with tensors (non-TPU)."""
        # This test will run for all backends as per original logic,
        # but will not explicitly use TPU.
        import jax.numpy as jnp

        x1 = ops.ones((1,), dtype=dtype1)
        x2 = ops.ones((1,), dtype=dtype2)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)

        out = backend.result_type(x1.dtype, x2.dtype)
        expected = jnp.result_type(x1_jax, x2_jax).name
        self.assertEqual(out, expected)

    # --- NEW TPU-ENABLED TEST METHOD (no fixtures/markers) ---
    @parameterized.named_parameters(
        named_product(dtype1=ALL_DTYPES_FOR_TPU_TESTS, dtype2=ALL_DTYPES_FOR_TPU_TESTS)
    )
    def test_result_type_with_tensor_on_tpu(self, dtype1, dtype2):
        """Test dtype result_type behavior specifically on TPU with supported dtypes."""
        # Check if backend is TensorFlow and TPU is available for this class
        if backend.backend() != "tensorflow":
            self.skipTest("TPU tests are only applicable for TensorFlow backend.")
        if not self.tpu_available:
            self.skipTest("TPU not available for this test class.")
            
        with self.tpu_strategy.scope(): # Use the class-level strategy object
            try:
                x1_on_tpu = ops.ones((1,), dtype=dtype1)
                x2_on_tpu = ops.ones((1,), dtype=dtype2)
                
                print(f"Initial (Eager Context) X1 Device : {x1_on_tpu.device}")
                print(f"Initial (Eager Context) X2 Device : {x2_on_tpu.device}")
                
                result_eager_attempt = ops.add(x1_on_tpu, x2_on_tpu)
                print(f"Initial (Eager Context) Add Result Device : {result_eager_attempt.device}")
                
                @tf.function
                def tpu_compute(a, b):
                    add_result = ops.add(a, b)
                    return add_result

                distributed_result = self.tpu_strategy.run(tpu_compute, args=(x1_on_tpu, x2_on_tpu))
                
                actual_result_dtype = None
                if isinstance(distributed_result, tf.distribute.DistributedValues):
                    replica_result = distributed_result.values[0]
                    print(f"Device of result from TPU replica 0: {replica_result.device}")
                    self.assertIn("TPU", replica_result.device)
                    actual_result_dtype = replica_result.dtype
                else:
                    print(f"Device of direct distributed result: {distributed_result.device}")
                    self.assertIn("TPU", distributed_result.device)
                    actual_result_dtype = distributed_result.dtype

                expected_keras_result_type = backend.result_type(x1_on_tpu.dtype, x2_on_tpu.dtype)
                
                print(f"Test case: dtype1={dtype1}, dtype2={dtype2}")
                print(f"Keras backend.result_type: {expected_keras_result_type}")
                print(f"Actual result dtype from TPU operation: {actual_result_dtype}")

                self.assertEqual(actual_result_dtype, expected_keras_result_type)
                
            except Exception as e:
                if "context_id" in str(e).lower() or "socket closed" in str(e).lower():
                    self.skipTest(f"TPU context issue or socket closed: {e}")
                else:
                    raise
    # --- END NEW TPU-ENABLED TEST METHOD ---


    # Original tests below remain unchanged
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

    def test_respect_weak_type_for_complex64(self):
        self.assertAllEqual(
            dtypes._respect_weak_type("complex64", True), "complex"
        )

    def test_respect_weak_type_for_complex128(self):
        self.assertAllEqual(
            dtypes._respect_weak_type("complex128", True), "complex"
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

    def test_invalid_dtype_in_least_upper_bound(self):
        invalid_dtype = "non_existent_dtype"
        with self.assertRaisesRegex(
            ValueError, "is not a valid dtype for Keras type promotion"
        ):
            dtypes._least_upper_bound(invalid_dtype)

    def test_empty_lub_in_least_upper_bound(self):
        dtype1 = "float32"
        dtype2 = "int32"
        with patch.dict(
            dtypes.LATTICE_UPPER_BOUNDS,
            {"float32": set(), "int32": set()},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError, "no available implicit dtype promotion path"
            ):
                dtypes._least_upper_bound(dtype1, dtype2)

    def test_valid_dtype_leading_to_single_lub_element(self):
        self.assertEqual(
            dtypes._least_upper_bound("float32", "int32"), "float32"
        )

    def test_valid_dtype_leading_to_keyerror_and_valueerror(self):
        invalid_dtype = "non_existent_dtype"
        with self.assertRaisesRegex(
            ValueError, "is not a valid dtype for Keras type promotion"
        ):
            dtypes._least_upper_bound(invalid_dtype)

    def test_resolve_weak_type_bool(self):
        self.assertEqual(dtypes._resolve_weak_type("bool"), "bool")

    def test_resolve_weak_type_int(self):
        self.assertEqual(
            dtypes._resolve_weak_type("int32", precision="32"), "int32"
        )
        self.assertEqual(
            dtypes._resolve_weak_type("int64", precision="64"), "int64"
        )

    def test_resolve_weak_type_uint(self):
        self.assertEqual(
            dtypes._resolve_weak_type("uint32", precision="32"), "uint32"
        )
        self.assertEqual(
            dtypes._resolve_weak_type("uint64", precision="64"), "uint64"
        )

    def test_resolve_weak_type_float(self):
        self.assertEqual(
            dtypes._resolve_weak_type("float32", precision="32"), "float32"
        )
        self.assertEqual(
            dtypes._resolve_weak_type("float64", precision="64"), "float64"
        )

    def test_least_upper_bound_ensure_order_independence(self):
        result1 = dtypes._least_upper_bound("float32", "int32")
        result2 = dtypes._least_upper_bound("int32", "float32")
        self.assertEqual(result1, result2)

    def test_least_upper_bound_single_element(self):
        dtypes.LATTICE_UPPER_BOUNDS["test_dtype"] = {"test_dtype"}
        self.assertEqual(dtypes._least_upper_bound("test_dtype"), "test_dtype")

    def test_least_upper_bound_no_element(self):
        dtypes.LATTICE_UPPER_BOUNDS["test_dtype"] = set()
        with self.assertRaisesRegex(
            ValueError, "no available implicit dtype promotion path"
        ):
            dtypes._least_upper_bound("test_dtype")

    def test_least_upper_bound_with_no_common_upper_bound(self):
        with patch.dict(
            dtypes.LATTICE_UPPER_BOUNDS,
            {"test_dtype1": set(), "test_dtype2": set()},
            clear=True,
        ):
            with self.assertRaisesRegex(
                ValueError, "no available implicit dtype promotion path"
            ):
                dtypes._least_upper_bound("test_dtype1", "test_dtype2")

    def test_invalid_float8_dtype(self):
        with self.assertRaisesRegex(
            ValueError, "There is no implicit conversions from float8 dtypes"
        ):
            dtypes.result_type("float8_e4m3fn", "bfloat16")
        with self.assertRaisesRegex(
            ValueError, "There is no implicit conversions from float8 dtypes"
        ):
            dtypes.result_type("float8_e5m2", "bfloat16")
