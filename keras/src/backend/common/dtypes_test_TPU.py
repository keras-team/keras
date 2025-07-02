import tensorflow as tf
from unittest.mock import patch
import os

from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src.backend.common import dtypes
from keras.src.testing import test_case
from keras.src.testing.test_utils import named_product


os.environ['TPU_NAME'] = 'harshith-tf-4'
os.environ['JAX_PLATFORMS'] = ''

class DtypesTPUTest(test_case.TestCase):
    """Test the dtype to verify that the behavior matches JAX, with TPU support."""

    # Configuration for TPU retry logic
    TPU_MAX_RETRIES = int(os.environ.get('TPU_MAX_RETRIES', '3'))
    TPU_BASE_DELAY = float(os.environ.get('TPU_BASE_DELAY', '2.0'))

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

    @classmethod
    def _cleanup_tpu_state(cls):
        """Clean up any partial TPU initialization state."""
        try:
            tf.config.experimental_disconnect_from_cluster()
        except:
            pass

        try:
            tf.config.experimental_reset_memory_stats('TPU_SYSTEM')
        except:
            pass

    @classmethod
    def setUpClass(cls):
        """Initialize TPU if available, with retry logic."""
        import time

        super().setUpClass()
        cls.tpu_available = False
        cls.tpu_strategy = None

        max_retries = cls.TPU_MAX_RETRIES
        base_delay = cls.TPU_BASE_DELAY

        for attempt in range(max_retries):
            try:
                print(f"TPU initialization attempt {attempt + 1}/{max_retries}")

                cls._cleanup_tpu_state()

                resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
                tf.config.experimental_connect_to_cluster(resolver)
                tf.tpu.experimental.initialize_tpu_system(resolver)

                tpu_devices = tf.config.list_logical_devices('TPU_SYSTEM')
                if not tpu_devices:
                    raise RuntimeError("No TPU devices found after initialization")

                cls.tpu_strategy = tf.distribute.TPUStrategy(resolver)
                cls.tpu_available = True

                print("✓ TPU initialization successful!")
                print("TPU devices found: ", tpu_devices)
                print(f"Number of TPU cores: {cls.tpu_strategy.num_replicas_in_sync}")
                break

            except (ValueError, RuntimeError, Exception) as e:
                print(f"✗ TPU initialization attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + (attempt * 0.5)
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    cls._cleanup_tpu_state()
                else:
                    print("All TPU initialization attempts failed. Falling back to CPU/GPU testing")
                    cls.tpu_available = False

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
        if not self.tpu_available:
            self.skipTest("TPU not available")

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