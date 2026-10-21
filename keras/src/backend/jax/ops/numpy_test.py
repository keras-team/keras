"""Tests for JAX backend numpy ops."""

import jax
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.ops import numpy as knp


@pytest.mark.skipif(
    backend.backend() != "jax" or testing.uses_gpu() or testing.uses_tpu(),
    reason="XLA CPU specific test",
)
class Int8MatmulTest(testing.TestCase):
    """XLA CPU runs a one-row int8 dot on an int32 copy of the right operand.

    The tests compile `matmul` and check that the compiled program holds no
    such copy. `keras/src/ops/numpy_test.py` checks the values on every
    backend.
    """

    @parameterized.named_parameters(
        ("rank1", (64,)),
        ("one_row", (1, 64)),
        ("two_rows", (2, 64)),
        ("one_row_rank3", (1, 1, 64)),
        ("three_rows_rank3", (3, 1, 64)),
        ("one_row_k16384", (1, 16384)),
    )
    def test_matmul_int8(self, x_shape):
        rng = np.random.default_rng(0)
        x = rng.integers(-128, 128, size=x_shape, dtype="int8")
        y = rng.integers(-128, 128, size=(x_shape[-1], 256), dtype="int8")
        expected = np.matmul(x.astype("int64"), y.astype("int64"))

        compiled = jax.jit(knp.matmul).lower(x, y).compile()
        self.assertNotIn(f"s32[{x_shape[-1]},256]", compiled.as_text())
        self.assertAllEqual(compiled(x, y), expected)

    @parameterized.named_parameters(
        ("rank1", (64,)),
        ("rank2", (1, 64)),
        ("rank3", (1, 1, 64)),
    )
    def test_matmul_int8_one_row_transposed(self, x_shape):
        # A tied `ReversibleEmbedding` contracts against the transpose of
        # its table.
        rng = np.random.default_rng(0)
        x = rng.integers(-128, 128, size=x_shape, dtype="int8")
        table = rng.integers(-128, 128, size=(256, 64), dtype="int8")
        expected = np.matmul(x.astype("int64"), table.T.astype("int64"))

        def matmul_transposed(x, table):
            return knp.matmul(x, knp.transpose(table))

        compiled = jax.jit(matmul_transposed).lower(x, table).compile()
        self.assertNotIn("s32[256,64]", compiled.as_text())
        self.assertNotIn("s32[64,256]", compiled.as_text())
        self.assertAllEqual(compiled(x, table), expected)
