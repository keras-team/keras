# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""See primitives_test docstring for how the Jax2TfLimitations are used."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import itertools
from typing import Any

import jax
from jax import lax
from jax import numpy as jnp
from jax._src import config
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.internal_test_util import test_harnesses
import numpy as np

DType = Any


class Jax2TfLimitation(test_harnesses.Limitation):
  """Specific primitive limitations for jax2tf.

  See the primitive_test module docstring for details.
  """

  # Bitmask values for encoding limitations specific to native lowering
  FOR_NATIVE = 1
  FOR_NON_NATIVE = 2

  def __init__(
      self,
      description: str,
      *,
      devices: str | Sequence[str] = ("cpu", "gpu", "tpu"),
      dtypes: Sequence[DType] = (),
      enabled: bool = True,
      # jax2tf specific
      modes=("eager", "graph", "compiled"),
      native_serialization=FOR_NON_NATIVE,
      skip_tf_run=False,
      expect_tf_error: bool = True,
      skip_comparison=False,
      custom_assert: Callable | None = None,
      tol=None):
    """See the test_harnesses.Limitation common arguments.

    Args :
      modes: one of "eager", "graph", "compiled"
      for_native_serialization: A bitmask with some of {FOR_NATIVE, FOR_NON_NATIVE}
        to specify how the limitation applies to native and non-native lowering.
      skip_tf_run: if set will skip the TF execution. Use this sparingly,
        prefer `expect_tf_error`. Use only when the test cannot recover from
        the TF error.
      expect_tf_error: if set, then expect a TF error in the given mode when
        executing the result of jax2tf conversion. If not set, then the
        limitation must have a custom_assert or non-default tol.
      skip_comparison: skips the numeric comparison.
      tol: a tolerance to use for both atol and rtol. We will use the maximum
        tolerance over all the applicable limitations, irrespective of their
        order.
      custom_assert: if given, then execute as
        `custom_assert(tst, result_jax, result_tf, args=args, tol=tol, err_msg)`
        , where `tst` is the current TestCase instance, and args are the input
        arguments that the harness created. The `tol` is the maximum tolerance
        based on the applicable limitations. `err_msg` is passed to NumPy
        assert methods.
        `result_tf` is already converted to NumPy arrays.
    """
    super().__init__(
        description, devices=devices, dtypes=dtypes, enabled=enabled)
    if isinstance(modes, str):
      modes = (modes,)
    assert all(m in ["eager", "graph", "compiled"] for m in modes), "Invalid modes: {modes}"
    self.modes = modes
    self.native_serialization = native_serialization
    self.expect_tf_error = expect_tf_error
    self.skip_tf_run = skip_tf_run
    self.custom_assert = custom_assert
    self.tol = tol
    self.skip_comparison = skip_comparison

  def get_max_tolerance_limitation(
      self, limitations: Sequence[Jax2TfLimitation]
  ) -> Jax2TfLimitation | None:
    """Pick the tolerance limitation that establishes the maximum tolerance."""
    # TODO: it would be best if the limitations with tolerance are mutually exclusive
    # and we don't have to compute the maximum
    # TODO: we made this an instance method only so that we don't have to import
    # this module from tf_test.util.
    max_tol_lim = None
    for l in limitations:
      if l.tol is not None:
        if max_tol_lim is None or l.tol > max_tol_lim.tol:
          max_tol_lim = l
    return max_tol_lim

  def filter(  # type: ignore[override]
      self,
      dtype: DType | None = None,
      device: str | None = None,
      mode: str | None = None) -> bool:
    """Checks if this limitation is enabled for dtype and device and mode."""
    native_serialization_mask = (
        Jax2TfLimitation.FOR_NATIVE
        if config.jax2tf_default_native_serialization.value
        else Jax2TfLimitation.FOR_NON_NATIVE)
    return ((mode is None or mode in self.modes) and
            (self.native_serialization & native_serialization_mask) and
            super().filter(device=device, dtype=dtype))

  @classmethod
  def limitations_for_harness(
      cls, harness: test_harnesses.Harness) -> Sequence[Jax2TfLimitation]:
    group_method = getattr(cls, harness.group_name, None)
    if harness.group_name in cls.harness_groups_no_limitations:
      assert group_method is None, (
          f"Harness group '{harness.group_name}' is both in "
          f"'harness_groups_no_limitations' and has a custom "
          f"Jax2TfLimitation.classmethod defined (see module docstring)")
      return []
    else:
      assert group_method is not None, (
          f"Harness group '{harness.group_name}' must be either part of "
          f"'harness_groups_no_limitations' or must have a custom "
          f"Jax2TfLimitation.classmethod defined (see module docstring)")
      limitations = group_method(harness)
      assert isinstance(limitations, (list, tuple))
      return limitations

  # We keep here the explicit set of groups for which we don't have limitations
  harness_groups_no_limitations = {
      "abs", "add", "add_any", "and", "atan2", "bitcast_convert_type",
      "broadcast", "broadcast_in_dim", "ceil", "clamp", "concatenate",
      "cos", "cosh", "complex", "conj", "convert_element_type", "cummax",
      "cummin", "device_put", "dynamic_slice", "dynamic_update_slice", "exp",
      "eq", "floor", "gather", "ge", "gt", "imag", "iota", "iota_2x32_shape",
      "is_finite", "le", "logistic", "lt", "log", "mul", "ne", "neg", "not",
      "or", "pad", "population_count", "random_categorical", "random_uniform",
      "random_randint", "reduce", "reduce_and", "reduce_precision",
      "reduce_prod", "reduce_or",
      "reduce_sum", "reduce_window_mul", "reduce_window_min",
      "reduce_window_max", "real", "reshape", "rev", "rsqrt", "select_n",
      "select_and_scatter_add", "shift_left", "shift_right_logical",
      "shift_right_arithmetic", "sign", "sin", "sinh", "slice", "sqrt",
      "squeeze", "stop_gradient", "sub", "tie_in", "transpose", "xor",
      "zeros_like"
  }

  @classmethod
  def helper_get_trig_custom_limitation(cls, np_inverse):

    def custom_assert(tst, result_jax, result_tf, *, args, tol, err_msg):
      operand, = args
      tst.assertAllClose(
          operand, np_inverse(result_tf), atol=tol, rtol=tol, err_msg=err_msg)

    return custom_numeric(
        description="May return different but still correct results",
        dtypes=[np.complex64, np.complex128],
        custom_assert=custom_assert)

  @classmethod
  def random_seed(cls, handess: test_harnesses.Harness):
    return [custom_random_keys_output()]

  @classmethod
  def random_split(cls, handess: test_harnesses.Harness):
    return [custom_random_keys_output()]

  @classmethod
  def random_fold_in(cls, handess: test_harnesses.Harness):
    return [custom_random_keys_output()]

  @classmethod
  def acos(cls, harness: test_harnesses.Harness):
    return [
        custom_numeric(
            dtypes=[np.complex64],
            devices=("cpu", "gpu"),
            tol=1e-4,
            modes=("eager", "graph", "compiled")),
        custom_numeric(
            dtypes=[np.complex128],
            devices=("cpu", "gpu"),
            tol=1e-13,
            modes=("eager", "graph", "compiled")),
        custom_numeric(
            dtypes=[np.complex64],
            devices=("tpu",),
            tol=1e-3,
            modes=("eager", "graph", "compiled"),
            native_serialization=Jax2TfLimitation.FOR_NON_NATIVE),
    ]

  @classmethod
  def acosh(cls, harness: test_harnesses.Harness):
    return [
        custom_numeric(dtypes=[np.complex64], devices=("cpu", "gpu", "tpu"),
                       tol=1e-3),
        custom_numeric(dtypes=[np.complex128], devices=("cpu", "gpu"), tol=1e-12),
        Jax2TfLimitation(
            "TF2XLA impl for Acosh doesn't properly handle large complex types,"
            " native serialization more closely matches numpy numerics.",
            dtypes=[np.complex64, np.complex128],
            devices=("cpu", "gpu", "tpu"),
            modes="compiled",
            expect_tf_error=False,
            skip_comparison=True,
            native_serialization=Jax2TfLimitation.FOR_NON_NATIVE,
        ),
        cls.helper_get_trig_custom_limitation(np.cosh),
    ]

  @classmethod
  def approx_top_k(cls, harness: test_harnesses.Harness):
    supported_dtypes = jtu.supported_dtypes()
    def custom_assert(tst, result_jax, result_tf, *, args, tol, err_msg):
      del tol, err_msg
      # Tests only that the indices correspond to the returned values
      jax_values, jax_indices = result_jax
      tf_values, tf_indices = result_tf
      operand, = args
      def operand_values(indices):
        if operand.ndim == 1:
          return operand[indices]
        elif operand.ndim == 2:
          return operand[np.arange(operand.shape[0]).reshape((-1, 1)), indices]
        else:
          assert False
      tst.assertAllClose(operand_values(jax_indices), jax_values)
      tst.assertAllClose(operand_values(tf_indices), tf_values)

    return [
        missing_tf_kernel(
            dtypes=[t for t in [jnp.bfloat16, np.float16, np.float32, np.float64]
                    if t in supported_dtypes],
            devices=("cpu", "gpu"),
            modes=("graph", "eager")),
        Jax2TfLimitation(
            "compilation not supported for float64.",
            dtypes=[np.float64],
            devices=("cpu", "gpu"),
            modes=("compiled",)),
        custom_numeric(
            dtypes=[t for t in [jnp.bfloat16, np.float16, np.float32, np.float64]
                    if t in supported_dtypes],
            devices=("cpu", "gpu"),
            modes=("eager", "graph"),
            custom_assert=custom_assert)]

  @classmethod
  def argmax(cls, harness: test_harnesses.Harness):
    return [
        Jax2TfLimitation(
            "different results when the input contains NaN and enable_xla=False",
            dtypes=jtu.dtypes.all_inexact,
            devices=("cpu", "gpu", "tpu"),
            modes=("eager", "graph", "compiled"),
            expect_tf_error=False,
            skip_comparison=True,
            enabled=("nan_" in harness.name and not harness.params["enable_xla"])),
    ]

  @classmethod
  def argmin(cls, harness: test_harnesses.Harness):
    return cls.argmax(harness)

  @classmethod
  def asin(cls, harness: test_harnesses.Harness):
    return [
        custom_numeric(dtypes=[np.complex64], devices=("cpu", "gpu"), tol=1e-4,
                       modes=("eager", "graph", "compiled")),
        custom_numeric(dtypes=[np.complex64], devices=("tpu", "gpu"), tol=2e-4,
                       modes=("eager", "graph", "compiled")),
        custom_numeric(dtypes=[np.complex128], devices=("cpu", "gpu"), tol=1e-12,
                       modes=("eager", "graph", "compiled")),
        cls.helper_get_trig_custom_limitation(np.sin)
    ]

  @classmethod
  def asinh(cls, harness: test_harnesses.Harness):
    return [
        custom_numeric(dtypes=[np.complex64], devices=("cpu", "gpu", "tpu"),
                       tol=1e-3),
        custom_numeric(dtypes=[np.complex128], devices=("cpu", "gpu"), tol=1e-12),
        custom_numeric(dtypes=[np.complex64, np.complex128],
                       devices=("cpu", "gpu", "tpu"),
                       modes=("compiled",),
                       tol=1e-3,
                       native_serialization=Jax2TfLimitation.FOR_NON_NATIVE),
        custom_numeric(dtypes=[np.complex128], devices=("cpu",),
                       modes=("eager", "compiled", "graph"),
                       tol=1e-13,
                       native_serialization=Jax2TfLimitation.FOR_NATIVE | Jax2TfLimitation.FOR_NON_NATIVE),
        cls.helper_get_trig_custom_limitation(np.sinh)
    ]

  @classmethod
  def atan(cls, harness: test_harnesses.Harness):
    return [
        custom_numeric(dtypes=[np.complex64], devices=("cpu", "gpu"), tol=1e-5),
        custom_numeric(dtypes=[np.complex64], devices=("tpu"), tol=1e-3),
        custom_numeric(dtypes=[np.complex128], devices=("cpu", "gpu"), tol=1e-12),
        cls.helper_get_trig_custom_limitation(np.tan)
    ]

  @classmethod
  def atanh(cls, harness: test_harnesses.Harness):
    return [
        custom_numeric(dtypes=[np.float64], tol=1e-14),
        custom_numeric(dtypes=[np.complex64], tol=1e-3),
        custom_numeric(dtypes=[np.complex128], devices=("cpu", "gpu"), tol=1e-12),
        cls.helper_get_trig_custom_limitation(np.tanh)
    ]

  @classmethod
  def bessel_i0e(cls, harness: test_harnesses.Harness):
    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph"))
    ]

  @classmethod
  def bessel_i1e(cls, harness: test_harnesses.Harness):
    return cls.bessel_i0e(harness)

  @classmethod
  def cbrt(cls, harness: test_harnesses.Harness):
    return [
        custom_numeric(dtypes=[np.float32], devices=("tpu"), tol=1e-5),
    ]

  @classmethod
  def cholesky(cls, harness: test_harnesses.Harness):

    def custom_assert(tst, result_jax, result_tf, *, tol, err_msg, **_):
      # cholesky_p returns garbage in the strictly upper triangular part of the
      # result, so we can safely ignore that part.
      tst.assertAllClose(
          jnp.tril(result_jax), result_tf, atol=tol, err_msg=err_msg)

    return [
        # TODO: very high tolerance
        custom_numeric(
            dtypes=[np.float32, np.complex64],
            tol=1e-2,
            devices=("cpu", "gpu"),
            modes=("eager", "graph", "compiled")),
        custom_numeric(
            dtypes=[np.float64, np.complex128],
            tol=1e-6,
            devices=("cpu", "gpu"),
            modes=("eager", "graph", "compiled")),
        custom_numeric(
            dtypes=[dtypes.bfloat16, np.float16],
            tol=5e-2,
            devices=("cpu", "gpu"),
            modes=("eager", "graph", "compiled")),
        custom_numeric(
            dtypes=[dtypes.bfloat16],
            tol=5e-5,
            # Error for GL
            devices=("tpu",),
            modes=("eager", "graph", "compiled"),
            native_serialization=Jax2TfLimitation.FOR_NATIVE),
        custom_numeric(
            custom_assert=custom_assert,
            description=(
                "May return different values in the strictly upper triangular "
                "part of the result. This does not matter for correctness, "
                "because this part of the matrix is not considered in the result."
            ),
            modes=("eager", "graph", "compiled"))
    ]

  @classmethod
  def conv_general_dilated(cls, harness: test_harnesses.Harness):
    prefer_elem = harness.params["preferred_element_type"]
    return [
        Jax2TfLimitation(
          "Non-deterministic NaN for conv_general_dilated with preferred_element_type",
          dtypes=[
            jnp.int32, np.int16, np.int64
          ],
          devices=["cpu", "gpu", "tpu"],
          modes=("eager", "graph", "compiled"),
          enabled=(prefer_elem is not None
                   and prefer_elem in [jnp.bfloat16, np.float16, np.float32, np.float64]),
          skip_comparison=True),
        # Even in compiled mode, for GPU we see a bit of discrepancy but
        # very minor.
        custom_numeric(dtypes=[np.float32], devices="gpu",
                       modes=("eager", "graph", "compiled"),
                       tol=1e-5),
        custom_numeric(dtypes=[np.float32], devices="cpu",
                       modes=("eager", "graph", "compiled"),
                       tol=1e-4,
                       native_serialization=Jax2TfLimitation.FOR_NATIVE | Jax2TfLimitation.FOR_NON_NATIVE),
        custom_numeric(description="higher numeric inaccuracy when `enable_xla=False`",
                       modes=("eager", "graph", "compiled"),
                       enabled=(not harness.params["enable_xla"]),
                       tol=5e-3)
    ]

  @classmethod
  def cumlogsumexp(cls, harness):
    return [
        custom_numeric(
            dtypes=(np.float16, jnp.bfloat16, np.float32),
            devices=("cpu", "gpu", "tpu"),
            modes=("eager", "graph", "compiled"),
            tol=5e-1,
        )
    ]

  @classmethod
  def cumprod(cls, harness):
    return [
        custom_numeric(
            dtypes=(np.float16, jnp.bfloat16),
            devices=("cpu", "gpu", "tpu"),
            modes=("eager", "graph", "compiled"),
            tol=5e-1,
        )
    ]

  @classmethod
  def cumsum(cls, harness):
    return [
        custom_numeric(
            dtypes=(np.float16, jnp.bfloat16),
            devices=("cpu", "gpu", "tpu"),
            modes=("eager", "graph", "compiled"),
            tol=5e-1,
        )
    ]

  @classmethod
  def custom_linear_solve(cls, harness: test_harnesses.Harness):
    return [
        Jax2TfLimitation(
            "TODO: large numerical discrepancy",
            dtypes=[np.float32],
            devices="tpu",
            expect_tf_error=False,
            skip_comparison=True),
        custom_numeric(dtypes=[np.float32], devices="tpu", tol=0.01),
        custom_numeric(tol=1e-3),
    ]

  @classmethod
  def digamma(cls, harness: test_harnesses.Harness):
    dtype = harness.dtype

    # In the bfloat16 case, TF and lax both return NaN in undefined cases.
    # digamma is not defined at 0 and -1
    def custom_assert(tst, result_jax, result_tf, *, args, tol, err_msg):
      # lax.digamma returns NaN and tf.math.digamma returns inf
      arg, = args
      special_cases = (arg == 0.) | (arg == -1.)
      nr_special_cases = np.count_nonzero(special_cases)
      tst.assertAllClose(
          np.full((nr_special_cases,), dtype(np.nan)),
          result_jax[special_cases],
          err_msg=err_msg)
      tst.assertAllClose(
          np.full((nr_special_cases,), dtype(np.inf)),
          result_tf[special_cases],
          err_msg=err_msg)
      # non-special cases are equal
      tst.assertAllClose(
          result_jax[~special_cases],
          result_tf[~special_cases],
          atol=tol,
          rtol=tol,
          err_msg=err_msg)

    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=[np.float64], tol=1e-13),
        custom_numeric(dtypes=[np.float32], devices=["cpu", "gpu"], tol=1e-3),
        custom_numeric(
            dtypes=[dtypes.bfloat16],
            custom_assert=custom_assert,
            description=(
                "May return different results at singularity points 0 and -1."
                "JAX returns nan and TF returns inf"))
    ]

  @classmethod
  def div(cls, harness: test_harnesses.Harness):
    return [
        Jax2TfLimitation(
            "TF integer division fails if divisor contains 0; JAX returns NaN",
            dtypes=[
                np.uint8, np.int8, np.uint16, np.uint32, np.uint64, np.int8,
                np.int16, np.int32, np.int64
            ],
            # Only the harnesses with "singularity" will have divide by 0
            enabled=("singularity" in harness.name))
    ]

  @classmethod
  def dot_general(cls, harness: test_harnesses.Harness):
    prefer_elem = harness.params["preferred_element_type"]
    return [
        missing_tf_kernel(dtypes=[np.bool_],),
        # TODO(b/189287598)
        Jax2TfLimitation(
            "Non-deterministic NaN for dot_general with preferred_element_type on GPU (b/189287598)",
            dtypes=[
                jnp.bfloat16, np.float16, np.float32, np.complex64
            ],
            devices="gpu",
            modes=("eager", "graph", "compiled"),
            enabled=(prefer_elem is not None),
            skip_comparison=True),
        # TODO(b/241740367) - note this only occurs when X64 is enabled.
        Jax2TfLimitation(
            "Large tolerances when upcasting with preferred_element_type on CPU (b/241740367)",
            devices=["cpu", "gpu", "tpu"],
            enabled=prefer_elem and np.dtype(harness.dtype) < np.dtype(prefer_elem),
            skip_comparison=True),
        # TODO(necula): look into this, but this is only for non-native serialization
        Jax2TfLimitation(
            "Errors when lhs_dtype != rhs_dtype for non-native serialization with 64-bit types",
            devices=["cpu", "gpu", "tpu"],
            enabled=(harness.dtype != harness.params["rhs_dtype"] and
                     (harness.dtype in [np.int64, np.uint64, np.float64] or
                      harness.params["rhs_dtype"] in [np.int64, np.uint64, np.float64])),
            skip_comparison=True),
      # TODO(necula): look into this, but this is only for non-native serialization and enable_xla=False
      Jax2TfLimitation(
        "Errors for non-native serialization with enable_xla=False for certain input dtype combinations",
        devices=["cpu", "gpu", "tpu"],
        enabled=(not harness.params["enable_xla"] and
                 (harness.dtype in [np.int16, np.uint32, np.uint16] or
                  harness.params["rhs_dtype"] in [np.int16, np.uint32, np.uint16] or
                  # Some combinations end up being widened to a larger type that is not
                  # supported
                  (harness.dtype, harness.params["rhs_dtype"]) in [
                    (np.float16, jnp.bfloat16),
                    (np.int32, np.float16),
                    (np.int8, np.float16),
                    (np.int8, np.uint8),
                  ])),
        skip_comparison=True,
        skip_tf_run=True),
        # TODO(necula): look into this, but this is only for non-native serialization
        Jax2TfLimitation(
            "Crash when lhs_dtype != rhs_dtype for non-native serialization on TPU for complex numbers",
            devices=["tpu"],
            enabled=(harness.dtype != harness.params["rhs_dtype"] and
                     (harness.dtype in [np.complex64, np.complex128] or
                      harness.params["rhs_dtype"] in [np.complex64, np.complex128])),
            skip_comparison=True,
            skip_tf_run=True),
        # JAX performs float16 matmuls in float32 on CPU, so the JAX result
        # may be more precise.
        custom_numeric(dtypes=[np.float16], devices=["cpu"], tol=1e-2,
                       modes=("eager", "graph", "compiled")),
        # Flakiness on different_dtypes_lhs_int16_4_3_rhs_float16_3_6_dimensionnumbers_1_0_enable_xla_True
        # Strangely, we only see the flakiness in primitives_graph_serialization_test_gpu_pjrt_c_api
        custom_numeric(dtypes=[np.int16], devices=["gpu"], tol=1e-2,
                       modes=("eager", "graph", "compiled"),
                       enabled=(harness.params["enable_xla"] and
                                harness.dtype != harness.params["rhs_dtype"])),
    ]

  @classmethod
  def eig(cls, harness: test_harnesses.Harness):
    compute_left_eigenvectors = harness.params["compute_left_eigenvectors"]
    compute_right_eigenvectors = harness.params["compute_right_eigenvectors"]
    dtype = harness.dtype

    def custom_assert(tst, result_jax, result_tf, *, args, tol, err_msg):
      operand, = args
      inner_dimension = operand.shape[-1]

      # Test ported from tests.linlag_test.testEig
      # Norm, adjusted for dimension and type.
      def norm(x):
        norm = np.linalg.norm(x, axis=(-2, -1))
        return norm / ((inner_dimension + 1) * jnp.finfo(dtype).eps)

      def check_right_eigenvectors(a, w, vr):
        tst.assertTrue(
            np.all(norm(np.matmul(a, vr) - w[..., None, :] * vr) < 100))

      def check_left_eigenvectors(a, w, vl):
        rank = len(a.shape)
        aH = jnp.conj(a.transpose(list(range(rank - 2)) + [rank - 1, rank - 2]))
        wC = jnp.conj(w)
        check_right_eigenvectors(aH, wC, vl)

      def check_eigenvalue_is_in_array(eigenvalue, eigenvalues_array):
        tol = None
        # TODO(bchetioui): numerical discrepancies
        if dtype in [np.float32, np.complex64]:
          tol = 1e-4
        elif dtype in [np.float64, np.complex128]:
          tol = 1e-13
        closest_diff = min(abs(eigenvalues_array - eigenvalue))
        tst.assertAllClose(
            closest_diff,
            np.array(0., closest_diff.dtype),
            atol=tol,
            err_msg=err_msg)

      all_w_jax, all_w_tf = result_jax[0], result_tf[0]
      for idx in itertools.product(*map(range, operand.shape[:-2])):
        w_jax, w_tf = all_w_jax[idx], all_w_tf[idx]
        for i in range(inner_dimension):
          check_eigenvalue_is_in_array(w_jax[i], w_tf)
          check_eigenvalue_is_in_array(w_tf[i], w_jax)

      if compute_left_eigenvectors:
        check_left_eigenvectors(operand, all_w_tf, result_tf[1])
      if compute_right_eigenvectors:
        check_right_eigenvectors(operand, all_w_tf,
                                 result_tf[1 + compute_left_eigenvectors])

    return [
        # Eig does not work in JAX on gpu or tpu
        Jax2TfLimitation(
            "function not compilable", modes="compiled", devices="cpu"),
        Jax2TfLimitation(
            "TF Conversion of eig is not implemented when both compute_left_eigenvectors and compute_right_eigenvectors are set to True",
            enabled=(compute_left_eigenvectors and compute_right_eigenvectors)),
        custom_numeric(
            custom_assert=custom_assert,
            description=("May return the eigenvalues and eigenvectors in a "
                         "potentially different order. The eigenvectors may "
                         "also be different, but equally valid."))
    ]

  @classmethod
  def eigh(cls, harness: test_harnesses.Harness):
    dtype = harness.dtype

    def custom_assert(tst, result_jax, result_tf, *, args, tol, err_msg):
      operand, = args
      inner_dimension = operand.shape[-1]

      def check_right_eigenvectors(a, w, vr):
        tol = 1e-16
        # TODO(bchetioui): tolerance needs to be very high in compiled mode,
        # specifically for eigenvectors.
        if dtype == np.float64:
          tol = 2e-5
        elif dtype == np.float32:
          tol = 1e-2
        elif dtype in [dtypes.bfloat16, np.complex64]:
          tol = 1e-3
        elif dtype == np.complex128:
          tol = 2e-5
        tst.assertAllClose(
            np.matmul(a, vr) - w[..., None, :] * vr,
            np.zeros(a.shape, dtype=vr.dtype),
            atol=tol,
            # For bfloat16 the np.matmul returns float32 result.
            check_dtypes=False,
            err_msg=err_msg)

      def check_eigenvalue_is_in_array(eigenvalue, eigenvalues_array):
        tol = None
        if dtype in [dtypes.bfloat16, np.float32, np.complex64]:
          tol = 1e-3
        elif dtype in [np.float64, np.complex128]:
          tol = 1e-5
        closest_diff = min(abs(eigenvalues_array - eigenvalue))
        tst.assertAllClose(
            closest_diff,
            np.array(0., closest_diff.dtype),
            atol=tol,
            err_msg=err_msg)

      _, all_w_jax = result_jax
      all_vr_tf, all_w_tf = result_tf

      for idx in itertools.product(*map(range, operand.shape[:-2])):
        w_jax, w_tf = all_w_jax[idx], all_w_tf[idx]
        for i in range(inner_dimension):
          check_eigenvalue_is_in_array(w_jax[i], w_tf)
          check_eigenvalue_is_in_array(w_tf[i], w_jax)

      check_right_eigenvectors(operand, all_w_tf, all_vr_tf)

    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices="tpu",
            enabled=(harness.params["shape"] != (0, 0)),  # This actually works!
        ),
        Jax2TfLimitation(
            "TODO: numeric discrepancies",
            dtypes=[np.float16],
            devices="tpu",
            expect_tf_error=False,
            skip_comparison=True),
        custom_numeric(
            custom_assert=custom_assert,
            description=("May return the eigenvalues and eigenvectors in a "
                         "potentially different order. The eigenvectors may "
                         "also be different, but equally valid."),
            modes=("eager", "graph", "compiled"))
    ]

  @classmethod
  def erf(cls, harness: test_harnesses.Harness):
    return []

  @classmethod
  def erfc(cls, harness: test_harnesses.Harness):
    return []

  @classmethod
  def erf_inv(cls, harness: test_harnesses.Harness):
    # erf_inv is not defined for arg <= -1 or arg >= 1
    def custom_assert(tst, result_jax, result_tf, *, args, tol,
                      err_msg):  # noqa: F811
      arg, = args
      # for arg < -1 or arg > 1
      # lax.erf_inv returns NaN; tf.math.erf_inv return +/- inf
      special_cases = (arg < -1.) | (arg > 1.)
      # non-special cases are equal
      tst.assertAllClose(
          result_jax[~special_cases],
          result_tf[~special_cases],
          atol=tol,
          rtol=tol,
          err_msg=err_msg)

    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16, np.float16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=[np.float32, np.float64], tol=1e-4),
        custom_numeric(
            dtypes=[np.float32, np.float64],
            custom_assert=custom_assert,
            description=(
                "May return different results at undefined points (< -1 or > 1):"
                " JAX returns `NaN` and TF returns `+inf` or `-inf`.")),
    ]

  @classmethod
  def expm1(cls, harness: test_harnesses.Harness):
    return [custom_numeric(dtypes=[np.float64], tol=1e-5)]

  @classmethod
  def fft(cls, harness):
    return [
        Jax2TfLimitation(
            "TF function not compilableble",
            devices=("cpu", "gpu"),
            dtypes=[np.float64],
            modes="compiled"),
        Jax2TfLimitation(
            "TF function not compilableble for IFFT and IRFFT",
            devices=("cpu", "gpu"),
            dtypes=[np.complex128],
            modes="compiled",
            enabled=(str(harness.params["fft_type"]) in ["FftType.IFFT",
                                                         "FftType.IRFFT"])),
        # TODO: very high tolerance
        custom_numeric(tol=1e-3, modes=("eager", "graph", "compiled"),
                       native_serialization=Jax2TfLimitation.FOR_NON_NATIVE),
        custom_numeric(tol=1e-5, modes=("eager", "graph", "compiled"),
                       native_serialization=Jax2TfLimitation.FOR_NATIVE,
                       devices=("cpu",)),
    ]

  @classmethod
  def _pow_test_util(cls, harness: test_harnesses.Harness):

    def custom_assert(tst, result_jax, result_tf, *, args, tol, err_msg):
      # NaNs are mismatched, but assertAllClose will also behave weirdly for
      # complex numbers containing np.inf as one of their components. See
      # https://github.com/numpy/numpy/issues/15959 for more details.
      mask = (
          np.isnan(result_jax) + np.isnan(result_tf) + np.isinf(result_jax) +
          np.isinf(result_tf))
      tst.assertAllClose(
          result_jax[~mask], result_tf[~mask], rtol=tol, err_msg=err_msg)

    return [
        custom_numeric(
            dtypes=[np.float32, np.complex64], devices=("cpu", "gpu"),
            tol=1e-3),
        custom_numeric(
            dtypes=[np.float64, np.complex128],
            devices=("cpu", "gpu"),
            tol=5e-5),
        custom_numeric(
            dtypes=[np.complex64, np.complex128],
            custom_assert=custom_assert,
        )
    ]

  @classmethod
  def igamma(cls, harness: test_harnesses.Harness):
    dtype = harness.dtype

    # igamma is not defined when the first argument is <=0
    def custom_assert(tst, result_jax, result_tf, *, args, tol, err_msg):
      arg1, arg2 = args
      # lax.igamma returns NaN when arg1 == arg2 == 0; tf.math.igamma returns 0
      special_cases = (arg1 == 0.) & (arg2 == 0.)
      nr_special_cases = np.count_nonzero(special_cases)
      tst.assertAllClose(
          np.full((nr_special_cases,), np.nan, dtype=dtype),
          result_jax[special_cases])
      tst.assertAllClose(
          np.full((nr_special_cases,), 0., dtype=dtype),
          result_tf[special_cases])
      if harness.dtype == np.float32:
        tol = 1e-5
      # non-special cases are equal
      tst.assertAllClose(
          result_jax[~special_cases],
          result_tf[~special_cases],
          atol=tol,
          rtol=tol,
          err_msg=err_msg)

    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16, np.float16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        custom_numeric(
            custom_assert=custom_assert,
            description=(
                "May return different results at undefined points "
                "(both arguments 0). JAX returns `NaN` and TF returns 0 or "
                "JAX returns 1 and TF returns `NaN`"))
    ]

  @classmethod
  def igammac(cls, harness: test_harnesses.Harness):
    dtype = harness.dtype

    # igammac is not defined when the first argument is <=0
    def custom_assert(tst, result_jax, result_tf, *, args, tol,
                      err_msg):  # noqa: F811
      arg1, arg2 = args
      # lax.igammac returns 1. when arg1 <= 0; tf.math.igammac returns NaN
      special_cases = (arg1 <= 0.) | (arg2 <= 0)
      nr_special_cases = np.count_nonzero(special_cases)
      tst.assertAllClose(
          np.full((nr_special_cases,), 1., dtype=dtype),
          result_jax[special_cases],
          err_msg=err_msg)
      tst.assertAllClose(
          np.full((nr_special_cases,), np.nan, dtype=dtype),
          result_tf[special_cases],
          err_msg=err_msg)
      # non-special cases are equal
      tst.assertAllClose(
          result_jax[~special_cases],
          result_tf[~special_cases],
          atol=tol,
          rtol=tol,
          err_msg=err_msg)

    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16, np.float16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=[np.float64], tol=1e-9),
        custom_numeric(devices="gpu", tol=1e-3),
        custom_numeric(
            custom_assert=custom_assert,
            devices=("cpu", "gpu"),
            description=(
                "May return different results at undefined points "
                "(both arguments less or equal 0). JAX returns `NaN` and TF returns 0 or "
                "JAX returns 1 and TF returns `NaN`")),
    ]

  @classmethod
  def integer_pow(cls, harness: test_harnesses.Harness):
    y = harness.params["y"]
    return [
        # TODO: on TPU, for f16, we get different results with eager mode
        # than with compiled mode.
        Jax2TfLimitation(
            "Different overflow behavior. ",
            dtypes=[np.float16, jnp.bfloat16],
            devices="tpu",
            expect_tf_error=False,
            modes=("eager", "graph"),
            skip_comparison=True),
        Jax2TfLimitation(
            "Different overflow behavior for large exponents. ",
            dtypes=[
                np.int8, np.int16, np.int32, np.int64, np.float16, jnp.bfloat16,
                np.float32, np.complex64, np.complex128
            ],
            enabled=(abs(y) > 10),
            expect_tf_error=False,
            modes=("eager", "graph"),
            skip_comparison=True),
        custom_numeric(dtypes=[dtypes.bfloat16], tol=2e-2)
    ] + list(cls._pow_test_util(harness))

  @classmethod
  def pow(cls, harness: test_harnesses.Harness):
    return cls._pow_test_util(harness)

  @classmethod
  def lgamma(cls, harness: test_harnesses.Harness):
    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=[np.float64], tol=1e-11),
        custom_numeric(dtypes=[np.float32], tol=1e-3)
    ]

  @classmethod
  def log1p(cls, harness: test_harnesses.Harness):
    return [
        custom_numeric(dtypes=[np.complex128], tol=3e-14),
        custom_numeric(dtypes=[np.float64], tol=1e-10),
        custom_numeric(dtypes=[np.float32], tol=1e-3)
    ]

  @classmethod
  def lu(cls, harness: test_harnesses.Harness):
    dtype = harness.dtype

    def custom_assert(tst, result_jax, result_tf, *, args, tol, err_msg):
      operand, = args
      lu, pivots, perm = result_tf
      batch_dims = operand.shape[:-2]
      m, n = operand.shape[-2], operand.shape[-1]

      def _make_permutation_matrix(perm):
        result = []
        for idx in itertools.product(*map(range, operand.shape[:-1])):
          result += [0 if c != perm[idx] else 1 for c in range(m)]
        result = np.reshape(np.array(result, dtype=dtype), [*batch_dims, m, m])
        return result

      k = min(m, n)
      l = jnp.tril(lu, -1)[..., :, :k] + jnp.eye(m, k, dtype=dtype)
      u = jnp.triu(lu)[..., :k, :]
      p_mat = _make_permutation_matrix(perm)

      tst.assertArraysEqual(
          lax.linalg.lu_pivots_to_permutation(pivots, m), perm)
      tst.assertAllClose(
          jnp.matmul(p_mat, operand),
          jnp.matmul(l, u),
          atol=tol,
          rtol=tol,
          err_msg=err_msg)

    return [
        custom_numeric(
            dtypes=[np.float32, np.complex64], devices="tpu", tol=0.1),
        custom_numeric(
            dtypes=[np.float32, np.complex64], devices=("cpu", "gpu"),
            tol=1e-5),
        custom_numeric(
            dtypes=[np.float64, np.complex128],
            modes=("eager", "graph"),
            tol=1e-13),
        custom_numeric(
            dtypes=[np.float64, np.complex128], modes=("compiled"), tol=1e-14),
        custom_numeric(
            custom_assert=custom_assert,
            description=("May return different, but also correct, results when "
                         "the decomposition is not unique"),
            devices=("cpu", "gpu"),
            modes=("eager", "graph", "compiled")),
    ]

  @classmethod
  def max(cls, harness: test_harnesses.Harness):
    # TODO(bchetioui): discrepancies between TF & JAX when comparing with NaN;
    # JAX always returns NaN, while TF returns the value NaN is compared with.
    def custom_assert(tst, result_jax, result_tf, err_msg, **_):
      mask = np.isnan(result_jax)
      tst.assertAllClose(result_jax[~mask], result_tf[~mask], err_msg=err_msg)

    return [
        custom_numeric(
            custom_assert=custom_assert,
            description=(
                "May return different values when one of the values is NaN. "
                "JAX always returns NaN, while TF returns the value NaN is compared with."
            ),
            modes=("eager", "graph", "compiled"),
            native_serialization=Jax2TfLimitation.FOR_NON_NATIVE),
        # TODO(b/269996580)
        custom_numeric(
            custom_assert=custom_assert,
            devices="cpu",
            description=(
                "TF and JAX use different values of the compiler flag "
                "xla_cpu_enable_fast_min_max compiler flag and therefore have "
                "different behavior of NaN propagation through min/max."
            ),
            modes=("eager", "graph", "compiled"),
            native_serialization=Jax2TfLimitation.FOR_NATIVE)
    ]

  @classmethod
  def min(cls, harness: test_harnesses.Harness):
    # TODO(bchetioui): discrepancies between TF & JAX when comparing with NaN;
    # JAX always returns NaN, while TF returns the value NaN is compared with.
    def custom_assert(tst, result_jax, result_tf, *, err_msg, **_):
      mask = np.isnan(result_jax)
      tst.assertAllClose(result_jax[~mask], result_tf[~mask], err_msg=err_msg)

    return [
        custom_numeric(
            custom_assert=custom_assert,
            description=(
                "May return different values when one of the values is NaN. "
                "JAX always returns NaN, while TF returns the value NaN is compared with."
            ),
            modes=("eager", "graph", "compiled"),
            native_serialization=Jax2TfLimitation.FOR_NON_NATIVE),
        # TODO(b/269996580)
        custom_numeric(
            custom_assert=custom_assert,
            devices="cpu",
            description=(
                "TF and JAX use different values of the compiler flag "
                "xla_cpu_enable_fast_min_max compiler flag and therefore have "
                "different behavior of NaN propagation through min/max."
            ),
            modes=("eager", "graph", "compiled"),
            native_serialization=Jax2TfLimitation.FOR_NATIVE)
    ]

  @classmethod
  def nextafter(cls, harness: test_harnesses.Harness):
    return [missing_tf_kernel(dtypes=[np.float16, dtypes.bfloat16])]

  @classmethod
  def qr(cls, harness: test_harnesses.Harness):
    # See https://github.com/jax-ml/jax/pull/3775#issuecomment-659407824;
    #     # jit_compile=True breaks for complex types.
    # TODO: see https://github.com/jax-ml/jax/pull/3775#issuecomment-659407824.
    # - for now, the performance of the HLO QR implementation called when
    #   compiling with TF is expected to have worse performance than the
    #   custom calls made in JAX.
    return [
        custom_numeric(
            dtypes=[np.float64, np.complex128],
            devices=("cpu", "gpu"),
            modes=("eager", "graph", "compiled"),
            tol=1e-13),
        custom_numeric(
            dtypes=[np.float32, np.complex64],
            devices=("cpu", "gpu"),
            modes=("eager", "graph", "compiled"),
            tol=1e-4),
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices="tpu",
        )
    ]

  @classmethod
  def random_gamma(cls, harness: test_harnesses.Harness):
    return [custom_numeric(devices="tpu", tol=1e-3)]

  @classmethod
  def reduce_max(cls, harness: test_harnesses.Harness):
    # Unlike reduce_window_max, we use a native TF op: tf.reduce_max, which
    # does not work for complex
    return [missing_tf_kernel(dtypes=[np.complex64, np.complex128])]

  @classmethod
  def reduce_min(cls, harness: test_harnesses.Harness):
    return cls.reduce_max(harness)

  @classmethod
  def reduce_window_add(cls, harness: test_harnesses.Harness):
    return [
        Jax2TfLimitation(
            "Small deviations on GPU for large inputs and enable_xla=False",
            dtypes=[np.float32],
            devices="gpu",
            modes=("eager", "graph", "compiled"),
            expect_tf_error=False,
            skip_comparison=False,
            enabled=not harness.params["enable_xla"],
            tol=3e-5),
        Jax2TfLimitation(
            "Large deviations on TPU for enable_xla=False",
            dtypes=[dtypes.bfloat16, np.float16, np.float32],
            devices="tpu",
            modes=("eager", "graph", "compiled"),
            expect_tf_error=False,
            skip_comparison=True,
            enabled=not harness.params["enable_xla"]),
      custom_numeric(devices="cpu", dtypes=[np.float32],
                     modes=("eager", "graph", "compiled",), tol=1e-5),
      custom_numeric(devices=("cpu", "gpu"), dtypes=[np.float16],
                     modes=("eager", "graph", "compiled",), tol=5e-3),
      custom_numeric(devices=("cpu", "gpu"), dtypes=[dtypes.bfloat16],
                     modes=("eager", "graph", "compiled",), tol=5e-1),
    ]

  @classmethod
  def regularized_incomplete_beta(cls, harness: test_harnesses.Harness):
    return [
        custom_numeric(dtypes=[np.float64], tol=1e-14),
        missing_tf_kernel(dtypes=[np.float16, dtypes.bfloat16])
    ]

  @classmethod
  def rem(cls, harness: test_harnesses.Harness):
    return [
        Jax2TfLimitation(
            "TF integer division fails if divisor contains 0; JAX returns NaN",
            dtypes=[
                np.uint8, np.int8, np.uint16, np.uint32, np.uint64, np.int8,
                np.int16, np.int32, np.int64
            ],
            skip_comparison=True,
            # Only the harnesses with "singularity" will have divide by 0
            enabled=("singularity" in harness.name)),
        Jax2TfLimitation(
            "TF division of inf by inf returns inf while in JAX returns nan",
            dtypes=[
                np.float32,
            ],
            devices="gpu",
            skip_comparison=True,
            enabled=("singularity_inf_by_inf" in harness.name)),
    ]

  @classmethod
  def rng_bit_generator(cls, harness: test_harnesses.Harness):
    return []

  @classmethod
  def round(cls, harness: test_harnesses.Harness):
    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices=("cpu", "gpu"),
            modes=("eager", "graph"))
    ]

  @classmethod
  def scatter(cls, harness):
    return [
        Jax2TfLimitation(
            "out-of-bounds scatters are not supported in graph and eager mode",
            dtypes=jtu.dtypes.all_inexact,
            devices=("cpu", "gpu", "tpu"),
            modes=("eager", "graph"),
            expect_tf_error=True,
            skip_comparison=True,
            enabled=("modes_out_of_bounds" in harness.name and not harness.params["enable_xla"])),
      custom_numeric(modes=("eager", "graph", "compiled"),
                     dtypes=[np.float16], tol=5e-3,
                     enabled=(not harness.params["enable_xla"])),
    ]

  @classmethod
  def scatter_add(cls, harness):
    return cls.scatter(harness)

  @classmethod
  def scatter_mul(cls, harness):
    return cls.scatter(harness)

  @classmethod
  def scatter_max(cls, harness):
    return cls.scatter(harness)

  @classmethod
  def scatter_min(cls, harness):
    return cls.scatter(harness)

  @classmethod
  def select_and_gather_add(cls, harness):
    return [
        # This JAX primitives is not exposed directly in the JAX API
        # but arises from JVP of `lax.reduce_window` for reducers
        # `lax.max` or `lax.min`. It also arises from second-order
        # VJP of the same. Implemented using XlaReduceWindow.
        Jax2TfLimitation((
            "jax2tf unimplemented for 64-bit inputs because the current implementation "
            "relies on packing two values into a single value. This can be "
            "fixed by using a variadic XlaReduceWindow, when available"),
                         dtypes=[np.float64],
                         devices=("cpu", "gpu"))
    ]

  @classmethod
  def sort(cls, harness: test_harnesses.Harness):
    return [
        Jax2TfLimitation(
            # I think that this is because TF is running on CPU even for GPU tests?
            "TODO: TF non-stable multiple-array sort",
            devices="gpu",
            enabled=(harness.params["num_arrays"] > 1 and
                     not harness.params["is_stable"]),
            expect_tf_error=False,
            skip_comparison=True),
    ]

  @classmethod
  def svd(cls, harness: test_harnesses.Harness):
    # TODO: slow test
    compute_uv = harness.params["compute_uv"]

    # Both `r_jax` and `r_tf` are 3-Tuples containing the SVD results:
    # `S` (singular values), `U` (left singular vectors), and `Vh` (the
    # adjoint of the right singular vectors). Note that the TF results are
    # obtained through `_svd` in jax/experimental/jax2tf/jax2tf.py.
    def custom_assert(tst, r_jax, r_tf, *, args, tol, err_msg):

      def reconstruct_operand(result):
        # Reconstructing operand as documented in numpy.linalg.svd (see
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html)
        s, u, v = result
        U = u[..., :s.shape[-1]]
        V = v[..., :s.shape[-1], :]
        S = s[..., None, :]
        return jnp.matmul(U * S, V, precision=lax.Precision.HIGHEST)

      # Compares the shapes.
      def compare_shapes(r_jax, r_tf):
        shapes_jax = [result.shape for result in r_jax]
        shapes_tf = [result.shape for result in r_tf]
        tst.assertEqual(shapes_jax, shapes_tf)

      # Compares reconstructed operand.
      # Computes backward error https://www.netlib.org/lapack/lug/node97.html
      # and uses the maximum backward error if there are batch dimensions.
      # The backward error is bounded by some constant multiplying the machine
      # precision.
      # TODO: Compares the operand instead of the reconstructed operand.
      def compare_reconstructed_operand(r_jax, r_tf, tol):
        operand_jax = reconstruct_operand(r_jax)
        operand_tf = reconstruct_operand(r_tf)
        error_norm = jnp.linalg.norm(operand_jax - operand_tf,
                                              axis=(-2, -1))
        backward_error = (error_norm /
                          jnp.linalg.norm(operand_jax, axis=(-2, -1)))
        max_backward_error = jnp.amax(backward_error)
        tst.assertLess(max_backward_error, tol)

      # Computes the absolute gap between singular value `\sigma_i` and the
      # nearest other singular value and for all singular values. The absolute
      # gap is used to approximate the upper bound of angular difference
      # between the computed and the true singular vectors. If the matrix is
      # rectangular `m != n`, the gap for the smallest nonzero singular value
      # should also consider the gap between it and zero. Note that this code
      # relies on the singular values being in descending order.
      def compute_absolute_gap(s, m, n):
        forward_appendant = np.inf if m == n else 0
        forward_diff = jnp.diff(s, axis=-1, append=forward_appendant)
        backward_diff = jnp.diff(
            s[..., ::-1], axis=-1, append=np.inf)[..., ::-1]
        absolute_gap = jnp.minimum(jnp.abs(forward_diff),
                                   jnp.abs(backward_diff))
        return absolute_gap

      # See `CompareSingularVectors` in
      # tensorflow/python/kernel_tests/linalg/svd_op_test.py
      def compare_singular_vectors(x, y, *, error_bound):
        # Singular vectors are only unique up to sign (complex phase factor for
        # complex matrices), so we normalize the sign first.
        sum_of_ratios = jnp.sum(jnp.divide(y, x), -2, keepdims=True)
        phases = jnp.divide(sum_of_ratios, jnp.abs(sum_of_ratios))
        x *= phases

        # Note that in general `sqrt(sum(squares))` is not a stable way to
        # compute l2 vector norms, but it should be OK for normalization
        # factors of vectors with norm ~= 1 as here.
        def dot_column_wise(a, b):
          output = jnp.sum(jnp.einsum('...ij,...ij->...ij', a.conj(), b,
                                      precision=lax.Precision.HIGHEST),
                           axis=-2)
          return jnp.real(output)

        cos_angular_diff = (
            dot_column_wise(x, y) /
            jnp.sqrt(dot_column_wise(x, x) * dot_column_wise(y, y)))

        # Values of `\cos(angular_diff)` outside the interval [0, 1] are clipped
        # to the interval edges. For example, `\cos(angular_diff)` could contain
        # values like 1.0000001 on float32, which are clipped to 1.0. It is
        # possible that anything other than `cos_angular_diff` can be outside
        # the interval [0, 1] due to roundoff.
        cos_angular_diff = jnp.clip(cos_angular_diff, min=0.0, max=1.0)

        angular_diff = jnp.arccos(cos_angular_diff)

        # TODO: removes the slack factor on the angular difference.
        # It is possible that the singular vectors are not accurate to much more
        # than O(\sqrt(eps)), which is likely a property of the SVD algorithms
        # in question; revisit with better understanding of the SVD algorithms.
        if x.dtype in [np.float32, np.complex64]:
          slack_factor = 2E4
        elif x.dtype in [np.float64, np.complex128]:
          slack_factor = 2E9

        np.testing.assert_array_less(angular_diff,
                                     slack_factor * error_bound)

      if compute_uv:
        # Compares the shapes.
        compare_shapes(r_jax, r_tf)

        # Compares the singular values. Each computed singular value `\sigma_i`
        # differs from the true `\sigma_i`* by at most
        # `|\sigma_i - \sigma_i*| <= \epsilon \sigma_1`, where `\sigma_1` is the
        # largest singular value and `\epsilon` denotes the machine precision.
        s_jax, s_tf = r_jax[0], r_tf[0]
        tst.assertAllClose(s_jax, s_tf, atol=tol, rtol=tol, err_msg=err_msg)

        # Compares the reconstructed operand.
        compare_reconstructed_operand(r_jax, r_tf, tol)

        # Compares the singular vectors.
        # We only compare the first `rank` singular vectors since the remainder
        # forms an arbitrary orthonormal basis for the (row- or column-) null
        # space, whose exact value depends on implementation details.
        # TODO: A better estimation on the rank?
        rank = r_jax[0].shape[-1]

        # Computes the upper bound for angular difference of singular vectors.
        # The upper bound has the shape of `[..., k]`, where `...` denotes the
        # batch dimensions and `k` is the number of nonzero singular values.
        m = r_jax[1].shape[-2]
        n = r_jax[2].shape[-2]
        absolute_gap = compute_absolute_gap(r_jax[0], m, n)
        epsilon = jnp.finfo(r_jax[0].dtype).eps
        sigma_largest = (r_jax[0][..., 0])[..., None]
        upperbound_singular_vectors = epsilon * sigma_largest / absolute_gap
        upperbound_singular_vectors = upperbound_singular_vectors[..., :rank]

        # Left singular vectors.
        u_jax = r_jax[1][..., :rank]
        u_tf = r_tf[1][..., :rank]
        compare_singular_vectors(u_jax, u_tf,
                                 error_bound=upperbound_singular_vectors)

        # Right singular vectors.
        v_jax = jnp.swapaxes(r_jax[2][..., :rank, :], -2, -1).conj()
        v_tf = jnp.swapaxes(r_tf[2][..., :rank, :], -2, -1).conj()
        compare_singular_vectors(v_jax, v_tf,
                                 error_bound=upperbound_singular_vectors)
      else:
        tst.assertAllClose(r_jax, r_tf, atol=tol, rtol=tol, err_msg=err_msg)

    return [
        # Works in JAX for complex due to custom calls on cpu and gpu
        Jax2TfLimitation(
            "function not compilable. Implemented using `tf.linalg.svd` and `tf.linalg.adjoint`",
            dtypes=[np.complex64, np.complex128],
            devices=("cpu", "gpu"),
            modes=("compiled",)),
        Jax2TfLimitation(
            "Large numerical discrepancy",
            dtypes=[np.float16],
            devices=("tpu"),
            modes=("eager", "graph", "compiled"),
            skip_comparison=True),
        missing_tf_kernel(dtypes=[dtypes.bfloat16], devices="tpu"),
        missing_tf_kernel(dtypes=[np.complex64, np.complex128],
                          modes=("compiled", "graph"),
                          devices="tpu"),
        custom_numeric(
            tol=1e-4,
            dtypes=[np.float32, np.complex64],
            devices=("cpu", "gpu"),
            modes=("eager", "graph", "compiled")),
        # TODO: this is very low tolerance for f64
        custom_numeric(
            tol=1e-4,
            dtypes=[np.float64, np.complex128],
            devices=("cpu", "gpu"),
            modes=("eager", "graph", "compiled")),
        custom_numeric(
            tol=1e-4,
            description="custom numeric comparison when compute_uv on CPU/GPU",
            custom_assert=custom_assert,
            devices=("cpu", "gpu"),
            modes=("eager", "graph", "compiled"),
            enabled=(compute_uv == True)),
        custom_numeric(
            tol=1e-5,
            description="custom numeric comparison when !compute_uv on TPU",
            dtypes=[np.float32, np.complex64],
            custom_assert=custom_assert,
            devices=("tpu"),
            modes=("eager", "graph", "compiled"),
            enabled=not compute_uv),
        custom_numeric(
            tol=1e-2,
            description="custom numeric comparison when compute_uv on TPU",
            dtypes=[np.float32, np.float64, np.complex64, np.complex128],
            custom_assert=custom_assert,
            devices=("tpu"),
            modes=("eager", "graph", "compiled"),
            enabled=(compute_uv == True)),
    ]

  @classmethod
  def tan(cls, harness):
    return [
        custom_numeric(dtypes=[np.complex64], devices="tpu", tol=1e-4),
        custom_numeric(dtypes=[np.complex64], devices=("cpu", "gpu"), tol=1e-3),
        custom_numeric(dtypes=[np.complex128], devices=("cpu", "gpu"), tol=1e-12)
    ]

  @classmethod
  def tanh(cls, harness):
    return [
        custom_numeric(dtypes=[np.complex128], tol=1e-7),
        custom_numeric(dtypes=[np.complex64], tol=1e-4)
    ]

  @classmethod
  def top_k(cls, harness):

    def custom_assert(tst, result_jax, result_tf, *, err_msg, **_):
      assert len(result_jax) == len(result_tf)
      # TODO: TF and JAX sort [inf, nan] differently.
      first_arr_jax, first_arr_tf = result_jax[0], result_tf[0]
      if np.all(first_arr_jax == first_arr_tf):
        for arr_jax, arr_tf in zip(result_jax, result_tf):
          tst.assertArraysEqual(arr_jax, arr_tf, err_msg=err_msg)
      else:
        mask_jax = np.isnan(first_arr_jax) | np.isinf(first_arr_jax)
        mask_tf = np.isnan(first_arr_tf) | np.isinf(first_arr_tf)
        tst.assertArraysEqual(
            first_arr_jax[~mask_jax], first_arr_tf[~mask_tf], err_msg=err_msg)

    return [
        custom_numeric(
            dtypes=[np.float16, dtypes.bfloat16, np.float32, np.float64],
            custom_assert=custom_assert,
            description=(
                "Produces different results when the array contains `inf` and `NaN`"
                " (they are sorted differently in TF vs. XLA)."))
    ]

  @classmethod
  def triangular_solve(cls, harness: test_harnesses.Harness):
    return [
        missing_tf_kernel(
            dtypes=[dtypes.bfloat16],
            devices=("gpu", "cpu"),
            modes=("eager", "graph")),
        missing_tf_kernel(
            dtypes=[np.float16],
            devices=("gpu", "cpu"),
            modes=("eager", "graph")),
        custom_numeric(dtypes=[np.float32], tol=5e-3,
                       modes=("eager", "graph", "compiled"))
    ]

  @classmethod
  def tridiagonal_solve(cls, harness: test_harnesses.Harness):
    return []

def custom_numeric(
    *,
    description="custom numeric comparison",
    dtypes=(),  # All
    modes=(
        "eager",
        "graph",
    ),  # By default we should not need tolerance for
    # "compiled"
    devices=("cpu", "gpu", "tpu"),
    custom_assert=None,
    enabled=True,
    native_serialization=Jax2TfLimitation.FOR_NON_NATIVE,
    tol=None) -> Jax2TfLimitation:

  return Jax2TfLimitation(
      description,
      expect_tf_error=False,
      dtypes=dtypes,
      devices=devices,
      modes=modes,
      custom_assert=custom_assert,
      enabled=enabled,
      native_serialization=native_serialization,
      tol=tol)

def custom_random_keys_output():
  def custom_assert(tst, result_jax, result_tf, *, args, tol, err_msg):
    # Here we handle both new-style and old-style keys; see JEP 9263
    def unwrap_keys(keys):
      if jax.dtypes.issubdtype(keys.dtype, jax.dtypes.prng_key):
        return jax._src.prng.random_unwrap(keys)
      else:
        return keys

    tst.assertAllClose(unwrap_keys(result_jax), result_tf,
                       atol=tol, rtol=tol, err_msg=err_msg)

  return custom_numeric(
      description="Returns JAX key arrays, so compare underlying base array",
      modes=("eager", "graph", "compiled"),
      custom_assert=custom_assert)


def missing_tf_kernel(*,
    description="op not defined for dtype",
    dtypes,
    modes=("eager", "graph", "compiled"),
    devices=("cpu", "gpu", "tpu"),
    native_serialization = Jax2TfLimitation.FOR_NON_NATIVE,
    enabled=True) -> Jax2TfLimitation:

  return Jax2TfLimitation(
      description, dtypes=dtypes, devices=devices, modes=modes, enabled=enabled,
      native_serialization=native_serialization)
