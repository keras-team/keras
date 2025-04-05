# Copyright 2023 The JAX Authors.
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
"""Utilities for testing backwards compatibility of custom calls.

Since we have to guarantee 6 months of backward compatibility for the
JAX serialized format, we need to guarantee that custom calls continue to
work as before. We test this here.

The tests in this file refer to the test data in
jax/_src/internal_test_util/export_back_compat_test_data.

There is one test for each version of a custom call target.
Only custom call targets tested here should be listed in
export._CUSTOM_CALL_TARGETS_GUARANTEED_STABLE. All other custom
call targets will result in an error when encountered during serialization.

Once we stop using a custom call target in JAX, you can remove it from the
_CUSTOM_CALL_TARGETS_GUARANTEED_STABLE and you can add a comment to the
test here to remove it after 6 months.

** To create a new test **

Write the JAX function `func` that exercises the custom call `foo_call` you
want, then pick some inputs, and then add this to the new test to get started.
Add the following code to your test file, e.g., `export_back_compat_test.py`.

  import dataclasses
  from jax._src.internal_test_util import export_back_compat_test_util as bctu

  class CompatTest(bctu.CompatTestBase)
    ...

    def test_foo_call(self):
      def func(...): ...
      inputs = (...,)  # Tuple of nd.array, keep it small, perhaps generate the
                      # inputs in `func`.
      data = self.starter_data(inputs)  # This is temporary, just for starting.
      self.run_one_test(func, data)

The test will fail, but will save to a file the test data you will need. The
file name will be printed in the logs. Create a new
file jax/_src/internal_test_util/export_back_compat_test_data/foo_call.py
and paste the test data that you will see printed in the logs.

Name the literal `data_YYYYY_MM_DD` to include the date of serialization
(for readability only). Then add to this file:

  from jax._src.internal_test_util.export_back_compat_test_data import foo_call

then update `test_custom_call_coverage`, and then update your `test_foo_call`:

  def test_foo_call(self):
    def func(...): ...
    data = self.load_testdata(foo_call.data_YYYY_MM_DD)  # <-- this is new
    self.run_one_test(func, data)

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import dataclasses
import datetime
import os
import re
import sys
from typing import Any

from absl import logging

import numpy as np
# Import some NumPy symbols so that we can parse repr(ndarray).
from numpy import array, float32

import jax
from jax import tree_util
from jax import export

from jax.experimental import pjit

from jax._src import core
from jax._src import test_util as jtu
from jax._src import xla_bridge as xb


CURRENT_TESTDATA_VERSION = 1

@dataclasses.dataclass
class CompatTestData:
  testdata_version: int
  platform: str  # One of: "cpu", "tpu", "cuda", "rocm"
  custom_call_targets: list[str]
  serialized_date: datetime.date  # e.g., datetime.date(2023, 3, 9)
  inputs: Sequence[np.ndarray]
  expected_outputs: Sequence[np.ndarray]
  mlir_module_text: str
  mlir_module_serialized: bytes
  xla_call_module_version: int  # The version of XlaCallModule to use for testing
  nr_devices: int = 1


# The dummy_data is used for getting started for adding a new test and for
# testing the helper functions.
dummy_data_dict = dict(
    testdata_version=CURRENT_TESTDATA_VERSION,
    platform="cpu",
    custom_call_targets=[],
    serialized_date=datetime.date(2023, 3, 15),
    inputs=(array(0.0, dtype=float32),),
    expected_outputs=(array(0.0, dtype=float32),),
    mlir_module_text=r"""
  module @jit_sin {
  func.func public @main(%arg0: tensor<f32>) -> tensor<f32> {
    %0 = stablehlo.sine %arg0 : tensor<f32>
    return %0 : tensor<f32>
  }
}
""",
    mlir_module_serialized=b"ML\xefR\x03MLIRxxx-trunk\x00\x01\x17\x05\x01\x05\x01\x03\x05\x03\x07\x07\t\x0b\x03K5\x07\x01\x1b\x07\x0b\x13\x0b3\x0b\x0b\x0b\x0b\x0f\x0b\x13\x0b\x03\x1b\x0f\x1b\x0b\x0b\x0b\x0b\x0b\x0f\x13\x0b\x0b\x0b\x0b\x03\x07\x0f\x17\x07\x02\xa7\x1f\x05\r\x03\x03\x03\x07\x05\x0f\x03\x0b\x0b\x1b\r'\x0f)\x031\x113\x05\x11\x05\x13\x05\x15\x05\x17\x1d\x15\x17\x05\x19\x17\x19\xef\x01\x05\x1b\x03\x03\x1d\r\x05\x1f!#%\x1d\x1d\x1d\x1f\x1d!\x1d##\x03\x03\x03+\r\x03-/\x1d%\x1d'\x1d)\x1d+)\x01\x05\x11\x03\x01\x03\x01\t\x04A\x05\x01\x11\x01\x05\x07\x03\x01\x05\x03\x11\x01\t\x05\x03\x05\x0b\x03\x01\x01\x05\x06\x13\x03\x01\x03\x01\x07\x04\x01\x03\x03\x06\x03\x01\x05\x01\x00\x9a\x04-\x0f\x0b\x03!\x1b\x1d\x05\x1b\x83/\x1f\x15\x1d\x15\x11\x13\x15\x11\x11\x0f\x0b\x11builtin\x00vhlo\x00module\x00func_v1\x00sine_v1\x00return_v1\x00sym_name\x00jit_sin\x00arg_attrs\x00function_type\x00res_attrs\x00sym_visibility\x00jit(sin)/jit(main)/sin\x00third_party/py/jax/experimental/jax2tf/tests/back_compat_test.py\x00jax.arg_info\x00x\x00mhlo.sharding\x00{replicated}\x00jax.result_info\x00\x00main\x00public\x00",
    xla_call_module_version=4,
)  # End paste


class CompatTestBase(jtu.JaxTestCase):
  """Base class with helper functions for backward compatibility tests."""
  def default_jax_backend(self) -> str:
    # Canonicalize to turn into "cuda" or "rocm"
    return xb.canonicalize_platform(jax.default_backend())

  def starter_data(self, inputs: Sequence[np.ndarray]) -> CompatTestData:
    # Helper for starting a test, see module docstring.
    assert isinstance(inputs, Sequence), f"{inputs}"
    return dataclasses.replace(self.load_testdata(dummy_data_dict),
                               inputs=inputs,
                               platform=self.default_jax_backend())

  def load_testdata(self, testdata_dict: dict[str, Any]) -> CompatTestData:
    if testdata_dict["testdata_version"] == CURRENT_TESTDATA_VERSION:
      return CompatTestData(**testdata_dict)
    else:
      raise NotImplementedError("testdata_version not recognized: " +
                                testdata_dict["testdata_version"])

  def load_testdata_nested(self, testdata_nest) -> Iterable[CompatTestData]:
    # Load all the CompatTestData in a Python nest.
    if isinstance(testdata_nest, dict) and "testdata_version" in testdata_nest:
      yield self.load_testdata(testdata_nest)
    elif isinstance(testdata_nest, dict):
      for e in testdata_nest.values():
        yield from self.load_testdata_nested(e)
    elif isinstance(testdata_nest, list):
      for e in testdata_nest:
        yield from self.load_testdata_nested(e)
    else:
      assert False, testdata_nest

  def run_one_test(self, func: Callable[..., jax.Array],
                   data: CompatTestData,
                   polymorphic_shapes: Sequence[str] | None = None,
                   rtol: float | None = None,
                   atol: float | None = None,
                   allow_unstable_custom_call_targets: Sequence[str] = (),
                   check_results: Callable[..., None] | None = None,
                   expect_current_custom_calls: Sequence[str] | None = None):
    """Run one compatibility test.

    Args:
      func: the JAX function to serialize and run
      data: the test data
      polymorphic_shapes: when using shape polymorphism, the specification for
        each argument of `func`.
      rtol: relative tolerance for numerical comparisons
      atol: absolute tolerance for numerical comparisons
      check_results: invoked with the results obtained from running the
        serialized code, and those stored in the test data, and the kwargs rtol
        and atol.
      allow_unstable_custom_call_targets: additional custom call targets to allow.
      expect_current_custom_calls: if `None` checks that the current serialization
        has the same custom calls as the saved one. This is the default, and
        will fail when the serialization changes. Otherwise, when checking old
        serializations you can specify what custom calls are expected in the
        current serialization.
      nr_devices: the number of devices for which the data was serialized.
    """
    if not isinstance(data, CompatTestData):
      raise ValueError(f"Expecting data: CompatTestData but got {data}. "
                       "Did you forget to `self.load_testdata`?")

    if self.default_jax_backend() != data.platform:
      self.skipTest(f"Test enabled only for {data.platform}")

    logging.info("Lowering and running the function at the current version")
    res_run_current = self.run_current(func, data)
    if not isinstance(res_run_current, (list, tuple)):
      res_run_current = (res_run_current,)
    res_run_current = tuple(np.array(a) for a in res_run_current)
    logging.info("Result of current version run is %s", res_run_current)

    serialized, module_str, module_version, nr_devices = self.serialize(
      func, data,
      polymorphic_shapes=polymorphic_shapes,
      allow_unstable_custom_call_targets=allow_unstable_custom_call_targets)

    custom_call_re = r"stablehlo.custom_call\s*@([^\(]+)\("
    current_custom_call_targets = sorted(
        set(re.findall(custom_call_re, module_str)))

    np.set_printoptions(threshold=sys.maxsize, floatmode="unique")
    # Print the current test data to simplify updating the test.
    updated_testdata = f"""
# Pasted from the test output (see export_back_compat_test_util.py module docstring)
data_{datetime.date.today().strftime('%Y_%m_%d')} = dict(
    testdata_version={CURRENT_TESTDATA_VERSION},
    platform={self.default_jax_backend()!r},
    custom_call_targets={current_custom_call_targets!r},
    serialized_date={datetime.date.today()!r},
    inputs={data.inputs!r},
    expected_outputs={res_run_current!r},
    mlir_module_text=r\"\"\"\n{module_str}\"\"\",
    mlir_module_serialized={serialized!r},
    xla_call_module_version={module_version},
    nr_devices={nr_devices},
)  # End paste

"""
    # Replace the word that should not appear.
    updated_testdata = re.sub(r"google.", "googlex", updated_testdata)
    output_dir = os.getenv("TEST_UNDECLARED_OUTPUTS_DIR",
                           "/tmp/back_compat_testdata")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"{self._testMethodName}.py")
    logging.info("Writing the updated testdata at %s", output_file)
    with open(output_file, "w") as f:
      f.write(updated_testdata)

    if rtol is None:
      rtol = 1.e-7
    if check_results is not None:
      check_results(res_run_current, data.expected_outputs, rtol=rtol,
                    atol=atol)
    else:
      self.assertAllClose(res_run_current, data.expected_outputs, rtol=rtol,
                          atol=atol)

    logging.info("Running the serialized module")
    res_run_serialized = self.run_serialized(
        data,
        polymorphic_shapes=polymorphic_shapes)
    logging.info("Result of serialized run is %s", res_run_serialized)
    if check_results is not None:
      check_results(res_run_serialized, data.expected_outputs,
                    rtol=rtol, atol=atol)
    else:
      self.assertAllClose(res_run_serialized, data.expected_outputs,
                          rtol=rtol, atol=atol)
    if expect_current_custom_calls is None:
      expect_current_custom_calls = data.custom_call_targets
    self.assertItemsEqual(expect_current_custom_calls, current_custom_call_targets)

  def run_current(self, func: Callable, data: CompatTestData):
    """Lowers and runs the test function at the current JAX version."""
    return jax.jit(func)(*data.inputs)

  def serialize(self,
                func: Callable, data: CompatTestData, *,
                polymorphic_shapes: Sequence[str] | None = None,
                allow_unstable_custom_call_targets: Sequence[str] = ()
                ) -> tuple[bytes, str, int, int]:
    """Serializes the test function.

    Args:
      func: the function to serialize
      polymorphic_shapes: the polymorphic_shapes to use for serialization
      allow_unstable_custom_call_targets: whether to allow additional
        custom call targets besides those known as stable.

    Returns: a tuple with the (a) serialization, (b) the module contents as
      a string (for debugging), (c) the module serialization version,
      (d) the number of devices for which the module was serialized.
    """
    # Use the native exporter, to make sure we get the proper serialization.
    args_specs = export.symbolic_args_specs(data.inputs, polymorphic_shapes)
    exported = export.export(
      jax.jit(func),
      platforms=(self.default_jax_backend(),),
      disabled_checks=tuple(
        export.DisabledSafetyCheck.custom_call(target)
        for target in allow_unstable_custom_call_targets)
    )(*args_specs)

    module_str = str(exported.mlir_module())
    serialized = exported.mlir_module_serialized
    module_version = exported.calling_convention_version
    nr_devices = exported.nr_devices
    return serialized, module_str, module_version, nr_devices

  def run_serialized(self, data: CompatTestData,
                     polymorphic_shapes: Sequence[str] | None = None):
    args_specs = export.symbolic_args_specs(data.inputs, polymorphic_shapes)
    def ndarray_to_aval(a: np.ndarray) -> core.ShapedArray:
      return core.ShapedArray(a.shape, a.dtype)
    in_avals_tree = tree_util.tree_map(ndarray_to_aval, args_specs)
    # TODO: we ought to ensure that out_avals are polymorphic if need be. We
    # could either save the in/out_avals (but we need to first implement that
    # support in export), or we can just re-use them from the current
    # exported.
    out_avals_tree = tree_util.tree_map(ndarray_to_aval, data.expected_outputs)
    # in_tree must be for (args, kwargs)
    in_avals, in_tree = tree_util.tree_flatten((in_avals_tree, {}))
    out_avals, out_tree = tree_util.tree_flatten(out_avals_tree)
    def _get_vjp(_):
      assert False  # We do not have and do not need VJP

    exported = export.Exported(
        fun_name="run_serialized",
        in_tree=in_tree,
        in_avals=tuple(in_avals),
        out_tree=out_tree,
        out_avals=tuple(out_avals),
        in_shardings_hlo=(None,) * len(in_avals),
        out_shardings_hlo=(None,) * len(out_avals),
        platforms=(data.platform,),
        ordered_effects=(),
        unordered_effects=(),
        disabled_safety_checks=(),
        mlir_module_serialized=data.mlir_module_serialized,
        calling_convention_version=data.xla_call_module_version,
        nr_devices=data.nr_devices,
        module_kept_var_idx=tuple(range(len(in_avals))),
        uses_global_constants=any(not core.is_constant_shape(a.shape)
                                    for a in in_avals),
      _get_vjp=_get_vjp)

      # We use pjit in case there are shardings in the exported module.
    return pjit.pjit(exported.call)(*data.inputs)
