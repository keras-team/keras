# Copyright 2020 The JAX Authors.
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

from __future__ import annotations

from collections.abc import Callable, Sequence
import contextlib
import dataclasses
import re
import os
from typing import Any

from absl.testing import absltest
from absl import logging

import jax
from jax import dtypes
from jax import numpy as jnp
from jax._src import test_util as jtu
from jax import tree_util

from jax.experimental import jax2tf
from jax import export
from jax._src import config
from jax._src import xla_bridge
import numpy as np
import tensorflow as tf
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.compiler.tf2xla.python import xla as tfxla

DType = Any

def _make_tf_input_signature(*tf_args) -> list[tf.TensorSpec]:
  # tf_args can be PyTrees
  def _make_one_array_signature(tf_arg):
    return tf.TensorSpec(np.shape(tf_arg), jax2tf.dtype_of_val(tf_arg))

  return tf.nest.map_structure(_make_one_array_signature, list(tf_args))

def _run_tf_function(func_tf: Callable, *tf_args, mode: str):
  if mode == "eager":
    return func_tf(*tf_args)  # EAGER
  elif mode == "graph":
    return tf.function(  # GRAPH
        func_tf,
        autograph=False,
        # Note that jit_compile defaults to True on TPU and False elsewhere
        input_signature=_make_tf_input_signature(*tf_args))(*tf_args)  # GRAPH
  elif mode == "compiled":
    # Adding an explicit input_signature prevents TF from constant-folding
    # the computation eagerly before compilation
    return tf.function(  # COMPILED
        func_tf,
        autograph=False,
        jit_compile=True,
        input_signature=_make_tf_input_signature(*tf_args))(
            *tf_args)  # COMPILED
  else:
    assert False, (
        f"Expected 'eager', 'graph', or 'compiled' for mode: got '{mode}'")


## Helper functions for matching OpMetadata in TF graphs
@dataclasses.dataclass(order=True, frozen=True)
class OpMetadataGraph:
  tf_type: str  # The standard Tf.Operation.type
  op_type: str  # The rest are OpMetadata fields from _Xla... attributes
  op_name: str
  source_file: str
  source_line: str


def SaveAndLoadModel(model: tf.Module,
                     save_gradients=True) -> tf.Module:
  # Roundtrip through saved model on disk.
  model_dir = os.path.join(absltest.get_default_test_tmpdir(), str(id(model)))
  tf.saved_model.save(
      model, model_dir,
      options=tf.saved_model.SaveOptions(experimental_custom_gradients=save_gradients))
  restored_model = tf.saved_model.load(model_dir)
  return restored_model

def SaveAndLoadFunction(f_tf: Callable, *,
                        input_signature: Sequence[tf.TensorSpec] | None = None,
                        input_args: Sequence[Any] | None = None,
                        variables: Sequence[tf.Variable] = (),
                        save_gradients=True) -> tuple[Callable, tf.train.Checkpoint]:
  # Roundtrip through saved model on disk. Return the Checkpoint also
  # for the cases when there are variables. If you don't pass input_signature
  # then it is created from the input_args.
  model = tf.train.Checkpoint()
  if input_signature is None:
    assert input_args is not None
    input_signature = tf.nest.map_structure(lambda a: tf.TensorSpec(a.shape, a.dtype),
                                            input_args)
  else:
    assert input_args is None
  model.f = tf.function(f_tf,
                        autograph=False,
                        input_signature=input_signature)
  model.variables = variables
  restored = SaveAndLoadModel(model, save_gradients=save_gradients)
  return restored.f, restored

def TransformJaxVJP(f: Callable, args, res_f_of_args):
  # Given `f` and its `args` tuple and `res_f_of_args=f(*args)` return a pair of a function
  # that computes the VJP of `f` and appropriate arguments tuple.
  def make_ct(res):
    res_dtype = np.result_type(res)
    assert res_dtype != dtypes.float0
    # We produce cotangents of the same type as the primal. It does not
    # seem to matter whether we feed float0, and avoiding float0 makes things
    # simpler with TF.
    return np.ones(np.shape(res), dtype=res_dtype)

  cts = tree_util.tree_map(make_ct, res_f_of_args)

  def f_vjp(args, cts):
    res, pullback = jax.vjp(f, *args)
    return pullback(cts)
  return (f_vjp, (args, cts))

def TransformTfValueAndGrad(tf_f: Callable, tf_args,
                          unconnected_gradients=tf.UnconnectedGradients.ZERO):
  # Given a TF function `tf_f` and its `tf_args` tuple,
  # return a pair of a function that computes both the value and the
  # gradient and appropriate arguments tuple.
  def wrapped(*tf_args):
    tf_vars = tf.nest.map_structure(tf.Variable, tf_args)
    with tf.GradientTape() as tape:
      res_tf = tf_f(*tf_vars)

    grad = tape.gradient(res_tf, tf_vars,
                         unconnected_gradients=unconnected_gradients)
    return (res_tf, grad)
  return wrapped, tf_args

def ComputeTfValueAndGrad(tf_f: Callable, tf_args: Sequence,
                          unconnected_gradients=tf.UnconnectedGradients.ZERO):
  assert isinstance(tf_args, Sequence), f"tf_args must be a tuple: {tf_args}"
  f1, args1 = TransformTfValueAndGrad(tf_f, tf_args,
                                      unconnected_gradients=unconnected_gradients)
  return f1(*args1)


# TODO(necula): clean up the test harnesses to not require these flags
@jtu.with_config(jax_numpy_rank_promotion="allow",
                 jax_numpy_dtype_promotion='standard',
                 jax_legacy_prng_key="allow",
                 jax_debug_key_reuse=False)
class JaxToTfTestCase(jtu.JaxTestCase):
  # We want most tests to use the maximum available version, from the locally
  # installed tfxla module and export.
  use_max_serialization_version = True

  def setUp(self):
    super().setUp()
    # Ensure that all TF ops are created on the proper device (TPU or GPU or CPU)
    tf_preferred_devices = (
        tf.config.list_logical_devices("TPU") +
        tf.config.list_logical_devices("GPU") +
        tf.config.list_logical_devices())
    self.tf_default_device = tf_preferred_devices[0]
    logging.info("Running jax2tf converted code on %s.", self.tf_default_device)
    # We need --config=cuda build flag for TF to see the GPUs
    self.assertEqual(jtu.device_under_test().upper(),
                     self.tf_default_device.device_type)

    # We run the tests using the maximum version supported, even though
    # the default serialization version may be held back for a while to
    # ensure compatibility
    version = config.jax_export_calling_convention_version.value
    if self.use_max_serialization_version:
      # Use the largest supported by both export and tfxla.call_module
      version = min(export.maximum_supported_calling_convention_version,
                    tfxla.call_module_maximum_supported_version())
      self.assertGreaterEqual(version,
                              export.minimum_supported_calling_convention_version)
      self.enter_context(config.jax_export_calling_convention_version(version))
    logging.info(
      "Using JAX serialization version %s (export.max_version %s, tf.XlaCallModule max version %s)",
      version,
      export.maximum_supported_calling_convention_version,
      tfxla.call_module_maximum_supported_version())

    with contextlib.ExitStack() as stack:
      stack.enter_context(tf.device(self.tf_default_device))
      self.addCleanup(stack.pop_all().close)

  def assertDtypesMatch(self, x, y, *, canonicalize_dtypes=True):
    """Compares dtypes across JAX and TF dtypes. Overrides super method."""

    def to_numpy_dtype(dt):
      return dt if isinstance(dt, np.dtype) else dt.as_numpy_dtype

    if not config.enable_x64.value and canonicalize_dtypes:
      self.assertEqual(
          dtypes.canonicalize_dtype(to_numpy_dtype(jtu._dtype(x))),
          dtypes.canonicalize_dtype(to_numpy_dtype(jtu._dtype(y))))
    else:
      self.assertEqual(
          to_numpy_dtype(jtu._dtype(x)), to_numpy_dtype(jtu._dtype(y)))

  def ConvertAndCompare(self,
                        func_jax: Callable,
                        *args,
                        enable_xla: bool = True,
                        limitations: Sequence = ()):
    """Compares jax_func(*args) with convert(jax_func)(*args).

    It compares the result of JAX, TF ("eager" mode),
    TF with tf.function ("graph" mode), and TF with
    tf.function(jit_compile=True) ("compiled" mode). In each mode,
    either we expect to encounter a known limitation, or the value should
    match the value from the JAX execution.

    Args:
      func_jax: the function to invoke (``func_jax(*args)``)
      args: the arguments.
      enable_xla: if True, allows the use of XLA ops in jax2tf.convert
        (default: True).
      limitations: the set of limitations for this harness (not yet filtered
        by mode).
    """
    # Run JAX. Should not fail, we assume that the harness has been filtered
    # already by JAX unimplemented primitives.
    result_jax = func_jax(*args)  # JAX
    result_tf = None

    func_tf = jax2tf.convert(func_jax, enable_xla=enable_xla)

    unexpected_successes: list[str] = []
    # Run the "compiled" mode first, it is most important
    for mode in ("compiled", "eager", "graph"):
      if mode == "graph" and jtu.device_under_test() == "tpu":
        continue  # The "graph" mode on TPU is the same as "compiled"
      def log_message(extra):
        return f"[{self._testMethodName}] {mode=}: {extra}"

      jax2tf_limits = tuple(filter(lambda l: l.filter(mode=mode), limitations))

      skip_tf_run = [l for l in jax2tf_limits if l.skip_tf_run]
      if skip_tf_run:
        logging.info(log_message(f"Skip TF run due to limitations {skip_tf_run}"))
        continue

      try:
        result_tf = _run_tf_function(func_tf, *args, mode=mode)
        tf_exception = None
      except Exception as e:
        tf_exception = e

      expect_tf_error = [l for l in jax2tf_limits if l.expect_tf_error]
      if tf_exception:
        if expect_tf_error:
          logging.info(log_message(
            "Found expected TF error with enabled limitations "
            f"{expect_tf_error}; TF error is {tf_exception}"))
          continue
        else:
          raise tf_exception
      else:
        if expect_tf_error:
          # It is more ergonomic to print all successful modes once
          logging.warning(log_message(
            f"Unexpected execution success with known limitations {expect_tf_error}"))
          unexpected_successes.append(f"{mode}: {expect_tf_error}")

      if (jtu.device_under_test() == "gpu" and
          "dot_general_preferred" in self._testMethodName):
        logging.info(log_message(f"Arguments are {args}, JAX result is {result_jax}\nand TF result is {result_tf}"))

      skip_comparison = [l for l in jax2tf_limits if l.skip_comparison]
      if skip_comparison:
        logging.warning(log_message(f"Skip result comparison due to {skip_comparison}"))
        continue

      max_tol = None
      max_tol_lim = None if not jax2tf_limits else jax2tf_limits[0].get_max_tolerance_limitation(jax2tf_limits)
      if max_tol_lim is not None:
        max_tol = max_tol_lim.tol
        logging.info(log_message(f"Using tol={max_tol} due to {max_tol_lim}"))

      # Convert results to np.arrays
      result_tf = tf.nest.map_structure(lambda t: t.numpy(), result_tf)

      custom_assert_lim = [l for l in jax2tf_limits if l.custom_assert]
      assert len(custom_assert_lim) <= 1, f"Expecting at most one applicable limitation with custom_assert, found {custom_assert_lim}"

      try:
        err_msg = f"TF mode {mode}."
        log_hlo_on_error = mode == "compiled" or jtu.device_under_test() == "tpu"
        if log_hlo_on_error:
          err_msg += " See the logs for JAX and TF HLO comparisons."
        if custom_assert_lim:
          logging.info(log_message(f"Running custom_assert with tol={max_tol} due to {custom_assert_lim[0]}"))
          custom_assert_lim[0].custom_assert(self, result_jax, result_tf,
                                             args=args, tol=max_tol,
                                             err_msg=err_msg)
        else:
          logging.info(log_message(f"Running default assert with tol={max_tol}"))
          self.assertAllClose(result_jax, result_tf, atol=max_tol, rtol=max_tol,
                              err_msg=err_msg)
      except AssertionError as e:
        # Print the HLO for comparison
        if not log_hlo_on_error:
          print(f"[{self._testMethodName}] Not logging HLO because the "
                f"mode was {mode}")
          raise

        logging.info("[%s] Logging HLO for exception in mode %s: %s",
                     self._testMethodName, mode, e)
        jax_lowered = jax.jit(func_jax).lower(*args)
        # We log the HLO dialect for easier comparison with TF
        logging.info("[%s] JAX NON_OPT HLO\n%s",
                     self._testMethodName,
                     jax_lowered.compiler_ir(dialect="hlo").as_hlo_text())  # type: ignore

        tf_args_signature = _make_tf_input_signature(*args)
        # If we give the signature, we cannot pass scalars
        tf_args_no_scalars = tuple(
            map(lambda a, sig: tf.convert_to_tensor(a, dtype=sig.dtype),
                args, tf_args_signature))

        tf_func_compiled = tf.function(
            func_tf,
            autograph=False,
            jit_compile=True,
            input_signature=tf_args_signature)
        tf_hlo = tf_func_compiled.experimental_get_compiler_ir(*tf_args_no_scalars)(
                    stage="hlo")
        logging.info("[%s] TF NON OPT HLO\n{%s}", self._testMethodName,
                     tf_hlo)

        backend = xla_bridge.get_backend()
        modules = backend.compile(str(jax_lowered.compiler_ir())).hlo_modules()
        jax_opt_hlo = modules[0].to_string()
        logging.info("[%s] JAX OPT HLO\n%s", self._testMethodName,
                     jax_opt_hlo)

        tf_opt_hlo = tf_func_compiled.experimental_get_compiler_ir(*tf_args_no_scalars)(
                    stage="optimized_hlo")
        logging.info("[%s] TF OPT HLO\n%s", self._testMethodName, tf_opt_hlo)

        raise

    # end "for mode"

    if unexpected_successes:
      msg = (f"[{self._testMethodName}] The following are unexpected "
             "successful modes:\n" + "\n".join(unexpected_successes))
      logging.warning(msg)
      # Uncomment the below if you want to see warnings as failures
      # self.assertEmpty(msg)
    return result_jax, result_tf

  def TransformConvertAndCompare(self, func: Callable, arg,
                                 transform: str | None):
    """Like ConvertAndCompare but first applies a transformation.

    `func` must be a function from one argument to one result. `arg` is
    the argument before the transformation.

    `transform` can be None, "jit", "jvp", "grad", "vmap", "jvp_vmap",
    "grad_vmap"
    """
    if transform is None:
      return self.ConvertAndCompare(func, arg)
    if transform == "jit":
      return self.ConvertAndCompare(jax.jit(func), arg)
    if transform == "jvp":
      t_func = lambda x, xt: jax.jvp(func, (x,), (xt,))
      return self.ConvertAndCompare(t_func, arg, np.full_like(arg, 0.1))
    if transform == "grad":
      return self.ConvertAndCompare(jax.grad(func), arg)
    if transform == "vmap":
      t_arg = np.stack([arg] * 4)
      return self.ConvertAndCompare(jax.vmap(func), t_arg)
    if transform == "jvp_vmap":
      jvp_func = lambda x, xt: jax.jvp(jax.vmap(func), (x,), (xt,))
      t_arg = np.stack([arg] * 4)
      return self.ConvertAndCompare(jvp_func, t_arg, np.full_like(t_arg, 0.1))
    if transform == "grad_vmap":
      grad_func = jax.grad(lambda x: jnp.sum(jax.vmap(func)(x)))
      t_arg = np.stack([arg] * 4)
      return self.ConvertAndCompare(grad_func, t_arg)
    assert False, transform

  def TfToHlo(self, tf_fun: Callable, *args):
    # Converts a tf.function to HLO text which we can inspect for occurrence of
    # substrings. This works whether we use native serialization or not.
    tf_function = tf.function(tf_fun, autograph=False, jit_compile=True)
    return tf_function.experimental_get_compiler_ir(*args)(
        stage="hlo", platform_name=jtu.device_under_test().upper()
    )

  def FindLargeTfConstants(self, tf_fun: Callable, *args,
                           at_least=256):
    # A hacky way to find the "large" constants that are embedded in the
    # graph. We count the number of characters in the textual representation
    # of the constant.
    f_tf_graph = tf.function(tf_fun, autograph=False).get_concrete_function(*args).graph.as_graph_def()
    if config.jax2tf_default_native_serialization.value:
      # This way of finding constants may be brittle, if the constant representation
      # contains >. It seems tobe hex-encoded, so this may be safe.
      large_consts = [m for m in re.findall(r"dense<([^>]+)>", str(f_tf_graph)) if len(m) >= at_least]
    else:
      # We cannot find the constants just with string matching because their
      # representation may contain escaped "
      large_consts = [str(n) for n in f_tf_graph.node if n.op == "Const" and len(str(n)) >= at_least]
    return large_consts

  def CheckOpMetadata(self, jax_fun, x,
                      expected: Sequence[OpMetadataGraph],
                      include_xla_op_metadata=True):
    """Checks that the tf.Graph obtained by converting `jax_fun` for argument
    `x` contains all the given OpMetadata.

    If `not include_xla_op_metadata` then disable the generation of the
    OpMetadata attributes, and check that we don't find any ops with
    metadata.
    """
    f_tf = tf.function(
        jax2tf.convert(jax_fun,
                       include_xla_op_metadata=include_xla_op_metadata),
        autograph=False,
        input_signature=[tf.TensorSpec(x.shape, x.dtype)])
    # Trace the TF function to a graph
    f_tf_concrete = f_tf.get_concrete_function(tf.convert_to_tensor(x))

    found_tf_ops = []
    def iter_nested_graph(graph: tf.Graph):
      for n in graph._nodes_by_id.values():
        try:
          op_metadata = n.get_attr("_XlaOpMetadata")
          op_metadata_proto = xla_data_pb2.OpMetadata()
          op_metadata_proto.ParseFromString(op_metadata)
          found_tf_ops.append(
              OpMetadataGraph(
                  tf_type=n.type,
                  op_name=op_metadata_proto.op_name,
                  op_type=op_metadata_proto.op_type,
                  source_file=op_metadata_proto.source_file,
                  source_line=op_metadata_proto.source_line))
        except ValueError:
          continue

        # Look for nested graphs. There probably is a better way!
        if n.type == "StatelessWhile":
          iter_nested_graph(n._body_graph)
          iter_nested_graph(n._cond_graph)
        if n.type == "StatelessCase":
          for idx in range(10):  # How can I tell how many cases there are?
            branch = getattr(n, f"_branch_graph_{idx}", None)
            if branch is None:
              break
            iter_nested_graph(branch)

    iter_nested_graph(f_tf_concrete.graph)
    try:
      if include_xla_op_metadata:
        self.assertContainsSubset(expected, found_tf_ops)
      else:
        self.assertEmpty(found_tf_ops)
    except Exception:
      print("Found nodes:\n  ", "\n   ".join([str(md) for md in found_tf_ops]))
      raise
