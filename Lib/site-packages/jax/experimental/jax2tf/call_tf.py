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
"""Allows JAX to call TensorFlow functions with support for autodiff.

**Experimental: please give feedback, and expect changes.**

This module introduces the function :func:`call_tf` that allows JAX to call
TensorFlow functions.

For examples and details, see
https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
from typing import Any

from absl import logging
import jax
from jax import dlpack
from jax import dtypes
from jax import numpy as jnp
from jax import tree_util
from jax._src import ad_util
from jax._src import core
from jax._src import effects
from jax._src import util
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src.lib.mlir.dialects import hlo
from jax.experimental.jax2tf import jax2tf as jax2tf_internal
from jax._src.interpreters import mlir
import numpy as np
import tensorflow as tf


map = util.safe_map
zip = util.safe_zip

TfConcreteFunction = Any
TfVal = jax2tf_internal.TfVal

# The platforms for which to use DLPack to avoid copying (only works on GPU
# and CPU at the moment, and only for Array). For CPU we don't need
# DLPack, if we are careful.
_DLPACK_PLATFORMS = ("gpu",)

class UnspecifiedOutputShapeDtype:
  pass

def call_tf(
    callable_tf: Callable,
    has_side_effects=True,
    ordered=False,
    output_shape_dtype=UnspecifiedOutputShapeDtype(),
    call_tf_graph=False,
) -> Callable:
  """Calls a TensorFlow function from JAX, with support for reverse autodiff.

  The ``callable_tf`` will be called with TensorFlow-compatible arguments (
  numpy.ndarray, ``tf.Tensor`` or ``tf.Variable``) or pytrees thereof. The
  function must return the same type of results.

  If ``call_tf`` appears in a JAX staging context (:func:`jax.jit`,
  or :func:`jax.pmap`, or a control-flow primitive) then
  ``callable_tf`` will be compiled with ``tf.function(callable_tf,
  jit_compile=True)``
  and the resulting XLA computation will be embedded in JAX's XLA computation.

  If ``call_tf`` appears outside a JAX staging context, it will be called inline
  using TensorFlow eager mode.

  The ``call_tf`` supports JAX's reverse-mode autodiff, in which case the
  ``callable_tf`` will be differentiated using ``tf.GradientTape``. This means
  that the gradient will be TensorFlow-accurate, e.g., will respect the
  custom gradients that may be defined for the code in ``callable_tf``.

  For an example and more details see the
  `README
  <https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#calling-tensorflow-functions-from-jax>`_.

  Args:
    callable_tf: a TensorFlow Callable that can take a pytree of TensorFlow
      arguments.
    has_side_effects: if True then it ensures that instances of this primitive
      are not removed or replicated by JAX optimizations such as dead-code
      elimination.
    ordered: If true, calls are modeled as having ordered effects.
    output_shape_dtype: An optional declaration of the expected shape and dtype
      of the result of the called TensorFlow function. If given it will be used
      during JAX tracing to form the abstract values of the results of the
      `call_tf`. If not given then we form a `tf.Graph` for the called
      TensorFlow function and we use the TensorFlow-inferred shapes and types.
      Must be a pytree matching the structure of the nested structure returned
      from the TensorFlow function, containing objects with `.shape` and
      `.dtype` attributes, e.g., `jax.ShapeDtypeStruct` or `jax.Array`.
    call_tf_graph: EXPERIMENTAL, DO NOT USE. We may change the name in the
      future.

  Returns: a JAX callable that can be invoked with JAX pytree arguments, in
    op-by-op mode or in a staged context. This callable can be used with JAX's
    reverse-mode autodiff (:func:`jax.grad`).
  """
  @jax.custom_vjp
  def make_call(*args_jax):
    """We wrap it all in `make_call` so that we can attach custom VJP."""

    args_flat_jax, args_treedef = tree_util.tree_flatten(args_jax)
    # Canonicalize the arguments; e.g., makes them x32 if JAX is in 32-bit mode
    def canonical_arg(v):
      v = v if getattr(v, "dtype", None) else np.asarray(v)
      dtype = dtypes.canonicalize_dtype(v.dtype)
      if dtype != v.dtype:
        v = v.astype(dtype)
      return v

    args_flat_jax = tuple(map(canonical_arg, args_flat_jax))
    def make_tensorspec(a_jax):
      a_tf_dtype = jax2tf_internal._to_tf_dtype(a_jax.dtype)
      a_tf_shape = [d if core.is_constant_dim(d) else None for d in a_jax.shape]
      return tf.TensorSpec(a_tf_shape, a_tf_dtype)
    args_flat_sig_tf = tuple(map(make_tensorspec, args_flat_jax))

    if not isinstance(output_shape_dtype, UnspecifiedOutputShapeDtype):
      output_shape_dtype_flat, output_shape_dtype_tree = tree_util.tree_flatten(output_shape_dtype)
      output_avals = tuple(core.ShapedArray(st.shape, st.dtype) for st in output_shape_dtype_flat)
    else:
      output_avals, output_shape_dtype_tree = None, None

    res_treedef = None  # We'll store here the result treedef
    res_tf_flat = None  # For error reporting
    # The function below will be called at least once, either in eager
    # mode during jax2tf_call_tf or in graph mode during _get_concrete_function_tf()
    def callable_flat_tf(*args_tf_flat: TfVal) -> Sequence[TfVal]:
      args_tf = args_treedef.unflatten(args_tf_flat)
      res_tf = callable_tf(*args_tf)

      # b/279454591: When `callable_tf` is a tf function with zero outputs, it
      # returns a `StatefulPartitionedCall` (if the function is stateful) or
      # `PartitionedCall` (if the function is stateless) op instead of
      # tf.Tensors. We work around this issue by replacing the output `res_tf`
      # with an empty list.

      if isinstance(res_tf, tf.Operation):
        assert (
            res_tf.type == "StatefulPartitionedCall"
            or res_tf.type == "PartitionedCall"
        )
        t_out = res_tf.get_attr("Tout")
        # t_out should be an empty list.
        assert not t_out, (
            "The TF function returned an unexpected result, please check its"
            f" function body. res_tf = {res_tf}"
        )
        res_tf = t_out

      nonlocal res_treedef, res_tf_flat
      res_tf_flat, res_treedef_now = tree_util.tree_flatten(res_tf)
      assert res_treedef is None or res_treedef == res_treedef_now, (
          f"Subsequent calls had different results. Previous {res_treedef} and now {res_treedef_now}")
      res_treedef = res_treedef_now
      if output_avals is not None:
        if res_treedef != output_shape_dtype_tree:
          raise ValueError(
              "The pytree of the TensorFlow function results does not match the "
              "pytree of the declared output_shape_dtype:\n"
              f"results pytree: {res_treedef}\noutput_shape_dtype tree: {output_shape_dtype_tree}")
        assert len(output_avals) == len(res_tf_flat)

      checked_res_tf_flat = [
          check_tf_result(i, r_tf, r_aval)
          for i, (r_tf, r_aval) in enumerate(
              zip(res_tf_flat,
                  (output_avals
                   if output_avals is not None
                   else (None,) * len(res_tf_flat))))]
      return checked_res_tf_flat

    # Prepare a tf.function ahead of time, to cache the concrete functions. This
    # won't be used in op-by-op execution mode.
    function_flat_tf = tf.function(
        callable_flat_tf, autograph=False, jit_compile=not call_tf_graph)

    res_jax_flat = call_tf_p.bind(
        *args_flat_jax,
        # Carry the actual function such that op-by-op call can call in TF eager mode.
        callable_flat_tf=callable_flat_tf,
        function_flat_tf=function_flat_tf,
        args_flat_sig_tf=args_flat_sig_tf,
        output_avals=output_avals,
        has_side_effects=has_side_effects,
        ordered=ordered,
        call_tf_graph=call_tf_graph,
    )

    # We must have called callable_flat_tf by nοw
    assert res_treedef is not None
    return res_treedef.unflatten(res_jax_flat)

  # Define the fwd and bwd custom_vjp functions
  def make_call_vjp_fwd(*args_jax):
    # Return the primal arguments as the residual
    return make_call(*args_jax), args_jax

  def make_call_vjp_bwd(residual_jax, ct_res_jax):
    args_jax = residual_jax  # residual is the primal argument

    def tf_vjp_fun(args_tf, ct_res_tf):
      """Invoke TF gradient."""

      # TF does not like us to watch non-float vars or Nones.
      def replace_non_float_or_none(arg_tf):
        if arg_tf is not None and (
            arg_tf.dtype.is_floating or arg_tf.dtype.is_complex
        ):
          return arg_tf
        else:
          # When watched, this will be ignored. When used in results it will
          # result in a floating 0. gradient, which JAX will ignore (and
          # replace it with a float0)
          return tf.zeros((), dtype=tf.float32)

      watched_args_tf = tf.nest.map_structure(
          replace_non_float_or_none, args_tf
      )
      with tf.GradientTape(persistent=True) as tape:
        tape.watch(watched_args_tf)
        res = callable_tf(*args_tf)

      tf.nest.assert_same_structure(res, ct_res_tf)
      dres_darg = tape.gradient(
          tf.nest.map_structure(replace_non_float_or_none, res),
          sources=watched_args_tf,
          output_gradients=ct_res_tf,
          unconnected_gradients=tf.UnconnectedGradients.ZERO,
      )

      dres_darg = tree_util.tree_map(
          lambda x: x if x is None else tf.convert_to_tensor(x),
          dres_darg,
      )

      # callable_tf may mutate (the structure of) args_tf, thus we check against
      # watched_args_tf which should be structurally the same as the original
      # args_tf.
      tf.nest.assert_same_structure(dres_darg, watched_args_tf)
      return dres_darg

    # Use call_tf to call the VJP function
    ct_args_jax = call_tf(tf_vjp_fun)(args_jax, ct_res_jax)
    # We must make the float0s that JAX expects
    def fix_float0(arg_jax, ct_arg_jax):
      if arg_jax is None:
        return None
      arg_dtype = dtypes.result_type(arg_jax)  # May be scalar
      ct_arg_dtype = core.primal_dtype_to_tangent_dtype(arg_dtype)
      if ct_arg_dtype != ct_arg_jax.dtype:
        return ad_util.zeros_like_aval(core.ShapedArray(np.shape(arg_jax),
                                                        ct_arg_dtype))
      return ct_arg_jax

    ct_args_jax_fixed = tree_util.tree_map(fix_float0, args_jax, ct_args_jax,
                                           is_leaf=lambda x: x is None)
    return ct_args_jax_fixed

  make_call.defvjp(make_call_vjp_fwd, make_call_vjp_bwd)
  return util.wraps(callable_tf)(make_call)


def check_tf_result(idx: int, r_tf: TfVal, r_aval: core.ShapedArray | None) -> TfVal:
  # Check that the TF function returns values of expected types. This
  # improves error reporting, preventing hard-to-diagnose errors downstream
  try:
    jax2tf_internal._tfval_to_tensor_jax_dtype(r_tf)
  except Exception as e:
    msg = ("The called TF function returns a result that is not "
           f"convertible to JAX: {r_tf}.")
    raise ValueError(msg) from e

  if r_aval is None:
    return r_tf
  # We convert to TF type, and canonicalize to 32-bit if necessary
  r_aval_dtype_tf = jax2tf_internal._to_tf_dtype(r_aval.dtype)
  # Checking shapes is trickier in presence of dynamic shapes. I wish we could
  # check at runtime that the returned shape matches the declared shape. I wish
  # that tf.ensure_shape did this, but it can only take shapes that contain None
  # not computed shapes. However, in eager mode we should be able to resolve
  # the declared shapes to constants and we get better checking.
  if tf.executing_eagerly():
    r_aval_shape_tf = jax2tf_internal._eval_shape(r_aval.shape)
  else:
    r_aval_shape_tf = jax2tf_internal._aval_to_tf_shape(r_aval)
  # We do as much checking as we can here, instead of relying on tf.ensure_shape
  # because the latter gives different errors in eager vs. compiled mode.
  # TODO(b/279454591): This strange error is from TF. Eager function suppose
  # return tf Val with concrete shape but not.  Here we change exception to warn
  # and bypass it. This case need revisit on TF side.
  try:
    _ = len(r_tf.shape)
  except ValueError as e:
    msg = (
        "The shape check test cannot be performed because the shape of the"
        "`r_tf` tensor cannot be obtained."
        f"r_tf = {r_tf}, r_aval = {r_aval}"
    )
    msg += str(e)
    logging.warning(msg)
    return r_tf
  if (r_tf.dtype != r_aval_dtype_tf or
      len(r_tf.shape) != len(r_aval_shape_tf) or
      any(r_aval_d is not None and r_tf_d is not None and r_aval_d != r_tf_d
          for r_tf_d, r_aval_d in zip(r_tf.shape, r_aval_shape_tf))):
    msg = ("The shapes or dtypes returned by the TensorFlow function "
           "do not match the declared output_shape_dtype:\n"
           f"Result[{idx}] is {r_tf.dtype}[{r_tf.shape}] vs. expected {r_aval_dtype_tf}[{r_aval_shape_tf}]")
    raise ValueError(msg)
  # At this point tf.ensure_shape does not do much, it should never throw an
  # error, albeit it may refine the shape a bit.
  return tf.ensure_shape(r_tf, r_aval_shape_tf)


call_tf_p = core.Primitive("call_tf")
call_tf_p.multiple_results = True

# The impl will be used in op-by-op mode and calls callable_tf in TF eager mode.
def _call_tf_impl(*args_jax_flat, callable_flat_tf, **_):
  # On GPU we use dlpack to avoid copies of data to the host.
  def _arg_jax_to_tf(arg_jax):
    if (isinstance(arg_jax, jax.Array) and
        list(arg_jax.devices())[0].platform in _DLPACK_PLATFORMS and
        arg_jax.dtype.type in dlpack.SUPPORTED_DTYPES):
      arg_dlpack = jax.dlpack.to_dlpack(arg_jax)
      return tf.experimental.dlpack.from_dlpack(arg_dlpack)
    # The following avoids copies to the host on CPU, always for Array
    # and even for ndarray if they are sufficiently aligned.
    # TODO(necula): on TPU this copies to the host!
    if getattr(arg_jax, 'dtype', None) == dtypes.float0:
      return tf.zeros(shape=arg_jax.shape,
                      dtype=jax2tf_internal._tf_np_dtype_for_float0)
    return tf.constant(np.asarray(arg_jax))

  args_tf_flat = tuple(map(_arg_jax_to_tf, args_jax_flat))
  with jax2tf_internal.inside_call_tf():
    # Call in TF eager mode
    res_tf_flat = callable_flat_tf(*args_tf_flat)

  def _res_tf_to_jax(res_tf: TfVal):
    res_tf, jax_dtype = jax2tf_internal._tfval_to_tensor_jax_dtype(res_tf)
    if isinstance(res_tf, tf.Tensor) and jax_dtype.type in dlpack.SUPPORTED_DTYPES:
      res_tf_platform = tf.DeviceSpec.from_string(res_tf.backing_device).device_type
      res_jax_platform = res_tf_platform.lower()
      if res_jax_platform in _DLPACK_PLATFORMS:
        res_dlpack = tf.experimental.dlpack.to_dlpack(res_tf)
        return jax.dlpack.from_dlpack(res_dlpack)

    # When working with a bfloat16 scalar tf.Tensor,np.asarray() can fail.
    # To handle this special case, we create a numpy copy.
    if res_tf.shape == tf.TensorShape([]) and res_tf.dtype == tf.bfloat16:
      return jax.device_put(jnp.array(res_tf.numpy()))
    else:
      return jax.device_put(np.asarray(res_tf))

  return list(map(_res_tf_to_jax, res_tf_flat))


call_tf_p.def_impl(_call_tf_impl)

@functools.lru_cache(maxsize=128)
def _get_concrete_function_tf(function_flat_tf, args_flat_sig_tf):  # -> tf.ConcreteFunction
  with jax2tf_internal.inside_call_tf():
    return function_flat_tf.get_concrete_function(*args_flat_sig_tf)


# Mark the effectful instances of call_tf
@dataclasses.dataclass(frozen=True)
class CallTfEffect(effects.Effect):
  __str__ = lambda _: "CallTfEffect"

call_tf_effect = CallTfEffect()

effects.lowerable_effects.add_type(CallTfEffect)
effects.control_flow_allowed_effects.add_type(CallTfEffect)
effects.remat_allowed_effects.add_type(CallTfEffect)
effects.custom_derivatives_allowed_effects.add_type(CallTfEffect)


class CallTfOrderedEffect(effects.Effect):
  __str__ = lambda _: "CallTfOrderedEffect"


call_tf_ordered_effect = CallTfOrderedEffect()

effects.lowerable_effects.add_type(CallTfOrderedEffect)
effects.control_flow_allowed_effects.add_type(CallTfOrderedEffect)
effects.remat_allowed_effects.add_type(CallTfOrderedEffect)
effects.custom_derivatives_allowed_effects.add_type(CallTfOrderedEffect)
effects.ordered_effects.add_type(CallTfOrderedEffect)
effects.shardable_ordered_effects.add_type(CallTfOrderedEffect)


def _call_tf_abstract_eval(
    *args_flat_avals,
    function_flat_tf,
    args_flat_sig_tf,
    has_side_effects,
    ordered,
    output_avals,
    call_tf_graph,
    **__,
):
  # Called only when we form a Jaxpr, i.e., under jit, scan, etc.
  effects = set()
  if ordered:
    effects.add(call_tf_ordered_effect)
  elif has_side_effects:
    effects.add(call_tf_effect)

  # If no output_avals is given, then we ask TF to infer the output shapes.
  # We call this even if output_avals is given because it will ensure that
  # callable_flat_tf is called. Since _get_concrete_function_tf is cached
  # there is a small cost of calling it more often than needed.
  concrete_function_flat_tf = _get_concrete_function_tf(function_flat_tf,
                                                        args_flat_sig_tf)

  # In the case that the tf.function has no return value
  if len(concrete_function_flat_tf.outputs) == 0:
    return (), effects

  if output_avals is not None:
    return output_avals, effects

  def is_fully_known_shape(s):
    return s.rank is not None and all(d is not None for d in s)

  if all(is_fully_known_shape(s)
        for s in concrete_function_flat_tf.output_shapes):
    avals_from_tf = tuple(
        # We convert to JAX type, and canonicalize to 32-bit if necessary
        core.ShapedArray(shape, jax2tf_internal._to_jax_dtype(dtype))
        for dtype, shape in zip(concrete_function_flat_tf.output_dtypes,
                                concrete_function_flat_tf.output_shapes))
    return avals_from_tf, effects

  msg = ("call_tf cannot call functions whose output has dynamic shape. "
    f"Found output shapes: {concrete_function_flat_tf.output_shapes}. "
    "Consider using the `output_shape_dtype` argument to call_tf. "
    "\nSee https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf"
      " for a discussion.")
  raise ValueError(msg)


call_tf_p.def_effectful_abstract_eval(_call_tf_abstract_eval)


def _call_tf_lowering(
    ctx: mlir.LoweringRuleContext,
    *args_op,
    platform,
    function_flat_tf,
    args_flat_sig_tf,
    has_side_effects,
    ordered,
    call_tf_graph,
    output_avals,
    **_,
):
  # We use the same TF lowering device as for the embedding JAX computation.
  # One example when this is needed is when the code refers to variables on one
  # device. Or, for sharding annotations (only supported on TPU).

  if platform in ["cpu", "tpu"]:
    tf_platform = platform.upper()
  elif platform == "cuda":
    tf_platform = "GPU"
  else:
    raise ValueError("platform {platform} not supported")

  concrete_function_flat_tf = _get_concrete_function_tf(function_flat_tf, args_flat_sig_tf)

  captured_inputs = []
  if concrete_function_flat_tf.captured_inputs:
    # The function uses either captured variables or tensors.
    msg = (
        "call_tf works best with a TensorFlow function that does not capture "
        "variables or tensors from the context. "
        "See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf for a discussion. "
        f"The following captures were found {concrete_function_flat_tf.captured_inputs}")
    logging.warning(msg)
    for inp in concrete_function_flat_tf.captured_inputs:
      if inp.dtype == tf.resource:  # A variable; lookup by handle
        inp_vars = [v for v in concrete_function_flat_tf.variables if inp is v.handle]
        assert len(inp_vars) == 1, f"Found {inp_vars}"
        captured_inputs.append(inp_vars[0])
      else:
        captured_inputs.append(inp)

  # The following use case happens when we call_tf a restored saved model that
  # includes parameters (hence functions closing over tf.Variable), and then
  # we jax2tf.convert it with native serialization, under tf.function (or
  # for saving to saved model). The `np.asarray(inp)` fails because it thinks
  # it is in TF graph mode. The `tf.init_scope()` lifts out of function-building
  # graph scopes, and allows us to read the values of the variables
  with tf.init_scope():
    captured_ops = tuple(
        mlir.ir_constant(np.asarray(inp))
        for inp in captured_inputs
    )

  if call_tf_graph:
    with jax2tf_internal.inside_call_tf():
      return emit_tf_embedded_graph_custom_call(
          ctx,
          concrete_function_flat_tf,
          tuple(args_op) + captured_ops,
          has_side_effects,
          ordered,
          output_avals,
      )

  def convert_to_spec(x):
    if isinstance(x, tf.TensorSpec):
      return x
    else:
      return tf.TensorSpec.from_tensor(x)

  args_tf_flat = [convert_to_spec(a) for a in args_flat_sig_tf]

  with jax2tf_internal.inside_call_tf():
    try:
      func_tf_hlo = function_flat_tf.experimental_get_compiler_ir(
          *args_tf_flat
      )(stage="hlo_serialized", platform_name=tf_platform)
    except Exception as e:
      msg = ("Error compiling TensorFlow function (see below for the caught exception)." +
             "\ncall_tf can used " +
              "in a staged context (under jax.jit, lax.scan, etc.) only with " +
              "compilable functions with static output shapes.\n" +
              "See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf for a discussion." +
             "\n\nCaught TensorFlow exception: " + str(e))
      raise ValueError(msg) from e

  xla_comp = xla_client.XlaComputation(func_tf_hlo)

  # Canonicalize the results; e.g., makes them x32 if JAX is in 32-bit mode
  def canonical_res_aval(res_shape: xla_client.Shape) -> core.ShapedArray:
    if not res_shape.is_static():
      msg = ("Compiled TensorFlow function has dynamic output shape " +
             f"{res_shape}. call_tf can used " +
             "in a staged context (under jax.jit, lax.scan, etc.) only with " +
             "compilable functions with static output shapes. " +
             "See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#limitations-of-call_tf for a discussion.")
      raise ValueError(msg)

    res_dtype = res_shape.numpy_dtype()
    jax_res_dtype = dtypes.canonicalize_dtype(res_dtype)
    return core.ShapedArray(res_shape.dimensions(), jax_res_dtype)

  result_shape = xla_comp.program_shape().result_shape()
  if not result_shape.is_tuple():
    # TF does not wrap singletons as tuples, but JAX expects tuples because
    # call_tf is a multiple_results primitive.
    result_shapes = (result_shape,)
  else:
    result_shapes = result_shape.tuple_shapes()  # type: ignore

  result_avals = tuple(map(canonical_res_aval, result_shapes))

  submodule = mlir.xla_computation_to_mlir_module(xla_comp)
  symtab = ir.SymbolTable(submodule.operation)
  callee_result_types = symtab["main"].type.results
  fn = mlir.merge_mlir_modules(ctx.module_context.module,
                               f"call_tf_{function_flat_tf.name}",
                               submodule,
                               dst_symtab=ctx.module_context.symbol_table)
  call = func_dialect.CallOp(callee_result_types,
                             ir.FlatSymbolRefAttr.get(fn),
                             tuple(args_op) + captured_ops)
  if result_shape.is_tuple():
    flat_results = [hlo.get_tuple_element(call, mlir.i32_attr(i))
                    for i in range(len(result_shapes))]
  else:
    flat_results = call.results

  if ordered:
    raise NotImplementedError(
        "ordered=True is not supported in the jitted context without"
        " `call_tf_graph=True`"
    )

  outputs = []
  for op, res_aval, res_shape in zip(flat_results, result_avals,
                                     result_shapes):
    if res_aval.dtype != res_shape.numpy_dtype():
      op = hlo.ConvertOp(mlir.aval_to_ir_type(res_aval), op).result
    outputs.append(op)
  return outputs


def _register_call_lowering(platform):
  mlir.register_lowering(call_tf_p, functools.partial(_call_tf_lowering,
                                                      platform=platform),
                         platform=platform)
for platform in ("cpu", "cuda", "tpu"):
  _register_call_lowering(platform)

# Support the call_tf under jax2tf.convert in eager mode
def _jax2tf_call_tf(*args: TfVal,
                    callable_flat_tf: Callable,
                    **_) -> TfVal:
  with jax2tf_internal.inside_call_tf():
    res_tf_flat = callable_flat_tf(*args)
  return res_tf_flat

jax2tf_internal.tf_impl[call_tf_p] = _jax2tf_call_tf


def emit_tf_embedded_graph_custom_call(
    ctx: mlir.LoweringRuleContext,
    concrete_function_flat_tf,
    operands: Sequence[ir.Value],
    has_side_effects,
    ordered,
    output_avals,
):
  """Emits a custom call referencing a tf.Graph embedding of the TF function.

  All call_tf called function information is stored in tf.metadata.
  This includes:
  (1) The called function name: This name will be used by the runtime to execute
  the callback.
  (2) The called function index in the XLACallModule `function_list` attribute.
  """
  call_tf_concrete_function_list = jax2tf_internal.get_thread_local_state_call_tf_concrete_function_list()
  if call_tf_concrete_function_list is None:
    raise ValueError(
        "call_tf_graph=True only support exporting by jax2tf.convert currently."
    )
  # TODO(necula): It is dangerous to modify global state when lowering because
  # there are a number of lowering caches that only cache the StableHLO.
  # See call_tf_test.py:test_multi_platform_call_tf_graph.
  called_index = add_to_call_tf_concrete_function_list(
      concrete_function_flat_tf, call_tf_concrete_function_list)
  tf_backend_config = {
      "has_token_input_output": ir.BoolAttr.get(ordered),
      "called_index": mlir.i64_attr(called_index),
  }
  result_avals = ctx.avals_out if ctx.avals_out is not None else ()

  operands = list(operands)
  result_types = list(
      mlir.flatten_ir_types([mlir.aval_to_ir_type(aval) for aval in result_avals])
  )
  if ordered:
    operands.insert(0, ctx.tokens_in.get(call_tf_ordered_effect))
    result_types.insert(0, mlir.token_type())

  custom_call = hlo.CustomCallOp(
      result_types,
      operands,
      call_target_name=ir.StringAttr.get("tf.call_tf_function"),
      has_side_effect=ir.BoolAttr.get(has_side_effects),
      api_version=mlir.i32_attr(2),
      called_computations=ir.ArrayAttr.get([]),
      backend_config=ir.StringAttr.get(""),
  )
  # Store TF metadata in unregistered attribute
  custom_call.attributes["tf.backend_config"] = ir.DictAttr.get(
      tf_backend_config
  )

  results = list(custom_call.results)
  if ordered:
    token = results.pop(0)
    ctx.set_tokens_out(mlir.TokenSet({call_tf_ordered_effect: token}))

  return results


def add_to_call_tf_concrete_function_list(concrete_tf_fn: Any, call_tf_concrete_function_list: list[Any]) -> int:
  try:
    called_index = call_tf_concrete_function_list.index(concrete_tf_fn)
  except ValueError:
    called_index = len(call_tf_concrete_function_list)
    call_tf_concrete_function_list.append(concrete_tf_fn)
  return called_index
