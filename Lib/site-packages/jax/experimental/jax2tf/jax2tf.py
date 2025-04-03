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
"""Provides JAX and TensorFlow interoperation APIs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import partial
import contextlib
import math
import operator
import os
import re
import threading
from typing import Any, Union
import warnings

from absl import logging
import numpy as np

import jax
from jax import lax
from jax import custom_derivatives
from jax import random
from jax import numpy as jnp
from jax import tree_util
from jax import sharding
from jax import export
from jax.experimental.jax2tf import impl_no_xla

from jax._src import ad_checkpoint
from jax._src import ad_util
from jax._src import api
from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import linear_util as lu
from jax._src import op_shardings
from jax._src import sharding_impls
from jax._src import mesh
from jax._src import pjit
from jax._src import prng
from jax._src import random as random_internal
from jax._src import source_info_util
from jax._src import util
from jax._src import shard_alike
from jax._src.export import _export
from jax._src.export import shape_poly
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src.lax import control_flow as lax_control_flow
from jax._src.lax import lax as lax_internal
from jax._src.lax import linalg as lax_linalg
from jax._src.lax import slicing as lax_slicing
from jax._src.lax import windowed_reductions as lax_windowed_reductions
from jax._src.lib import xla_client
from jax._src.numpy.ufuncs import logaddexp

import tensorflow as tf

# These don't have public equivalents.
# pylint: disable=g-direct-tensorflow-import
from tensorflow.compiler.tf2xla.python import xla as tfxla
from tensorflow.compiler.xla import xla_data_pb2
from tensorflow.core.framework import attr_value_pb2
try:
  from tensorflow.python.compiler.xla.experimental import xla_sharding
except ModuleNotFoundError:
  # This can be removed when TF 2.10 support is no longer needed.
  from tensorflow.compiler.xla.experimental.xla_sharding import xla_sharding
from tensorflow.python.framework import ops as tf_ops
from tensorflow.python.eager import context as tf_context
# pylint: enable=g-direct-tensorflow-import

NameStack = source_info_util.NameStack
PolyShape = shape_poly.PolyShape  # TODO: deprecate
DType = Any

DisabledSafetyCheck = export.DisabledSafetyCheck

# A temporary internal flag, to enable the wrapping of jax.jit functions
# with tf.function(jit_compile=True). See #7389. This change has triggered a
# number of failures in TF. We keep this until we are confident that it does
# not create problems.
# TODO(b/207464757): figure out why this change breaks test
_WRAP_JAX_JIT_WITH_TF_FUNCTION = False

# The scope name need to be a valid TensorFlow name. See
# https://github.com/tensorflow/tensorflow/blob/r2.3/tensorflow/core/framework/node_def_util.cc#L731
_VALID_SCOPE_REGEX = re.compile("^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$")
_INVALID_SCOPE_CHAR = re.compile("[^A-Za-z0-9_.\\/-]")

map = util.safe_map
zip = util.safe_zip


def _sanitize_scope_name(name):
  scope_name = _INVALID_SCOPE_CHAR.sub("_", name)
  if not _VALID_SCOPE_REGEX.match(scope_name):
    scope_name = f".{scope_name}"
  return scope_name


# TODO(b/353437394): Deprecate support for `enable_xla=False`.
# Line below is different externally and internally.
allow_enable_xla_false = lambda: True

# TODO(b/353437398): Deprecate support for `native_serialization=False`.
# Line below is different externally and internally.
allow_native_serialization_false = lambda: True

# A value suitable in a TF tracing context: tf.Tensor, tf.Variable,
# or Python scalar or numpy.ndarray. (A tf.EagerTensor is a tf.Tensor.)
TfVal = Any
PrecisionType = int  # Enum xla_data.PrecisionConfig.Precision

def _is_tfval(v: TfVal) -> bool:
  if isinstance(v, (tf.Tensor, tf.Variable)):
    return True
  try:
    # Include all convertible types, even if not supported on accelerators.
    with tf.device("CPU"):
      tf.constant(v)
    return True
  except:
    return False

class _DefaultNativeSerialization:
  pass
DEFAULT_NATIVE_SERIALIZATION = _DefaultNativeSerialization()

# The implementation rules for primitives. The rule will be called with the
# arguments (TfVal) and must return TfVal (or a sequence thereof,
# if primitive.multiple_results). The exception are primarily the
# control-flow primitives.
tf_impl: dict[core.Primitive, Callable[..., Any]] = {}

# Some primitive implementation rules need the abstract values of arguments
# and the results. This is the case for the primitives implemented using
# _convert_jax_impl and those that need to adjust the shape of the outputs
# due to missing TF shape inference rules for TFXLA ops. The rules for these
# primitives should be added to `tf_impl_with_avals`.
# The abstract value are passed to the implementation as two special kwargs
# `_in_avals` (a tuple of core.ShapedArray) and `_out_aval` (a
# core.ShapedArray, or a tuple thereof when primitive.multiple_results).
tf_impl_with_avals: dict[core.Primitive, Callable[..., Any]] = {}

# XLA is not linked in all environments when converting a primitive. If this is
# the case, we first search for implementation rules for primitives in the
# following map. These implementations are workarounds, making use of TF ops
# that do work when XLA is not linked in.
tf_impl_no_xla = impl_no_xla.tf_impl_no_xla

# In order to ensure that JAX picks up the proper user-frame for source
# locations we will register the TensorFlow source path as an internal
# path with source_info_util. The typical stack when a JAX primitive
# conversion happens is:
#    jax2tf.process_primitive  (top of stack)
#    jax tracing machinery ...
#    tf.custom_gradient machinery ...
#    jax2tf.converted_fun
#    tf function machinery ...
#    user code invokes the converted function on TF tensors
#
# We need to skip over not only JAX internal frames, but TF internal frames
# also.
# We register the TensorFlow source path lazily
_has_registered_tf_source_path = False

class _ThreadLocalState(threading.local):
  def __init__(self):
    # XLA is not linked in all environments; when converting a primitive, if this
    # variable is disabled, we try harder to use only standard TF ops if they are
    # applicable to the concrete use case; if the resulting conversion path ends up
    # requiring a TFXLA operation, an exception is thrown instead.
    self.enable_xla = True

    # Keep track if we are inside a call_tf. In that context we disable the
    # safety check that we are not inside JAX transformations.
    self.inside_call_tf = False

    # Maps dimension variables to TF expressions, for non-native lowering
    self.shape_env: Sequence[tuple[str, TfVal]] = ()

    # Whether to actually include XLA op metadata in the generated TF ops
    # TODO(b/189306134): implement support for XLA metadata
    self.include_xla_op_metadata = False

    # A cache for the tf.convert_to_tensor for constants. We try to preserve
    # sharing for constants, to enable tf.Graph to take advantage of it.
    # See https://github.com/jax-ml/jax/issues/7992.
    self.constant_cache = None  # None means that we don't use a cache. We
    # may be outside a conversion scope.

    # A cache for the outside tf name_scope when the converted
    # function is running. We will add this as the prefix to the generated tf op
    # name. For example, the tf op name will be like
    # "{tf_outer_name_scope}/JAX_NAME_STACKS"
    self.tf_outer_name_scope = ""

    # A dict collecting all tf concrete_functions called by stablehlo.custom_call
    # This is used only by native serialization (unlike all the other
    # thread-local state).
    self.call_tf_concrete_function_list: list[Any] | None = None

_thread_local_state = _ThreadLocalState()

def _get_current_name_stack() -> NameStack | str:
  return source_info_util.current_name_stack()

@contextlib.contextmanager
def inside_call_tf():
  # Set the inside_call_tf flag for a context.
  prev = _thread_local_state.inside_call_tf
  _thread_local_state.inside_call_tf = True
  try:
    yield
  finally:
    _thread_local_state.inside_call_tf = prev


def get_thread_local_state_call_tf_concrete_function_list() -> (
    list[Any] | None
):
  return _thread_local_state.call_tf_concrete_function_list


@partial(api_util.api_hook, tag="jax2tf_convert")
def convert(fun_jax: Callable,
            *,
            polymorphic_shapes: str | None = None,
            polymorphic_constraints: Sequence[str] = (),
            with_gradient: bool = True,
            enable_xla: bool = True,
            native_serialization: bool | _DefaultNativeSerialization = DEFAULT_NATIVE_SERIALIZATION,
            native_serialization_platforms: Sequence[str] | None = None,
            native_serialization_disabled_checks: Sequence[DisabledSafetyCheck] = (),
            ) -> Callable:
  """Allows calling a JAX function from a TensorFlow program.

  See
  [README](https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md)
  for more details about usage and common problems.

  Args:
    fun_jax: target JAX function to be called. Its arguments and return value
      should be JAX arrays, or nested standard Python containers
      (tuple/list/dict) thereof (pytrees).
    polymorphic_shapes: Specifies input shapes to be treated polymorphically
      during lowering.

      .. warning:: The shape-polymorphic lowering is an experimental feature.
        It is meant to be sound, but it is known to reject some JAX programs
        that are shape polymorphic. The details of this feature can change.

      It should be `None` (all arguments are monomorphic), a single PolyShape
      or string (applies to all arguments), or a tuple/list of the same length
      as the function arguments. For each argument the shape specification
      should be `None` (monomorphic argument), or a Python object with the
      same pytree structure as the argument.
      See [how optional parameters are matched to
      arguments](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees).

      A shape specification for an array argument should be an object
      `PolyShape(dim0, dim1, ..., dimn)`
      where each `dim` is a dimension specification: a positive integer denoting
      a monomorphic dimension of the given size, or a string denoting a
      dimension variable assumed to range over non-zero dimension sizes, or
      the special placeholder string "_" denoting a monomorphic dimension
      whose size is given by the actual argument. As a shortcut, an Ellipsis
      suffix in the list of dimension specifications stands for a list of "_"
      placeholders.

      For convenience, a shape specification can also be given as a string
      representation, e.g.: "batch, ...", "batch, height, width, _", possibly
      with surrounding parentheses: "(batch, ...)".

      The lowering fails if it cannot ensure that the it would produce the same
      sequence of TF ops for any non-zero values of the dimension variables.

      polymorphic_shapes are only supported for positional arguments; shape
      polymorphism is not supported for keyword arguments.

      See [the README](https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#shape-polymorphic-conversion)
      for more details.

    polymorphic_constraints: a sequence of constraints on symbolic dimension
      expressions, of the form `e1 >= e2` or `e1 <= e2`.
      See more details at https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#user-specified-symbolic-constraints.
    with_gradient: if set (default), add a tf.custom_gradient to the lowered
      function, by converting the ``jax.vjp(fun)``. This means that reverse-mode
      TensorFlow AD is supported for the output TensorFlow function, and the
      value of the gradient will be JAX-accurate.
    enable_xla: if set (default), use the simplest conversion
      and use XLA TF ops when necessary. These ops are known to create issues
      for the TFLite and TFjs converters. For those cases, unset this parameter
      so the lowering tries harder to use non-XLA TF ops to lower the
      function and aborts if this is not possible. Cannot be set to `False`
      when using `native_serialization`.
      Starting with JAX 0.4.31 support for `enable_xla=False` is deprecated.
    native_serialization: serialize the JAX function natively to
      StableHLO with compatibility guarantees. This makes it easier to have
      confidence that the code executed when calling this function from
      TensorFlow is exactly the same as JAX would run natively.
      The DEFAULT_NATIVE_SERIALIZATION value defers to `False` if `enable_xla`
      is set to `False` or to the configuration flag
      `--jax2tf_default_native_serialization` otherwise.
      Native serialization cannot be used with `enable_xla=False`.
      Starting with JAX 0.4.31 support for non-native serialization is deprecated.
    native_serialization_platforms: In conjunction with
      `native_serialization`, specify the platform(s)
      for which to lower the code. Must be a tuple of
      strings, including a subset of: 'cpu', 'cuda', 'rocm', 'tpu'.
      The default (`None``), specifies the JAX default
      backend on the machine where the lowering is done.
    native_serialization_disabled_checks: In conjunction with
      `native_serialization`, disable the specified safety checks.
      See docstring of `DisabledSafetyCheck`.

  Returns:
    A version of `fun_jax` that expects TfVals as arguments (or
    tuple/lists/dicts thereof), and returns TfVals as outputs, and uses
    only TensorFlow ops and thus can be called from a TensorFlow program.
  """
  if native_serialization is DEFAULT_NATIVE_SERIALIZATION:
    if not enable_xla:
      native_serialization = False
    else:
      native_serialization = config.jax2tf_default_native_serialization.value

  if not enable_xla:
    if allow_enable_xla_false():
      warnings.warn(
          "jax2tf.convert with enable_xla=False has been deprecated "
          "since July 2024.",
          DeprecationWarning,
          stacklevel=2)
      if native_serialization:
        raise ValueError(
            "native_serialization is not supported with enable_xla=False")
    else:
      raise ValueError(
          "jax2tf.convert with enable_xla=False has been deprecated "
          "since July 2024 and it is not supported anymore.")

  elif not native_serialization:
    if allow_native_serialization_false():
      warnings.warn(
          "jax2tf.convert with native_serialization=False has been deprecated "
          "since July 2024.",
          DeprecationWarning,
          stacklevel=2)
    else:
      raise ValueError(
          "jax2tf.convert with native_serialization=False has been deprecated "
          "since July 2024 and it is not supported anymore.")

  if not native_serialization and polymorphic_constraints:
    raise ValueError(
        "polymorphic_constraints are supported only with native serialization"
    )

  if native_serialization_platforms:
    if not native_serialization:
      warnings.warn(
          "using native_serialization_platforms without native_serialization. "
          "The parameter will have no effect, since the same code is serialized "
          "for all platforms without native_serialization.")

    if (not isinstance(native_serialization_platforms, (list, tuple)) or
        not all(p in ["cpu", "cuda", "rocm", "tpu"]
                for p in native_serialization_platforms)):
      raise ValueError(
          "native_serialization_platforms must be a sequence "
          "containing a subset of {'cpu', 'cuda', 'rocm', 'tpu'}. "
          f"Got: {native_serialization_platforms}")
    native_serialization_platforms = tuple(native_serialization_platforms)

  api.check_callable(fun_jax)

  def converted_fun_tf(*args_tf: TfVal, **kwargs_tf: TfVal) -> TfVal:

    # TODO: is there a better way to check if we are inside a transformation?
    if not core.trace_state_clean() and not _thread_local_state.inside_call_tf:
      # It is Ok to nest convert when we are inside a call_tf
      raise ValueError(
          "convert must be used outside all JAX transformations." +
          f"Trace state: {core.trace_ctx}")

    global _has_registered_tf_source_path
    if not _has_registered_tf_source_path:
      source_info_util.register_exclusion(os.path.dirname(tf.__file__))
      _has_registered_tf_source_path = True

    def jax_arg_spec_from_tf(a: TfVal) -> jax.ShapeDtypeStruct:
      # The shape and JAX dtype for a TF argument
      tf_arg_shape = np.shape(a)
      # Fix the shape for TF1
      tf_arg_shape = tuple(d.value
                           if isinstance(d, tf.compat.v1.Dimension) else d
                           for d in tf_arg_shape)
      _, a_jax_dtype = _tfval_to_tensor_jax_dtype(a)
      # We count on the fact that jax.ShapeDtypeStruct allows shapes that
      # contain None.
      return jax.ShapeDtypeStruct(tf_arg_shape, a_jax_dtype)

    args_jax_specs = tree_util.tree_map(jax_arg_spec_from_tf, args_tf)
    args_specs = export.symbolic_args_specs(
        args_jax_specs, polymorphic_shapes,
        constraints=polymorphic_constraints)
    # The polymorphic_shapes argument refers to positional arguments only.
    # We assume None for the kwargs.
    kwargs_jax_specs = tree_util.tree_map(jax_arg_spec_from_tf, kwargs_tf)
    kwargs_specs = export.symbolic_args_specs(
        kwargs_jax_specs, None)
    combined_args_tf = (args_tf, kwargs_tf)
    args_flat_tf: Sequence[TfVal]
    args_flat_tf, args_kwargs_tree = tree_util.tree_flatten(combined_args_tf)

    args_flat_tf = tuple(
        map(preprocess_arg_tf, range(len(args_flat_tf)), args_flat_tf))

    impl: SerializationImpl
    if native_serialization:
      impl = NativeSerializationImpl(
          fun_jax,
          args_specs=args_specs, kwargs_specs=kwargs_specs,
          native_serialization_platforms=native_serialization_platforms,
          native_serialization_disabled_checks=native_serialization_disabled_checks)
    else:
      impl = GraphSerializationImpl(
          fun_jax,
          args_specs=args_specs, kwargs_specs=kwargs_specs,
          args_flat_tf=args_flat_tf,
          enable_xla=enable_xla)
    try:
      impl.before_conversion()

      outs_tree: tree_util.PyTreeDef = None  # type: ignore
      if with_gradient:
        @tf.custom_gradient
        def converted_fun_flat_with_custom_gradient_tf(*args_flat_tf: TfVal) -> TfVal:
          nonlocal outs_tree
          outs_tf, outs_avals, outs_tree = impl.run_fun_tf(args_flat_tf)
          return (tuple(outs_tf),
                  _make_custom_gradient_fn_tf(
                      fun_jax,
                      impl=impl,
                      with_gradient=with_gradient,
                      args_specs=args_specs, kwargs_specs=kwargs_specs,
                      args_tf=args_flat_tf,
                      outs_avals=outs_avals,
                      outs_tf=outs_tf))

        outs_flat_tf = converted_fun_flat_with_custom_gradient_tf(*args_flat_tf)
      else:
        outs_tf, _, outs_tree = impl.run_fun_tf(args_flat_tf)
        message = ("The jax2tf-converted function does not support gradients. "
                   "Use `with_gradient` parameter to enable gradients")
        # We use PreventGradient, which is propagated through a SavedModel.
        outs_flat_tf = [
            tf.raw_ops.PreventGradient(input=o, message=message)
            for o in outs_tf
        ]
    finally:
      impl.after_conversion()

    outs_flat_tf = [tf.identity(x, "jax2tf_out") for x in outs_flat_tf]
    out_tf = tree_util.tree_unflatten(outs_tree, outs_flat_tf)
    return out_tf

  return converted_fun_tf

class SerializationImpl:
  """Implementation details for jax2tf serialization.

  Abstract superclass for subclassing.
  """
  def before_conversion(self):
    """Called in the resulting TF function, before any other method.

    Useful to set any global context."""
    raise NotImplementedError

  def after_conversion(self):
    """Called in the resulting TF function, after conversion is done.

    Useful to restore any global context set up by `before_conversion`."""
    raise NotImplementedError

  def run_fun_tf(self,
                 args_flat_tf: Sequence[TfVal]
                 ) -> tuple[Sequence[TfVal], Sequence[core.ShapedArray], tree_util.PyTreeDef]:
    """Runs the resulting TF function.

    Args:
      args_flat_tf: a flat tuple of tf.Tensor arguments

    Returns: a tuple with:
      outs_tfs: a flat tuple of tf.Tensor results
      outs_avals: a flat tuple of JAX abstract values for the underlying JAX
        function.
      outs_tree: the PyTreeDef for the outputs
    """
    raise NotImplementedError

  def get_vjp_fun(self) -> tuple[Callable,
                                 Sequence[core.AbstractValue]]:
    """Returns the VJP function, and the VJP in_avals."""
    raise NotImplementedError


class NativeSerializationImpl(SerializationImpl):
  def __init__(self, fun_jax, *,
               args_specs, kwargs_specs,
               native_serialization_platforms: Sequence[str] | None,
               native_serialization_disabled_checks: Sequence[DisabledSafetyCheck]):
    self.convert_kwargs = dict(native_serialization=True,
                               native_serialization_platforms=native_serialization_platforms,
                               native_serialization_disabled_checks=native_serialization_disabled_checks)
    if hasattr(fun_jax, "trace"):
      # If we have a pjit or pmap already we do not wrap with another, and we
      # allow shardings.
      fun_jit = fun_jax
    else:
      # We support convert(pjit(f_jax)) and convert(jit(f_jax)) but also
      # convert(f_jax), in which case a "jit" is implied. In that case we raise
      # an error if the lowered function contains non-replicated sharding annotations.
      fun_jit = jax.jit(fun_jax)
    self.fun_jax = fun_jit
    self.args_specs = args_specs
    self.kwargs_specs = kwargs_specs
    self.native_serialization_disabled_checks = native_serialization_disabled_checks
    self.native_serialization_platforms = native_serialization_platforms

  def before_conversion(self):
    _prev_func_list = _thread_local_state.call_tf_concrete_function_list
    _thread_local_state.call_tf_concrete_function_list = []

    def _restore_context():
      _thread_local_state.call_tf_concrete_function_list = _prev_func_list

    self._restore_context = _restore_context
    _exported_device_assignment = [None]
    self.exported = _export._export_internal(
        self.fun_jax,
        platforms=self.native_serialization_platforms,
        disabled_checks=self.native_serialization_disabled_checks,
        _device_assignment_for_internal_jax2tf_use_only=_exported_device_assignment,
    )(*self.args_specs, **self.kwargs_specs)
    assert(_exported_device_assignment[0] is not None)
    self.device_assignment = _exported_device_assignment[0]

  def after_conversion(self):
    self._restore_context()

  def run_fun_tf(self,
                 args_flat_tf: Sequence[TfVal]
                 ) -> tuple[Sequence[TfVal], Sequence[core.ShapedArray], tree_util.PyTreeDef]:
    results = _run_exported_as_tf(args_flat_tf, self.exported)
    return results, tuple(self.exported.out_avals), self.exported.out_tree

  def get_vjp_fun(self) -> tuple[Callable,
                                 Sequence[core.AbstractValue]]:
    return _export._get_vjp_fun(self.fun_jax,
                                in_tree=self.exported.in_tree,
                                in_avals=self.exported.in_avals,
                                in_shardings_hlo=self.exported.in_shardings_hlo,
                                out_avals=self.exported.out_avals,
                                out_shardings_hlo=self.exported.out_shardings_hlo,
                                device_assignment=self.device_assignment,
                                apply_jit=True)

class GraphSerializationImpl(SerializationImpl):
  def __init__(self, fun_jax, *,
               args_specs, kwargs_specs,
               args_flat_tf: Sequence[TfVal],
               enable_xla: bool):
    self.convert_kwargs = dict(native_serialization=False)
    self.fun_jax = fun_jax
    self.args_specs = args_specs
    self.kwargs_specs = kwargs_specs
    self.enable_xla = enable_xla

    fun_name = getattr(fun_jax, "__name__", "unknown")
    name_stack = util.wrap_name(fun_name, "jax2tf")
    self.name_stack = name_stack
    self.args_flat_tf = args_flat_tf

  def before_conversion(self):
    prev_enable_xla = _thread_local_state.enable_xla
    prev_include_xla_op_metadata = _thread_local_state.include_xla_op_metadata
    prev_tf_outer_name_scope = _thread_local_state.tf_outer_name_scope
    def _restore_context():
      _thread_local_state.enable_xla = prev_enable_xla
      _thread_local_state.include_xla_op_metadata = prev_include_xla_op_metadata
      _thread_local_state.tf_outer_name_scope = prev_tf_outer_name_scope
      _thread_local_state.shape_env = ()
    self._restore_context = _restore_context
    _thread_local_state.enable_xla = self.enable_xla
    # TODO(b/189306134): implement support for XLA metadata
    _thread_local_state.include_xla_op_metadata = False
    _thread_local_state.tf_outer_name_scope = tf.get_current_name_scope()
    assert not _thread_local_state.shape_env, f"Unexpected shape environment {_thread_local_state.shape_env}"
    args_specs_flat, self.in_tree = tree_util.tree_flatten(
        (self.args_specs, self.kwargs_specs))
    self.args_avals_flat = tuple(
        map(core.get_aval, args_specs_flat))
    dim_vars = shape_poly.all_dim_vars(self.args_avals_flat)
    dim_values, _ = _interpret_fun_jax(
        partial(shape_poly.compute_dim_vars_from_arg_shapes,
                self.args_avals_flat, args_kwargs_tree=self.in_tree),
        self.args_flat_tf, self.args_avals_flat, self.name_stack)

    _thread_local_state.shape_env = zip(dim_vars, dim_values)

  def after_conversion(self):
    self._restore_context()

  def run_fun_tf(self,
      args_flat_tf: Sequence[TfVal]
      ) -> tuple[Sequence[TfVal], Sequence[core.ShapedArray], tree_util.PyTreeDef]:
    fun_flat_jax, out_tree_thunk = flatten_fun_jax(self.fun_jax, self.in_tree)
    # out_tree_thunk will be ready after we _interpret_fun_jax below
    outs_tf, self.outs_avals = _interpret_fun_jax(
        fun_flat_jax,
        args_flat_tf, self.args_avals_flat,
        self.name_stack,
        fresh_constant_cache=True)
    return outs_tf, self.outs_avals, out_tree_thunk()

  def get_vjp_fun(self) -> tuple[Callable,
                                 Sequence[core.AbstractValue]]:
    # We reuse the code for native serialization to get the VJP functions,
    # except we use unspecified shardings, and we do not apply a jit on the
    # VJP. This matches the older behavior of jax2tf for graph serialization.
    return _export._get_vjp_fun(self.fun_jax,
                                in_tree=self.in_tree,
                                in_avals=self.args_avals_flat,
                                in_shardings_hlo=(None,) * len(self.args_avals_flat),
                                out_avals=self.outs_avals,
                                out_shardings_hlo=(None,) * len(self.outs_avals),
                                device_assignment=None,  # Not used when apply_jit = False
                                apply_jit=False)


def dtype_of_val(val: TfVal) -> DType:
  """Computes the TensorFlow dtype using JAX's typing rules.

  If the value is a tf.Tensor, it starts with its dtype. If the value is a
  constant it uses JAX to infer its dtype. The resulting dtype follows the
  JAX type inference rules, and depends on the value of the
  JAX_ENABLE_X64 flag.

  See README.md for how 64-bit values are treated.
  """
  tval, _ = _tfval_to_tensor_jax_dtype(val)
  return tval.dtype

@partial(api_util.api_hook, tag="jax2tf_eval_polymorphic_shapes")
def eval_polymorphic_shape(fun_jax: Callable,
               *,
               polymorphic_shapes=None) -> Callable:
  """Evaluates the output shape in presence of shape polymorphism.

  This is done without lowering or executing the function, same as for
  `jax.eval_shape`.

  Args:
    fun_jax: target JAX function to be called. Its arguments and return value
      should be JAX arrays, or nested standard Python containers
      (tuple/list/dict) thereof (pytrees).
    polymorphic_shapes: Specifies input shapes to be treated polymorphically
      during shape evaluation. See discussion for `jax2tf.convert`.

      .. warning:: The shape-polymorphic lowering is an experimental feature.

  Returns: a function that takes `jax.ShapeDtypeStruct`s (or any values
    with `.shape` and `.dtype` attributes) corresponding to the inputs for
    `fun_jax`, and returns a tuple with:

      * the jax.ShapeDtypeStruct corresponding to the result, as for
       `jax.eval_shape`. The shape may contain symbolic dimension expressions.
      * the value that can be passed to `polymorphic_shapes` for a subsequent
        call to `jax2tf.eval_polymorphic_shape`, or `jax2tf.convert`.

  For example:

  >>> import jax
  >>> from jax.experimental import jax2tf
  >>> from jax import numpy as jnp
  >>>
  >>> f = lambda A, x: jnp.sin(jnp.dot(A, x))
  >>> A = jax.ShapeDtypeStruct((2000, 3000), jnp.float32)
  >>> x = jax.ShapeDtypeStruct((3000, 1000), jnp.float32)
  >>> out_spec, out_poly_shape = jax2tf.eval_polymorphic_shape(f, polymorphic_shapes=["a, b", "b, c"])(A, x)
  >>> print(out_spec.shape)
  ("a", "c")
  >>> print(out_poly_shape)
  (a, c)
  >>> res_spec, res_poly_shape = jax2tf.eval_polymorphic_shape(lambda x: x.T, polymorphic_shapes=[out_poly_shape])(out_spec)
  >>> print(res_poly_shape)
  (c, a)
  """
  def do_eval_polymorphic_shape(*args_specs) -> Any:
    args_poly_specs = export.symbolic_args_specs(
        args_specs, polymorphic_shapes)
    res_poly_spec = jax.eval_shape(fun_jax, *args_poly_specs)
    # TODO(necula): For now we export the polymorphic shapes using `str`.
    res_polymorphic_shape = tree_util.tree_map(lambda r: str(r.shape), res_poly_spec)
    return res_poly_spec, res_polymorphic_shape

  return do_eval_polymorphic_shape


# Internals

def flatten_fun_jax(fun_jax: Callable,
                    in_tree,
                    ) -> tuple[Callable, Callable]:
  """Wraps the function to take a (flat) list of positional args.

  jax2tf works better and is simpler when the JAX function takes and returns
  just a tuple of values (no pytrees, no kwargs). This is in part because
  jax.vjp does not support kwargs and we can only set
  tf.custom_gradient on functions with flat arguments and results

  Returns:
     * the wrapped JAX function taking and returning a flat list of arguments
     * a thunk that can be called after the wrapped function has been called
       to return the output pytree.
  """
  out_tree_ref = None
  def fun_flat_jax(*args_flat_jax):
    tree_args, tree_kwargs = tree_util.tree_unflatten(in_tree, args_flat_jax)
    tree_res = fun_jax(*tree_args, **tree_kwargs)
    res_flat_jax, out_tree = tree_util.tree_flatten(tree_res)
    nonlocal out_tree_ref
    assert out_tree_ref is None or out_tree_ref == out_tree
    out_tree_ref = out_tree
    return res_flat_jax

  return fun_flat_jax, lambda: out_tree_ref

def preprocess_arg_tf(arg_idx: int,
                      arg_tf: TfVal) -> TfVal:
  """Pre-processes the TF args.

  Returns: a tuple with the pre-processed TF arg, the TF shape, and the
      JAX dtype.
  """
  if not _is_tfval(arg_tf):
    msg = (f"Argument {arg_tf} of type {type(arg_tf)} of jax2tf.convert(f) should "
           "be NumPy array, scalar, tf.Variable, or tf.Tensor")
    raise TypeError(msg)

  # May cast the args_flat to JAX types, using JAX's interpretation
  # of types of constants.
  arg_tf, _ = _tfval_to_tensor_jax_dtype(arg_tf)
  # Name input tensors; do this after we have cast the arguments
  arg_tf = tf.identity(arg_tf, f"jax2tf_arg_{arg_idx}")
  return arg_tf


def _make_custom_gradient_fn_tf(fun_jax,
                                *,
                                impl: SerializationImpl,
                                with_gradient: bool,
                                args_specs, kwargs_specs,
                                args_tf: Sequence[TfVal],
                                outs_avals: Sequence[core.ShapedArray],
                                outs_tf: Sequence[TfVal]):
  """Prepares the TF function to be used with tf.custom_gradient.

  Args:
    impl: the serialization implementation details
    with_gradient: whether to include a tf.custom_gradient
    args_specs, kwargs_specs: the jax.ShapeDtypeArrays for the args and kwargs
    args_tf: the flattened TF arguments of the primal function
    outs_avals: the flattened output JAX abstract values of the primal function
    outs_tf: the flattened TF outputs of the primal function
  """
  def grad_fn_tf(*out_cts_flat_tf: TfVal,
                 variables=None):
    if variables:
      raise ValueError(
          "Unexpected variables used in forward pass. "
          "This should not happen for first-order differentiation. "
          f"{variables=}")

    # TODO: enable higher-order gradients
    with tf.name_scope("jax2tf_vjp"):
      def fix_out_ct(out_ct_tf, out_ct_aval: core.ShapedArray, out_tf: TfVal):
        # If the primal function has outputs of integer or bool types, and if we are
        # under a tf.function context, then TF will pass None in _out_cts_flat
        # in place of these values. We should change these to float0 or
        # else JAX gets unhappy. See issue #6975.
        if out_ct_tf is not None:
          return out_ct_tf
        assert core.primal_dtype_to_tangent_dtype(out_ct_aval.dtype) == dtypes.float0, f"{out_ct_tf=}"
        # Note that out_ct_aval.shape contains dimension variable from the
        # primal function scope. We use tf.zeros_like to make a 0 of the right shape.
        return tf.zeros_like(out_tf, dtype=_tf_np_dtype_for_float0)

      out_cts_fixed_flat_tf = tuple(map(fix_out_ct, out_cts_flat_tf, outs_avals, outs_tf))
      vjp_args_flat_tf = tuple(args_tf) + out_cts_fixed_flat_tf

      fun_vjp_jax, vjp_in_avals = impl.get_vjp_fun()

      vjp_polymorphic_shapes = tuple(
        str(a.shape)  # Note: may be _DimExpr, not just DimVar
        for a in vjp_in_avals)
      in_cts_flat = convert(
        fun_vjp_jax,
        with_gradient=with_gradient,
        polymorphic_shapes=vjp_polymorphic_shapes,
        **impl.convert_kwargs)(*vjp_args_flat_tf)

    # We do not need to fix the in_cts because the TF gradient machinery
    # will adjust the unconnected gradients and those for integer types.
    return in_cts_flat

  return grad_fn_tf

@contextlib.contextmanager
def _extended_name_stack(extra_name_stack: str | None):
  name_ctx = (source_info_util.extend_name_stack(extra_name_stack)
      if extra_name_stack
      else contextlib.nullcontext())
  with name_ctx:
    yield
  return


def _interpret_fun_jax(
    fun_jax: Callable,
    args_tf: Sequence[TfVal],
    args_avals: Sequence[core.ShapedArray],
    extra_name_stack: str | None,
    fresh_constant_cache: bool = False,
) -> tuple[tuple[TfVal, ...], tuple[core.ShapedArray, ...]]:
  subtrace_fun = _interpret_subtrace(lu.wrap_init(fun_jax), args_avals)
  with _extended_name_stack(extra_name_stack):
    out_vals: Sequence[tuple[TfVal, core.ShapedArray]] = \
        _call_wrapped_with_new_constant_cache(subtrace_fun, args_tf,
                                                fresh_constant_cache=fresh_constant_cache)
  return util.unzip2(out_vals)


def _run_exported_as_tf(args_flat_tf: Sequence[TfVal],
                        exported: export.Exported,
                        ) -> Sequence[TfVal]:
  """Runs the `exported` as an XlaCallModule TF op.

  Returns: the flattened tuple of results.
  """
  args_avals = exported.in_avals

  # TF values may be integer types for float0
  def _convert_value(val, aval):
    # Check the shape
    assert all(d_aval == d_val
               for d_aval, d_val in zip(aval.shape, val.shape)
               if core.is_constant_dim(d_aval)), (aval, val)
    conversion_dtype = _to_tf_dtype(aval.dtype)
    if conversion_dtype != aval.dtype:
      return tf.cast(val, conversion_dtype)
    else:
      return val

  args_flat_tf = tuple(map(_convert_value, args_flat_tf, args_avals))

  out_shapes_tf = tuple(
      tuple(d if core.is_constant_dim(d) else None
            for d in out_aval.shape)
      for out_aval in exported.out_avals)

  out_types = tuple(_to_tf_dtype(out_aval.dtype) for out_aval in exported.out_avals)

  kept_args_avals = [aval for i, aval in enumerate(exported.in_avals) if i in exported.module_kept_var_idx]
  kept_args_flat_tf = [atf for i, atf in enumerate(args_flat_tf) if i in exported.module_kept_var_idx]

  version = exported.calling_convention_version

  try:
    get_max_supported_version = tfxla.call_module_maximum_supported_version
  except AttributeError:
    get_max_supported_version = None

  if get_max_supported_version:
    max_supported_version = get_max_supported_version()
  else:
    max_supported_version = 6

  if version > max_supported_version:
    raise NotImplementedError(
      "XlaCallModule from your TensorFlow installation supports up to "
      f"serialization version {max_supported_version} but the serialized "
      f"module needs version {version}. "
      "You should upgrade TensorFlow, e.g., to tf_nightly."
    )

  call_module_attrs = dict(
      version=version,
      Tout=out_types,
      Sout=out_shapes_tf,
      function_list=[
          concrete_fn.function_def.signature.name
          for concrete_fn in _thread_local_state.call_tf_concrete_function_list
      ] if _thread_local_state.call_tf_concrete_function_list is not None else [],
      # We always set has_token_input_output because it requires real tokens
      # for versions less than 9 and is not used starting with version 9.
      has_token_input_output=False
  )

  call_module_attrs["platforms"] = tuple(p.upper() for p in exported.platforms)
  if version >= 6:
    call_module_attrs["disabled_checks"] = tuple(
        str(dc)
        for dc in exported.disabled_safety_checks)
  else:
    if version >= 3:
      if DisabledSafetyCheck.platform() in exported.disabled_safety_checks:
        call_module_attrs["platforms"] = ()  # No platform checking

  if logging.vlog_is_on(3):
    # We already logged the MLIR module when we exported it.
    logging.vlog(3, "XlaCallModule %s", str(call_module_attrs))

  call_module_attrs["module"] = exported.mlir_module_serialized

  # Apply the shardings on arguments and results for pjit. This is redundant
  # because the mlir_module_text will already contain the shardings, but it
  # makes it easier for tools like the TPU inference converter to see the
  # sharding without digging into the `module` attribute of the `XlaCallModule`
  # op, in the same way as it is done for the legacy jax2tf conversion.
  # Do not apply XlaSharding for REPLICATED, on inputs and outputs.
  # This is an agreed convention, and also improves usability under TF eager.
  # See b/255511660.
  kept_in_shardings = []
  for i in exported.module_kept_var_idx:
    kept_in_shardings.append(exported.in_shardings_hlo[i])
  args_flat_tf = tuple(
    map(partial(_shard_value,
                skip_replicated_sharding=tf.executing_eagerly()),
        kept_args_flat_tf, kept_in_shardings))
  res = tfxla.call_module(args_flat_tf, **call_module_attrs)
  # TODO(b/278940799): Replace the TF v1 API with public TF2 API.
  # Add the custom call tf.function into the default graph, so those functions
  # will be available during tf.SavedModel.save.
  if _thread_local_state.call_tf_concrete_function_list is not None:
    for concrete_fn in _thread_local_state.call_tf_concrete_function_list:
      tf.compat.v1.get_default_graph()._add_function_recursive(
          concrete_fn._inference_function
      )

  res = list(map(partial(_shard_value,
                         skip_replicated_sharding=tf.executing_eagerly()),
                 res, exported.out_shardings_hlo))
  res = tuple(map(_convert_value, res, exported.out_avals))
  return res


def _call_wrapped_with_new_constant_cache(fun: lu.WrappedFun,
                                          in_vals: Sequence[TfVal],
                                          fresh_constant_cache: bool = False
                                          ) -> Sequence[tuple[TfVal, core.ShapedArray]]:
  try:
    prev_constant_cache = _thread_local_state.constant_cache
    # Start a new cache, so that we don't share constants across tf.function
    # boundaries.
    if fresh_constant_cache:
      _thread_local_state.constant_cache = {}
    else:
      prev_constant_cache_keys = set(prev_constant_cache.keys()) if prev_constant_cache is not None else set()
    out_vals: Sequence[tuple[TfVal, core.ShapedArray]] = \
        fun.call_wrapped(*in_vals)
  finally:
    if (not fresh_constant_cache and
        prev_constant_cache is not None and
        _WRAP_JAX_JIT_WITH_TF_FUNCTION):
      newly_added_keys = set(prev_constant_cache.keys()) - prev_constant_cache_keys
      # Delete the newly added keys
      for k in newly_added_keys:
        del prev_constant_cache[k]
    _thread_local_state.constant_cache = prev_constant_cache
  return out_vals

def _convert_jax_impl(impl_jax: Callable, *,
                      multiple_results=True,
                      with_physical_avals=False,
                      extra_name_stack: str | None = None) -> Callable:
  """Convert the JAX implementation of a primitive.

  Args:
    impl_jax: typically the impl-rule for a primitive, with signature
      `(*args_jax: JaxVal, **kwargs) -> Sequence[JaxVal]`. This function implements
        a primitive in terms of other primitives.
    multiple_results: whether `impl_jax` returns a sequence of results.
    extra_name_stack: additional element to add to the name stack for the
      converted ops.

  Returns:
     a function with signature `(*args_tf: TfVal, _in_avals, _out_aval, **kwargs)
     -> Sequence[TfVal]`.
  """

  def wrapped_tf(*args_tf: TfVal, _in_avals: Sequence[core.ShapedArray],
                 _out_aval: core.ShapedArray,
                 **kwargs) -> Sequence[TfVal]:

    if with_physical_avals:
      _in_avals = map(_jax_physical_aval, _in_avals)
      _out_aval = _jax_physical_aval(_out_aval)

    # We wrap the impl_jax to always return a tuple of results.
    def impl_multiple_results_jax(*args_jax):
      results_jax = impl_jax(*args_jax, **kwargs)
      return results_jax if multiple_results else [results_jax]

    results_tf, _ = _interpret_fun_jax(
        impl_multiple_results_jax, args_tf, _in_avals,
        extra_name_stack)
    return results_tf if multiple_results else results_tf[0]

  return wrapped_tf


@lu.transformation2
def _interpret_subtrace(f, in_avals: Sequence[core.ShapedArray],
                        *in_vals: TfVal):
  trace = TensorFlowTrace()
  in_tracers = tuple(
      TensorFlowTracer(trace, val, aval)
      for val, aval in zip(in_vals, in_avals))
  with core.set_current_trace(trace):
    outs = f(*in_tracers)
  out_tracers: Iterable[TensorFlowTracer] = (
      map(trace.to_tf_tracer, outs))
  out_vals_with_avals: Sequence[tuple[TfVal, core.ShapedArray]] = (
      tuple((t.val, t.aval) for t in out_tracers))
  return out_vals_with_avals


def _interpret_jaxpr(jaxpr: core.ClosedJaxpr, *args_tf: TfVal,
                     extra_name_stack: str | None,
                     fresh_constant_cache: bool = True) -> Sequence[TfVal]:
  """Evaluates a Jaxpr with tf.Tensor arguments.

  This is most often used as the body of a tf.function, or tf.switch_case,
  in which case it should use a fresh constant cache.
  The output is a sequence of TfVal, suitable for use with TF.
  """
  outs_tf, _ = _interpret_fun_jax(core.jaxpr_as_fun(jaxpr),
                                  args_tf, jaxpr.in_avals, extra_name_stack,
                                  fresh_constant_cache=fresh_constant_cache)
  return outs_tf


def _jax_physical_aval(aval: core.ShapedArray) -> core.ShapedArray:
  """Converts JAX avals from logical to physical, if relevant.

  JAX might have avals whose logical vs physical shape/dtype may
  differ, and only the physical view is expected to possibly
  relate to TF. TF impl rules should operate on the physical form.

  A JAX logical aval might even correspond, in principle, to several
  physical avals, but we don't support those here. Instead we assert
  there is only one and return it.
  """
  physical_aval = core.physical_aval(aval)
  assert (len(physical_aval.shape) >= len(aval.shape) and
          physical_aval.shape[:len(aval.shape)] == aval.shape), (physical_aval, aval)
  return physical_aval

def _jax_physical_dtype(dtype):
  # assuming () is a fine stand-in shape
  return _jax_physical_aval(core.ShapedArray((), dtype)).dtype


def _aval_to_tf_shape(aval: core.ShapedArray) -> tuple[int | None, ...]:

  """Generate a TF shape, possibly containing None for polymorphic dimensions."""
  aval = _jax_physical_aval(aval)
  return tuple(map(lambda d: None if export.is_symbolic_dim(d) else d,
                   aval.shape))

# In the TF world, we represent float0 as zeros of this type.
# We pick bool because this is what JAX uses when it lowers float0 to HLO.
_tf_np_dtype_for_float0 = np.bool_

def _to_tf_dtype(jax_dtype):
  # Note that converting _to_tf_dtype and _to_jax_dtype are not inverses,
  # due to float0 and 64-bit behavior.
  try:
    jax_dtype = _jax_physical_dtype(jax_dtype)
  except TypeError:
    # `jax_dtype` isn't actually a valid jax dtype (e.g. it is
    # tf.float32), so there is no physical dtype anyway
    pass
  if jax_dtype == dtypes.float0:
    jax_dtype = _tf_np_dtype_for_float0
  return tf.dtypes.as_dtype(jax_dtype)


def _to_jax_dtype(tf_dtype):
  # Note that converting _to_tf_dtype and _to_jax_dtype are not inverses,
  # due to float0 and 64-bit behavior.
  dt = dtypes.canonicalize_dtype(tf_dtype.as_numpy_dtype)
  if dt not in dtypes._jax_dtype_set:
    raise TypeError(f"dtype {dt} is not a valid JAX array "
                    "type. Only arrays of numeric types are supported by JAX.")
  return dt


def _tfval_to_tensor_jax_dtype(val: TfVal,
                               jax_dtype: DType | None = None,
                               memoize_constants=False) -> tuple[TfVal, DType]:
  """Converts a scalar, ndarray, or tf.Tensor to a tf.Tensor with proper type.

  If `jax_dtype` is missing, uses JAX typing rules.
  See README.md for details regarding 64-bit values.

  Args:
    val: a scalar, ndarray, tf.Tensor, or tf.Variable
    jax_dtype: an optional dtype to use. If missing, uses JAX type inference
      rules for constants.
    memoize_constants: whether to memoize TF constants. We can't do this
      everywhere, we may be outside of a conversion scope.

  Returns:
    a tuple with a tf.Tensor with the type as needed by JAX, and the JAX type.
  """
  if isinstance(val, (tf.Tensor, tf.Variable)):
    jax_dtype = jax_dtype or _to_jax_dtype(val.dtype)  # Give JAX a chance to pick the type
    conversion_dtype = _to_tf_dtype(jax_dtype)
    if conversion_dtype != val.dtype:  # May need to cast for 64-bit values
      return tf.cast(val, conversion_dtype), jax_dtype
    else:
      return val, jax_dtype
  else:  # A constant
    jax_dtype = jax_dtype or core.abstractify(val).dtype
    # TODO(document): We assume that the value of a constant does not
    # change through the scope of the function. But it may be an ndarray, ...
    # JAX has the same problem when generating HLO.
    const_key = (id(val), jax_dtype)
    # Since we use id(val) as a cache key, we have to make sure that we keep
    # the previous `val` alive. Otherwise, for a ndarray, it can get garbage
    # collected and reused for a different value, which would create correctness
    # issues. We keep the `val` alive by storing in the cache the pair
    # `(val, tf_val)`.
    # Only memoize non-scalars. JAX will lift all non-scalar constants as
    # Jaxpr consts, to the top level of the Jaxpr. This ensures that we see them
    # early, when entering the Jaxpr, so we create the tf.const early and its
    # scope is the entire Jaxpr.
    do_memoize = (memoize_constants and np.size(val) > 1 and _thread_local_state.constant_cache is not None)
    if do_memoize:
      _, tf_val = _thread_local_state.constant_cache.get(const_key, (None, None))
    else:
      tf_val = None
    if tf_val is None:
      conversion_dtype = _to_tf_dtype(jax_dtype)
      # The float0 type is not known to TF.
      if jax_dtype == dtypes.float0:
        val = np.zeros(np.shape(val), conversion_dtype.as_numpy_dtype)
      if hasattr(val, 'dtype') and dtypes.issubdtype(val.dtype, dtypes.extended):
        val = val.dtype._rules.physical_const(val)
      tf_val = tf.convert_to_tensor(val, dtype=conversion_dtype)
      if do_memoize:
        _thread_local_state.constant_cache[const_key] = (val, tf_val)
    return tf_val, jax_dtype


def _eval_shape(shape: Sequence[shape_poly.DimSize], dtype=None) -> Sequence[TfVal]:
  # Returns a tuple of shape_poly.dim_as_value_dtype
  # Used only for non-native lowering
  assert all(map(lambda x: x is not None, shape)), (
      f"Argument shape should be a valid JAX shape but got {shape}")
  if dtype is not None:
    shape = _jax_physical_aval(core.ShapedArray(shape, dtype)).shape
  if core.is_constant_shape(shape):
    return tuple(int(d) for d in shape)

  dim_vars, dim_values = util.unzip2(_thread_local_state.shape_env)
  shape_values_tf, _ = _interpret_fun_jax(
      partial(core.evaluate_shape, shape, dim_vars),
      dim_values, [core.dim_value_aval()] * len(dim_values), "")  # type: ignore
  # Keep only the non-constant dimensions
  return tuple(operator.index(d) if core.is_constant_dim(d) else d_tf  # type: ignore
               for d, d_tf in zip(shape, shape_values_tf))


def _ensure_tf_shape_if_dynamic(x: TfVal, shape):
  # Update TF tensor `x` with shape `shape` if the shape of `x`` is dynamic.
  if x.shape.is_fully_defined():
    return x
  return tf.ensure_shape(x, shape)


def _assert_matching_abstract_shape(x: TfVal, shape: Sequence[shape_poly.DimSize]):
  """Asserts that shape matches x.shape in the known dimensions and has
  dimension polynomials elsewhere."""
  # Ensures that the shape does not contain None; it should contain symbolic expressions.
  def check_one(xd: int | None, sd: Any):
    if core.is_constant_dim(sd):
      return xd == sd
    else:
      assert export.is_symbolic_dim(sd)
      return True
  assert (len(x.shape) == len(shape) and
          all(check_one(xd, sd)
              for xd, sd in zip(x.shape, shape))), \
    f"Shape {shape} does not match x.shape {x.shape}"

# TODO(b/26854495): pylint doesn't understand slots and inheritance.
# pylint: disable=assigning-non-slot


class TensorFlowTracer(core.Tracer):
  """Tracer class that boxes a TF value and a JAX abstract value.

  In addition to the TF value we carry the JAX abstract value because
  there are some cases when it cannot be recovered from the value:
  when we are converting with polymorphic shapes or when the JAX aval
  has a custom element type. In these cases the shape of the value may
  have dimensions set to `None`, or it may only correspond to the JAX
  "physical" (TF/lowering-compatible) shape, so the JAX abstract value
  may contain more precise information.

  When the value has a partially-known shape, the dimensions marked as `None`
  must correspond to non-constant dimensions in the abstract value.

  See README.md for details.
  """
  # val: TfVal
  # _aval: core.ShapedArray
  __slots__ = ["val", "_aval"]

  def __init__(self, trace: TensorFlowTrace, val: TfVal,
               aval: core.AbstractValue):
    self._trace = trace
    self._aval = aval
    phys_aval = _jax_physical_aval(self._aval)  # type: ignore[arg-type]

    if isinstance(val, (tf.Tensor, tf.Variable)):
      val_shape = val.shape

      if config.enable_checks.value:
        assert len(phys_aval.shape) == len(val_shape), f"_aval.shape={phys_aval.shape} different rank than {val_shape=}"
        # To compare types, we must handle float0 in JAX and x64 in TF
        if phys_aval.dtype == dtypes.float0:
          assert _to_tf_dtype(phys_aval.dtype) == val.dtype, f"expected {phys_aval.dtype} == {val.dtype}"
        else:
          assert phys_aval.dtype == _to_jax_dtype(val.dtype), f"expected {phys_aval.dtype} == {val.dtype}"

        for aval_dim, val_dim in zip(phys_aval.shape, val_shape):
          if val_dim is None:
            assert export.is_symbolic_dim(aval_dim), f"expected {phys_aval.shape} == {val_shape}"
          elif not export.is_symbolic_dim(aval_dim):
            assert aval_dim == val_dim, f"expected {phys_aval.shape} == {val_shape}"
          else:
            # We have a TF value with known shape, and the abstract shape is a shape variable.
            try:
              aval_int = int(_eval_shape([aval_dim]))  # type: ignore
            except (TypeError, KeyError, shape_poly.UnexpectedDimVar):
              continue
            assert aval_int == val_dim, f"expected {phys_aval.shape} == {val_shape}. Found {aval_int} != {val_dim}."

    self.val = _tfval_to_tensor_jax_dtype(val,
                                          phys_aval.dtype,
                                          memoize_constants=True)[0]

  @property
  def aval(self):
    return self._aval

  def full_lower(self):
    return self

def _make_op_metadata(primitive: core.Primitive,
                      params: dict, *,
                      source_info: source_info_util.SourceInfo,
                      ) -> xla_client.OpMetadata:
  eqn_str = (str(source_info.name_stack) + '/'
             + core.str_eqn_compact(primitive, params))
  frame = source_info_util.user_frame(source_info)
  return xla_client.OpMetadata(
        op_type=primitive.name,
        op_name=eqn_str,
        source_file=mlir.get_canonical_source_file(
            frame.file_name if frame else "", mlir.TracebackCaches()),
        source_line=frame.start_line if frame else None)


class TensorFlowTrace(core.Trace):
  """Trace class that underlies the jax2tf transformation.

  We are going to ensure that jax2tf.convert is never nested inside other
  transformations. This is sufficient for intended use cases (converting
  fully-transformed JAX code). It also simplifies our job because we do not have
  to handle situations where we apply primitives on a mix of TF values and
  JAX tracers from an outer transformation. E.g., for addition both the TF
  values
  and the JAX tracers have an override and they get confused if they see values
  from the other world.

  Hence a TFT trace does not interact with non-TFT traces at lower-level. For
  higher-order control-flow primitives we invoke recursively
  _interpret_fun on the body of the conditional, which will create a nested TFT.

  We do want to allow transformations nested inside a TensorFlowTrace (TFT), but
  those will introduce their own MainTrace, and any operations involving those
  will be done on those traces, i.e., not a concern for TFT.
  """
  def to_tf_tracer(self, val: TfVal) -> TensorFlowTracer:
    """Lifts a non-Tracer into the TensorFlowTracer.
    """
    if isinstance(val, TensorFlowTracer):
      return val
    if hasattr(val, "__jax_array__"):
      with core.set_current_trace(self):
        val = val.__jax_array__()
      if isinstance(val, TensorFlowTracer):
        return val
    tf_val, jax_dtype = _tfval_to_tensor_jax_dtype(val, memoize_constants=True)
    return TensorFlowTracer(
        self, tf_val, core.ShapedArray(np.shape(val), jax_dtype,
                                       weak_type=dtypes.is_weakly_typed(val)))

  def process_primitive(self, primitive: core.Primitive,
                        tracers: Sequence[TensorFlowTracer],
                        params) -> TensorFlowTracer:
    tracers = map(self.to_tf_tracer, tracers)
    impl, impl_needs_avals = self.get_primitive_impl(primitive)
    args_avals: Sequence[core.ShapedArray] = tuple(t.aval for t in tracers)
    # This is a bit conservative, doing abstract_eval even in op-by-op execution
    # but we needed it for, e.g., shape_polymorphism where only JAX's
    # abstract evaluation rules can properly track polymorphic shapes.
    # Unfortunately under op-by-op execution this is a rare occasion where we
    # need abstract evaluation.
    out_aval, _ = primitive.abstract_eval(*args_avals, **params)
    args_tf: Sequence[TfVal] = [t.val for t in tracers]
    def invoke_impl() -> TfVal:
      if impl_needs_avals:
        return impl(
            *args_tf,
            _in_avals=args_avals,
            _out_aval=out_aval,
            **params)
      else:
        return impl(*args_tf, **params)

    current_name_stack = _get_current_name_stack()
    # We don't use `str(name_stack)` because it uses parentheses for
    # transformations, which aren't allowed in `name_scope`.
    scope = '/'.join([s.name for s in current_name_stack.stack])  # type: ignore[union-attr]

    # Here we reset the name scope to the memorized TF name scope
    # + JAX name stack by using absolute scope.
    # We need to add a '/' to the name stack string to force `tf.name_scope`
    # to interpret it as an absolute scope, not a relative scope.
    if _thread_local_state.tf_outer_name_scope:
      scope = f"{_thread_local_state.tf_outer_name_scope}/{scope}"

    if not scope.endswith("/"):
      scope = scope + "/"

    with tf.name_scope(_sanitize_scope_name(scope)):
      if _thread_local_state.include_xla_op_metadata:
        op_metadata = _make_op_metadata(primitive, params,
                                        source_info=source_info_util.current())
        op_metadata_proto = xla_data_pb2.OpMetadata(
            op_type=op_metadata.op_type,
            op_name=op_metadata.op_name,
            source_file=op_metadata.source_file,
            source_line=op_metadata.source_line
        )
        with tf_ops.get_default_graph()._attr_scope(
            {"_XlaOpMetadata": attr_value_pb2.AttrValue(
                s=op_metadata_proto.SerializeToString())}):
          val_out = invoke_impl()
      else:
        val_out = invoke_impl()

    if primitive.multiple_results:
      out = [
          TensorFlowTracer(self, v, a)
          for v, a in zip(val_out, out_aval)
      ]
    else:
      out = TensorFlowTracer(self, val_out, out_aval)  # type: ignore

    # Check that the impl rule returned a value of expected shape and dtype
    # TODO: adapt this to match polymorphic shapes
    if config.enable_checks.value:
      if primitive.multiple_results:
        for o, expected_aval in zip(out, out_aval):
          assert o.aval.strip_weak_type() == expected_aval.strip_weak_type(), (
              f"{primitive}: out.aval = {o.aval}; expected {expected_aval}")
      else:
        assert out.aval == out_aval, (  # type: ignore
            f"{primitive}: out.aval = {out.aval}; expected {out_aval}"
        )
    return out  # type: ignore

  def process_call(self, call_primitive: core.Primitive, fun: lu.WrappedFun,
                   tracers: Sequence[TensorFlowTracer], params):
    assert call_primitive.multiple_results
    tracers = map(self.to_tf_tracer, tracers)
    vals: Sequence[TfVal] = [t.val for t in tracers]
    avals: Sequence[core.ShapedArray] = tuple(t.aval for t in tracers)
    interpreted_fun = _interpret_subtrace(fun, avals)
    extra_name_stack = None
    with _extended_name_stack(extra_name_stack):
      vals_out = interpreted_fun.call_wrapped(*vals)
    return [TensorFlowTracer(self, v, a) for v, a in vals_out]

  def process_map(self, map_primitive, f, tracers, params):
    raise NotImplementedError("process_map")

  def process_custom_jvp_call(self, prim, fun, jvp, tracers, *, symbolic_zeros):
    # Drop the custom differentiation rule and act like a call primitive. This
    # behavior is desirable because jax2tf stages code out of the JAX system, so
    # there are no more JAX differentiation transformations to be applied.
    del jvp, symbolic_zeros  # Unused.
    return self.process_call(core.call_p, fun, tracers, {})

  def process_custom_vjp_call(self, prim, fun, fwd, bwd, tracers, out_trees,
                              symbolic_zeros):
    # Drop the custom differentiation rule and act like a call primitive. This
    # behavior is desirable because jax2tf stages code out of the JAX system, so
    # there are no more JAX differentiation transformations to be applied.
    del fwd, bwd, out_trees, symbolic_zeros  # Unused.
    return self.process_call(core.call_p, fun, tracers, {})

  def get_primitive_impl(self, p: core.Primitive) -> tuple[Callable, bool]:
    # Returns the primitive implementation and whether the implementation
    # takes abstract values (see definition of tf_impl_with_avals)
    if not _thread_local_state.enable_xla:
      try:
        return tf_impl_no_xla[p], True  # Always require avals.
      except KeyError:
        pass
    try:
      return tf_impl[p], False
    except KeyError:
      try:
        return tf_impl_with_avals[p], True
      except KeyError as err:
        msg = "TensorFlow interpretation rule for '{}' not implemented"
        raise NotImplementedError(msg.format(p)) from err

def _unexpected_primitive(p: core.Primitive, *args, **kwargs):
  assert False, f"Encountered unexpected primitive {p}"

for unexpected in [core.call_p]:
  tf_impl[unexpected] = partial(_unexpected_primitive, unexpected)

tf_impl[lax_control_flow.loops.eval_jaxpr_p] = \
    lambda *args, jaxpr: _interpret_jaxpr(
        jaxpr, *args, fresh_constant_cache=False, extra_name_stack=None)

# Primitives that are not yet implemented must be explicitly declared here.
tf_not_yet_impl = [
    "clz",
    "igamma_grad_a",
    "random_gamma_grad",
    "polygamma",
    "reduce_xor",
    "schur",
    "closed_call",
    "unreachable",
    "bint",
    "getslice",
    "full_to_shard",
    "shard_to_full",
    "pure_callback",
    "run_state",
    "for",
    "inspect_sharding",
    "io_callback",
    "broadcast_to",
    "shard_map",
    "global_array_to_host_local_array",
    "host_local_array_to_global_array",
    "call_exported",
    "zeta",
    # Not high priority?
    "after_all",
    "all_to_all",
    "check",
    "create_token",
    "custom_transpose_call",
    "custom_vmap_call",
    "infeed",
    "linear_call",
    "outfeed",
    "pmax_p",
    "pmin",
    "ppermute",
    "psum",
    "psum2",
    "pbroadcast",
    "pmax",
    "pgather",
    "reduce_scatter",
    "axis_index",
    "all_gather",
    "lu_pivots_to_permutation",
    "xla_pmap",
    "geqrf",
    "geqp3",
    "householder_product",
    "hessenberg",
    "tridiagonal",
    "eigh_jacobi",
    "platform_index",
    "assert_consumed_value",
    "consume",
    "ragged_dot",
    "cholesky_update",
    "symmetric_update",
    "from_edtype",
    "to_edtype",
    # Pallas TPU primitives
    "bitcast",
    "repeat",
    "roll",
    # temporary pending cudnn fix, see https://github.com/jax-ml/jax/pull/23740
    "bias_fwd",
    "bias_bwd",
]

tf_impl[random_internal.random_clone_p] = lambda x: x

tf_impl[ad_util.stop_gradient_p] = tf.stop_gradient


def _add(x: TfVal, y: TfVal) -> TfVal:
  return tf.raw_ops.AddV2(x=x, y=y)


tf_impl[ad_util.add_jaxvals_p] = _add
tf_impl[dispatch.device_put_p] = lambda *xs, devices=None, srcs=None, copy_semantics=None: xs
tf_impl[lax_internal.copy_p] = lambda x: x

def _shard_alike(*args: TfVal, **_):
  return tuple(args)
tf_impl[shard_alike.shard_alike_p] = _shard_alike

def _neg(x: TfVal) -> TfVal:
  if x.dtype.is_unsigned:
    signed_dtype = _UNSIGNED_TO_SIGNED_TABLE[x.dtype]
    x_signed = tf.cast(x, signed_dtype)
    res_signed = tf.math.negative(x_signed)
    return tf.cast(res_signed, x.dtype)
  else:
    return tf.math.negative(x)

tf_impl[lax.neg_p] = _neg


def _sign(x: TfVal) -> TfVal:
  if x.dtype.is_unsigned:
    # TF and XLA do not support tf.math.sign for unsigned types.
    return tf.where(
        tf.math.equal(x, 0), tf.constant(0, dtype=x.dtype),
        tf.constant(1, dtype=x.dtype))
  else:
    return tf.math.sign(x)


tf_impl[lax.sign_p] = _sign
tf_impl[lax.floor_p] = tf.math.floor
tf_impl[lax.ceil_p] = tf.math.ceil


def _round(operand, *, rounding_method,
           _in_avals: Sequence[core.ShapedArray],
           _out_aval: core.ShapedArray):
  if rounding_method is lax.RoundingMethod.AWAY_FROM_ZERO:
    # JAX uses a single HLO op Round here
    sign = _sign(operand)
    operand *= sign
    floor = tf.math.floor(operand)
    operand -= floor
    cond = tf.math.equal(operand, tf.constant(np.array(0.5), operand.dtype))
    return sign * (
        tf.where(cond, tf.constant(np.array(1), operand.dtype),
                 tf.math.round(operand)) + floor)
  else:  # rounding_method is RoundingMethod.TO_NEAREST_EVEN
    return tf.math.round(operand)

tf_impl_with_avals[lax.round_p] = _round
tf_impl[lax.nextafter_p] = tf.math.nextafter


def _population_count(x):
  orig_dtype = x.dtype
  return tf.cast(tf.raw_ops.PopulationCount(x=x), orig_dtype)


tf_impl[lax.population_count_p] = _population_count
tf_impl[lax.is_finite_p] = tf.math.is_finite


def _abs(x: TfVal) -> TfVal:
  # TF and XLA do not support tf.math.abs for unsigned types.
  return tf.math.abs(x) if not x.dtype.is_unsigned else x


tf_impl[lax.abs_p] = _abs


def _pow(x: TfVal, y: TfVal, *, _in_avals, _out_aval) -> TfVal:
  x = tf.dtypes.cast(x, _to_tf_dtype(_out_aval.dtype))
  y = tf.dtypes.cast(y, _to_tf_dtype(_out_aval.dtype))
  return tf.math.pow(x, y)


tf_impl_with_avals[lax.pow_p] = _pow


def _integer_pow(x, *, y: int, _in_avals: Sequence[core.ShapedArray],
                 _out_aval: core.ShapedArray):
  # Follows the implementation in lax._integer_pow_translation_rule
  if y == 0:
    return tf.broadcast_to(
        tf.constant(1, dtype=x.dtype, shape=()), _eval_shape(_out_aval.shape))
  is_reciprocal = y < 0
  if is_reciprocal:
    y = -y
  acc = None
  while y > 0:
    if y & 1:
      acc = x if acc is None else tf.math.multiply(acc, x)
    y >>= 1
    if y > 0:
      x = tf.math.multiply(x, x)
  return tf.math.reciprocal(acc) if is_reciprocal else acc


tf_impl_with_avals[lax.integer_pow_p] = _integer_pow
tf_impl[lax.exp_p] = tf.math.exp
tf_impl[lax_internal.exp2_p] = lambda x: \
    tf.math.exp(tf.math.multiply(tf.math.log(tf.constant(2, x.dtype)), x))
tf_impl[lax.expm1_p] = tf.math.expm1
tf_impl[lax.log_p] = tf.math.log
tf_impl[lax.log1p_p] = tf.math.log1p
tf_impl[lax.tan_p] = tf.math.tan
tf_impl[lax.tanh_p] = tf.math.tanh
tf_impl[lax.sin_p] = tf.math.sin
tf_impl[lax.sinh_p] = tf.math.sinh
tf_impl[lax.cos_p] = tf.math.cos
tf_impl[lax.cosh_p] = tf.math.cosh
tf_impl_with_avals[lax.atan_p] = _convert_jax_impl(
    lax_internal.atan_impl, multiple_results=False)

# TODO(phawkins): use tf.math.sigmoid here instead.
tf_impl_with_avals[lax.logistic_p] = _convert_jax_impl(
    lax_internal.logistic_impl, multiple_results=False)

def _atan2(y, x, **kwargs):
  if x.dtype.is_complex or y.dtype.is_complex:
    complex_component_dtype = {
      tf.complex64: tf.float32,
      tf.complex128: tf.float64
    }.get(y.dtype)
    zero = tf.constant(0, complex_component_dtype)
    one = tf.constant(1, complex_component_dtype)
    i = tf.complex(zero, one)
    return -i * tf.math.log((x + i * y)/tf.math.sqrt(x * x + y * y))
  else:
    return tf.math.atan2(y, x)


tf_impl[lax.atan2_p] = _atan2
tf_impl[lax.acosh_p] = tf.math.acosh
tf_impl[lax.atanh_p] = tf.math.atanh
tf_impl[lax.asinh_p] = tf.math.asinh
tf_impl[lax.asin_p] = tf.math.asin
tf_impl[lax.acos_p] = tf.math.acos

tf_impl[lax.sqrt_p] = tf.math.sqrt
tf_impl[lax.square_p] = tf.math.square
tf_impl[lax.rsqrt_p] = tf.math.rsqrt

def _cbrt(x):
  return tf.math.sign(x) * tf.math.pow(tf.math.abs(x), 1/3)

tf_impl[lax.cbrt_p] = _cbrt

tf_impl[lax.lgamma_p] = tf.math.lgamma
tf_impl[lax.digamma_p] = tf.math.digamma
tf_impl[lax.igamma_p] = tf.math.igamma
tf_impl[lax.igammac_p] = tf.math.igammac
tf_impl[lax.regularized_incomplete_beta_p] = tf.math.betainc
tf_impl[lax.erf_p] = tf.math.erf
tf_impl[lax.erfc_p] = tf.math.erfc
tf_impl[lax.erf_inv_p] = tf.math.erfinv
tf_impl[lax.bessel_i0e_p] = tf.math.bessel_i0e
tf_impl[lax.bessel_i1e_p] = tf.math.bessel_i1e

tf_impl[lax.complex_p] = tf.complex


def _conj(x, **kwargs):
  # The only dtypes that are allowed are: float32, float64, complex64, and
  # complex128.
  if x.dtype == tf.float32:
    return tf.cast(x, tf.complex64)
  elif x.dtype == tf.float64:
    return tf.cast(x, tf.complex128)
  else:
    return tf.math.conj(x)


tf_impl[lax.conj_p] = _conj
tf_impl[lax.real_p] = tf.math.real
tf_impl[lax.imag_p] = tf.math.imag

tf_impl[lax.add_p] = _add
tf_impl[lax.sub_p] = tf.math.subtract
tf_impl[lax.mul_p] = tf.math.multiply


def _iota(*, dtype, shape, dimension, sharding=None):
  dtype = _to_tf_dtype(dtype)
  # Some dtypes are unsupported, like uint32, so we just fall back to int32.
  # TODO(mattjj, necula): improve tf.range dtype handling
  shape_tf = _eval_shape(shape)
  vec = tf.range(tf.cast(shape_tf[dimension], tf.int32), dtype=tf.int32)
  vec_shape = [-1 if i == dimension else 1 for i in range(len(shape))]
  return tf.cast(tf.broadcast_to(tf.reshape(vec, vec_shape), shape_tf), dtype)


tf_impl[lax.iota_p] = _iota


def _div(lhs, rhs):
  if lhs.dtype.is_integer:
    quotient = tf.math.floordiv(lhs, rhs)
    select = tf.math.logical_and(
        tf.not_equal(_sign(lhs), _sign(rhs)),
        tf.not_equal(tf.math.floormod(lhs, rhs), 0))
    return tf.where(select, quotient + 1, quotient)
  else:
    return tf.math.truediv(lhs, rhs)


def _rem(lhs, rhs):
  return _sign(lhs) * tf.math.floormod(_abs(lhs), _abs(rhs))


tf_impl[lax.div_p] = _div
tf_impl[lax.rem_p] = _rem


def _minmax(x: TfVal, y: TfVal, *, is_min: bool,
            _in_avals: Sequence[core.ShapedArray],
            _out_aval: core.ShapedArray,) -> TfVal:
  # For complex numbers use lexicographic ordering, like JAX
  if dtypes.issubdtype(x.dtype.as_numpy_dtype, np.complexfloating):
    return _convert_jax_impl(
        partial(lax_internal._minmax_complex_lowering,
                lax_cmp_pick_x=lax.lt if is_min else lax.gt),
        multiple_results=False)(x, y, _in_avals=_in_avals, _out_aval=_out_aval)
  elif x.dtype.as_numpy_dtype == np.bool_:
    return (tf.math.logical_and if is_min else tf.math.logical_or)(x, y)
  else:
    return (tf.math.minimum if is_min else tf.math.maximum)(x, y)

def _minmax_scalar(x: TfVal, y: TfVal, *, is_min: bool) -> TfVal:
  # For reducers we will need min/max for scalars only. In that case we
  # can construct the AbstractValues ourselves, even in the presence of
  # shape polymorphism.
  assert len(x.shape) == 0 and len(y.shape) == 0, f"x: {x.shape}, y: {y.shape}"
  aval = core.ShapedArray((), _to_jax_dtype(x.dtype))
  return _minmax(x, y, is_min=is_min,
                 _in_avals=[aval, aval], _out_aval=aval)

tf_impl_with_avals[lax.max_p] = partial(_minmax, is_min=False)
tf_impl_with_avals[lax.min_p] = partial(_minmax, is_min=True)

# Map from TF signed types to TF unsigned types.
_SIGNED_TO_UNSIGNED_TABLE = {
    tf.int8: tf.uint8,
    tf.int16: tf.uint16,
    tf.int32: tf.uint32,
    tf.int64: tf.uint64,
}

# Map from TF unsigned types to TF signed types.
_UNSIGNED_TO_SIGNED_TABLE = {u: s for s, u in _SIGNED_TO_UNSIGNED_TABLE.items()}


# Note: Bitwise operations only yield identical results on unsigned integers!
# pylint: disable=protected-access
def _shift_right_arithmetic_raw(x, y):
  if x.dtype.is_unsigned:
    assert x.dtype == y.dtype
    orig_dtype = x.dtype
    signed_dtype = _UNSIGNED_TO_SIGNED_TABLE[orig_dtype]
    x = tf.cast(x, signed_dtype)
    y = tf.cast(y, signed_dtype)
    res = tf.bitwise.right_shift(x, y)
    return tf.cast(res, orig_dtype)
  else:
    return tf.bitwise.right_shift(x, y)


def _shift_right_arithmetic(x, y):
  # TF shift is "implementation defined" if the shift amount is negative
  # or larger or equal to the size of the value. We implement the XLA
  # semantics to return the shift by the max value (x_bits - 1).
  # TODO: it is likely better to add XlaOps for shifts
  x_bits = 8 * x.dtype.size
  clamp_y = tf.where(_shift_in_bounds(x, y), y, x_bits - 1)
  return _shift_right_arithmetic_raw(x, clamp_y)


tf_impl[lax.shift_right_arithmetic_p] = _shift_right_arithmetic


def _shift_right_logical_raw(x, y):
  if x.dtype.is_unsigned:
    return tf.bitwise.right_shift(x, y)
  else:
    assert x.dtype == y.dtype
    orig_dtype = x.dtype
    unsigned_dtype = _SIGNED_TO_UNSIGNED_TABLE[orig_dtype]
    x = tf.cast(x, unsigned_dtype)
    y = tf.cast(y, unsigned_dtype)
    res = tf.bitwise.right_shift(x, y)
    return tf.cast(res, orig_dtype)


def _shift_right_logical(x, y):
  # TF shift is "implementation defined" if the shift amount is negative
  # or larger or equal to the size of the value. We implement the XLA semantics
  # to return 0.
  # TODO: it is likely better to add XlaOps for shifts
  return tf.where(
      _shift_in_bounds(x, y), _shift_right_logical_raw(x, y), tf.zeros_like(x))


tf_impl[lax.shift_right_logical_p] = _shift_right_logical


def _shift_left(x, y):
  # TF shift is "implementation defined" if the shift amount is negative
  # or larger or equal to the size of the value. We implement the XLA semantics
  # to return 0.
  # TODO: it is likely better to add XlaOps for shifts
  return tf.where(
      _shift_in_bounds(x, y), tf.bitwise.left_shift(x, y), tf.zeros_like(x))


tf_impl[lax.shift_left_p] = _shift_left


def _shift_in_bounds(x: TfVal, y: TfVal) -> TfVal:
  # Return the TF expression for when y is within bounds (0 <= y < |x|)
  x_bits = 8 * x.dtype.size
  # TF does not have comparisons for uint16 and uint32 (despite what the
  # documentation says)
  y_comp = tf.cast(
      y, _UNSIGNED_TO_SIGNED_TABLE[y.dtype]) if y.dtype.is_unsigned else y
  y_lt_x_bits = tf.math.less(y_comp, x_bits)
  y_ge_0 = tf.math.greater_equal(y_comp, 0)
  return tf.logical_and(y_lt_x_bits, y_ge_0)


def _not(x):
  """Computes bitwise not with support for booleans.

  Numpy and JAX support bitwise not for booleans by applying a logical not!
  This means that applying bitwise_not yields an unexpected result:
    jnp.bitwise_not(jnp.array([True, False]))
    >> Array([False,  True], dtype=bool)

  if you assume that booleans are simply casted to integers.
    jnp.bitwise_not(jnp.array([True, False]).astype(np.int32)).astype(bool)
    >> Array([True,  True], dtype=bool)
  """
  if x.dtype == tf.bool:
    return tf.logical_not(x)
  else:
    return tf.bitwise.invert(x)


tf_impl[lax.not_p] = _not


def handle_boolean_args(f, argnums: Sequence[int], boolean_f=None):
  """Computes functions with some bool args and bool results using int8.

  This is needed because some TF ops do not work for bool args, e.g.,
  inequalities, min/max.

  Args:
    f: a TF callable to wrap. It will be called with non-boolean arguments.
    argnums: the positional arguments that may be booleans.
    boolean_f: [Optional] a TF callable compatible with boolean
      arguments.

  Returns: a TF callable that can take a mix of boolean positional arguments
    (in the positions specified by `argnums`) and some non-boolean positional
    arguments. If there are no boolean arguments, just calls `f`. Otherwise,
    it calls `boolean_f` if defined. Otherwise, casts the boolean
    arguments to `int8`, calls `f`, then casts the result to `bool`.
  """
  argnums = tf.nest.flatten(argnums)

  def wrapper(*args: TfVal, **kwargs):
    argnum_types = {args[i].dtype for i in argnums}
    if tf.bool not in argnum_types:
      return f(*args, **kwargs)
    else:
      # All argnums should be boolean
      assert len(argnum_types) == 1, argnum_types
      if boolean_f != None:
        return boolean_f(*args, **kwargs)
      else:
        args_cast = [(tf.cast(a, tf.int8) if i in argnums else a)
                    for i, a in enumerate(args)]
        if "_in_avals" in kwargs:

          def cast_aval(aval):
            assert aval.dtype == np.bool_
            return core.ShapedArray(aval.shape, np.int8)

          _in_avals_cast = [
              cast_aval(aval) if i in argnums else aval
              for i, aval in enumerate(kwargs["_in_avals"])
          ]
          _out_aval_cast = tf.nest.map_structure(cast_aval, kwargs["_out_aval"])
          kwargs = dict(
              kwargs, _in_avals=_in_avals_cast, _out_aval=_out_aval_cast)
        out = f(*args_cast, **kwargs)
        return tf.nest.map_structure(lambda o: tf.cast(o, tf.bool), out)

  return wrapper


tf_impl[lax.or_p] = handle_boolean_args(tf.bitwise.bitwise_or, argnums=(0, 1), boolean_f=tf.logical_or)
tf_impl[lax.and_p] = handle_boolean_args(tf.bitwise.bitwise_and, argnums=(0, 1), boolean_f=tf.logical_and)
tf_impl[lax.xor_p] = handle_boolean_args(tf.bitwise.bitwise_xor, argnums=(0, 1), boolean_f=tf.math.logical_xor)

tf_impl[lax.eq_p] = tf.math.equal
tf_impl[lax.ne_p] = tf.math.not_equal


def _total_order_adjustment(x):
  if not dtypes.issubdtype(x.dtype.as_numpy_dtype, np.inexact):
    return x
  assert dtypes.issubdtype(x.dtype.as_numpy_dtype, np.floating)
  # Switch from a floating point value to a integer value in such a way that
  # when using the integer value to compare, we get the same result for normal
  # values, and -nan is treated as the smallest value, and nan is treated as
  # the largest value.
  # If f is a float, and
  # x = bit_cast<int32>(f);
  # y = x < 0 ? int32_max - x : x;
  # then y is ordered as an int32 such that finite values have the obvious
  # order. In this scheme, -0 would be before 0, and -NaN and NaN appear at
  # the beginning and end of the ordering.
  nbits = dtypes.finfo(x.dtype.as_numpy_dtype).bits
  signed_dtype = lax_internal._INT_DTYPES[nbits]
  unsigned_dtype = lax_internal._UINT_DTYPES[nbits]

  signed = tf.bitcast(x, signed_dtype)
  sign_mask = tf.bitcast(tf.bitwise.right_shift(signed, nbits - 1), unsigned_dtype)
  sign_magnitude_mask = tf.bitcast(tf.bitwise.right_shift(sign_mask, 1), signed_dtype)
  return tf.bitwise.bitwise_xor(signed, sign_magnitude_mask)

def _total_order_equal(x, y):
  if dtypes.issubdtype(x.dtype.as_numpy_dtype, np.complexfloating):
    return _total_order_equal(tf.math.real(x), tf.math.real(y)) and _total_order_equal(tf.math.imag(x), tf.math.imag(y))
  return tf.math.equal(_total_order_adjustment(x), _total_order_adjustment(y))

tf_impl[lax.eq_to_p] = _total_order_equal

boolean_greater = lambda x,y: tf.logical_and(x, tf.logical_not(y)) # Only one combo: T,F -> T
boolean_less = lambda x,y: tf.logical_and(tf.logical_not(x), y) # Only one combo: F,T -> T
boolean_greater_or_equal = lambda x, y: tf.logical_not(boolean_less(x,y)) # All cases except F,T
boolean_less_or_equal = lambda x, y: tf.logical_not(boolean_greater(x,y)) # All cases except T,F

tf_impl[lax.gt_p] = handle_boolean_args(tf.math.greater, argnums=(0, 1), boolean_f=boolean_greater)
tf_impl[lax.lt_p] = handle_boolean_args(tf.math.less, argnums=(0, 1), boolean_f=boolean_less)
tf_impl[lax.ge_p] = handle_boolean_args(tf.math.greater_equal, argnums=(0, 1), boolean_f=boolean_greater_or_equal)
tf_impl[lax.le_p] = handle_boolean_args(tf.math.less_equal, argnums=(0, 1), boolean_f=boolean_less_or_equal)

def _total_order_cond(cond, x, y):
  return cond(_total_order_adjustment(x), _total_order_adjustment(y))

tf_impl[lax.lt_to_p] = handle_boolean_args(partial(_total_order_cond, tf.math.less), argnums=(0, 1), boolean_f=boolean_less)
tf_impl[lax.le_to_p] = handle_boolean_args(partial(_total_order_cond, tf.math.less_equal), argnums=(0, 1), boolean_f=boolean_less_or_equal)

tf_impl[lax.linalg.cholesky_p] = tf.linalg.cholesky


def _convert_element_type(operand, *, new_dtype, weak_type=False, sharding=None):
  old_dtype = operand.dtype.as_numpy_dtype
  if (dtypes.issubdtype(old_dtype, np.complexfloating) and
      not dtypes.issubdtype(new_dtype, np.complexfloating)):
    operand = tf.math.real(operand)
  if (dtypes.issubdtype(old_dtype, np.floating) and
      not (dtypes.issubdtype(new_dtype, np.floating) or dtypes.issubdtype(
          new_dtype, np.complexfloating) or new_dtype == np.bool_)):
    sign = _sign(operand)
    operand = sign * tf.math.floor(sign * operand)
  return tf.dtypes.cast(operand, _to_tf_dtype(new_dtype))


tf_impl[lax.convert_element_type_p] = _convert_element_type


def _bitcast_convert_type(operand, new_dtype):
  if operand.dtype == new_dtype:
    return operand
  return tf.bitcast(operand, _to_tf_dtype(new_dtype))


tf_impl[lax.bitcast_convert_type_p] = _bitcast_convert_type


def _clamp(minval, operand, maxval, *, _in_avals, _out_aval):
  # The below permits mirroring the behavior of JAX when maxval < minval
  op_shape_tf_val = _eval_shape(_in_avals[1].shape, _in_avals[1].dtype)
  maxval = tf.broadcast_to(maxval, op_shape_tf_val)
  minval = tf.math.minimum(tf.broadcast_to(minval, op_shape_tf_val), maxval)
  return tf.clip_by_value(operand, minval, maxval)


tf_impl_with_avals[lax.clamp_p] = _clamp


def _concatenate(*operands, dimension):
  return tf.concat(operands, axis=tf.cast(dimension, tf.int32))


tf_impl[lax.concatenate_p] = _concatenate


def _split(operand, *, sizes, axis):
  return tf.split(operand, _eval_shape(sizes), axis=axis)

tf_impl[lax.split_p] = _split


def _conv_general_dimension_numbers_proto(dimension_numbers):
  """Converts a ConvDimensionNumbers to an XLA ConvolutionDimensionNumbers."""
  assert isinstance(dimension_numbers, lax.ConvDimensionNumbers)
  lhs_spec, rhs_spec, out_spec = dimension_numbers
  proto = xla_data_pb2.ConvolutionDimensionNumbers()
  proto.input_batch_dimension = lhs_spec[0]
  proto.input_feature_dimension = lhs_spec[1]
  proto.output_batch_dimension = out_spec[0]
  proto.output_feature_dimension = out_spec[1]
  proto.kernel_output_feature_dimension = rhs_spec[0]
  proto.kernel_input_feature_dimension = rhs_spec[1]
  proto.input_spatial_dimensions.extend(lhs_spec[2:])
  proto.kernel_spatial_dimensions.extend(rhs_spec[2:])
  proto.output_spatial_dimensions.extend(out_spec[2:])
  return proto


def _precision_config_proto(precision: None | (tuple[PrecisionType,
                                                      PrecisionType])):
  """Convert an integer to an XLA.PrecisionConfig."""
  if precision is None:
    return None

  proto = xla_data_pb2.PrecisionConfig()
  proto.operand_precision.append(precision[0].value)
  proto.operand_precision.append(precision[1].value)
  return proto


def _conv_general_dilated(lhs, rhs, *,
                          window_strides, padding, lhs_dilation,
                          rhs_dilation,
                          dimension_numbers: lax.ConvDimensionNumbers,
                          feature_group_count: int,
                          batch_group_count: int,
                          precision: tuple[PrecisionType, PrecisionType] | None,
                          preferred_element_type: DType | None,
                          _in_avals: Sequence[core.ShapedArray],
                          _out_aval: core.ShapedArray):
  """Implementation of lax.conv_general_dilated_p using XlaConv."""
  out_tf_shape = _aval_to_tf_shape(_out_aval)
  dnums_proto = _conv_general_dimension_numbers_proto(dimension_numbers)
  precision_config_proto = _precision_config_proto(precision)

  def gen_conv(lhs, rhs, preferred_element_type: DType | None):
    tf_version = tuple(int(v) for v in tf.__version__.split(".")[:2])
    if tf_version >= (2, 8):
      # TODO(necula): remove when 2.8.0 is the stable TF version (and supports
      # batch_group_count.
      padding_tf = [_eval_shape(p) for p in padding]
      out = tfxla.conv(
          lhs, rhs, window_strides, padding_tf, lhs_dilation, rhs_dilation,
          dnums_proto,
          feature_group_count=feature_group_count,
          batch_group_count=batch_group_count,
          precision_config=precision_config_proto,
          preferred_element_type=preferred_element_type,
          use_v2=True)
    else:
      if batch_group_count != 1:
        raise ValueError(
            "The batch_group_count parameter for conv requires TF version "
            "at least 2.8.0. You may want to use tf-nightly.")
      padding_tf = [_eval_shape(p) for p in padding]
      out = tfxla.conv(
          lhs, rhs, window_strides, padding_tf, lhs_dilation, rhs_dilation,
          dnums_proto,
          feature_group_count=feature_group_count,
          precision_config=precision_config_proto,
          preferred_element_type=preferred_element_type,
          use_v2=True)
    # TODO: implement shape inference for XlaConv
    out = _ensure_tf_shape_if_dynamic(out, out_tf_shape)
    if _WRAP_JAX_JIT_WITH_TF_FUNCTION:
      out = tf.stop_gradient(out)  # See #7839
    return out

  # Follow the lowering for complex convolutions from
  # lax._conv_general_dilated_translation. We can use the same conversion on all
  # platforms because on XLA:TPU the compiler does the same as a rewrite.
  preferred_float_et: Any | None
  if np.issubdtype(_in_avals[0].dtype, np.complexfloating):
    if preferred_element_type is not None:
      # Convert complex dtype to types used for real and imaginary parts
      assert np.issubdtype(preferred_element_type, np.complexfloating)
      preferred_float_et = (
          np.float64 if preferred_element_type == np.complex128 else np.float32)
    else:
      preferred_float_et = None
    lhs_real, lhs_imag = tf.math.real(lhs), tf.math.imag(lhs)
    rhs_real, rhs_imag = tf.math.real(rhs), tf.math.imag(rhs)
    k1 = gen_conv(_add(lhs_real, lhs_imag), rhs_real, preferred_float_et)
    k2 = gen_conv(lhs_real, tf.math.subtract(rhs_imag, rhs_real),
                  preferred_float_et)
    k3 = gen_conv(lhs_imag, _add(rhs_real, rhs_imag), preferred_float_et)
    return tf.complex(tf.math.subtract(k1, k3), _add(k1, k2))
  else:
    return gen_conv(lhs, rhs, preferred_element_type)


tf_impl_with_avals[lax.conv_general_dilated_p] = _conv_general_dilated


def _dot_general(lhs, rhs, *, dimension_numbers,
                 precision: lax_internal.CanonicalPrecision,
                 preferred_element_type: DType | None,
                 out_sharding=None,
                 _in_avals: Sequence[core.ShapedArray],
                 _out_aval: core.ShapedArray):
  """Implementation of lax.dot_general_p in terms of tf.linalg.einsum."""
  # TODO(b/293247337): we ought to turn on this safety check, but this leads to
  # failures. Since we are going to turn on native serialization soon, wait
  # until then to turn on this check.
  # lhs_aval, rhs_aval = _in_avals
  # if lhs_aval.dtype != rhs_aval.dtype:
  #   # There are multiple kinds of errors: handling jnp.bfloat16 in xla.py and
  #   # returning different result dtype than JAX expects for various combinations
  #   # of types. We ought to implement the same workarounds as in the
  #   # native dot_general lowering rules, but this is not a high priority now
  #   # that we deprecate non-native serialization.
  #   raise NotImplementedError(
  #     "dot_general with different lhs_dtype and rhs_dtype is not supported "
  #     "in non-native serialization")

  if precision == lax.DotAlgorithmPreset.DEFAULT:
    precision = None
  if precision is not None and not (isinstance(precision, tuple) and
                                    len(precision) == 2):
    raise NotImplementedError(
        f"Unsupported precision in dot_general: {precision}")

  lhs, rhs, convert_result = _dot_general_convert_to_common_dtype(
    lhs, _in_avals[0], rhs, _in_avals[1], _out_aval)

  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  dnums_proto = xla_data_pb2.DotDimensionNumbers()
  dnums_proto.lhs_contracting_dimensions.extend(lhs_contracting)
  dnums_proto.rhs_contracting_dimensions.extend(rhs_contracting)
  dnums_proto.lhs_batch_dimensions.extend(lhs_batch)
  dnums_proto.rhs_batch_dimensions.extend(rhs_batch)
  precision_config_proto = _precision_config_proto(precision)  # type: ignore
  res = tfxla.dot_general(
      lhs,
      rhs,
      dnums_proto,
      precision_config_proto,
      preferred_element_type=preferred_element_type,
      use_v2=True)
  res = convert_result(res)
  if _WRAP_JAX_JIT_WITH_TF_FUNCTION:
    res = tf.stop_gradient(res)  # See #7839
  return res


tf_impl_with_avals[lax.dot_general_p] = _dot_general

def _dot_general_convert_to_common_dtype(
  lhs: TfVal, lhs_aval: core.ShapedArray,
  rhs: TfVal, rhs_aval: core.ShapedArray,
  out_aval: core.ShapedArray) -> tuple[TfVal, TfVal, Callable[[TfVal], TfVal]]:
  # Returns the converted lhs, rhs, and the converter for the result.
  # tfxla.dot_general does not handle arguments of different types.
  # We convert the arguments and the result.
  # Use native serialization for a more JAX-native behavior.
  if lhs_aval.dtype != rhs_aval.dtype:

    common_dtype = dtypes.result_type(lhs_aval, rhs_aval)
    if common_dtype != lhs_aval.dtype:
      lhs = _convert_element_type(lhs, new_dtype=common_dtype)
    if common_dtype != rhs_aval.dtype:
      rhs = _convert_element_type(rhs, new_dtype=common_dtype)
    convert_result = lambda res: _convert_element_type(res, new_dtype=out_aval.dtype)
  else:
    convert_result = lambda res: res
  return (lhs, rhs, convert_result)

def _broadcast_in_dim(operand, *, shape, broadcast_dimensions, sharding=None,
                      _in_avals: Sequence[core.ShapedArray],
                      _out_aval: core.ShapedArray):
  # for i in range(len(operand.shape)):
  #   result.shape[bcast_dims[i]] <- operand.shape[i]
  # bcast_dims must be strictly increasing.
  # len(bcast_dims) == len(operand.shape)
  op_shape = _in_avals[0].shape
  dtype = _in_avals[0].dtype
  add_1s_shape = [1] * len(shape)
  for i, broadcast_dim_i in enumerate(broadcast_dimensions):
    add_1s_shape[broadcast_dim_i] = op_shape[i]
  with_1s = tf.reshape(operand, _eval_shape(add_1s_shape, dtype=dtype))
  return tf.broadcast_to(with_1s, _eval_shape(shape, dtype=dtype))


tf_impl_with_avals[lax.broadcast_in_dim_p] = _broadcast_in_dim


def _empty(*, dtype):
  if dtypes.issubdtype(dtype, dtypes.extended):
    raise NotImplementedError  # TODO(frostig,mattjj): jax2tf handlers
  return tf.constant(np.array(0, dtype=dtype))


tf_impl[lax_internal.empty_p] = _empty


def _reshape(operand, *, new_sizes, dimensions, sharding, _in_avals, _out_aval):
  if dimensions is None:
    dimensions = tf.range(tf.rank(operand))
  new_sizes_tf = _eval_shape(new_sizes, _in_avals[0].dtype)
  return tf.reshape(tf.transpose(operand, dimensions), new_sizes_tf)


tf_impl_with_avals[lax.reshape_p] = _reshape


def _squeeze(operand, *, dimensions, _in_avals, _out_aval):
  op_aval = _jax_physical_aval(_in_avals[0])
  op_shape = op_aval.shape
  new_shape = tuple(d for i, d in enumerate(op_shape) if i not in dimensions)
  new_shape_tf = _eval_shape(new_shape, op_aval.dtype)
  return tf.reshape(operand, new_shape_tf)


tf_impl_with_avals[lax.squeeze_p] = _squeeze


def _pad(operand, padding_value, *, padding_config,
         _in_avals: Sequence[core.ShapedArray],
         _out_aval: core.ShapedArray):
  low, high, interior = util.unzip3(map(_eval_shape, padding_config))  # type: ignore
  out = tfxla.pad(operand, padding_value, low, high, interior)
  # TODO: implement shape inference for XlaPad (when some padding_config is constant)
  out = _ensure_tf_shape_if_dynamic(out, _aval_to_tf_shape(_out_aval))
  if _WRAP_JAX_JIT_WITH_TF_FUNCTION:
    out = tf.stop_gradient(out)  # See #7839
  return out


tf_impl_with_avals[lax.pad_p] = _pad


def _rev(operand, *, dimensions):
  return tf.reverse(operand, dimensions)


tf_impl[lax.rev_p] = _rev


def _where(which, *cases):
  if which.dtype == tf.bool:
    assert len(cases) <= 2
    return cases if len(cases) == 1 else tf.where(which, cases[1], cases[0])

  def _select(offset, cases):
    assert len(cases) > 0
    if len(cases) == 1:
      return cases[0]
    mid = len(cases) // 2
    return tf.where(tf.less(which, offset + mid),
                    _select(offset, cases[:mid]),
                    _select(mid, cases[mid:]))

  return _select(0, cases)


tf_impl[lax.select_n_p] = _where


def _transpose(operand, *, permutation):
  return tf.transpose(operand, perm=permutation)


tf_impl[lax.transpose_p] = _transpose

axes_to_axis = lambda func: lambda operand, axes: func(operand, axis=axes)

# reduce_sum and reduce_prod are not supported for bool
tf_impl[lax.reduce_sum_p] = axes_to_axis(tf.reduce_sum)
tf_impl[lax.reduce_prod_p] = axes_to_axis(tf.reduce_prod)
tf_impl[lax.reduce_max_p] = handle_boolean_args(
  axes_to_axis(tf.reduce_max), argnums=[0],
  boolean_f=axes_to_axis(tf.reduce_any)) # Max is T if any one is T
tf_impl[lax.reduce_min_p] = handle_boolean_args(
  axes_to_axis(tf.reduce_min), argnums=[0],
  boolean_f=axes_to_axis(tf.reduce_all)) # Min is F if not all are T
tf_impl[lax.reduce_or_p] = axes_to_axis(tf.reduce_any)
tf_impl[lax.reduce_and_p] = axes_to_axis(tf.reduce_all)


def _argminmax(is_min: bool, operand: TfVal, axes: Sequence[int],
               index_dtype: DType,
               _in_avals: Sequence[core.ShapedArray],
               _out_aval: core.ShapedArray):
  # Follow the JAX implementation, using a XlaReduce with a custom comparator
  if is_min:
    extra_name_stack = "argmin"
    value_comparator = lax.lt
    get_identity = lax_internal._get_min_identity
  else:
    extra_name_stack = "argmax"
    value_comparator = lax.gt
    get_identity = lax_internal._get_max_identity

  res = _convert_jax_impl(
      partial(lax_internal._compute_argminmax, value_comparator, get_identity),
      multiple_results=False,
      extra_name_stack=extra_name_stack)(
          operand,
          index_dtype=index_dtype,
          axes=axes,
          _in_avals=_in_avals,
          _out_aval=_out_aval)
  return res


tf_impl_with_avals[lax.argmin_p] = partial(_argminmax, True)
tf_impl_with_avals[lax.argmax_p] = partial(_argminmax, False)


_add_fn = tf.function(_add, autograph=False)
_ge_fn = tf.function(tf.math.greater_equal, autograph=False)


def _select_and_gather_add(
    tangents: TfVal, operand: TfVal, select_prim: core.Primitive,
    window_dimensions: Sequence[int], window_strides: Sequence[int],
    base_dilation: Sequence[int], window_dilation: Sequence[int],
    padding: Sequence[tuple[int, int]], _in_avals: Sequence[core.ShapedArray],
    _out_aval: core.ShapedArray):
  # Note: this function follows the pattern in
  # jax.lax._select_and_gather_add_translation.
  dtype = operand.dtype
  nbits = dtypes.finfo(dtype.as_numpy_dtype).bits

  # Specializing the function for 64 bits. Only up to 32 bits are supported on TPU,
  # we thus intend to let the code throw a different exception on this platform.
  max_bits = 64

  assert nbits <= max_bits
  double_word_reduction = nbits * 2 <= max_bits

  const = lambda dtype, x: tf.constant(np.array(x), dtype)

  if double_word_reduction:
    word_dtype = lax_internal._UINT_DTYPES[nbits]
    double_word_dtype = lax_internal._UINT_DTYPES[nbits * 2]

    # Packs two values into a tuple.
    def pack(a, b):
      a = _bitcast_convert_type(a, word_dtype)
      b = _bitcast_convert_type(b, word_dtype)
      a = _convert_element_type(a, new_dtype=double_word_dtype)
      b = _convert_element_type(b, new_dtype=double_word_dtype)
      a = tf.bitwise.left_shift(a, const(double_word_dtype, nbits))
      return tf.bitwise.bitwise_or(a, b)

    # Unpacks the first element of a tuple.
    def fst(t):
      assert t.dtype == double_word_dtype
      st = _shift_right_logical(t, const(double_word_dtype, nbits))
      return _bitcast_convert_type(
          _convert_element_type(st, new_dtype=word_dtype), dtype)

    # Unpacks the second element of a tuple.
    def snd(t):
      return _bitcast_convert_type(
          _convert_element_type(t, new_dtype=word_dtype), dtype)

  else:
    raise NotImplementedError(
        f"TODO: need to pack {nbits * 2} bits but this platform can only go up to {max_bits} bits."
    )

  assert select_prim is lax.ge_p or select_prim is lax.le_p, select_prim

  def reducer(x, y):
    which = tf_impl[select_prim]
    return tf_impl[lax.select_n_p](which(fst(x), fst(y)), y, x)

  init = -np.inf if select_prim is lax.ge_p else np.inf
  init_identity = lambda x: pack(const(dtype, init), const(dtype, 0))

  out = _specialized_reduce_window(
      reducer,
      init_identity,
      pack(operand, tangents),
      window_dimensions=window_dimensions,
      window_strides=window_strides,
      padding=padding,
      base_dilation=base_dilation,
      window_dilation=window_dilation,
      _in_avals=_in_avals,
      _out_aval=_out_aval)

  return snd(out)


tf_impl_with_avals[lax.select_and_gather_add_p] = _select_and_gather_add


def _common_reduce_window(operand, init_val, reducer, window_dimensions,
                          window_strides, padding, base_dilation,
                          window_dilation, _in_avals, _out_aval):
  o_spec = tf.TensorSpec((), dtype=operand.dtype)
  reducer_fn = tf.function(
      reducer, autograph=False).get_concrete_function(o_spec, o_spec)

  if not isinstance(init_val, (tf.Tensor, tf.Variable)):
    init_val = tf.constant(init_val, operand.dtype)
  window_dimensions_tf = _eval_shape(window_dimensions)
  window_strides_tf = _eval_shape(window_strides)
  window_dilation_tf = _eval_shape(window_dilation)
  base_dilation_tf = _eval_shape(base_dilation)
  padding_tf = [_eval_shape(p) for p in padding]
  out = tfxla.reduce_window(
      operand,
      init_val,
      reducer_fn,
      window_dimensions_tf,
      window_strides_tf,
      base_dilations=base_dilation_tf,
      window_dilations=window_dilation_tf,
      padding=padding_tf)
  # TODO: implement shape inference for XlaReduceWindow
  out = _ensure_tf_shape_if_dynamic(out, _aval_to_tf_shape(_out_aval))
  if _WRAP_JAX_JIT_WITH_TF_FUNCTION:
    out = tf.stop_gradient(out)  # See #7839
  return out


def _reduce_window(*args, jaxpr, consts, window_dimensions,
                   window_strides, padding, base_dilation, window_dilation,
                   _in_avals, _out_aval):
  """TensorFlow implementation of reduce_window.

  Args:
    operands: N dimensional arrays containing elements of type T
    init_values: starting values of the reduction
    jaxpr: the jaxpr corresponding to the reduction function
    consts: the constants associated with jaxpr.
    window_dimensions: array of integers for window dimension values
    window_strides: array of integers for window stride values
    padding: array of pairs of integers for padding values
    base_dilation: array of integers for base dilation values
    window_dilation: array of integers for window dilation values

  Returns:
    The reduced operand.
  """
  assert len(consts) == 0, "Reduction computation cannot have constants"
  operands, init_values = util.split_list(args, [len(args) // 2])

  if len(operands) != 1:
    raise NotImplementedError("jax2tf does not support variadic reduce_window")

  def reducer(arg1: TfVal, arg2: TfVal) -> TfVal:
    closed_jaxpr = core.ClosedJaxpr(jaxpr, consts)
    res, = _interpret_jaxpr(closed_jaxpr, arg1, arg2, extra_name_stack=None)
    return res

  return (_common_reduce_window(operands[0], init_values[0], reducer,
                                window_dimensions, window_strides, padding,
                                base_dilation, window_dilation, _in_avals,
                                _out_aval[0]),)


def _specialized_reduce_window(reducer,
                               identity,
                               operand,
                               *,
                               window_dimensions,
                               window_strides,
                               padding,
                               base_dilation,
                               window_dilation,
                               _in_avals,
                               _out_aval,
                               name=None):
  """Wraps the TensorFlow reduce window operation based on a reducer and an

  identity function defining the initial value of the reduction depending on
  the dtype of the operand.

  Args:
    reducer: reduction function of type TfVal -> TfVal -> TfVal
    identity: function that takes a TensorFlow dtype as a parameter and returns
      the starting value of the reduction.
    operand: N dimensional array containing elements of type T
    window_dimensions: array of integers for window dimension values
    window_strides: array of integers for window stride values
    padding: array of pairs of integers for padding values
    base_dilation: array of integers for base dilation values
    window_dilation: array of integers for window dilation values
    name: the name of the specialized reduce window primitive for which this
      conversion function is called. This information may help to choose a
      different conversion path (optional)

  Returns:
    The reduced operand.
  """
  return _common_reduce_window(operand, identity(operand.dtype), reducer,
                               window_dimensions, window_strides, padding,
                               base_dilation, window_dilation, _in_avals,
                               _out_aval)


def _get_max_identity(tf_dtype):
  numpy_tf_dtype = tf_dtype.as_numpy_dtype
  if tf_dtype == tf.bfloat16 or dtypes.issubdtype(numpy_tf_dtype, np.inexact):
    return numpy_tf_dtype(-np.inf)
  elif dtypes.issubdtype(numpy_tf_dtype, np.integer):
    return dtypes.iinfo(numpy_tf_dtype).min
  else:
    assert dtypes.issubdtype(
        numpy_tf_dtype, np.bool_), (f"{tf_dtype} has no defined max identity")
    return False


def _get_min_identity(tf_dtype):
  numpy_tf_dtype = tf_dtype.as_numpy_dtype
  if tf_dtype == tf.bfloat16 or dtypes.issubdtype(numpy_tf_dtype, np.inexact):
    return numpy_tf_dtype(np.inf)
  elif dtypes.issubdtype(numpy_tf_dtype, np.integer):
    return dtypes.iinfo(numpy_tf_dtype).max
  else:
    assert dtypes.issubdtype(
        numpy_tf_dtype, np.bool_), (f"{tf_dtype} has no defined min identity")
    return True


# pylint: disable=protected-access
tf_impl_with_avals[lax.reduce_window_sum_p] = (
    partial(_specialized_reduce_window, _add, lambda x: 0,
            name="reduce_window_sum"))
tf_impl_with_avals[lax.reduce_window_min_p] = (
    partial(_specialized_reduce_window,
            partial(_minmax_scalar, is_min=True),
            _get_min_identity,
            name="reduce_window_min"))
tf_impl_with_avals[lax.reduce_window_max_p] = (
    partial(_specialized_reduce_window,
            partial(_minmax_scalar, is_min=False),
            _get_max_identity,
            name="reduce_window_max"))
tf_impl_with_avals[lax.reduce_window_p] = _reduce_window
# pylint: enable=protected-access

def _reduce(*operands: TfVal,
            computation: Callable,
            jaxpr: core.ClosedJaxpr,
            dimensions: Sequence[int],
            _in_avals: Sequence[core.ShapedArray],
            _out_aval: core.ShapedArray) -> Sequence[TfVal]:
  del computation
  assert not jaxpr.consts
  assert len(operands) % 2 == 0
  # operands: op1, op2, ..., init_val1, init_val2, ...
  # reducer takes op1[i], op2[i], ..., init_val1, init_val2, ...
  nr_operands = len(operands) // 2
  init_vals = operands[nr_operands:]
  operands = operands[0:nr_operands]

  reducer_arg_spec = tuple([tf.TensorSpec((), op.dtype) for op in init_vals] * 2)

  def reducer_computation(*args: TfVal) -> TfVal:
    res = _interpret_jaxpr(jaxpr, *args, extra_name_stack=None)
    return res

  xla_reducer_computation = (
      tf.function(reducer_computation,
                  autograph=False).get_concrete_function(*reducer_arg_spec))

  outs = tfxla.variadic_reduce(operands, init_vals,
                               dimensions_to_reduce=dimensions,
                               reducer=xla_reducer_computation)
  if _WRAP_JAX_JIT_WITH_TF_FUNCTION:
    outs = tuple(tf.stop_gradient(out) for out in outs)  # See #7839
  return outs

tf_impl_with_avals[lax.reduce_p] = _reduce


# We use lax.cumred_reduce_window_impl to convert cummax,
# cummin, cumsum and cumprod. This is efficient on TPU, but the complexity is
# O(n^2) on other backends. This may be implemented using associative_scan
# instead to favor different backends.
def _cumred(lax_reduce_fn: Callable,
            lax_reduce_window_fn: Callable,
            extra_name_stack: str):
  associative_scan = partial(lax_control_flow.associative_scan, lax_reduce_fn)
  reduce_window = partial(
      lax_control_flow.cumred_reduce_window_impl, lax_reduce_window_fn
  )

  def _call_impl(*args, **kwargs):
    # Vary which implementation to use when cumulation is called. This cannot be
    # done during import time because the caller may later use a python context
    # to switch the implementation to use.
    associative = config.jax2tf_associative_scan_reductions.value
    return (associative_scan if associative else reduce_window)(*args, **kwargs)

  return _convert_jax_impl(
      _call_impl, multiple_results=False, extra_name_stack=extra_name_stack
  )


tf_impl_with_avals[lax.cummax_p] = _cumred(
    lax_reduce_window_fn=lax_windowed_reductions._reduce_window_max,
    lax_reduce_fn=lax.max,
    extra_name_stack="cummax")
tf_impl_with_avals[lax.cummin_p] = _cumred(
    lax_reduce_window_fn=lax_windowed_reductions._reduce_window_min,
    lax_reduce_fn=lax.min,
    extra_name_stack="cummin")
tf_impl_with_avals[lax.cumlogsumexp_p] = _cumred(
    lax_reduce_window_fn=lax_windowed_reductions._reduce_window_logaddexp,
    lax_reduce_fn=logaddexp,
    extra_name_stack="cumlogsumexp")
tf_impl_with_avals[lax.cumsum_p] = _cumred(
    lax_reduce_window_fn=lax_windowed_reductions._reduce_window_sum,
    lax_reduce_fn=lax.add,
    extra_name_stack="cumsum")
tf_impl_with_avals[lax.cumprod_p] = _cumred(
    lax_reduce_window_fn=lax_windowed_reductions._reduce_window_prod,
    lax_reduce_fn=lax.mul,
    extra_name_stack="cumprod")


def _select_and_scatter(operand, source, init_value, select_jaxpr,
                        select_consts, scatter_jaxpr, scatter_consts,
                        window_dimensions, window_strides, padding):
  raise NotImplementedError("TODO: jax2tf can not convert _select_and_scatter")


tf_impl[lax.select_and_scatter_p] = _select_and_scatter


@partial(handle_boolean_args, argnums=(0, 1))
def _select_and_scatter_add(source, operand, *, select_prim, window_dimensions,
                            window_strides, padding, _in_avals, _out_aval):
  init_value = tf.zeros((), operand.dtype)
  select_fn = (
      tf.function(tf_impl[select_prim], autograph=False).get_concrete_function(
          init_value, init_value))
  scatter_fn = _add_fn.get_concrete_function(init_value, init_value)
  out = tfxla.select_and_scatter(operand, window_dimensions, window_strides,
                                 padding, source, init_value, select_fn,
                                 scatter_fn)
  out = _ensure_tf_shape_if_dynamic(out, _aval_to_tf_shape(_out_aval))
  if _WRAP_JAX_JIT_WITH_TF_FUNCTION:
    out = tf.stop_gradient(out)  # See #7839
  return out


tf_impl_with_avals[lax.select_and_scatter_add_p] = _select_and_scatter_add


def _random_seed_impl(seeds: TfVal, *, impl, _in_avals, _out_aval):

  def impl_wrapper(seeds: TfVal, *, impl):
    return prng.random_seed_impl_base(seeds, impl=impl)

  converted_impl = _convert_jax_impl(
      impl_wrapper, multiple_results=False, with_physical_avals=True,
      extra_name_stack="random_seed")
  return converted_impl(
      seeds, impl=impl, _in_avals=_in_avals, _out_aval=_out_aval)

tf_impl_with_avals[prng.random_seed_p] = _random_seed_impl


def _random_split_impl(keys: TfVal, *, shape, _in_avals, _out_aval):
  keys_aval, = _in_avals

  def impl_wrapper(keys: TfVal, *, shape):
    return prng.random_split_impl_base(
        keys_aval.dtype._impl, keys, keys_aval.ndim, shape=shape)

  converted_impl = _convert_jax_impl(
      impl_wrapper, multiple_results=False, with_physical_avals=True,
      extra_name_stack="random_split")
  return converted_impl(
      keys, shape=shape, _in_avals=_in_avals, _out_aval=_out_aval)

tf_impl_with_avals[prng.random_split_p] = _random_split_impl


def _random_fold_in_impl(keys: TfVal, msgs: TfVal, *, _in_avals, _out_aval):
  keys_aval, _ = _in_avals

  def impl_wrapper(keys: TfVal, msgs: TfVal):
    return prng.random_fold_in_impl_base(
        keys_aval.dtype._impl, keys, msgs, keys_aval.shape)

  converted_impl = _convert_jax_impl(
      impl_wrapper, multiple_results=False, with_physical_avals=True,
      extra_name_stack="random_fold_in")
  return converted_impl(
      keys, msgs, _in_avals=_in_avals, _out_aval=_out_aval)

tf_impl_with_avals[prng.random_fold_in_p] = _random_fold_in_impl


def _random_bits_impl(keys: TfVal, *, bit_width, shape, _in_avals, _out_aval):
  keys_aval, = _in_avals

  def impl_wrapper(keys: TfVal, **kwargs):
    return prng.random_bits_impl_base(
        keys_aval.dtype._impl, keys, keys_aval.ndim,
        bit_width=bit_width, shape=shape)

  converted_impl = _convert_jax_impl(
      impl_wrapper, multiple_results=False, with_physical_avals=True,
      extra_name_stack="random_bits")
  return converted_impl(keys, bit_width=bit_width, shape=shape,
                        _in_avals=_in_avals, _out_aval=_out_aval)

tf_impl_with_avals[prng.random_bits_p] = _random_bits_impl


def _random_wrap_impl(base_arr: TfVal, *, impl, _in_avals, _out_aval):
  return base_arr

tf_impl_with_avals[prng.random_wrap_p] = _random_wrap_impl


def _random_unwrap_impl(keys: TfVal, *, _in_avals, _out_aval):
  return keys

tf_impl_with_avals[prng.random_unwrap_p] = _random_unwrap_impl


def _threefry2x32_jax_impl(*args: TfVal, _in_avals, _out_aval):
  res = _convert_jax_impl(
      partial(prng._threefry2x32_lowering, use_rolled_loops=False),
      multiple_results=True, extra_name_stack="threefry")(
          *args, _in_avals=_in_avals, _out_aval=_out_aval)
  return res


tf_impl_with_avals[prng.threefry2x32_p] = _threefry2x32_jax_impl

# Use the vmap implementation, otherwise on TPU the performance is really bad
# With use_vmap=True on, we get about the same performance for JAX and jax2tf.
tf_impl_with_avals[random.random_gamma_p] = _convert_jax_impl(
    partial(random_internal._gamma_impl, use_vmap=True),
    multiple_results=False, extra_name_stack="random_gamma")


def _rng_bit_generator(key: TfVal, *, shape, dtype, algorithm) -> Sequence[TfVal]:
  is_uint32_key = key.dtype == _to_tf_dtype(jnp.uint32)
  if is_uint32_key:
    key = tf.reshape(key, (2, 2))
    key = tfxla.bitcast_convert_type(key, _to_tf_dtype(jnp.uint64))
  shape_tf = _eval_shape(shape)
  # JAX uses XLA algorithm enums; tfxla uses tf.random.Algorithm
  if algorithm == lax.RandomAlgorithm.RNG_THREE_FRY:
    algorithm_tf = tf.random.Algorithm.THREEFRY
  elif algorithm == lax.RandomAlgorithm.RNG_PHILOX:
    algorithm_tf = tf.random.Algorithm.PHILOX
  elif algorithm == lax.RandomAlgorithm.RNG_DEFAULT:
    algorithm_tf = tf.random.Algorithm.AUTO_SELECT
  else:
    assert False
  (new_key, res) = tfxla.rng_bit_generator(algorithm_tf.value, key, shape_tf,
                                           dtype=_to_tf_dtype(dtype))
  if is_uint32_key:
    new_key = tfxla.bitcast_convert_type(new_key, _to_tf_dtype(jnp.uint32))
    new_key = tf.reshape(new_key, (4,))
  if _WRAP_JAX_JIT_WITH_TF_FUNCTION:
    # See #7839
    new_key = tf.stop_gradient(new_key)
    res = tf.stop_gradient(res)
  return new_key, res


tf_impl[lax.rng_bit_generator_p] = _rng_bit_generator


def _rng_uniform(minval: TfVal, maxval: TfVal, *, shape) -> TfVal:
  shape_tf = _eval_shape(shape)
  return tf.random.uniform(shape_tf, minval=minval, maxval=maxval, dtype=minval.dtype)

tf_impl[lax.rng_uniform_p] = _rng_uniform


def _iota_2x32_shape(*, shape):
  def _add(x, y):
    return x + y
  def _mul(x, y):
    if not core.is_constant_dim(x):
      x = tf.cast(_eval_shape((x,))[0], y.dtype)
      x = tf.broadcast_to(x, tf.shape(y))
    return x * y
  def _cast32(xs): return tf.cast(xs, _to_tf_dtype(jnp.uint32))
  iotas = [_iota(dtype=jnp.uint64, shape=shape, dimension=dimension)
           for dimension in range(len(shape))]
  counts = prng.bcast_iotas_to_reshaped_iota(_add, _mul, shape, iotas)
  counts_lo = _cast32(counts)
  counts_hi = _cast32(tf.bitwise.right_shift(counts, 32))
  return counts_hi, counts_lo

tf_impl[prng.iota_2x32_shape_p] = _iota_2x32_shape


def _gather_dimensions_proto(indices_shape, dimension_numbers):
  proto = xla_data_pb2.GatherDimensionNumbers()
  proto.offset_dims.extend(dimension_numbers.offset_dims)
  proto.collapsed_slice_dims.extend(dimension_numbers.collapsed_slice_dims)
  proto.start_index_map.extend(dimension_numbers.start_index_map)
  proto.operand_batching_dims.extend(dimension_numbers.operand_batching_dims)
  proto.start_indices_batching_dims.extend(
      dimension_numbers.start_indices_batching_dims)
  assert indices_shape
  proto.index_vector_dim = len(indices_shape) - 1
  return proto


def _maybe_cast_to_int64(x: TfVal) -> TfVal:
  if x.dtype != tf.int32 and x.dtype != tf.int64:
    return tf.cast(x, tf.int64)
  return x


@partial(handle_boolean_args, argnums=[0])
def _gather(operand, start_indices, *, dimension_numbers, slice_sizes: core.Shape,
            indices_are_sorted, unique_indices, mode, fill_value,
            _in_avals: Sequence[core.ShapedArray],
            _out_aval: core.ShapedArray):
  """Tensorflow implementation of gather."""
  if mode == lax.GatherScatterMode.FILL_OR_DROP:
    gather_fill_fn = _convert_jax_impl(lax_slicing._gather_fill,
                                       multiple_results=False)
    return gather_fill_fn(
        operand, start_indices, dimension_numbers=dimension_numbers,
        slice_sizes=slice_sizes, unique_indices=unique_indices,
        indices_are_sorted=indices_are_sorted, fill_value=fill_value,
        output_shape=_out_aval.shape, _in_avals=_in_avals, _out_aval=_out_aval)

  operand_aval = _in_avals[0]
  start_indices = _maybe_cast_to_int64(start_indices)
  if dtypes.issubdtype(operand_aval.dtype, dtypes.extended):
    opaque_shape = _jax_physical_aval(operand_aval).shape[len(operand_aval.shape):]
    trailing_offset_dims = [len(_out_aval.shape) + i for i in range(len(opaque_shape))]
    dimension_numbers = dimension_numbers._replace(
        offset_dims=(*dimension_numbers.offset_dims, *trailing_offset_dims))
    slice_sizes = (*slice_sizes, *opaque_shape)
  proto = _gather_dimensions_proto(start_indices.shape, dimension_numbers)
  slice_sizes_tf = _eval_shape(slice_sizes)
  out = tfxla.gather(operand, start_indices, proto, slice_sizes_tf,
                     indices_are_sorted)
  out = _ensure_tf_shape_if_dynamic(out, _aval_to_tf_shape(_out_aval))
  if _WRAP_JAX_JIT_WITH_TF_FUNCTION:
    out = tf.stop_gradient(out)  # See #7839
  return out


tf_impl_with_avals[lax.gather_p] = _gather


def _slice(operand, start_indices, limit_indices, strides, _in_avals,
           _out_aval):
  if strides is None:
    strides = [1] * len(start_indices)
  slices = tuple(
      map(slice, _eval_shape(start_indices), _eval_shape(limit_indices),
          _eval_shape(strides)))
  out = operand[slices]
  # TODO(b/184503314): improve shape inference for __getitem__
  # E.g., operand.shape=(b, 5, 3), start_indices=(0, 1, 1), limit_indices=(b, 5, 3), strides=(1, 2, 1)
  out = _ensure_tf_shape_if_dynamic(out, _aval_to_tf_shape(_out_aval))
  return out


tf_impl_with_avals[lax.slice_p] = _slice


def _dynamic_slice(operand, *start_indices, slice_sizes: core.Shape,
                   _in_avals: Sequence[core.ShapedArray],
                   _out_aval: core.ShapedArray):
  start_indices = _maybe_cast_to_int64(tf.stack(start_indices))
  operand_aval = _in_avals[0]
  if dtypes.issubdtype(operand_aval.dtype, dtypes.extended):
    opaque_shape = _jax_physical_aval(operand_aval).shape[len(operand_aval.shape):]
    slice_sizes = (*slice_sizes, *opaque_shape)
    start_indices = tf.concat([start_indices, tf.zeros((len(opaque_shape),),
                                                       dtype=start_indices.dtype)],
                              axis=0)

  slice_sizes_tf = _eval_shape(slice_sizes)
  res = tfxla.dynamic_slice(operand, start_indices, size_indices=slice_sizes_tf)
  if _WRAP_JAX_JIT_WITH_TF_FUNCTION:
    res = tf.stop_gradient(res)  # See #7839
  return res


tf_impl_with_avals[lax.dynamic_slice_p] = _dynamic_slice


def _dynamic_update_slice(operand, update, *start_indices,
                          _in_avals: Sequence[core.ShapedArray],
                          _out_aval: core.ShapedArray):
  start_indices = _maybe_cast_to_int64(tf.stack(start_indices))
  operand_aval = _in_avals[0]
  if dtypes.issubdtype(operand_aval.dtype, dtypes.extended):
    opaque_shape = _jax_physical_aval(operand_aval).shape[len(operand_aval.shape):]
    start_indices = tf.concat([start_indices, tf.zeros((len(opaque_shape),),
                                                       dtype=start_indices.dtype)],
                              axis=0)
  out = tfxla.dynamic_update_slice(operand, update, start_indices)
  if _WRAP_JAX_JIT_WITH_TF_FUNCTION:
    out = tf.stop_gradient(out)  # See #7839
  return out


tf_impl_with_avals[lax.dynamic_update_slice_p] = _dynamic_update_slice


def _scatter_dimensions_proto(indices_shape, dimension_numbers):
  proto = xla_data_pb2.ScatterDimensionNumbers()
  proto.update_window_dims.extend(dimension_numbers.update_window_dims)
  proto.inserted_window_dims.extend(dimension_numbers.inserted_window_dims)
  proto.scatter_dims_to_operand_dims.extend(
      dimension_numbers.scatter_dims_to_operand_dims)
  proto.input_batching_dims.extend(dimension_numbers.operand_batching_dims)
  proto.scatter_indices_batching_dims.extend(
      dimension_numbers.scatter_indices_batching_dims)
  assert indices_shape
  proto.index_vector_dim = len(indices_shape) - 1
  return proto


_scatter_reduction_computation = lambda x, y: y


def _scatter(operand, scatter_indices, updates, *, update_jaxpr, update_consts,
             dimension_numbers, indices_are_sorted, unique_indices, mode,
             _in_avals: Sequence[core.ShapedArray],
             _out_aval: core.ShapedArray):
  del unique_indices
  if update_jaxpr is None:
    assert not update_consts
    update_jaxpr, update_consts = lax_internal._reduction_jaxpr(
        _scatter_reduction_computation,
        core.ShapedArray((), operand.dtype.as_numpy_dtype))

  if mode == lax.GatherScatterMode.CLIP:
    clip_fn = _convert_jax_impl(lax_slicing._clamp_scatter_indices,
                                multiple_results=False)
    scatter_indices = clip_fn(
        operand, scatter_indices, updates, dnums=dimension_numbers,
        _in_avals=_in_avals, _out_aval=_in_avals[1])

  assert len(update_consts) == 0, "Update computation cannot have constants"

  proto = _scatter_dimensions_proto(scatter_indices.shape, dimension_numbers)

  def update_computation(arg1: TfVal, arg2: TfVal) -> TfVal:
    closed_jaxpr = core.ClosedJaxpr(update_jaxpr, update_consts)
    res, = _interpret_jaxpr(closed_jaxpr, arg1, arg2, extra_name_stack=None)
    return res

  o_spec = tf.TensorSpec((), dtype=operand.dtype)
  xla_update_computation = (
      tf.function(update_computation,
                  autograph=False).get_concrete_function(o_spec, o_spec))
  out = tfxla.scatter(
      operand,
      scatter_indices,
      updates,
      xla_update_computation,
      proto,
      indices_are_sorted=indices_are_sorted)
  if _WRAP_JAX_JIT_WITH_TF_FUNCTION:
    out = tf.stop_gradient(out)  # See #7839
  return out


tf_impl_with_avals[lax.scatter_p] = _scatter
tf_impl_with_avals[lax.scatter_min_p] = _scatter
tf_impl_with_avals[lax.scatter_max_p] = _scatter
tf_impl_with_avals[lax.scatter_mul_p] = _scatter
tf_impl_with_avals[lax.scatter_add_p] = _scatter
tf_impl_with_avals[lax.scatter_sub_p] = _scatter


def _cond(
    index: TfVal, *operands: TfVal, branches: Sequence[core.ClosedJaxpr]
) -> Sequence[TfVal]:
  # tf.cond needs lambdas with no arguments.
  branches_tf = [
      partial(_interpret_jaxpr, jaxpr, *operands,
              # Same name stack as the XLA translation of cond_p
              extra_name_stack=f"branch_{i}_fun")
      for i, jaxpr in enumerate(branches)
  ]
  # Same name stack as XLA translation of cond_p
  # Note: extend_name_stack is a contextmanager, which is callable as a decorator.
  branches_tf = list(map(source_info_util.extend_name_stack("cond"),
      branches_tf))
  if len(branches) == 2:
    # `index` comes with tf.int32 type of casted boolean parameter.
    return tf.cond(tf.cast(index, tf.bool), branches_tf[1], branches_tf[0])
  else:
    return tf.switch_case(index, branches_tf)


tf_impl[lax.cond_p] = _cond


def _while(*args: TfVal, cond_nconsts: int, cond_jaxpr: core.ClosedJaxpr,
           body_nconsts: int, body_jaxpr: core.ClosedJaxpr) -> Sequence[TfVal]:
  cond_consts, body_consts, init_carry = util.split_list(
      args, [cond_nconsts, body_nconsts])
  if cond_jaxpr.out_avals[0].shape:
    # The conditional is not a scalar, this must be a batched while
    return _batched_cond_while(
        *args,
        cond_nconsts=cond_nconsts,
        cond_jaxpr=cond_jaxpr,
        body_nconsts=body_nconsts,
        body_jaxpr=body_jaxpr)

  # The conditional must return a single value to TF
  def cond_tf_func(*args: TfVal) -> TfVal:
    pred, = _interpret_jaxpr(cond_jaxpr, *cond_consts, *args,
                             # Same name stack as the XLA translation of while_p
                             extra_name_stack="while/cond")
    return pred

  body_tf_func = partial(_interpret_jaxpr, body_jaxpr, *body_consts,
                         extra_name_stack="while/body")
  # Sometimes TF infers more specific shapes for the init_carry, and this has
  # led to errors: "enters the loop with shape (1,), but has shape (None,) after one iteration"
  shape_invariants = [tf.TensorShape(_aval_to_tf_shape(_out_aval))
                      for _out_aval in body_jaxpr.out_avals]
  return tf.while_loop(cond_tf_func, body_tf_func, init_carry,
                       shape_invariants=shape_invariants)


def _batched_cond_while(*args: TfVal, cond_nconsts: int,
                        cond_jaxpr: core.ClosedJaxpr, body_nconsts: int,
                        body_jaxpr: core.ClosedJaxpr) -> Sequence[TfVal]:
  """Interprets a while_loop with a batched condition.

  A batched while has a conditional that returns a tensor of booleans, and
  a body that returns a list of tensors whose leading dimensions match those
  of the conditional tensor.

  We need to turn it into a while with scalar boolean conditional. We will
  expand the loop carry to include a prefix with the current tensor boolean
  condition. We prepend to the loop the first calculation of the tensor boolean
  condition. The loop condition will use a "reduce_any" to calculate a scalar
  boolean from the tensor boolean condition. The end of the loop body will
  compute the new carry using a "tf.where", and we compute the new tensor
  boolean condition.
  """
  cond_consts, body_consts, init_carry = util.split_list(
      args, [cond_nconsts, body_nconsts])
  # Initial computation of batched condition
  init_pred_b, = _interpret_jaxpr(cond_jaxpr, *cond_consts, *init_carry,
                                  extra_name_stack="while/body_pred")

  def new_cond_tf_func(pred_b: TfVal, *carry: TfVal) -> TfVal:
    pred = tf.reduce_any(pred_b, axis=list(range(len(pred_b.shape))))
    return pred

  def new_body_tf_func(pred_b: TfVal, *carry: TfVal) -> Sequence[TfVal]:
    new_carry: Sequence[TfVal] = _interpret_jaxpr(body_jaxpr, *body_consts,
                                                  *carry,
                                                  extra_name_stack="while/body")
    # We repeat those carries for which the loop termination condition is false
    def select_one_carry(new_c: TfVal, c: TfVal, c_aval: core.ShapedArray) -> TfVal:
      pred_b_bcast = _broadcast_in_dim(
          pred_b,
          shape=_jax_physical_aval(c_aval).shape,  # a JAX shape
          broadcast_dimensions=list(range(len(pred_b.shape))),
          _in_avals=cond_jaxpr.out_avals,
          _out_aval=core.ShapedArray(c_aval.shape, np.bool_))
      return tf.where(pred_b_bcast, new_c, c)

    selected_carry: Sequence[TfVal] = list(map(select_one_carry, new_carry, carry, body_jaxpr.out_avals))
    next_pred_b, = _interpret_jaxpr(cond_jaxpr, *cond_consts, *selected_carry,
                                    extra_name_stack="body_pred")
    return (next_pred_b, *selected_carry)

  _, *res_carry = tf.while_loop(new_cond_tf_func, new_body_tf_func,
                                (init_pred_b, *init_carry))
  return res_carry


tf_impl[lax.while_p] = _while

# We use the scan impl rule to rewrite in terms of while.
tf_impl_with_avals[lax.scan_p] = _convert_jax_impl(
    lax_control_flow._scan_impl,
    extra_name_stack="scan")

tf_impl_with_avals[ad_checkpoint.remat_p] = \
  _convert_jax_impl(partial(ad_checkpoint.remat_expansion,
                            # TODO: jax2tf cannot discriminate by platform
                            is_gpu_platform=False),
                    multiple_results=True,
                    extra_name_stack="checkpoint")

tf_impl[ad_checkpoint.name_p] = lambda x, *, name: x

# TODO: Remove once tensorflow is 2.10.0 everywhere.
if hasattr(tfxla, 'optimization_barrier'):
  tf_impl[lax_control_flow.optimization_barrier_p] = tfxla.optimization_barrier

def _top_k(operand: TfVal, k: int) -> tuple[TfVal, TfVal]:
  # Some types originally incompatible with tf.math.top_k can be promoted
  # to a compatible type without loss of precision.
  def promote_tf_dtype(tf_dtype):
    if tf_dtype in [tf.bool, tf.uint8, tf.uint16]:
      return tf.uint32
    if tf_dtype in [tf.int8, tf.int16]:
      return tf.int32
    if tf_dtype is tf.float16:
      return tf.float32
    return None

  conversion_dtype = promote_tf_dtype(operand.dtype)
  if not core.is_constant_dim(k):
    k_tf = _eval_shape((k,))[0]
    k_tf = tf.cast(k_tf, tf.int32)  # TopK works only for int32
  else:
    k_tf = k
  if conversion_dtype:
    values, indices = tf.math.top_k(
        tf.dtypes.cast(operand, conversion_dtype), k=k_tf, sorted=True)
    return tf.dtypes.cast(values, operand.dtype), indices
  else:
    return tf.math.top_k(operand, k=k_tf, sorted=True)


tf_impl[lax.top_k_p] = _top_k


def _approx_top_k(operand: TfVal, k: int, reduction_dimension: int,
                  recall_target: float, is_max_k: bool,
                  reduction_input_size_override: int,
                  aggregate_to_topk: bool) -> tuple[TfVal, TfVal]:
  k_tf = _eval_shape((k,))[0]
  if is_max_k:
    return tf.math.approx_max_k(operand, k_tf, reduction_dimension, recall_target,
                                reduction_input_size_override,
                                aggregate_to_topk)
  else:
    return tf.math.approx_min_k(operand, k_tf, reduction_dimension, recall_target,
                                reduction_input_size_override,
                                aggregate_to_topk)


tf_impl[lax.approx_top_k_p] = _approx_top_k


def _sort(*operands: TfVal, dimension: int, is_stable: bool,
          num_keys: int) -> tuple[TfVal, ...]:
  assert 1 <= num_keys <= len(operands)
  assert 0 <= dimension < len(
      operands[0].shape
  ), f"Invalid {dimension} for ndim {len(operands[0].shape)}"

  comparator_spec: list[tf.TensorSpec] = []
  comparator_jax_in_avals: list[core.ShapedArray] = []
  for op in operands:
    o_spec = tf.TensorSpec((), dtype=op.dtype)
    comparator_spec.extend([o_spec, o_spec])
    o_aval = core.ShapedArray((), _to_jax_dtype(op.dtype))
    comparator_jax_in_avals.extend([o_aval, o_aval])

  # Use the same comparator that JAX uses when compiling to XLA, to get the
  # proper NaN/Inf total order, and the lexicographic ordering.
  # The comparator is a 2N-argument TF function, with arguments [2k] and [2k +1]
  # corresponding to two scalars from operand[k].
  def lexicographic_comparator(*tf_args: TfVal) -> TfVal:
    return _convert_jax_impl(
        lax_internal._sort_lt_comparator, multiple_results=False)(
            *tf_args,
            _in_avals=comparator_jax_in_avals,
            _out_aval=core.ShapedArray((), np.bool_),
            num_keys=num_keys)

  xla_comparator_computation = (
      tf.function(lexicographic_comparator,
                  autograph=False).get_concrete_function(*comparator_spec))
  results = tfxla.variadic_sort(
      operands,
      dimension=dimension,
      is_stable=is_stable,
      comparator=xla_comparator_computation)
  if _WRAP_JAX_JIT_WITH_TF_FUNCTION:
    results = tuple(tf.stop_gradient(out) for out in results)  # See #7839
  return results


tf_impl[lax.sort_p] = _sort


def _fft(x, *, fft_type, fft_lengths,
         _in_avals: Sequence[core.ShapedArray],
         _out_aval: core.ShapedArray):
  FFT, IFFT, RFFT, IRFFT = list(map(lax.FftType, [0, 1, 2, 3]))
  tf_funcs = {
      FFT: [tf.signal.fft, tf.signal.fft2d, tf.signal.fft3d],
      IFFT: [tf.signal.ifft, tf.signal.ifft2d, tf.signal.ifft3d],
      RFFT: [tf.signal.rfft, tf.signal.rfft2d, tf.signal.rfft3d],
      IRFFT: [tf.signal.irfft, tf.signal.irfft2d, tf.signal.irfft3d]
  }
  tf_func = tf_funcs[fft_type][len(fft_lengths) - 1]
  if fft_type in (RFFT, IRFFT):
    # https://www.tensorflow.org/api_docs/python/tf/signal/irfft
    # Here we only set `fft_lengths` argument for non-default value.
    (x_aval,) = _in_avals
    x_shape = x_aval.shape
    expected_lengths = x_shape[-len(fft_lengths) : -1] + (
        (x_shape[-1] - 1) * 2,
    )
    if fft_lengths != expected_lengths:
      tf_func = partial(tf_func, fft_length=_eval_shape(fft_lengths))
  res = tf_func(x)
  return _ensure_tf_shape_if_dynamic(res, _aval_to_tf_shape(_out_aval))
tf_impl_with_avals[lax.fft_p] = _fft


def _qr(operand, full_matrices):
  return tf.linalg.qr(operand, full_matrices=full_matrices)


tf_impl[lax.linalg.qr_p] = _qr


def _svd(
    operand: TfVal,
    full_matrices: bool,
    compute_uv: bool,
    subset_by_index: tuple[int, int] | None = None,
    algorithm: lax.linalg.SvdAlgorithm | None = None,
):
  if not (
      subset_by_index is None
      or subset_by_index == (0, min(operand.shape[-1], operand.shape[-2]))
  ):
    raise NotImplementedError("subset_by_index is not implemented")

  if algorithm is not None and algorithm != lax.linalg.SvdAlgorithm.DEFAULT:
    raise NotImplementedError("SVD algorithm is not implemented")

  result = tf.linalg.svd(operand, full_matrices, compute_uv)
  if not compute_uv:
    return result,
  s, u, v = result
  return s, u, tf.linalg.adjoint(v)


tf_impl[lax.linalg.svd_p] = _svd


def _eig(operand: TfVal, compute_left_eigenvectors: bool,
         compute_right_eigenvectors: bool):
  if compute_left_eigenvectors and compute_right_eigenvectors:
    # TODO(bchetioui): didn't find a 100% reliable, easy and satisfying way to
    # sort the left eigenvectors in the right order. The jax.numpy.linalg API
    # suggests to me that left eigenvectors are anyway seldom used, so I
    # think it is acceptable to leave as unimplemented for now.
    msg = ("Conversion of eig is not implemented when both "
           "compute_left_eigenvectors and compute_right_eigenvectors are set "
           "to True.")
    raise NotImplementedError(msg)
  elif not (compute_left_eigenvectors or compute_right_eigenvectors):
    return (tf.linalg.eigvals(operand),)
  elif compute_right_eigenvectors:
    return tuple(tf.linalg.eig(operand))
  else:  # compute_left_eigenvectors == True
    wH, vl = tf.linalg.eig(tf.linalg.adjoint(operand))
    wHH = tf.math.conj(wH)
    return (wHH, vl)


tf_impl[lax.linalg.eig_p] = _eig


def _eigh(
    operand: TfVal,
    lower: bool,
    sort_eigenvalues: bool,
    subset_by_index: tuple,
    _in_avals,
    _out_aval,
):
  del sort_eigenvalues
  if operand.shape[-1] == 0:
    v, w = operand, tf.reshape(operand, _eval_shape(_in_avals[0].shape[:-1]))
  else:
    if not lower:
      operand = tf.linalg.adjoint(operand)
    w, v = tf.linalg.eigh(operand)
  cast_type = {
      tf.complex64: tf.float32,
      tf.complex128: tf.float64
  }.get(operand.dtype)
  if not (subset_by_index is None or subset_by_index == (0, operand.shape[-1])):
    raise NotImplementedError("subset_by_index is not implemented")
  if cast_type is not None:
    w = tf.cast(w, cast_type)
  return v, w


tf_impl_with_avals[lax.linalg.eigh_p] = _eigh


def _lu(operand: TfVal, _in_avals, _out_aval):
  return _convert_jax_impl(lax_linalg._lu_python, extra_name_stack="lu")(
      operand, _in_avals=_in_avals, _out_aval=_out_aval)


tf_impl_with_avals[lax.linalg.lu_p] = _lu


def _triangular_solve(a: TfVal, b: TfVal, *, left_side: bool, lower: bool,
                      transpose_a: bool, conjugate_a: bool, unit_diagonal: bool,
                      _in_avals: Sequence[core.ShapedArray],
                      _out_aval: core.ShapedArray):
  if unit_diagonal:
    a_aval, _ = _in_avals
    a_shape = _eval_shape(a_aval.shape)
    a = tf.linalg.set_diag(a, tf.ones(a_shape[:-1], dtype=a.dtype))
  if not left_side:
    rank = len(a.shape)
    transpose_dimensions = list(range(rank - 2)) + [rank - 1, rank - 2]
    a = tf.transpose(a, transpose_dimensions)
    b = tf.transpose(b, transpose_dimensions)
    lower = not lower
  # adjoint == transpose for real dtypes, so special care need only be taken
  # for complex types.
  if a.dtype in [tf.complex64, tf.complex128]:
    if (transpose_a and not conjugate_a) or (not transpose_a and conjugate_a):
      a = tf.math.conj(a)
  result = tf.linalg.triangular_solve(a, b, lower=lower, adjoint=transpose_a)
  if not left_side:
    result = tf.transpose(result, transpose_dimensions)
  return result


tf_impl_with_avals[lax.linalg.triangular_solve_p] = _triangular_solve


def _linear_solve(*args: TfVal, const_lengths, jaxprs, _in_avals, _out_aval):
  return _convert_jax_impl(lax_control_flow._custom_linear_solve_impl,
                           extra_name_stack="linear_solve")(
      *args,
      const_lengths=const_lengths,
      jaxprs=jaxprs,
      _in_avals=_in_avals,
      _out_aval=_out_aval)


tf_impl_with_avals[lax.linear_solve_p] = _linear_solve

def _tridiagonal_solve(*args: TfVal, _in_avals, _out_aval, **params):
  return _convert_jax_impl(lax_linalg._tridiagonal_solve_jax,
                           multiple_results=False,
                           extra_name_stack="tridiagonal_solve")(
      *args,
      _in_avals=_in_avals,
      _out_aval=_out_aval)


tf_impl_with_avals[lax.linalg.tridiagonal_solve_p] = _tridiagonal_solve

def _custom_jvp_call(*args: TfVal, call_jaxpr: core.ClosedJaxpr,
                           jvp_jaxpr_thunk: Callable,
                           num_consts: int) -> Sequence[TfVal]:
  # TODO(necula): ensure that there is no AD transformation in scope
  del jvp_jaxpr_thunk, num_consts
  return _interpret_jaxpr(call_jaxpr, *args, extra_name_stack="custom_jvp",
                          fresh_constant_cache=False)


tf_impl[custom_derivatives.custom_jvp_call_p] = _custom_jvp_call


def _custom_vjp_call_jaxpr(*args: TfVal, fun_jaxpr: core.ClosedJaxpr,
                           **_) -> Sequence[TfVal]:
  # TODO(necula): ensure that there is no AD transformation in scope
  return _interpret_jaxpr(fun_jaxpr, *args, extra_name_stack="custom_vjp",
                          fresh_constant_cache=False)


tf_impl[custom_derivatives.custom_vjp_call_jaxpr_p] = _custom_vjp_call_jaxpr


def _custom_lin(*args: TfVal, **_) -> Sequence[TfVal]:
  raise TypeError("can't apply forward-mode autodiff (jvp) to a custom_vjp "
                  "function.")


tf_impl[ad.custom_lin_p] = _custom_lin


def _remat_opt(*args: TfVal, num_consts: int, num_res: int,
                    fwd_jaxpr: core.ClosedJaxpr,
                    fun_jaxpr_thunk: Callable) -> Sequence[TfVal]:
  del num_consts, num_res, fun_jaxpr_thunk
  return _interpret_jaxpr(fwd_jaxpr, *args, extra_name_stack="remat_opt",
                          fresh_constant_cache=False)


tf_impl[custom_derivatives.remat_opt_p] = _remat_opt


PartitionsOrReplicated = Union[tuple[int, ...], None]

def split_to_logical_devices(tensor: TfVal,
                             partition_dimensions: PartitionsOrReplicated):
  """Like TPUMPStrategy.experimental_split_to_logical_devices.

  For jax2tf purposes we want to avoid needing to thread the `strategy` object
  through the generated computation. It seems that the original function needs
  the strategy object only for error checking, which we assume is done upstream
  by JAX.

  Args:
    tensor: Input tensor to annotate.
    partition_dimensions: A list of integers, with one integer per tensor
      dimension, specifying in how many parts the dimension should be split. The
      product of integers must equal the number of devices per replica.
    use_sharding_op: whether to use a sharding op, or not.

  Returns:
    an annotated tensor.
  """
  # TODO: this is only for sharded_jit. Either remove, or implement in terms
  # of _shard_values.
  if partition_dimensions is None:
    return xla_sharding.replicate(tensor, use_sharding_op=True)
  num_partition_splits = math.prod(partition_dimensions)
  tile_assignment = np.arange(num_partition_splits).reshape(
      partition_dimensions)
  return xla_sharding.tile(tensor, tile_assignment, use_sharding_op=True)


def _xla_compatible_sharding_to_hlo_sharding(
    s: sharding.Sharding,
    aval: core.ShapedArray) -> xla_client.HloSharding | None:
  if isinstance(s, sharding_impls.UnspecifiedValue):
    return None
  return s._to_xla_hlo_sharding(aval.ndim)

def _shard_value(val: TfVal,
                 sd: xla_client.HloSharding | None, *,
                 skip_replicated_sharding: bool) -> TfVal:
  """Apply sharding to a TfVal."""
  if sd is None:
    return val

  sharding_proto = sd.to_proto()
  if (skip_replicated_sharding and
      op_shardings.is_op_sharding_replicated(sharding_proto)):
    return val

  # Tensorflow heavily relies on tile_assignment_devices proto fields specific
  # to V1 sharding format, falling back to this format.
  if (
      not sharding_proto.tile_assignment_devices
      and sharding_proto.iota_reshape_dims
  ):
    tad = list(
        np.arange(math.prod(sharding_proto.tile_assignment_dimensions))
        .reshape(sharding_proto.iota_reshape_dims)
        .transpose(sharding_proto.iota_transpose_perm)
        .flat
    )
  else:
    tad = sharding_proto.tile_assignment_devices  # type: ignore

  # To use xla_sharding.py, we must have a xla_data_pb2.OpSharding.
  xla_sharding_proto: xla_data_pb2.OpSharding = xla_data_pb2.OpSharding(
      type=int(sharding_proto.type),
      tile_assignment_dimensions=sharding_proto.tile_assignment_dimensions,
      tile_assignment_devices=tad,
      replicate_on_last_tile_dim=sharding_proto.replicate_on_last_tile_dim,
      last_tile_dims=sharding_proto.last_tile_dims,
  )
  if tf_context.executing_eagerly():
    raise ValueError(
        "A jit function with sharded arguments or results must be used under a `tf.function` context. "
        "See https://github.com/jax-ml/jax/blob/main/jax/experimental/jax2tf/README.md#support-for-partitioning for a discussion")

  return xla_sharding.Sharding(proto=xla_sharding_proto).apply_to_tensor(
      val, use_sharding_op=True)


def _pjit(*args: TfVal,
          jaxpr: core.ClosedJaxpr,
          in_shardings: Sequence[sharding.Sharding],
          out_shardings: Sequence[sharding.Sharding],
          in_layouts, out_layouts,
          resource_env: mesh.ResourceEnv,
          donated_invars,
          name: str,
          keep_unused: bool,
          inline: bool,
          compiler_options_kvs,
          _in_avals: Sequence[core.ShapedArray],
          _out_aval: Sequence[core.ShapedArray]) -> TfVal:
  del donated_invars
  # Apply sharding annotation to the arguments
  in_hlo_shardings: Sequence[xla_client.HloSharding | None] = map(
    _xla_compatible_sharding_to_hlo_sharding, in_shardings, _in_avals)
  sharded_args: Sequence[TfVal] = tuple(
      map(partial(_shard_value,
                  skip_replicated_sharding=not _thread_local_state.enable_xla),
          args, in_hlo_shardings))
  results = _interpret_jaxpr(jaxpr, *sharded_args,
                              extra_name_stack=util.wrap_name(name, "pjit"),
                              fresh_constant_cache=False)
  out_hlo_shardings: Sequence[xla_client.HloSharding | None] = map(
    _xla_compatible_sharding_to_hlo_sharding, out_shardings, _out_aval)
  sharded_results: Sequence[TfVal] = tuple(
      map(partial(_shard_value,
                  skip_replicated_sharding=not _thread_local_state.enable_xla),
          results, out_hlo_shardings))
  return tuple(sharded_results)


tf_impl_with_avals[pjit.pjit_p] = _pjit


def _pjit_sharding_constraint(arg: TfVal, *,
                              sharding: sharding.Sharding,
                              resource_env: mesh.ResourceEnv,
                              _in_avals: Sequence[core.ShapedArray],
                              _out_aval: core.ShapedArray,
                              **kwargs) -> TfVal:
  hlo_sharding = _xla_compatible_sharding_to_hlo_sharding(sharding, _in_avals[0])
  return _shard_value(arg, hlo_sharding,
                      skip_replicated_sharding=False)


tf_impl_with_avals[pjit.sharding_constraint_p] = _pjit_sharding_constraint

def _dimension_size_jax2tf(op: TfVal, *, dimension, _in_avals, _out_aval):
  dim_tf = tf.shape(op)[dimension]
  if dim_tf.dtype != _to_tf_dtype(_out_aval.dtype):
    return _convert_element_type(dim_tf, new_dtype=_out_aval.dtype,
                                 weak_type=_out_aval.weak_type)
  else:
    return dim_tf

tf_impl_with_avals[shape_poly.dimension_size_p] = _dimension_size_jax2tf

def _dim_as_value_jax2tf(dim: shape_poly.DimSize):
  dim_tf, = _eval_shape((dim,))
  return dim_tf

tf_impl[shape_poly.dim_as_value_p] = _dim_as_value_jax2tf

def _shape_assertion_jax2tf(assert_what, *error_message_inputs,
                            error_message: str):

  tf.debugging.assert_equal(
    assert_what, True,
    message=error_message.format(*error_message_inputs))
  return []

tf_impl[shape_poly.shape_assertion_p] = _shape_assertion_jax2tf

def _reduce_precision(x, *, exponent_bits, mantissa_bits):
  return tfxla.reduce_precision(x, exponent_bits=exponent_bits,
                                mantissa_bits=mantissa_bits)

tf_impl[lax.reduce_precision_p] = _reduce_precision

tf_impl[lax_internal.tie_p] = lambda x, y: y


def _register_checkpoint_pytrees():
  """Registers TF custom container types as pytrees."""
  m = tf.Module()
  # The types here are automagically changed by TensorFlow's checkpointing
  # infrastructure.
  m.a = (tf.Module(), tf.Module())
  m.b = [tf.Module(), tf.Module()]
  m.c = {"a": tf.Module()}
  tuple_wrapper = type(m.a)
  list_wrapper = type(m.b)
  dict_wrapper = type(m.c)

  # TF AutoTrackable swaps container types out for wrappers.
  assert tuple_wrapper is not tuple
  assert list_wrapper is not list
  assert dict_wrapper is not dict

  jax.tree_util.register_pytree_node(tuple_wrapper, lambda xs:
                                     (tuple(xs), None), lambda _, xs: tuple(xs))

  jax.tree_util.register_pytree_node(list_wrapper, lambda xs: (tuple(xs), None),
                                     lambda _, xs: list(xs))

  jax.tree_util.register_pytree_node(
      dict_wrapper,
      lambda s: (tuple(s.values()), tuple(s.keys())),
      lambda k, xs: dict_wrapper(zip(k, xs)))


_register_checkpoint_pytrees()
