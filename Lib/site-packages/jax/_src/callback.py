# Copyright 2022 The JAX Authors.
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
"""Module for JAX callbacks."""
from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import functools
import logging
from typing import Any

import jax
from jax._src import config
from jax._src import core
from jax._src import deprecations
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src import sharding_impls
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import lax
from jax._src.lax.control_flow.loops import map as lax_map
from jax._src.lib import xla_client as xc
from jax._src.sharding_impls import SingleDeviceSharding
from jax._src.typing import DeprecatedArg
import numpy as np

logger = logging.getLogger(__name__)

# TODO(dfm): Remove after 6 months.
# Added Oct 1, 2024
deprecations.register("jax-callback-vectorized")

# `pure_callback_p` is the main primitive for staging out Python pure callbacks.
pure_callback_p = core.Primitive("pure_callback")
pure_callback_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(pure_callback_p)

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


@dataclasses.dataclass(frozen=True)
class _FlatCallback:
  """A Python function callable with flat arguments and results.

  An instance of this class is used as a parameter for the callback primitives.
  We prefer it to an anonymous flattened function because it produces
  equal objects when we call the same Python function with the same argument
  structure.
  """
  callback_func: Callable[..., Any]
  in_tree: tree_util.PyTreeDef  # (args, kwargs) pytree for `callback_func`.

  def __call__(self, *flat_args: jax.Array) -> Sequence[jax.Array]:
    args, kwargs = tree_util.tree_unflatten(self.in_tree, flat_args)
    return tree_util.tree_leaves(self.callback_func(*args, **kwargs))


def pure_callback_impl(
    *args,
    result_avals,
    callback: _FlatCallback,
    sharding: SingleDeviceSharding | None,
    vectorized: bool | DeprecatedArg,
    vmap_method: str | None,
):
  del sharding, vectorized, vmap_method, result_avals
  try:
    cpu_device, *_ = jax.local_devices(backend="cpu")
  except RuntimeError as e:
    raise RuntimeError(
        "jax.pure_callback failed to find a local CPU device to place the"
        " inputs on. Make sure \"cpu\" is listed in --jax_platforms or the"
        " JAX_PLATFORMS environment variable."
    ) from e
  args = jax.device_put(args, cpu_device)
  with jax.default_device(cpu_device):
    try:
      return tree_util.tree_map(np.asarray, callback(*args))
    except BaseException:
      logger.exception("jax.pure_callback failed")
      raise


pure_callback_p.def_impl(functools.partial(dispatch.apply_primitive,
                                           pure_callback_p))


@pure_callback_p.def_abstract_eval
def pure_callback_abstract_eval(
    *avals,
    callback: _FlatCallback,
    result_avals,
    sharding: SingleDeviceSharding | None,
    vectorized: bool | DeprecatedArg,
    vmap_method: str | None,
):
  del avals, callback, sharding, vectorized, vmap_method
  return result_avals


def pure_callback_jvp_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "Pure callbacks do not support JVP. "
      "Please use `jax.custom_jvp` to use callbacks while taking gradients.")


ad.primitive_jvps[pure_callback_p] = pure_callback_jvp_rule


def pure_callback_transpose_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError(
      "Pure callbacks do not support transpose. "
      "Please use `jax.custom_vjp` to use callbacks while taking gradients.")

ad.primitive_transposes[pure_callback_p] = pure_callback_transpose_rule


def callback_batching_rule(
    prim,
    args,
    dims,
    *,
    vectorized: bool | None | DeprecatedArg,
    vmap_method: str | None,
    result_avals: Sequence[core.ShapedArray],
    **kwargs: Any,
):
  if isinstance(vectorized, DeprecatedArg) and vmap_method is None:
    deprecations.warn(
        "jax-callback-vectorized",
        f"The default behavior of {prim.name} under vmap will soon "
        "change. Currently, the default behavior is to generate a sequential "
        "vmap (i.e. a loop), but in the future the default will be to raise "
        "an error. To keep the current default, set vmap_method='sequential'.",
        stacklevel=6)
    vmap_method = "sequential"

  axis_size, = {a.shape[d] for a, d in zip(args, dims)
                if d is not batching.not_mapped}
  new_args = [arg if dim is batching.not_mapped else
              batching.moveaxis(arg, dim, 0) for arg, dim in zip(args, dims)]
  batched_result_avals = tuple(
      core.unmapped_aval(axis_size, core.no_axis_name, 0, aval)
      for aval in result_avals)

  # For FFI calls we must update the layouts. We handle the output layouts
  # here, but the input layout updates depend on the vmap_method parameter.
  if vmap_method != "sequential" and kwargs.get("output_layouts") is not None:
    kwargs["output_layouts"] = tuple(
        None if layout is None else tuple(n + 1 for n in layout) + (0,)
        for layout in kwargs["output_layouts"])

  if vmap_method == "legacy_vectorized":
    # This method is kept to support the behavior that was previously exposed
    # when using `vectorized=True`.
    if kwargs.get("input_layouts") is not None:
      kwargs["input_layouts"] = tuple(
          layout if d is batching.not_mapped else
          (None if layout is None else tuple(n + 1 for n in layout) + (0,))
          for layout, d in zip(kwargs["input_layouts"], dims))
    outvals = prim.bind(
        *new_args,
        vectorized=vectorized,
        vmap_method=vmap_method,
        result_avals=batched_result_avals,
        **kwargs,
    )
  elif vmap_method == "expand_dims" or vmap_method == "broadcast_all":
    size = axis_size if vmap_method == "broadcast_all" else 1
    bcast_args = [
        lax.broadcast(x, (size,)) if d is batching.not_mapped else x
        for x, d in zip(new_args, dims)]
    if kwargs.get("input_layouts") is not None:
      kwargs["input_layouts"] = tuple(
          None if layout is None else tuple(n + 1 for n in layout) + (0,)
          for layout in kwargs["input_layouts"])
    outvals = prim.bind(
      *bcast_args,
      vectorized=vectorized,
      vmap_method=vmap_method,
      result_avals=batched_result_avals,
      **kwargs,
    )
  elif vmap_method == "sequential":
    is_batched = [d is not batching.not_mapped for d in dims]
    unbatched_args, batched_args = util.partition_list(is_batched, new_args)
    def _batch_fun(batched_args):
      merged_args = util.merge_lists(is_batched, unbatched_args, batched_args)
      return prim.bind(
          *merged_args,
          result_avals=result_avals,
          vectorized=vectorized,
          vmap_method=vmap_method,
          **kwargs,
      )
    outvals = lax_map(_batch_fun, batched_args)
  else:
    raise NotImplementedError(
        f"vmap is only supported for the {prim.name} primitive when vmap_method "
        "is one of 'sequential', 'expand_dims', 'broadcast_all', or "
        "'legacy_vectorized'.")
  return tuple(outvals), (0,) * len(outvals)


batching.primitive_batchers[pure_callback_p] = functools.partial(
    callback_batching_rule, pure_callback_p
)


def _callback_op_sharding(
    axis_context, sharding: SingleDeviceSharding | None, avals_out
):
  if isinstance(axis_context, sharding_impls.SPMDAxisContext):
    # If we have fully manual sharding during lowering, that means the JAX
    # program has per-device semantics, so we run the callback on each device.
    if axis_context.manual_axes != frozenset(axis_context.mesh.axis_names):
      raise NotImplementedError(
          "callbacks are only supported in spmd computations when all mesh"
          " axes are partitioned manually (no partial automatic sharding)."
      )
    if sharding is not None:
      raise NotImplementedError(
          "callbacks do not support specifying sharding inside spmd"
          " computations"
      )
    if config.use_shardy_partitioner.value:
      assert len(avals_out) == 1
      op_sharding = sharding_impls.SdyArrayShardingList([
          sharding_impls.SdyArraySharding(
              mesh_shape=(),
              dimension_shardings=[
                  sharding_impls.SdyDimSharding(axes=[], is_closed=True)
              ] * avals_out[0].ndim,
              logical_device_ids=())])
    else:
      op_sharding = xc.OpSharding()  # type: ignore[assignment]
      op_sharding.type = xc.OpSharding.Type.MANUAL
    return op_sharding

  if isinstance(axis_context, sharding_impls.ShardingContext):
    if sharding is not None:
      if not isinstance(sharding, SingleDeviceSharding):
        raise NotImplementedError(
            "pure_callback only supports SingleDeviceSharding, but got"
            f" {type(sharding)}"
        )
      device = next(iter(sharding.device_set))
      device_assignment = axis_context.device_assignment
      if device_assignment is None:
        raise AssertionError(
            "Please file a bug at https://github.com/jax-ml/jax/issues")
      try:
        device_index = device_assignment.index(device)
      except IndexError as e:
        raise ValueError(
            "Sharding provided to pure_callback specifies a device"
            f" {device} that is not in the device assignment"
            f" ({device_assignment})") from e
    else:
      device_index = 0

    # If we have fully automatic sharding during lowering, that means the JAX
    # program has bulk array semantics, so we run the callback with a MAXIMAL
    # sharding and hence execute it only once on the full logical value).
    if config.use_shardy_partitioner.value:
      op_sharding = sharding_impls.SdyArrayShardingList([
          sharding_impls.SdyArraySharding(
              mesh_shape=(),
              dimension_shardings=[],
              logical_device_ids=(device_index,))])
    else:
      op_sharding = xc.OpSharding()  # type: ignore[assignment]
      op_sharding.type = xc.OpSharding.Type.MAXIMAL
      op_sharding.tile_assignment_dimensions = [1]
      op_sharding.tile_assignment_devices = [device_index]
    return op_sharding

  # When there's no SPMD partitioning going on, don't annotate a sharding.
  return None


def pure_callback_lowering(
    ctx, *args, callback: _FlatCallback, sharding: SingleDeviceSharding | None, **params
):
  def _callback(*flat_args):
    return tuple(
        pure_callback_impl(
            *flat_args,
            callback=callback,
            sharding=None,  # unused.
            **params,
        )
    )

  op_sharding = _callback_op_sharding(
      ctx.module_context.axis_context, sharding, ctx.avals_out)
  result, _, _ = mlir.emit_python_callback(
      ctx,
      _callback,
      None,
      list(args),
      ctx.avals_in,
      ctx.avals_out,
      has_side_effect=False,
      sharding=op_sharding,
  )
  return result


mlir.register_lowering(pure_callback_p, pure_callback_lowering)

def _check_shape_dtype(shape_dtype):
  dt = np.dtype(shape_dtype.dtype)
  if dtypes.canonicalize_dtype(dt) != dt:
    raise ValueError(
        "result_shape_dtypes cannot specify 64-bit types when `jax_enable_x64` is disabled")


def pure_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    sharding: SingleDeviceSharding | None = None,
    vectorized: bool | None | DeprecatedArg = DeprecatedArg(),
    vmap_method: str | None = None,
    **kwargs: Any,
):
  """Calls a pure Python callback. Works under :func:`jit`/:func:`~vmap`/etc.

  For more explanation, see `External Callbacks`_.

  ``pure_callback`` enables calling a Python function in JIT-ed JAX functions.
  The input ``callback`` will be passed JAX arrays placed on a local CPU, and
  it should also return JAX arrays on CPU.

  The callback is treated as functionally pure, meaning it has no side-effects
  and its output value depends only on its argument values. As a consequence, it
  is safe to be called multiple times (e.g. when transformed by :func:`~vmap` or
  :func:`~pmap`), or not to be called at all when e.g. the output of a
  `jit`-decorated function has no data dependence on its value. Pure callbacks
  may also be reordered if data-dependence allows.

  When `vmap`-ed the behavior will depend on the value of the ``vmap_method``.

  * Calling :func:`~jax.vmap` on a callback without an explicit ``vmap_method``
    is deprecated and it will eventually raise ``NotImplementedError``.
  * ``vmap_method="sequential"`` uses :func:`~jax.lax.map` to loop over
    the batched arguments, calling ``callback`` once for each batch element.
  * ``vmap_method="expand_dims"`` calls ``callback`` with new axes of size ``1``
    added as the leading dimension unbatched inputs.
  * ``vmap_method="broadcast_all"`` behaves like ``expand_dims``, but the
    inputs are tiled to the expected batched shape.

  If necessary, the legacy behavior provided by the deprecated
  ``vectorized=True`` argument can be recovered using
  ``vmap_method="legacy_vectorized"``.

  The current default behavior is to use ``vmap_method="sequential"`` when
  not specified, but this behavior is deprecated, and in the future, the
  default will be to raise a ``NotImplementedError`` unless ``vmap_method`` is
  explicitly specified.

  Args:
    callback: function to execute on the host. The callback is assumed to be a pure
      function (i.e. one without side-effects): if an impure function is passed, it
      may behave in unexpected ways, particularly under transformation. The callable
      will be passed PyTrees of arrays as arguments, and should return a PyTree of
      arrays that matches ``result_shape_dtypes``.
    result_shape_dtypes: pytree whose leaves have ``shape`` and ``dtype`` attributes,
      whose structure matches the expected output of the callback function at runtime.
      :class:`jax.ShapeDtypeStruct` is often used to define leaf values.
    *args: arguments to be passed to the callback function
    sharding: optional sharding that specifies the device from which the callback should
      be invoked.
    vmap_method: string specifying how the callback transforms under
      :func:`~jax.vmap` as described above.
    **kwargs: keyword arguments to be passed to the callback function

  Returns:
    result: a pytree of :class:`jax.Array` objects whose structure matches that of
      ``result_shape_dtypes``.

  See Also:
    - :func:`jax.experimental.io_callback`: callback designed for impure functions.
    - :func:`jax.debug.callback`: callback designed for general-purpose debugging.
    - :func:`jax.debug.print`: callback designed for printing.

  Examples:
    The behavior of ``pure_callback`` under :func:`~jax.vmap` is controlled by
    the ``vmap_method`` argument as described above. It is useful to consider
    some explicit examples that demonstrate the semantics. For example,
    consider the following function:

    >>> def callback(x, y):
    ...   print(jnp.shape(x), jnp.shape(y))
    ...   return x + y

    >>> def fun(x, y, *, vmap_method):
    ...   shape = jnp.broadcast_shapes(jnp.shape(x), jnp.shape(y))
    ...   dtype = jnp.result_type(x, y)
    ...   out_type = jax.ShapeDtypeStruct(shape, dtype)
    ...   return jax.pure_callback(callback, out_type, x, y,
    ...                            vmap_method=vmap_method)

    Calling this with ``vmap_method="expand_dims"`` adds a new axis of size ``1``
    to ``y``:

    >>> from functools import partial
    >>> x = jnp.arange(4)
    >>> y = 1.0
    >>> jax.vmap(partial(fun, vmap_method="expand_dims"), in_axes=(0, None))(x, y)
    (4,) (1,)
    Array([1., 2., 3., 4.], dtype=float32)

    Whereas, ``vmap_method="broadcast_all"`` adds an axis of size ``4`` to
    ``y``:

    >>> jax.vmap(partial(fun, vmap_method="broadcast_all"),
    ...          in_axes=(0, None))(x, y)
    (4,) (4,)
    Array([1., 2., 3., 4.], dtype=float32)

  .. _External Callbacks: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html
  """
  if not isinstance(vectorized, DeprecatedArg) and not vectorized is None:
    deprecations.warn(
        "jax-callback-vectorized",
        "The vectorized argument of jax.pure_callback is deprecated and setting "
        "it will soon raise an error. To avoid an error in the future, and to "
        "suppress this warning, please use the vmap_method argument instead.",
        stacklevel=2)
    if vmap_method is not None:
      raise ValueError(
          "the vectorized and vmap_method arguments of jax.pure_callback cannot "
          "be used together. Please use the vmap_method argument.")
    vmap_method = "legacy_vectorized" if vectorized else "sequential"
  allowed_vmap_methods = ["sequential", "expand_dims", "broadcast_all",
                          "legacy_vectorized", None]
  if vmap_method not in allowed_vmap_methods:
    raise ValueError(
        f"vmap_method must be on of the allowed methods {allowed_vmap_methods}, "
        f"but got: {vmap_method}")

  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  tree_util.tree_map(_check_shape_dtype, result_shape_dtypes)
  result_avals = tree_util.tree_map(
      lambda x: core.ShapedArray(x.shape, x.dtype), result_shape_dtypes)
  flat_result_avals, out_tree = tree_util.tree_flatten(result_avals)
  out_flat = pure_callback_p.bind(
      *flat_args,
      callback=_FlatCallback(callback, in_tree),
      result_avals=tuple(flat_result_avals),
      sharding=sharding,
      vectorized=vectorized,
      vmap_method=vmap_method,
  )
  return tree_util.tree_unflatten(out_tree, out_flat)


# IO Callback

io_callback_p = core.Primitive("io_callback")
io_callback_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(io_callback_p)

class IOEffect(effects.Effect):
  __str__ = lambda _: "IO"

class OrderedIOEffect(effects.Effect):
  __str__ = lambda _: "OrderedIO"

_IOEffect = IOEffect()
_OrderedIOEffect = OrderedIOEffect()
effects.lowerable_effects.add_type(IOEffect)
effects.lowerable_effects.add_type(OrderedIOEffect)
effects.control_flow_allowed_effects.add_type(IOEffect)
effects.control_flow_allowed_effects.add_type(OrderedIOEffect)
effects.ordered_effects.add_type(OrderedIOEffect)
effects.shardable_ordered_effects.add_type(OrderedIOEffect)


def io_callback_impl(
    *args,
    result_avals,
    callback: _FlatCallback,
    sharding: SingleDeviceSharding | None,
    ordered: bool,
):
  del result_avals, sharding, ordered
  try:
    cpu_device, *_ = jax.local_devices(backend="cpu")
  except RuntimeError as e:
    raise RuntimeError(
        "jax.io_callback failed to find a local CPU device to place the"
        " inputs on. Make sure \"cpu\" is listed in --jax_platforms or the"
        " JAX_PLATFORMS environment variable."
    ) from e
  args = jax.device_put(args, cpu_device)
  with jax.default_device(cpu_device):
    try:
      return tree_util.tree_map(np.asarray, callback(*args))
    except BaseException:
      logger.exception("jax.io_callback failed")
      raise


io_callback_p.def_impl(functools.partial(dispatch.apply_primitive,
                                         io_callback_p))


@io_callback_p.def_effectful_abstract_eval
def io_callback_abstract_eval(
    *avals,
    callback: _FlatCallback,
    result_avals,
    sharding: SingleDeviceSharding | None,
    ordered: bool,
):
  del avals, sharding, callback
  effect = _OrderedIOEffect if ordered else _IOEffect
  return result_avals, {effect}


def io_callback_jvp_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError("IO callbacks do not support JVP.")
ad.primitive_jvps[io_callback_p] = io_callback_jvp_rule


def io_callback_transpose_rule(*args, **kwargs):
  del args, kwargs
  raise ValueError("IO callbacks do not support transpose.")
ad.primitive_transposes[io_callback_p] = io_callback_transpose_rule


def io_callback_batching_rule(
    args, dims, callback, result_avals, sharding, ordered
):
  if ordered:
    raise ValueError("Cannot `vmap` ordered IO callback.")
  is_batched = [d is not batching.not_mapped for d in dims]
  new_args = [arg if dim is batching.not_mapped else
              batching.moveaxis(arg, dim, 0) for arg, dim in zip(args, dims)]
  unbatched_args, batched_args = util.partition_list(is_batched, new_args)
  def _batch_fun(batched_args):
    merged = util.merge_lists(is_batched, unbatched_args, batched_args)
    return io_callback_p.bind(*merged, callback=callback, sharding=sharding,
                              result_avals=result_avals, ordered=False)
  out_vals = lax_map(_batch_fun, batched_args)
  return out_vals, (0,) * len(out_vals)
batching.primitive_batchers[io_callback_p] = io_callback_batching_rule


def io_callback_lowering(ctx, *args, callback, sharding, ordered, **params):
  def _callback(*flat_args):
    return tuple(
        io_callback_impl(
            *flat_args,
            callback=callback,
            sharding=None,  # unused.
            ordered=ordered,
            **params,
        )
    )

  op_sharding = _callback_op_sharding(
      ctx.module_context.axis_context, sharding, ctx.avals_out)
  if ordered:
    token = ctx.tokens_in.get(_OrderedIOEffect)
    result, token, _ = mlir.emit_python_callback(
        ctx,
        _callback,
        token,
        list(args),
        ctx.avals_in,
        ctx.avals_out,
        has_side_effect=True,
        sharding=op_sharding,
    )
    ctx.set_tokens_out(mlir.TokenSet({_OrderedIOEffect: token}))
  else:
    result, _, _ = mlir.emit_python_callback(
        ctx,
        _callback,
        None,
        list(args),
        ctx.avals_in,
        ctx.avals_out,
        has_side_effect=True,
        sharding=op_sharding,
    )
  return result


mlir.register_lowering(io_callback_p, io_callback_lowering)


def io_callback(
    callback: Callable[..., Any],
    result_shape_dtypes: Any,
    *args: Any,
    sharding: SingleDeviceSharding | None = None,
    ordered: bool = False,
    **kwargs: Any,
):
  """Calls an impure Python callback.

  For more explanation, see `External Callbacks`_.

  Args:
    callback: function to execute on the host. It is assumed to be an impure function.
      If ``callback`` is pure, using :func:`jax.pure_callback` instead may lead to
      more efficient execution.
    result_shape_dtypes: pytree whose leaves have ``shape`` and ``dtype`` attributes,
      whose structure matches the expected output of the callback function at runtime.
      :class:`jax.ShapeDtypeStruct` is often used to define leaf values.
    *args: arguments to be passed to the callback function
    sharding: optional sharding that specifies the device from which the callback should
      be invoked.
    ordered: boolean specifying whether sequential calls to callback must be ordered.
    **kwargs: keyword arguments to be passed to the callback function

  Returns:
    result: a pytree of :class:`jax.Array` objects whose structure matches that of
      ``result_shape_dtypes``.

  See Also:
    - :func:`jax.pure_callback`: callback designed for pure functions.
    - :func:`jax.debug.callback`: callback designed for general-purpose debugging.
    - :func:`jax.debug.print`: callback designed for printing.

  .. _External Callbacks: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html
  """
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  tree_util.tree_map(_check_shape_dtype, result_shape_dtypes)
  flat_shape_dtypes, out_tree = tree_util.tree_flatten(result_shape_dtypes)
  flat_result_avals = map(lambda x: core.ShapedArray(x.shape, x.dtype),
                          flat_shape_dtypes)
  out_flat = io_callback_p.bind(
      *flat_args,
      callback=_FlatCallback(callback, in_tree),
      result_avals=tuple(flat_result_avals),
      sharding=sharding,
      ordered=ordered,
  )
  return tree_util.tree_unflatten(out_tree, out_flat)
