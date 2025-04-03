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
"""Module for JAX debugging primitives and related functionality."""

from __future__ import annotations

from collections.abc import Callable, Sequence
import functools
import importlib.util
import logging
import string
import sys
from typing import Any, Union
import weakref

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import effects
from jax._src import mesh as mesh_lib
from jax._src import sharding_impls
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.sharding import Sharding
from jax._src.sharding_impls import NamedSharding, parse_flatten_op_sharding
from jax._src.state import discharge as state_discharge

logger = logging.getLogger(__name__)

class DebugEffect(effects.Effect):
  __str__ = lambda self: "Debug"
debug_effect = DebugEffect()

class OrderedDebugEffect(effects.Effect):
  __str__ = lambda self: "OrderedDebug"
ordered_debug_effect = OrderedDebugEffect()

effects.ordered_effects.add_type(OrderedDebugEffect)
effects.lowerable_effects.add_type(DebugEffect)
effects.lowerable_effects.add_type(OrderedDebugEffect)
effects.control_flow_allowed_effects.add_type(DebugEffect)
effects.control_flow_allowed_effects.add_type(OrderedDebugEffect)
effects.remat_allowed_effects.add_type(DebugEffect)
effects.remat_allowed_effects.add_type(OrderedDebugEffect)
effects.custom_derivatives_allowed_effects.add_type(DebugEffect)
effects.custom_derivatives_allowed_effects.add_type(OrderedDebugEffect)

# `debug_callback_p` is the main primitive for staging out Python callbacks.
debug_callback_p = core.Primitive('debug_callback')
debug_callback_p.multiple_results = True

map, unsafe_map = util.safe_map, map

@debug_callback_p.def_impl
def debug_callback_impl(*args, callback: Callable[..., Any],
                        effect: DebugEffect):
  del effect
  try:
    cpu_device, *_ = jax.local_devices(backend="cpu")
  except RuntimeError as e:
    raise RuntimeError(
        "jax.debug.callback failed to find a local CPU device to place the"
        " inputs on. Make sure \"cpu\" is listed in --jax_platforms or the"
        " JAX_PLATFORMS environment variable."
    ) from e
  args = jax.device_put(args, cpu_device)
  with jax.default_device(cpu_device):
    try:
      callback(*args)
    except BaseException:
      logger.exception("jax.debug.callback failed")
      raise
  return ()

@debug_callback_p.def_effectful_abstract_eval
def debug_callback_abstract_eval(*flat_avals, callback: Callable[..., Any],
                                 effect: DebugEffect):
  del flat_avals, callback
  return [], {effect}

def debug_callback_batching_rule(args, dims, **params):
  """Unrolls the debug callback across the mapped axis."""
  axis_size = next(x.shape[i] for x, i in zip(args, dims)
                   if i is not None)
  # TODO(sharadmv): implement in terms of rolled loop unstead of
  # unrolled.
  def get_arg_at_dim(i, dim, arg):
    if dim is batching.not_mapped:
      # Broadcast unmapped argument
      return arg
    return lax.index_in_dim(arg, i, axis=dim, keepdims=False)
  outs = []
  for i in range(axis_size):
    args_idx = map(functools.partial(get_arg_at_dim, i), dims, args)
    outs.append(debug_callback_p.bind(*args_idx, **params))
  outs = [jnp.stack(xs) for xs in zip(*outs)]
  return outs, (0,) * len(outs)
batching.primitive_batchers[debug_callback_p] = debug_callback_batching_rule

def debug_callback_jvp_rule(primals, tangents, **params):
  return debug_callback_p.bind(*primals, **params), []
ad.primitive_jvps[debug_callback_p] = debug_callback_jvp_rule

def debug_callback_transpose_rule(*flat_args, callback: Callable[..., Any],
    effect: DebugEffect):
  del flat_args, callback, effect
  raise ValueError("Transpose doesn't support debugging callbacks.")
ad.primitive_transposes[debug_callback_p] = debug_callback_transpose_rule

def debug_callback_lowering(ctx, *args, effect, callback, **params):

  axis_context = ctx.module_context.axis_context
  if (isinstance(axis_context, sharding_impls.SPMDAxisContext) and
        set(axis_context.manual_axes) == set(axis_context.mesh.axis_names)):
    if config.use_shardy_partitioner.value:
      assert len(ctx.avals_out) == 1
      sharding = sharding_impls.SdyArrayShardingList([
          sharding_impls.SdyArraySharding(
              mesh_shape=(),
              dimension_shardings=[
                  sharding_impls.SdyDimSharding(axes=[], is_closed=True)
              ] * ctx.avals_out[0].ndim,
              logical_device_ids=())])
    else:
      # If we have fully manual sharding during lowering, that means the JAX
      # program has per-device semantics, so we run the callback on each device.
      sharding = xc.OpSharding()
      sharding.type = xc.OpSharding.Type.MANUAL
  elif isinstance(
      axis_context,
      (sharding_impls.ShardingContext, sharding_impls.SPMDAxisContext),
  ):
    if config.use_shardy_partitioner.value:
      sharding = sharding_impls.SdyArrayShardingList([
          sharding_impls.SdyArraySharding(
              mesh_shape=(), dimension_shardings=[], logical_device_ids=(0,))])
    else:
      # If we have fully automatic sharding during lowering, that means the JAX
      # program has bulk array semantics, so we run the callback with a MAXIMAL
      # sharding and hence execute it only once on the full logical value).
      # If we have partially automatic sharding, we do this too... not sure why!
      sharding = xc.OpSharding()
      sharding.type = xc.OpSharding.Type.MAXIMAL
      sharding.tile_assignment_dimensions = [1]
      sharding.tile_assignment_devices = [0]
  else:
    # When there's no SPMD partitioning going on, don't annotate a sharding.
    sharding = None

  def _callback(*flat_args):
    debug_callback_p.impl(
        *flat_args, effect=effect, callback=callback, **params)
    return ()
  if effects.ordered_effects.contains(effect):
    token = ctx.tokens_in.get(effect)
    result, token, _ = mlir.emit_python_callback(
        ctx, _callback, token, list(args), ctx.avals_in, ctx.avals_out,
        has_side_effect=True)
    ctx.set_tokens_out(mlir.TokenSet({effect: token}))
  else:
    result, _, _ = mlir.emit_python_callback(
        ctx, _callback, None, list(args), ctx.avals_in, ctx.avals_out,
        has_side_effect=True, sharding=sharding)
  return result
mlir.register_lowering(debug_callback_p, debug_callback_lowering,
                       platform="cpu")
mlir.register_lowering(
    debug_callback_p, debug_callback_lowering, platform="gpu")
mlir.register_lowering(
    debug_callback_p, debug_callback_lowering, platform="tpu")

def _debug_callback_partial_eval_custom(saveable, unks_in, inst_in, eqn):
  # The default behavior for effectful primitives is to not stage them if
  # possible. For debug callback, we actually want it to be staged to
  # provide more information to the user. This rule bypasses partial_eval's
  # regular behavior to do that. Specifically, we will stage the callback
  # if:
  # 1) the policy says debug_callbacks are not saveable
  # 2) the policy says debug_callbacks are saveable BUT all of the input
  #    values are instantiated.
  # The purpose is to call back with as much information as possible while
  # avoiding unnecessarily staging out other values.
  if any(unks_in):
    # The usual case (if we have any unknowns, we need to stage it out)
    res = [v for v, inst in zip(eqn.invars, inst_in) if not inst]
    return None, eqn, [], [], res
  if saveable(debug_callback_p, *[v.aval for v in eqn.invars], **eqn.params):
    # The policy is telling us we can save the debug callback.
    if all(inst_in):
      # If all of the inputs are instantiated, we also stage out the
      # debug_callback.
      return eqn, eqn, [], [], []
    else:
      # If any are not instantiated, we don't do any extra staging to avoid
      # affecting the computation.
      return eqn, None, [], [], []
  # If we can't save the debug callback (thanks to the policy) we listen to
  # the policy and stage out the debug callback.
  return eqn, eqn, [], [], []
pe.partial_eval_jaxpr_custom_rules[debug_callback_p] = (
    _debug_callback_partial_eval_custom)

@state_discharge.register_discharge_rule(debug_callback_p)
def _debug_callback_state_discharge_rule(
    in_avals, out_avals, *args, effect, callback, **params
):
  del in_avals, out_avals  # Unused.
  out = debug_callback_p.bind(*args, effect=effect, callback=callback, **params)
  return args, out

def debug_callback(callback: Callable[..., None], *args: Any,
                   ordered: bool = False, **kwargs: Any) -> None:
  """Calls a stageable Python callback.

  For more explanation, see `External Callbacks`_.

  ``jax.debug.callback`` enables you to pass in a Python function that can be called
  inside of a staged JAX program. A ``jax.debug.callback`` follows existing JAX
  transformation *pure* operational semantics, which are therefore unaware of
  side-effects. This means the effect could be dropped, duplicated, or
  potentially reordered in the presence of higher-order primitives and
  transformations.

  We want this behavior because we'd like ``jax.debug.callback`` to be "innocuous",
  i.e. we want these primitives to change the JAX computation as little as
  possible while revealing as much about them as possible, such as which parts
  of the computation are duplicated or dropped.

  Args:
    callback: A Python callable returning None.
    *args: The positional arguments to the callback.
    ordered: A keyword only argument used to indicate whether or not the
      staged out computation will enforce ordering of this callback w.r.t.
      other ordered callbacks.
    **kwargs: The keyword arguments to the callback.

  Returns:
    None

  See Also:
    - :func:`jax.experimental.io_callback`: callback designed for impure functions.
    - :func:`jax.pure_callback`: callback designed for pure functions.
    - :func:`jax.debug.print`: callback designed for printing.

  .. _External Callbacks: https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html
  """
  if not callable(callback):
    raise TypeError("first argument to jax.debug.callback must be callable, "
                    f"but got an object of type {type(callback)}")
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  static_args, dyn_args = {}, []
  for i, a in enumerate(flat_args):
    try:
      core.shaped_abstractify(a)
      dyn_args.append(a)
    except (AssertionError, TypeError):
      static_args[i] = a

  def _flat_callback(*dyn_args):
    all_args = [None] * (len(static_args) + len(dyn_args))
    di = iter(dyn_args)
    for i in range(len(all_args)):
      if i in static_args:
        all_args[i] = static_args[i]
      else:
        all_args[i] = next(di)
    assert next(di, None) is None
    args, kwargs = tree_util.tree_unflatten(in_tree, all_args)
    callback(*args, **kwargs)
    return ()

  effect = ordered_debug_effect if ordered else debug_effect
  debug_callback_p.bind(*dyn_args, callback=_flat_callback, effect=effect)

class _DebugPrintFormatChecker(string.Formatter):

  def format_field(self, value, format_spec):
    del value, format_spec
    return ""  # No formatting is done.

  def check_unused_args(self, used_args, args, kwargs):
    unused_args = [arg for i, arg in enumerate(args) if i not in used_args]
    unused_kwargs = [k for k in kwargs if k not in used_args]
    if unused_args:
      raise ValueError(
          f"Unused positional arguments to `jax.debug.print`: {unused_args}")
    if unused_kwargs:
      raise ValueError(
          f"Unused keyword arguments to `jax.debug.print`: {unused_kwargs}. "
          "You may be passing an f-string (i.e, `f\"{x}\"`) into "
          "`jax.debug.print` and instead should pass in a regular string.")

formatter = _DebugPrintFormatChecker()

def _format_print_callback(fmt: str, np_printoptions, *args, **kwargs):
  with np.printoptions(**np_printoptions):
    sys.stdout.write(fmt.format(*args, **kwargs) + "\n")

def debug_print(fmt: str, *args, ordered: bool = False, **kwargs) -> None:
  """Prints values and works in staged out JAX functions.

  This function does *not* work with f-strings because formatting is delayed.
  So instead of ``jax.debug.print(f"hello {bar}")``, write
  ``jax.debug.print("hello {bar}", bar=bar)``.

  This function is a thin convenience wrapper around :func:`jax.debug.callback`.
  The implementation is essentially::

    def debug_print(fmt: str, *args, **kwargs):
      jax.debug.callback(
          lambda *args, **kwargs: print(fmt.format(*args, **kwargs)),
          *args, **kwargs)

  It may be useful to call :func:`jax.debug.callback` directly instead of this
  convenience wrapper. For example, to get debug printing in logs, you might
  use :func:`jax.debug.callback` together with ``logging.log``.

  Args:
    fmt: A format string, e.g. ``"hello {x}"``, that will be used to format
      input arguments, like ``str.format``. See the Python docs on
      `string formatting <https://docs.python.org/3/library/stdtypes.html#str.format>`_
      and `format string syntax <https://docs.python.org/3/library/string.html#formatstrings>`_.
    *args: A list of positional arguments to be formatted, as if passed to
      ``fmt.format``.
    ordered: A keyword only argument used to indicate whether or not the
      staged out computation will enforce ordering of this ``jax.debug.print``
      w.r.t. other ordered ``jax.debug.print`` calls.
    **kwargs: Additional keyword arguments to be formatted, as if passed to
      ``fmt.format``.
  """
  # Check that we provide the correct arguments to be formatted.
  formatter.format(fmt, *args, **kwargs)

  debug_callback(functools.partial(_format_print_callback, fmt, np.get_printoptions()),
                 *args, **kwargs, ordered=ordered)


# Sharding visualization

inspect_sharding_p = core.Primitive("inspect_sharding")
inspect_sharding_p.multiple_results = True
dispatch.prim_requires_devices_during_lowering.add(inspect_sharding_p)

def _inspect_sharding_impl(value, *, callback):
  callback(value.sharding)
  return []
inspect_sharding_p.def_impl(_inspect_sharding_impl)

def _inspect_sharding_abstract_eval(aval, **_):
  del aval
  # Effectful abstract avoids DCE
  return [], {debug_effect}
inspect_sharding_p.def_effectful_abstract_eval(_inspect_sharding_abstract_eval)

def _inspect_sharding_batching_rule(args, _, *, callback):
  value, = args
  inspect_sharding_p.bind(value, callback=callback)
  return [], []
batching.primitive_batchers[inspect_sharding_p] = (
    _inspect_sharding_batching_rule)

def _inspect_sharding_jvp_rule(primals, _, **params):
  return inspect_sharding_p.bind(*primals, **params), []
ad.primitive_jvps[inspect_sharding_p] = _inspect_sharding_jvp_rule

sharding_callbacks = weakref.WeakValueDictionary()  # type: ignore
_INSPECT_SHARDING_CALL_NAME = "InspectSharding"

class ShardingCallbackInfo:
  def __init__(self, callback, module_context):
    self.callback = callback
    self.module_context = module_context

def _inspect_sharding_lowering_rule(ctx: mlir.LoweringRuleContext, value, *,
                                    callback):

  mesh = mesh_lib.thread_resources.env.physical_mesh
  axis_context = ctx.module_context.axis_context

  if isinstance(axis_context, sharding_impls.ShardingContext):
    devices = axis_context.device_assignment
    if devices is None:
      raise AssertionError(
          'Please file a bug at https://github.com/jax-ml/jax/issues')
    am = axis_context.abstract_mesh
    if am is not None:
      mesh = mesh_lib.Mesh(np.array(devices).reshape(am.axis_sizes),
                           am.axis_names)
  elif isinstance(axis_context, sharding_impls.SPMDAxisContext):
    devices = axis_context.mesh._flat_devices_tuple
  else:
    raise NotImplementedError(type(axis_context))
  assert devices is not None

  # If we have a nontrivial parallel computation, we need to wait until the SPMD
  # partitioner calls back with the `HloSharding.
  def _hlo_sharding_callback(hlo_sharding: xc.HloSharding):
    if mesh.empty:
      return callback(
          sharding_impls._op_sharding_to_pos_sharding(hlo_sharding, devices))
    pspec = parse_flatten_op_sharding(hlo_sharding, mesh)[0].get_partition_spec()
    return callback(NamedSharding(mesh, pspec))

  if len(devices) == 1:
    # If we only have one device in our computation, we can construct a
    # replicated HloSharding and call it right now.
    _hlo_sharding_callback(sharding_impls.get_replicated_hlo_sharding())
    return []

  key = xc.encode_inspect_sharding_callback(_hlo_sharding_callback)
  # We need to make sure `_hlo_sharding_callback` is still alive when the SPMD
  # partitioner runs so we keep it alive by attaching it to the executable.    #
  ctx.module_context.add_keepalive(_hlo_sharding_callback)

  hlo.CustomCallOp([value.type], [value],
                   call_target_name=ir.StringAttr.get(
                     _INSPECT_SHARDING_CALL_NAME),
                   has_side_effect=ir.BoolAttr.get(True),
                   api_version=mlir.i32_attr(1),
                   called_computations=ir.ArrayAttr.get([]),
                   backend_config=ir.StringAttr.get(key),
                   operand_layouts=None,
                   result_layouts=None)
  return []
mlir.register_lowering(inspect_sharding_p, _inspect_sharding_lowering_rule)

def inspect_sharding_prop_user_sharding(sharding, backend_string):
  del sharding, backend_string
  return []

def inspect_sharding_infer_sharding_from_operands(arg_shapes, arg_shardings,
                                                  shape, backend_string):
  del arg_shapes, shape, backend_string
  return arg_shardings[0]

def _slice_to_chunk_idx(size: int, slc: slice) -> int:
  if slc.stop == slc.start == None:
    return 0
  slice_size = slc.stop - slc.start
  assert slc.start % slice_size == 0
  assert size % slice_size == 0
  return slc.start // slice_size

def _raise_to_slice(slc: slice | int):
  if isinstance(slc, int):
    return slice(slc, slc + 1)
  return slc

Color = Union[tuple[float, float, float], str]
ColorMap = Callable[[float], tuple[float, float, float, float]]

def _canonicalize_color(color: Color) -> str:
  if isinstance(color, str):
    return color
  r, g, b = (int(a * 255) for a in color)
  return f"#{r:02X}{g:02X}{b:02X}"

def _get_text_color(color: str) -> str:
  r, g, b = map(lambda x: int(x, 16), (color[1:3], color[3:5], color[5:7]))
  if (r * 0.299 + g * 0.587 + b * 0.114) > 186:
    return "#000000"
  return "#ffffff"

def make_color_iter(color_map, num_rows, num_cols):
  num_colors = num_rows * num_cols
  color_values = np.linspace(0, 1, num_colors)
  idx = 0
  for _ in range(num_colors):
    yield color_map(color_values[idx])
    idx = (idx + num_colors // 2 + bool(num_colors % 2 == 0)) % num_colors

def visualize_sharding(shape: Sequence[int], sharding: Sharding, *,
                       use_color: bool = True, scale: float = 1.,
                       min_width: int = 9, max_width: int = 80,
                       color_map: ColorMap | None = None):
  """Visualizes a ``Sharding`` using ``rich``."""
  if not importlib.util.find_spec("rich"):
    raise ValueError("`visualize_sharding` requires `rich` to be installed.")

  # These imports are local so that they don't affect JAX import times.
  # pytype: disable=import-error
  import rich.align
  import rich.console
  import rich.box
  import rich.padding
  import rich.style
  import rich.table
  # pytype: enable=import-error

  if len(shape) > 2 or len(shape) < 1:
    raise ValueError(
        "`visualize_sharding` only works for shapes with 1 and 2 dimensions.")
  console = rich.console.Console(width=max_width)
  use_color = use_color and console.color_system is not None
  if use_color and not color_map:
    try:
      import matplotlib as mpl  # pytype: disable=import-error
      color_map = mpl.colormaps["tab20b"]
    except ModuleNotFoundError:
      use_color = False

  base_height = int(10 * scale)
  aspect_ratio = (shape[1] if len(shape) == 2 else 1) / shape[0]
  base_width = int(base_height * aspect_ratio)
  height_to_width_ratio = 2.5

  # Grab the device kind from the first device
  device_kind = next(iter(sharding.device_set)).platform.upper()

  device_indices_map = sharding.devices_indices_map(tuple(shape))
  slices: dict[tuple[int, ...], set[int]] = {}
  heights: dict[tuple[int, ...], float | None] = {}
  widths: dict[tuple[int, ...], float] = {}

  for i, (dev, slcs) in enumerate(device_indices_map.items()):
    assert slcs is not None
    slcs = tuple(map(_raise_to_slice, slcs))
    chunk_idxs = tuple(map(_slice_to_chunk_idx, shape, slcs))
    if slcs is None:
      raise NotImplementedError
    if len(slcs) == 2:
      vert, horiz = slcs
      vert_size  = ((vert.stop  - vert.start ) if vert.stop  is not None
                    else shape[0])
      horiz_size = ((horiz.stop - horiz.start) if horiz.stop is not None
                    else shape[1])
      chunk_height = vert_size / shape[0]
      chunk_width = horiz_size / shape[1]
      heights[chunk_idxs] = chunk_height
      widths[chunk_idxs] = chunk_width
    else:
      # In the 1D case, we set the height to 1.
      horiz, = slcs
      vert = slice(0, 1, None)
      horiz_size = (
          (horiz.stop - horiz.start) if horiz.stop is not None else shape[0])
      chunk_idxs = (0, *chunk_idxs)
      heights[chunk_idxs] = None
      widths[chunk_idxs]  = horiz_size / shape[0]
    slices.setdefault(chunk_idxs, set()).add(dev.id)
  num_rows = max(a[0] for a in slices.keys()) + 1
  if len(list(slices.keys())[0]) == 1:
    num_cols = 1
  else:
    num_cols = max(a[1] for a in slices.keys()) + 1

  color_iter = make_color_iter(color_map, num_rows, num_cols)
  table = rich.table.Table(show_header=False, show_lines=not use_color,
                           padding=0,
                           highlight=not use_color, pad_edge=False,
                           box=rich.box.SQUARE if not use_color else None)
  for i in range(num_rows):
    col = []
    for j in range(num_cols):
      entry = f"{device_kind} "+",".join([str(s) for s in sorted(slices[i, j])])
      width, maybe_height = widths[i, j], heights[i, j]
      width = int(width * base_width * height_to_width_ratio)
      if maybe_height is None:
        height = 1
      else:
        height = int(maybe_height * base_height)
      width = min(max(width, min_width), max_width)
      left_padding, remainder = divmod(width - len(entry) - 2, 2)
      right_padding = left_padding + remainder
      top_padding, remainder = divmod(height - 2, 2)
      bottom_padding = top_padding + remainder
      if use_color:
        color = _canonicalize_color(next(color_iter)[:3])
        text_color = _get_text_color(color)
        top_padding += 1
        bottom_padding += 1
        left_padding += 1
        right_padding += 1
      else:
        color = None
        text_color = None
      padding = (top_padding, right_padding, bottom_padding, left_padding)
      padding = tuple(max(x, 0) for x in padding)  # type: ignore
      col.append(
          rich.padding.Padding(
            rich.align.Align(entry, "center", vertical="middle"), padding,
            style=rich.style.Style(bgcolor=color,
              color=text_color)))
    table.add_row(*col)
  console.print(table, end='\n\n')

def inspect_array_sharding(value, *, callback: Callable[[Sharding], None]):
  """Enables inspecting array sharding inside JIT-ted functions.

  This function, when provided with a Pytree of arrays, calls back with each of
  their shardings and works in ``pjit``-ted computations, enabling inspecting
  the chosen intermediate shardings.

  The policy for when ``callback`` is called is *as early as possible* when the
  sharding information is available. This means if ``inspect_array_callback`` is
  called without any transformations, the callback will happen immediately
  since we have the array and its sharding readily available. Inside of a
  ``jax.jit``, the callback will happen at lowering time, meaning you can
  trigger the callback using the AOT API (``jit(f).lower(...)``). When inside of
  a ``pjit``, the callback happens *at compile time* since the sharding is
  determined by XLA. You can trigger the callback by using JAX's AOT API
  (``pjit(f).lower(...).compile()``). In all cases, the callback will be
  triggered by running the function, since running a function entails lowering
  and compiling it first. However, once the function is compiled and cached,
  the callback will no longer occur.

  This function is experimental and its behavior may change in the future.

  Args:
    value: A Pytree of JAX arrays.
    callback: A callable that takes in a ``Sharding`` and doesn't return a value.

  In the following example, we print out the sharding of an intermediate value
  in a ``pjit``-ted computation:

  >>> import jax
  >>> import jax.numpy as jnp
  >>> from jax.experimental.pjit import pjit
  >>> from jax.sharding import Mesh, PartitionSpec
  >>>
  >>> x = jnp.arange(8, dtype=jnp.float32)
  >>> def f_(x):
  ...   x = jnp.sin(x)
  ...   jax.debug.inspect_array_sharding(x, callback=print)
  ...   return jnp.square(x)
  >>> f = pjit(f_, in_shardings=PartitionSpec('dev'),
  ...          out_shardings=PartitionSpec('dev'))
  >>> with Mesh(jax.devices(), ('dev',)):
  ...   f.lower(x).compile()  # doctest: +SKIP
  ...
  NamedSharding(mesh={'dev': 8}, partition_spec=PartitionSpec(('dev',),))
  """
  def _inspect(val):
    inspect_sharding_p.bind(val, callback=callback)
  tree_util.tree_map(_inspect, value)

def visualize_array_sharding(arr, **kwargs):
  """Visualizes an array's sharding."""
  def _visualize(sharding):
    return visualize_sharding(arr.shape, sharding, **kwargs)
  inspect_array_sharding(arr, callback=_visualize)
