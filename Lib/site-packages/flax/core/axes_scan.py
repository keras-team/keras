# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper around jax.lax.scan with in_axes/out_axes API."""
from collections.abc import Callable
import functools
from typing import Any, Optional

import jax
from jax import core
from jax import lax
from jax.extend import linear_util as lu
from jax.interpreters import partial_eval as pe
import jax.numpy as jnp
import numpy as np

ScanAxis = Optional[int]


class _Broadcast:
  pass


broadcast = _Broadcast()


def scan(
    fn: Callable[..., Any],
    in_axes: Any,
    out_axes: Any,
    length: int | None = None,
    reverse: bool = False,
    unroll: int = 1,
    _split_transpose: bool = False,
    check_constancy_invariants: bool = True,
):
  """A wrapper around `jax.lax.scan` with in_axes/out_axes api.

  Example::
    def body_fn(b, c, x):
      return b + 2, c + 1, 2 * x

    loop = scan(body_fn, in_axes=0, out_axes=0)
    broadcast_in = 1
    carry = 2
    xs = jnp.arange(3)
    broadcast_out, carry, ys = loop(broadcast_in, carry, xs)
    print(broadcast_out)  # prints: 3
    print(carry)  # prints: 5
    print(ys)  # prints: [0, 2, 4]


  Args:
    fn: the body function of the scan loop of the form
      `(broadcast_in, carry, *args) -> (broadcast_out, carry, scan_out)`.
      the broadcast argument allows for loop independent inputs/outputs to
      be computed inside `fn`. `fn` will be called once to compute
      `broadcast_out`. The actual loop will receive `broadcast_out` as the new
      `broadcast_in`. This is useful for initializing values inside the loop.
    in_axes: specifies the axis along which arguments are scanned.
      Use `broadcast` to use the same value across iterations.
    out_axes: specifies the axis along which outputs are concatenated.
      Use `broadcast` if a return value should not be concatenated and
      is independent of the loop body.
    length: number of iterations. Only needs to be specified if there
      is no scan axis from which it can be derived.
    reverse: scan in reverse order from end to start.
    unroll: how many scan iterations to unroll within a single
      iteration of a loop (default: 1).
    _split_transpose: An experimental feature to split the transpose of scan
       into a scan and a map, backed by an experimental Jax lax.scan() feature.
    check_constancy_invariants: If true, the scan will verify that the
      broadcast constants are true loop invariants, and further supports
      broadcast function (non-carry) outputs.  This requires an extra jax
      tracing step however, so setting to false can reduce trace time on larger
      models.
  Returns:
     the function that performs the scan of the form:
     (broadcast_in, carry_in, *args) -> (broadcast_out, carry_out, scan_out).
  """

  def transpose_to_front(ax, xs):
    if ax is broadcast:
      return ()
    if ax == 0:
      return xs

    def trans(x):
      perm = tuple(range(x.ndim))
      perm = (ax,) + tuple(np.delete(perm, ax))
      return jnp.transpose(x, perm)

    return jax.tree_util.tree_map(trans, xs)

  def transpose_from_front(ax, xs):
    if ax is broadcast:
      return ()
    if ax == 0:
      return xs

    def trans(x):
      if ax < 0:
        pax = x.ndim + ax
      else:
        pax = ax
      assert pax < x.ndim
      perm = tuple(range(1, pax + 1)) + (0,) + tuple(range(pax + 1, x.ndim))
      return jnp.transpose(x, perm)

    return jax.tree_util.tree_map(trans, xs)

  def scan_fn(broadcast_in, init, *args):
    # Requires one extra tracing operation to test invariants:
    # Verifies that broadcast constants are true loop invariants, and further
    # supports broadcast function (non-carry) outputs.

    xs = jax.tree_util.tree_map(transpose_to_front, in_axes, args)

    def body_fn(c, xs, init_mode=False):
      # inject constants
      xs = jax.tree_util.tree_map(
          lambda ax, arg, x: (arg if ax is broadcast else x), in_axes, args, xs
      )
      broadcast_out, c, ys = fn(broadcast_in, c, *xs)

      if init_mode:
        ys = jax.tree_util.tree_map(
            lambda ax, y: (y if ax is broadcast else ()), out_axes, ys
        )
        return broadcast_out, ys
      else:
        ys = jax.tree_util.tree_map(
            lambda ax, y: (() if ax is broadcast else y), out_axes, ys
        )
        return c, ys

    broadcast_body = functools.partial(body_fn, init_mode=True)

    carry_avals = jax.tree_util.tree_map(
        lambda x: core.ShapedArray(jnp.shape(x), jnp.result_type(x)), init
    )
    scan_avals = jax.tree_util.tree_map(
        lambda x: core.ShapedArray(jnp.shape(x)[1:], jnp.result_type(x)), xs
    )
    input_avals = (carry_avals, scan_avals)

    in_avals, in_tree = jax.tree_util.tree_flatten(input_avals)
    debug_info = jax.api_util.debug_info("flax scan", broadcast_body,
                                         (in_tree,), {})
    f_flat, out_tree = jax.api_util.flatten_fun_nokwargs(
        lu.wrap_init(broadcast_body, debug_info=debug_info), in_tree
    )
    in_pvals = list(map(pe.PartialVal.unknown, in_avals))
    _, out_pvals, _ = pe.trace_to_jaxpr_nounits(f_flat, in_pvals)

    out_flat = []
    for pv, const in out_pvals:
      if pv is not None:
        raise ValueError(
            'broadcasted variable has a data dependency on the scan body.'
        )
      out_flat.append(const)
    broadcast_in, constants_out = jax.tree_util.tree_unflatten(
        out_tree(), out_flat
    )

    if jax.version.__version_info__ > (0, 4, 25):
      c, ys = lax.scan(
          body_fn, init, xs, length=length, reverse=reverse, unroll=unroll,
          _split_transpose=_split_transpose
      )
    else:
      c, ys = lax.scan(
          body_fn, init, xs, length=length, reverse=reverse, unroll=unroll
      )
    ys = jax.tree_util.tree_map(transpose_from_front, out_axes, ys)
    ys = jax.tree_util.tree_map(
        lambda ax, const, y: (const if ax is broadcast else y),
        out_axes,
        constants_out,
        ys,
    )
    return broadcast_in, c, ys

  def simple_scan_fn(broadcast_in, init, *args):
    # Saves an extra tracing operation.
    # No verification of constancy, and no support for non-carry broadcast
    # function outputs.
    xs = jax.tree_util.tree_map(transpose_to_front, in_axes, args)

    if broadcast in jax.tree_util.tree_leaves(out_axes):
      raise ValueError(f"nn.scan run with check_constancy_invariants=False "
                       f"does not support broadcast non-carry function "
                       f"outputs.  out_axes was given as {out_axes}")

    def body_fn(c, xs):
      # inject constants
      xs = jax.tree_util.tree_map(
          lambda ax, arg, x: (arg if ax is broadcast else x), in_axes, args, xs
      )
      _, c, ys = fn(broadcast_in, c, *xs)
      return c, ys

    if jax.version.__version_info__ > (0, 4, 25):
      c, ys = lax.scan(
          body_fn, init, xs, length=length, reverse=reverse, unroll=unroll,
          _split_transpose=_split_transpose
      )
    else:
      c, ys = lax.scan(
          body_fn, init, xs, length=length, reverse=reverse, unroll=unroll
      )
    ys = jax.tree_util.tree_map(transpose_from_front, out_axes, ys)
    return broadcast_in, c, ys

  if check_constancy_invariants:
    return scan_fn
  else:
    return simple_scan_fn
