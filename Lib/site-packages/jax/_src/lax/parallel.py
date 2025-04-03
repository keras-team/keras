# Copyright 2019 The JAX Authors.
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
"""
Parallelization primitives.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial
import itertools
import math

from jax import tree_util
from jax._src import core
from jax._src import config
from jax._src import dispatch
from jax._src import dtypes
from jax._src.sharding_impls import (SPMDAxisContext, ShardingContext,
                                     NamedSharding, PartitionSpec as P)
from jax._src.core import AxisName, ShapedArray
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lax import lax
from jax._src.lax import slicing
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.util import (canonicalize_axis, moveaxis, safe_map, safe_zip,
                           unzip2)
import numpy as np

unsafe_map, map = map, safe_map  # type: ignore
unsafe_zip, zip = zip, safe_zip  # type: ignore


### parallel traceables

def psum(x, axis_name, *, axis_index_groups=None):
  """Compute an all-reduce sum on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Inputs of boolean dtype are converted to integers before the reduction.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would perform psums over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce sum along the axis ``axis_name``.

  Examples:
    For example, with 4 XLA devices available:

    >>> x = np.arange(4)
    >>> y = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(x)
    >>> print(y)
    [6 6 6 6]
    >>> y = jax.pmap(lambda x: x / jax.lax.psum(x, 'i'), axis_name='i')(x)
    >>> print(y)
    [0.         0.16666667 0.33333334 0.5       ]

    Suppose we want to perform ``psum`` among two groups, one with ``device0`` and ``device1``, the other with ``device2`` and ``device3``,

    >>> y = jax.pmap(lambda x: jax.lax.psum(x, 'i', axis_index_groups=[[0, 1], [2, 3]]), axis_name='i')(x)
    >>> print(y)
    [1 1 5 5]

    An example using 2D-shaped x. Each row is data from one device.

    >>> x = np.arange(16).reshape(4, 4)
    >>> print(x)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]

    Full ``psum`` across all devices:

    >>> y = jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(x)
    >>> print(y)
    [[24 28 32 36]
     [24 28 32 36]
     [24 28 32 36]
     [24 28 32 36]]

    Perform ``psum`` among two groups:

    >>> y = jax.pmap(lambda x: jax.lax.psum(x, 'i', axis_index_groups=[[0, 1], [2, 3]]), axis_name='i')(x)
    >>> print(y)
    [[ 4  6  8 10]
     [ 4  6  8 10]
     [20 22 24 26]
     [20 22 24 26]]
  """
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if any(isinstance(axis, int) for axis in axis_name) and axis_index_groups is not None:
    raise ValueError("axis_index_groups only supported for sums over just named axes")
  _validate_reduce_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  leaves = [lax.convert_element_type(l, np.int32)
            if dtypes.dtype(l) == np.bool_ else l for l in leaves]
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  # handle the constant case specially
  if all(not isinstance(leaf, core.Tracer) for leaf in leaves):
    named_axes, pos_axes = axes_partition = [], []
    for axis in axis_name:
      axes_partition[isinstance(axis, int)].append(axis)
    def pos_reduce(x):
      if not pos_axes:
        return x
      return lax._reduce_sum(x, [canonicalize_axis(axis, getattr(x, 'ndim', 0))
                                 for axis in pos_axes])
    if axis_index_groups is not None:
      assert not pos_axes
      size = len(axis_index_groups[0])
    else:
      size = math.prod([core.get_axis_env().axis_size(name) for name in named_axes])
    out_flat = tuple(lax._const(leaf, size) * pos_reduce(leaf) for leaf in leaves)
  else:
    out_flat = psum_p.bind(
        *leaves, axes=tuple(axis_name), axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, out_flat)

def pmean(x, axis_name, *, axis_index_groups=None):
  """Compute an all-reduce mean on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would perform pmeans over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and on TPUs all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce mean along the axis ``axis_name``.

  For example, with 4 XLA devices available:

  >>> x = np.arange(4)
  >>> y = jax.pmap(lambda x: jax.lax.pmean(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [1.5 1.5 1.5 1.5]
  >>> y = jax.pmap(lambda x: x / jax.lax.pmean(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [0.        0.6666667 1.3333334 2.       ]
  """
  x = psum(x, axis_name=axis_name, axis_index_groups=axis_index_groups)
  n = psum(1, axis_name=axis_name, axis_index_groups=axis_index_groups)
  return tree_util.tree_map(lambda v: v / n, x)

def pmax(x, axis_name, *, axis_index_groups=None):
  """Compute an all-reduce max on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would perform pmaxes over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and on TPUs all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce max along the axis ``axis_name``.
  """
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if any(isinstance(axis, int) for axis in axis_name) and axis_index_groups is not None:
    raise ValueError("axis_index_groups only supported for sums over just named axes")
  _validate_reduce_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  out_flat = pmax_p.bind(*leaves, axes=axis_name,
                         axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, out_flat)

def pmin(x, axis_name, *, axis_index_groups=None):
  """Compute an all-reduce min on ``x`` over the pmapped axis ``axis_name``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would perform pmins over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and on TPUs all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x`` representing the result of an
    all-reduce min along the axis ``axis_name``.
  """
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if any(isinstance(axis, int) for axis in axis_name) and axis_index_groups is not None:
    raise ValueError("axis_index_groups only supported for sums over just named axes")
  _validate_reduce_axis_index_groups(axis_index_groups)
  leaves, treedef = tree_util.tree_flatten(x)
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  out_flat = pmin_p.bind(*leaves, axes=axis_name,
                         axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, out_flat)

# TODO(mattjj): add a pargmin_p, or add named axis support to lax.argmin_p
def pargmin(x, axis_name):
  if isinstance(axis_name, (tuple, list)):
    raise TypeError(f"pargmin only accepts a single axis, got {axis_name}")
  return _axis_index_of_val(x, pmin(x, axis_name), axis_name)

# TODO(mattjj): add a pargmax_p, or add named axis support to lax.argmax_p
def pargmax(x, axis_name):
  if isinstance(axis_name, (tuple, list)):
    raise TypeError(f"pargmin only accepts a single axis, got {axis_name}")
  return _axis_index_of_val(x, pmax(x, axis_name), axis_name)

def _axis_index_of_val(x, val, axis_name):
  idx = axis_index(axis_name)
  mask = (val == x)
  validx = lax.select(mask,
                      lax.full(mask.shape, idx),
                      lax.full(mask.shape, dtypes.iinfo(dtypes.dtype(idx)).max, dtypes.dtype(idx)))
  return pmin(validx, axis_name)

def _validate_reduce_axis_index_groups(axis_index_groups):
  if axis_index_groups is None:
    return
  axis_space = range(sum(len(group) for group in axis_index_groups))
  if {i for g in axis_index_groups for i in g} != set(axis_space):
    raise ValueError("axis_index_groups must cover all indices exactly once")

def _canonicalize_axis_index_groups(axis_index_groups):
  if axis_index_groups is None:
    return
  return tuple(map(tuple, axis_index_groups))


def pbroadcast(x, axis_name, source):
  """Perform a collective broadcast and replicate from ``source``.

  This is equivalent to
  ```
  def pbroadcast(x, axis_name, source):
    masked = jnp.where(axis_index(axis_name) == source, x, zeros_like(x))
    return psum(masked, axis_name)
  ```
  but implemented in a hardware optimized way.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  This function is an analog of the CollectiveBroadcast HLO.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    source: int, representing which index into ``axis_name`` that should be copied.

  Returns:
    Array(s) with ``x`` being copied from the ``source`` index slice of ``axis_name``.
  """
  return tree_util.tree_map(
      partial(pbroadcast_p.bind, axis_name=axis_name, source=source), x)


def ppermute(x, axis_name, perm):
  """Perform a collective permutation according to the permutation ``perm``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  This function is an analog of the CollectivePermute HLO.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    perm: list of pairs of ints, representing
      ``(source_index, destination_index)``
      pairs that encode how the mapped axis named ``axis_name`` should be
      shuffled. The integer values are treated as indices into the mapped axis
      ``axis_name``. Any two pairs should not have the same source index or the
      same destination index. For each index of the axis ``axis_name`` that does
      not correspond to a destination index in ``perm``, the corresponding
      values in the result are filled with zeros of the appropriate type.

  Returns:
    Array(s) with the same shape as ``x`` with slices along the axis
    ``axis_name`` gathered from ``x`` according to the permutation ``perm``.
  """
  if not isinstance(axis_name, (list, tuple)):
    axis_name = (axis_name,)
  return tree_util.tree_map(
      partial(ppermute_p.bind, axis_name=axis_name,
              perm=tuple(map(tuple, perm))), x)

def pshuffle(x, axis_name, perm):
  """Convenience wrapper of jax.lax.ppermute with alternate permutation encoding

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    perm: list of ints encoding sources for the permutation to be applied to
      the axis named ``axis_name``, so that the output at axis index i
      comes from the input at axis index perm[i]. Every integer in [0, N) should
      be included exactly once for axis size N.

  Returns:
    Array(s) with the same shape as ``x`` with slices along the axis
    ``axis_name`` gathered from ``x`` according to the permutation ``perm``.
  """
  if set(perm) != set(range(len(perm))):
    raise ValueError(f"`perm` does not represent a permutation: {perm}")
  return ppermute(x, axis_name, list(zip(perm, range(len(perm)))))


def pswapaxes(x, axis_name, axis, *, axis_index_groups=None):
  """Swap the pmapped axis ``axis_name`` with the unmapped axis ``axis``.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  The group size of the mapped axis size must be equal to the size of the
  unmapped axis; that is, we must have
  ``lax.psum(1, axis_name, axis_index_groups=axis_index_groups) == x.shape[axis]``.
  By default, when ``axis_index_groups=None``, this encompasses all the devices.

  This function is a special case of ``all_to_all`` where the pmapped axis of
  the input is placed at the position ``axis`` in the output. That is, it is
  equivalent to ``all_to_all(x, axis_name, axis, axis)``.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would run pswapaxes over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and all groups must be the same size.

  Returns:
    Array(s) with the same shape as ``x``.
  """
  return all_to_all(x, axis_name, axis, axis, axis_index_groups=axis_index_groups)

def all_to_all(x, axis_name, split_axis, concat_axis, *, axis_index_groups=None, tiled=False):
  """Materialize the mapped axis and map a different axis.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  In the output, the input mapped axis ``axis_name`` is materialized at the
  logical axis position ``concat_axis``, and the input unmapped axis at position
  ``split_axis`` is mapped with the name ``axis_name``.

  The group size of the mapped axis size must be equal to the size of the
  unmapped axis; that is, we must have
  ``lax.psum(1, axis_name, axis_index_groups=axis_index_groups) == x.shape[axis]``.
  By default, when ``axis_index_groups=None``, this encompasses all the devices.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    split_axis: int indicating the unmapped axis of ``x`` to map with the name
      ``axis_name``.
    concat_axis: int indicating the position in the output to materialize the
      mapped axis of the input with the name ``axis_name``.
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would run all_to_all over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and all groups must be the same size.
    tiled: when True, all_to_all will divide split_axis into chunks and concatenate
      them along concat_axis. In particular, no dimensions are added or removed.
      False by default.

  Returns:
    When tiled is False, array(s) with shape given by the expression::

      np.insert(np.delete(x.shape, split_axis), concat_axis, axis_size)

    where ``axis_size`` is the size of the mapped axis named ``axis_name`` in
    the input ``x``, i.e. ``axis_size = lax.psum(1, axis_name)``.

    Otherwise array with shape similar to the input shape, except with split_axis
    divided by axis size and concat_axis multiplied by axis size.
  """
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  def bind(x, split_axis=split_axis, concat_axis=concat_axis):
    group_size = psum(1, axis_name, axis_index_groups=axis_index_groups)
    if tiled:
      if x.shape[split_axis] % group_size != 0:
        raise ValueError(f"The size of all_to_all split_axis ({x.shape[split_axis]}) "
                         f"has to be divisible by the size of the named axis "
                         f"{axis_name} ({group_size})")
    else:
      if group_size != x.shape[split_axis]:
        msg = ("all_to_all requires the size of the mapped axis axis_name to "
               "equal x.shape[split_axis], but they are {} and {} respectively.")
        raise ValueError(msg.format(group_size, x.shape[split_axis]))
      if split_axis < concat_axis:
        concat_axis += 1  # concat_axis gives a position _after_ split_axis is removed
        x = lax.expand_dims(x, (concat_axis,))  # insert the new axis
      elif split_axis == concat_axis:
        pass
      else:  # concat_axis < split_axis
        x = lax.expand_dims(x, (concat_axis,))  # insert the new axis
        split_axis += 1   # we have a new axis before split_axis now
    result = all_to_all_p.bind(x, split_axis=split_axis, concat_axis=concat_axis,
                               axis_name=axis_name,
                               axis_index_groups=axis_index_groups,
                               tiled=tiled)
    if not tiled and split_axis != concat_axis:
      result = lax.squeeze(result, (split_axis,))
    return result

  return tree_util.tree_map(bind, x)

def ragged_all_to_all(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes):
  """Ragged version of :func:`all_to_all`.

  For now, ``split_axis`` and ``concat_axis`` from `all_to_all` are equivalent
  and the outermost (ragged) dimension. ``axis_index_groups`` is default to all
  replicas (e.g. there is only one group and covers all axis indices).

  Ragged arrays are defined by a set of three arrays:
  * ``data``: the ``data`` array is "ragged" along its outermost dimension,
    along which each indexed element has variable size.
  * ``offsets``: the ``offsets`` array indexes the outermost dimension of the
    ``data`` array, and represents the starting offset of each ragged element of
    the ``data`` array.
  * ``sizes``: the ``sizes`` array represents the size of each ragged element of
    the ``data`` array, where the size is specified in units of sub-elements. A
    sub-element is defined as the suffix of the ``data`` array shape obtained by
    removing the outermost "ragged" dimension.
  The ``offsets`` and ``sizes`` arrays must have the same size.

  # Example ragged tensor
  data: [8,3] = {{a,b,c},{d,e,f},{g,h,i},{j,k,l},{m,n,o},{p,q,r},{s,t,u},{v,w,x}}
  offsets: [3] = {0, 1, 4}
  sizes: [3] = {1, 3, 4}

  # Index 'data' at 'offsets'[0], 'sizes'[0]'
  {a,b,c}

  # Index 'data' at 'offsets'[1], 'sizes'[1]'
  {d,e,f},{g,h,i},{j,k,l}

  # Index 'data' at 'offsets'[2], 'sizes'[2]'
  {m,n,o},{p,q,r},{s,t,u},{v,w,x}


  ``output_offsets`` must be sharded in a way that each replica has offsets in
  the target replica output perspective.

  For i-th output offset, the current replica will send
  `operand[input_offsets[i]:input_offsets[i]+input_sizes[i]]` update to `i`-th
  replica that will be written to
  `output_i[output_offsets[i]:output_offsets[i]+send_sizes[i]]` in `i`-th
  replica ``output``.

  For example, if we have 2 replicas:

  replica 0:
    operand: [1, 2, 2]
    output: [0, 0, 0, 0]
    input_offsets: [0, 1]
    send_sizes: [1, 2]
    output_offsets: [0, 0]
    recv_sizes: [1, 1]

  replica 1:
    operand: [3, 4, 0]
    output: [0, 0, 0, 0]
    input_offsets: [0, 1]
    send_sizes: [1, 1]
    output_offsets: [1, 2]
    recv_sizes: [2, 1]

  replica 0's result will be: [1, 3, 0, 0]
  replica 1's result will be: [2, 2, 4, 0]

  Args:
    operand: array with ragged dimension along its outermost dimension.
    output: array of ragged input offsets.
    input_offsets: array of ragged input send sizes.
    send_sizes: array of ragged output data.
    output_offsets: array of ragged offsets in the target replica output.
    recv_sizes: array of ragged output receive sizes.

  Returns:
    array with shape equal to ``output``.
  """
  return ragged_all_to_all_p.bind(operand, output, input_offsets, send_sizes,
                                  output_offsets, recv_sizes)

ragged_all_to_all_p = core.Primitive('ragged_all_to_all')


def axis_index(axis_name):
  """Return the index along the mapped axis ``axis_name``.

  Args:
    axis_name: hashable Python object used to name the mapped axis.

  Returns:
    An integer representing the index.

  For example, with 8 XLA devices available:

  >>> from functools import partial
  >>> @partial(jax.pmap, axis_name='i')
  ... def f(_):
  ...   return lax.axis_index('i')
  ...
  >>> f(np.zeros(4))
  Array([0, 1, 2, 3], dtype=int32)
  >>> f(np.zeros(8))
  Array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int32)
  >>> @partial(jax.pmap, axis_name='i')
  ... @partial(jax.pmap, axis_name='j')
  ... def f(_):
  ...   return lax.axis_index('i'), lax.axis_index('j')
  ...
  >>> x, y = f(np.zeros((4, 2)))
  >>> print(x)
  [[0 0]
  [1 1]
  [2 2]
  [3 3]]
  >>> print(y)
  [[0 1]
  [0 1]
  [0 1]
  [0 1]]
  """
  if not isinstance(axis_name, (tuple, list)):
    return axis_index_p.bind(axis_name=axis_name)
  else:
    inner_size = 1
    index = 0
    for name in reversed(axis_name):
      index += axis_index(name) * inner_size
      inner_size *= psum(1, name)
    return index

def pgather(src, idx, axes: int | AxisName):
  """Uses the last positional axis of idx to index into src's axes."""
  if not isinstance(axes, (tuple, list)):
    axes = (axes,)
  # TODO: Canonicalize exes!
  return pgather_p.bind(src, idx, axes=tuple(axes))


### parallel primitives

def _names_in_param(pname: str, params: core.ParamDict) -> tuple[str]:
  axis_names = params[pname]
  if isinstance(axis_names, (tuple, list)):
    return tuple(axis_names)
  else:
    return (axis_names,)

def _constant_reduction(prim, axis_data, args, axes, axis_index_groups):
  assert axis_data.name in axes
  if axis_index_groups: raise NotImplementedError
  new_axes = tuple(n for n in axes if n != axis_data.name)
  if new_axes:
    args = prim.bind(*args, axes=new_axes, axis_index_groups=axis_index_groups)
  if prim is psum_p:
    outs = [lax._const(x, axis_data.size) * x for x in args]
  elif prim in (pmin_p, pmax_p):
    outs = args
  else:
    raise Exception(f"Unrecognized reducer: {prim}")

  return outs, [None] * len(outs)

def _reduction_with_positional_batcher(
    prim, vals_in, dims_in, axis_index_groups,
    transform_unmapped, transform_mapped):
  if axis_index_groups is not None:
    raise NotImplementedError("axis_index_groups not supported in vmap collectives. "
                              "Please open a feature request!")
  vals_in = [val if d is batching.not_mapped or d == 0 else _moveaxis(d, 0, val)
             for val, d in zip(vals_in, dims_in)]
  mapped_vals_in, unmapped_vals_in = partitioned_vals_in = [], []
  mapped_idxs, unmapped_idxs = partitioned_idxs = [], []
  for i, (val, d) in enumerate(zip(vals_in, dims_in)):
    partitioned_vals_in[d is batching.not_mapped].append(val)
    partitioned_idxs[d is batching.not_mapped].append(i)
  vals_out = [None] * len(vals_in)
  if unmapped_vals_in:
    unmapped_axes, unmapped_vals_in = transform_unmapped(0, unmapped_vals_in)
    unmapped_vals_out = prim.bind(*unmapped_vals_in, axes=unmapped_axes, axis_index_groups=None)
    for i, val in zip(unmapped_idxs, unmapped_vals_out):
      vals_out[i] = val
  if mapped_vals_in:
    mapped_axes, mapped_vals_in = transform_mapped(0, mapped_vals_in)
    mapped_vals_out = prim.bind(*mapped_vals_in, axes=mapped_axes, axis_index_groups=None)
    for i, val in zip(mapped_idxs, mapped_vals_out):
      vals_out[i] = val
  assert all(v is not None for v in vals_out)
  return vals_out

def _reduction_batcher(prim, vals_in, dims_in, *, axes, axis_index_groups):
  assert prim.multiple_results
  if not any(isinstance(axis, int) for axis in axes):
    return prim.bind(*vals_in, axes=axes, axis_index_groups=axis_index_groups), dims_in
  vals_out = _reduction_with_positional_batcher(
      prim, vals_in, dims_in, axis_index_groups,
      lambda d, d_vals_in: (axes, d_vals_in),
      lambda d, d_vals_in: (tuple(axis + (axis >= d) if isinstance(axis, int) else axis
                                  for axis in axes),
                            d_vals_in))
  # _reduction_with_positional_batcher moves all map dims to 0
  return vals_out, [d if d is batching.not_mapped else 0 for d in dims_in]

def _batched_reduction_collective(
    prim, if_unmapped, axis_data, vals_in, dims_in, axes,
    axis_index_groups):
  assert prim.multiple_results
  if all(d is None for d in dims_in):
    if axis_data.name in axes:
      return _constant_reduction(prim, axis_data, vals_in, axes, axis_index_groups)
    else:
      return prim.bind(*vals_in, axes=axes, axis_index_groups=axis_index_groups), dims_in

  if axis_data.name not in axes:
    return _reduction_batcher(prim, vals_in, dims_in, axes=axes,
                              axis_index_groups=axis_index_groups)

  # Note that we have a choice here. We can either unfuse the reduction into one
  # that handles the batched dims and then another one that handles the rest.
  # Alternatively, we can keep the dimension reduction fused with the rest, but
  # we have to split the primitive into one for unmapped inputs and another
  # one for mapped, because they differ in their `axes` parameter.
  # We choose the second strategy here.
  vals_out = _reduction_with_positional_batcher(
      prim, vals_in, dims_in, axis_index_groups,
      lambda d, d_vals_in: (tuple(axis for axis in axes if axis != axis_data.name),
                            [if_unmapped(v, axis_data.size) for v in d_vals_in]),
      lambda d, d_vals_in: (tuple(axis + (axis >= d) if isinstance(axis, int) else
                                  axis if axis != axis_data.name else
                                  d for axis in axes),
                            d_vals_in))
  return vals_out, [batching.not_mapped] * len(vals_out)

def _replica_groups(axis_env, axis_name, axis_index_groups):
  replica_groups = pxla.axis_groups(axis_env, axis_name)
  if axis_index_groups is not None:
    replica_groups = [[axis_group[i] for i in axis_index_group]
                      for axis_group in replica_groups
                      for axis_index_group in axis_index_groups]
  return replica_groups

def _replica_groups_hlo(replica_groups: Sequence[Sequence[int]]
                        ) -> ir.DenseElementsAttr:
  # Uneven replica groups are padded with -1.
  groups = np.array(list(itertools.zip_longest(*replica_groups, fillvalue=-1)),
                    dtype=np.int64).T
  return ir.DenseIntElementsAttr.get(np.ascontiguousarray(groups))

def _allreduce_impl(prim, pos_reducer, *args, axes, axis_index_groups):
  assert axis_index_groups is None
  if not all(isinstance(axis, int) for axis in axes):
     return dispatch.apply_primitive(prim, *args, axes=axes,
                                     axis_index_groups=axis_index_groups)
  assert all(isinstance(axis, int) for axis in axes)
  return [pos_reducer(arg, axes) for arg in args]

def _allreduce_effectful_abstract_eval(*args, axes, axis_index_groups):
  _check_axis_names(axes)
  named_axes = tuple(axis for axis in axes if not isinstance(axis, int))
  pos_axes = tuple(axis for axis in axes if isinstance(axis, int))
  if axis_index_groups is not None:
    if len(pos_axes) != 0:
      raise ValueError(f"axis_index_groups can only be used with reductions over "
                       f"named axes, but got: {axes}")
  if config.sharding_in_types.value:
    core.check_avals_context_mesh(args, 'all_reduce')
    out_avals = [
        ShapedArray(lax._reduce_op_shape_rule(arg, axes=pos_axes), arg.dtype,
                    sharding=lax._reduce_op_sharding_rule(arg, axes=pos_axes))
        for arg in args
    ]
  else:
    out_avals = [ShapedArray(lax._reduce_op_shape_rule(arg, axes=pos_axes), arg.dtype)
                 for arg in args]
  return out_avals, {core.NamedAxisEffect(axis) for axis in named_axes}

def _check_axis_names(axes):
  named_axes = tuple(axis for axis in axes if not isinstance(axis, int))
  axis_env = core.get_axis_env()
  for name in named_axes:
    if not axis_env.axis_exists(name):
      raise NameError(f"unbound axis name: {name}")

def _allreduce_lowering(prim, pos_fn, ctx, *args, axes, axis_index_groups):
  if axis_index_groups is not None and ("tpu" in ctx.module_context.platforms):
    len_0 = len(axis_index_groups[0])
    if any(len(g) != len_0 for g in axis_index_groups):
      raise ValueError("axis_index_groups must all be the same size for TPU lowering")
  named_axes, positional_axes = axes_partition = [], []
  for axis in axes:
    axes_partition[isinstance(axis, int)].append(axis)

  if positional_axes:
    reducer = mlir.lower_fun(pos_fn, multiple_results=False)
    def _positional_reduce(aval, arg):
      aval_out = aval.update(
          shape=np.delete(np.array(aval.shape, dtype=np.int64),
                          positional_axes))
      reducer_ctx = ctx.replace(primitive=None, avals_in=[aval], avals_out=[aval_out])
      out, = reducer(reducer_ctx, arg, axes=tuple(positional_axes))
      return out
    args = map(_positional_reduce, ctx.avals_in, args)
  if not named_axes:
    return args

  replica_groups = _replica_groups_hlo(
      _replica_groups(ctx.module_context.axis_env, named_axes,
                      axis_index_groups))
  axis_context = ctx.module_context.axis_context
  is_spmd = isinstance(axis_context, (SPMDAxisContext, ShardingContext))

  def all_reduce(aval, x):
    if is_spmd:
      channel = ctx.module_context.new_channel()
      other_args = dict(
          channel_handle=hlo.ChannelHandle.get(
              channel, mlir.DEVICE_TO_DEVICE_TYPE),
          use_global_device_ids=ir.BoolAttr.get(True))
    else:
      other_args = {}

    if hlo.get_api_version() < 8:
      op = hlo.AllReduceOp(
          x.type, x, replica_groups=replica_groups, **other_args)
    else:
      op = hlo.AllReduceOp(
          [x.type], [x], replica_groups=replica_groups, **other_args)
    if config.sharding_in_types.value:
      scalar_aval = core.ShapedArray(
          (), aval.dtype, sharding=NamedSharding(aval.sharding.mesh, P()))
    else:
      scalar_aval = core.ShapedArray((), aval.dtype)
    scalar_type = mlir.aval_to_ir_type(scalar_aval)
    reducer_block = op.regions[0].blocks.append(scalar_type, scalar_type)
    with ir.InsertionPoint(reducer_block):
      lower_reducer = mlir.lower_fun(prim.bind, multiple_results=False)
      reducer_ctx = ctx.replace(primitive=None,
                                avals_in=[scalar_aval] * 2, avals_out=[scalar_aval])
      out_nodes = lower_reducer(reducer_ctx, *reducer_block.arguments)
      hlo.return_(mlir.flatten_ir_values(out_nodes))
    return op.result

  return [all_reduce(aval, x) for aval, x in zip(ctx.avals_in, args)]


def _psum_transpose_rule(cts, *args, axes, axis_index_groups):
  named_axes, pos_axes = axes_partition = [], []
  for axis in axes:
    axes_partition[isinstance(axis, int)].append(axis)

  if pos_axes:
    def broadcast_positional(ct, arg):
      assert ad.is_undefined_primal(arg)
      if type(ct) is ad.Zero: return ad.Zero(arg.aval)
      return lax._reduce_sum_transpose_rule(ct, arg, axes=pos_axes)[0]
    cts = map(broadcast_positional, cts, args)

  # We treat psum as psum + pbroadcast, which is why the transpose reduces
  # over the named axes again (unlike for positional axes).
  nonzero_out_cts, treedef = tree_util.tree_flatten(cts)
  nonzero_in_cts = psum_p.bind(*nonzero_out_cts, axes=tuple(named_axes),
                               axis_index_groups=axis_index_groups)
  return tree_util.tree_unflatten(treedef, nonzero_in_cts)

psum_p = core.Primitive('psum')
psum_p.multiple_results = True
psum_p.def_impl(partial(_allreduce_impl, psum_p, lax._reduce_sum))
psum_p.def_effectful_abstract_eval(_allreduce_effectful_abstract_eval)
mlir.register_lowering(
    psum_p, partial(_allreduce_lowering, lax.add_p, lax._reduce_sum))
ad.deflinear2(psum_p, _psum_transpose_rule)
batching.fancy_primitive_batchers[psum_p] = \
  partial(_batched_reduction_collective, psum_p, lambda v, axis_size: axis_size * v)
batching.skippable_batchers[psum_p] = partial(_names_in_param, 'axes')

pmax_p = core.Primitive('pmax')
pmax_p.multiple_results = True
pmax_p.def_impl(partial(_allreduce_impl, pmax_p, lax._reduce_max))
pmax_p.def_effectful_abstract_eval(_allreduce_effectful_abstract_eval)
mlir.register_lowering(
    pmax_p, partial(_allreduce_lowering, lax.max_p, lax._reduce_max))
batching.fancy_primitive_batchers[pmax_p] = \
  partial(_batched_reduction_collective, pmax_p, lambda v, axis_size: v)
batching.skippable_batchers[pmax_p] = partial(_names_in_param, 'axes')


pmin_p = core.Primitive('pmin')
pmin_p.multiple_results = True
pmin_p.def_impl(partial(_allreduce_impl, pmin_p, lax._reduce_min))
pmin_p.def_effectful_abstract_eval(_allreduce_effectful_abstract_eval)
mlir.register_lowering(
    pmin_p, partial(_allreduce_lowering, lax.min_p, lax._reduce_min))
batching.fancy_primitive_batchers[pmin_p] = \
  partial(_batched_reduction_collective, pmin_p, lambda v, axis_size: v)
batching.skippable_batchers[pmin_p] = partial(_names_in_param, 'axes')


def _ppermute_lowering(ctx, x, *, axis_name, perm):
  replica_groups = _replica_groups(ctx.module_context.axis_env, axis_name, None)
  group_size = len(replica_groups[0])
  srcs, dsts = unzip2((src % group_size, dst % group_size) for src, dst in perm)
  if not (len(srcs) == len(set(srcs)) and len(dsts) == len(set(dsts))):
    msg = "ppermute sources and destinations must be unique, got {}."
    raise ValueError(msg.format(perm))

  full_perm = np.zeros((len(replica_groups), len(perm), 2), np.int64)
  for i, grp in enumerate(replica_groups):
    grp = sorted(grp)
    for j, (src, dst) in enumerate(perm):
      full_perm[i, j, 0] = grp[src]
      full_perm[i, j, 1] = grp[dst]
  full_perm = full_perm.reshape((-1, 2))

  axis_context = ctx.module_context.axis_context
  is_manual = (
      isinstance(axis_context, SPMDAxisContext)
      and axis_context.manual_axes
  )
  if is_manual:
    channel = ctx.module_context.new_channel()
    other_args = dict(
        channel_handle=hlo.ChannelHandle.get(channel, mlir.DEVICE_TO_DEVICE_TYPE))
  else:
    other_args = {}

  return hlo.CollectivePermuteOp(
      x, mlir.dense_int_elements(full_perm), **other_args).results

def _ppermute_transpose_rule(t, x, perm, axis_name):
  srcs, dsts = unzip2(perm)
  inverse_perm = list(zip(dsts, srcs))
  return [ppermute(t, axis_name=axis_name, perm=inverse_perm)]

def _ppermute_batcher(axis_data, vals_in, dims_in, axis_name, perm):
  axis_size, frame_name = axis_data.size, axis_data.name
  (v,), (d,) = vals_in, dims_in
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if axis_data.name not in axis_name:
    return ppermute_p.bind(v, perm=perm, axis_name=axis_name), d
  remaining_axes = tuple(axis for axis in axis_name if axis != frame_name)
  if remaining_axes:
    return ppermute_p.bind(v, perm=perm, axis_name=remaining_axes), d
  assert axis_name[0] == frame_name, "ppermute batcher called with a wrong axis!"
  assert len(perm) == axis_size, "Permutation doesn't match the axis size!"
  if d is batching.not_mapped:
    return v, d
  perm_indices = np.zeros(axis_size, dtype=int)
  for src, dst in perm:
    perm_indices[dst] = src
  return v.take(perm_indices, d), d

def _raise_to_shaped_abstract_eval(x, *, axis_name, **params):
  _check_axis_names(axis_name)
  return x

ppermute_p = core.Primitive('ppermute')
ppermute_p.def_abstract_eval(_raise_to_shaped_abstract_eval)
ad.deflinear2(ppermute_p, _ppermute_transpose_rule)
mlir.register_lowering(ppermute_p, _ppermute_lowering)
batching.fancy_primitive_batchers[ppermute_p] = _ppermute_batcher
batching.skippable_batchers[ppermute_p] = partial(_names_in_param, 'axis_name')

def _pbroadcast_transpose_rule(t, x, source, axis_name):
  is_source = axis_index(axis_name) == source
  tsum = psum(t, axis_name)
  return [lax.select(is_source, lax.full_like(t, tsum), lax.full_like(t, 0))]

def _pbroadcast_batcher(axis_data, vals_in, dims_in, axis_name, source):
  axis_size = axis_data.size
  (v,), (d,) = vals_in, dims_in
  if not isinstance(axis_name, (tuple, list)):
    axis_name = (axis_name,)
  if axis_data.name not in axis_name:
    return pbroadcast_p.bind(v, axis_name=axis_name, source=source), d
  remaining_axes = tuple(axis for axis in axis_name if axis != axis_data.name)
  if remaining_axes:
    raise NotImplementedError("pbroadcast batcher only supports a single axis")
  assert axis_name[0] == axis_data.name, "pbroadcast batcher called with a wrong axis!"
  assert source >= 0 and source < axis_size, "collective broadcast doesn't fit in the axis size!"
  if axis_size == 1 and remaining_axes:
    return pbroadcast_p.bind(v, source=source, axis_name=remaining_axes), d
  if d is batching.not_mapped:
    return v, d
  return v.take([source] * axis_size, d), d

def _pbroadcast_lowering(ctx, x, *, axis_name, source):
  replica_groups = _replica_groups(ctx.module_context.axis_env, axis_name, None)
  def source_to_front(group):
    return [group[source]] + list(group[:source]) + list(group[source + 1:])
  replica_groups = [source_to_front(group) for group in replica_groups]
  channel = ctx.module_context.new_channel()
  return hlo.CollectiveBroadcastOp(
      x, replica_groups=_replica_groups_hlo(replica_groups)).results

pbroadcast_p = core.Primitive('pbroadcast')
pbroadcast_p.def_abstract_eval(_raise_to_shaped_abstract_eval)
ad.deflinear2(pbroadcast_p, _pbroadcast_transpose_rule)
mlir.register_lowering(pbroadcast_p, _pbroadcast_lowering)
batching.fancy_primitive_batchers[pbroadcast_p] = _pbroadcast_batcher
batching.skippable_batchers[pbroadcast_p] = partial(_names_in_param, 'axis_name')


def _moveaxis(src, dst, x):
  perm = [i for i in range(x.ndim) if i != src]
  perm.insert(dst, src)
  return lax.transpose(x, perm)

def _splitaxis(axis, factor, x):
  new_shape = list(x.shape)
  assert new_shape[axis] % factor == 0, (new_shape[axis], factor)
  new_shape[axis:axis+1] = [factor, new_shape[axis] // factor]
  return x.reshape(new_shape)

def _foldaxis(axis, x):
  new_shape = list(x.shape)
  new_shape[axis:axis+2] = [x.shape[axis] * x.shape[axis + 1]]
  return x.reshape(new_shape)

def _all_to_all_lowering(
    ctx, x, *, split_axis, concat_axis, axis_name, axis_index_groups, tiled
):
  del tiled  # expand_dims and squeeze is done in `all_to_all` if `True`
  # Workaround for AllToAll not being implemented on CPU.
  replica_groups = _replica_groups(ctx.module_context.axis_env, axis_name,
                                   axis_index_groups)
  if len(replica_groups[0]) == 1:
    return [x]
  split_count = len(replica_groups[0])
  if not all(split_count == len(g) for g in replica_groups):
    raise ValueError('Replica groups must be equally sized')
  is_spmd = isinstance(
      ctx.module_context.axis_context,
      (SPMDAxisContext, ShardingContext),
  )
  if is_spmd:
    # We want to emit the all-gather with global device IDs and a unique
    # channel ID, as otherwise it interprets the devices as replicas instead
    # of partitions - and XLA is configured with only a single replica.
    channel = ctx.module_context.new_channel()
    channel_handle = hlo.ChannelHandle.get(channel, mlir.DEVICE_TO_DEVICE_TYPE)
    other_args = dict(channel_handle=channel_handle)
  else:
    other_args = {}
  if hlo.get_api_version() < 8:
    return hlo.AllToAllOp(
        x,
        split_dimension=mlir.i64_attr(split_axis),
        concat_dimension=mlir.i64_attr(concat_axis),
        split_count=mlir.i64_attr(split_count),
        replica_groups=_replica_groups_hlo(replica_groups),
        **other_args).results
  return hlo.AllToAllOp(
    [x],
    split_dimension=mlir.i64_attr(split_axis),
    concat_dimension=mlir.i64_attr(concat_axis),
    split_count=mlir.i64_attr(split_count),
    replica_groups=_replica_groups_hlo(replica_groups),
    **other_args).results

def _all_to_all_transpose_rule(
    cts, x, axis_name, split_axis, concat_axis, axis_index_groups, tiled
):
  return (all_to_all(
      cts,
      axis_name=axis_name,
      split_axis=concat_axis,
      concat_axis=split_axis,
      axis_index_groups=axis_index_groups,
      tiled=tiled),)

def _all_to_all_batcher(vals_in, dims_in, *, axis_name, split_axis, concat_axis, axis_index_groups,
                        tiled):
  x, = vals_in
  d, = dims_in
  result = all_to_all_p.bind(
      x,
      axis_name=axis_name,
      split_axis=split_axis + (d <= split_axis),
      concat_axis=concat_axis + (d <= concat_axis),
      axis_index_groups=axis_index_groups,
      tiled=tiled,
  )
  return result, d

def _all_to_all_batched_collective(axis_data, vals_in, dims_in,
                                   axis_name, split_axis, concat_axis,
                                   axis_index_groups, tiled):
  axis_size, frame_name = axis_data.size, axis_data.name
  if axis_index_groups is not None:
    raise NotImplementedError("Please open a feature request!")

  if isinstance(axis_name, (list, tuple)):
    axes_names = axis_name
  else:
    axes_names = [axis_name]
  if axis_data.name not in axes_names:
    return _all_to_all_batcher(
      vals_in, dims_in, axis_name=axis_name, split_axis=split_axis,
      concat_axis=concat_axis, axis_index_groups=axis_index_groups, tiled=tiled)

  x, = vals_in
  d, = dims_in
  if d is batching.not_mapped:
    # TODO(sharadmv,apaszke): Remove this broadcast that comes from
    # all_gather_transpose and instead avoid using all_to_all in
    # all_gather_transpose.
    x = lax.broadcast(x, (axis_size, *x.shape))
    d = 0
  if isinstance(axis_name, (list, tuple)):
    pos = axis_name.index(frame_name)
    major_axes, minor_axes = axis_name[:pos], axis_name[pos + 1:]
  else:
    major_axes, minor_axes = (), ()
  # Optimized case when no splitting is necessary
  if not major_axes and not minor_axes:
    if split_axis == concat_axis:
      axis = split_axis + (d <= split_axis)
      d_pre_split = d
      x = _splitaxis(axis, axis_size, x)
      d += (axis <= d)
      return _foldaxis(axis, moveaxis(x, (d, axis), (axis, d))), d_pre_split
    else:
      x_concat = _foldaxis(concat_axis, _moveaxis(d, concat_axis, x))
      return _splitaxis(split_axis, axis_size, x_concat), split_axis
  # Here we have to handle either the major or the minor dimensions
  # We will be accumulating chunks into the three leading dims: [Major, Current, Minor, ...]
  x, d = lax.expand_dims(_moveaxis(d, 0, x), (0, 2)), 1
  split_axis += 3; concat_axis += 3  # Offset by extra three leading dims

  if major_axes:
    x = all_to_all_p.bind(x, axis_name=major_axes,
                          split_axis=split_axis, concat_axis=0,
                          axis_index_groups=axis_index_groups,
                          tiled=tiled)
  # Split out the local part into axis new_d (NOTE: d is already in axis 1)
  x = _splitaxis(split_axis, axis_size, x)
  new_d = split_axis
  concat_axis += (split_axis <= concat_axis)  # Offset the existing axes by the new batch axis
  split_axis += 1
  if minor_axes:
    x = all_to_all_p.bind(x, axis_name=minor_axes,
                          split_axis=split_axis, concat_axis=2,
                          axis_index_groups=axis_index_groups,
                          tiled=tiled)

  # Fold the chunk axes into a single one
  x = _foldaxis(0, _foldaxis(0, x))
  split_axis -= 2; concat_axis -= 2; new_d -= 2
  # Fold gathered axes into concat_axis
  x = _foldaxis(concat_axis - 1, _moveaxis(0, concat_axis - 1, x))
  new_d -= 1  # We've removed 0th dimension, so new_d needs to be adjusted
  return x, new_d


def _all_to_all_effectful_abstract_eval(
    input_aval, axis_name, split_axis, concat_axis, axis_index_groups, tiled
):
  del tiled  # expand_dims and squeeze is done in `all_to_all` if `True`
  if not isinstance(axis_name, (list, tuple)):
    axis_name = (axis_name,)
  _check_axis_names(axis_name)
  shape = list(input_aval.shape)
  axis_size = psum(1, axis_name) if axis_index_groups is None else len(axis_index_groups[0])
  assert shape[split_axis] % axis_size == 0, (shape[split_axis], axis_size)
  shape[split_axis] //= axis_size
  shape[concat_axis] *= axis_size
  out_aval = input_aval.update(shape=tuple(shape), weak_type=False)
  effects = {*map(core.NamedAxisEffect, axis_name)}
  return out_aval, effects


all_to_all_p = core.Primitive('all_to_all')
all_to_all_p.def_effectful_abstract_eval(_all_to_all_effectful_abstract_eval)
mlir.register_lowering(all_to_all_p, _all_to_all_lowering)
ad.deflinear2(all_to_all_p, _all_to_all_transpose_rule)
batching.fancy_primitive_batchers[all_to_all_p] = _all_to_all_batched_collective
batching.skippable_batchers[all_to_all_p] = partial(_names_in_param, 'axis_name')


def _ragged_all_to_all_lowering(ctx, operand, output, input_offsets, send_sizes, output_offsets, recv_sizes):
  N = input_offsets.type.shape[0]
  backend_config = ir.DictAttr.get({
      'replica_groups': ir.DenseIntElementsAttr.get(
          np.arange(0, N, 1, dtype=np.int64), shape=[1, N]
      )
  })
  return hlo.CustomCallOp(
      result=[output.type],
      inputs=[operand, output, input_offsets, send_sizes, output_offsets,
              recv_sizes],
      call_target_name=ir.StringAttr.get('ragged_all_to_all'),
      backend_config=backend_config,
      api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 4),
  ).results

@ragged_all_to_all_p.def_abstract_eval
def _ragged_all_to_all_abstract_eval(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes):
  if operand.shape[1:] != output.shape[1:]:
    raise ValueError(
        "ragged_all_to_all input and output shapes must be equal, except for"
        " the outermost dimension."
    )
  if not dtypes.issubdtype(input_offsets.dtype, np.integer):
    raise ValueError("ragged_all_to_all input_offsets must be integer type.")
  if not dtypes.issubdtype(send_sizes.dtype, np.integer):
    raise ValueError("ragged_all_to_all send_sizes must be integer type.")
  if not dtypes.issubdtype(output_offsets.dtype, np.integer):
    raise ValueError("ragged_all_to_all output_offsets must be integer type.")
  if not dtypes.issubdtype(recv_sizes.dtype, np.integer):
    raise ValueError("ragged_all_to_all recv_sizes must be integer type.")
  if len(input_offsets.shape) != 1 or input_offsets.shape[0] < 1:
    raise ValueError(
        "ragged_all_to_all input_offsets must be rank 1 with positive dimension"
        " size, but got shape {}".format(input_offsets.shape)
    )
  if len(send_sizes.shape) != 1 or send_sizes.shape[0] < 1:
    raise ValueError(
        "ragged_all_to_all send_sizes must be rank 1 with positive dimension"
        " size, but got shape {}".format(send_sizes.shape)
    )
  if len(output_offsets.shape) != 1 or output_offsets.shape[0] < 1:
    raise ValueError(
        "ragged_all_to_all output_offsets must be rank 1 with positive"
        " dimension size, but got shape {}".format(output_offsets.shape)
    )
  if len(recv_sizes.shape) != 1 or recv_sizes.shape[0] < 1:
    raise ValueError(
        "ragged_all_to_all recv_sizes must be rank 1 with positive dimension"
        " size, but got shape {}".format(recv_sizes.shape)
    )
  return output.update(
      shape=list(output.shape),
      dtype=output.dtype,
      weak_type=output.weak_type,
  )

ragged_all_to_all_p.def_impl(partial(dispatch.apply_primitive, ragged_all_to_all_p))
mlir.register_lowering(ragged_all_to_all_p, _ragged_all_to_all_lowering)


def all_gather(x, axis_name, *, axis_index_groups=None, axis=0, tiled=False):
  """Gather values of x across all replicas.

  If ``x`` is a pytree then the result is equivalent to mapping this function to
  each leaf in the tree.

  This is equivalent to, but faster than, all_to_all(broadcast(x)).

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a pmapped axis (see the
      :func:`jax.pmap` documentation for more details).
    axis_index_groups: optional list of lists containing axis indices (e.g. for
      an axis of size 4, [[0, 1], [2, 3]] would run all gather over the first
      two and last two replicas). Groups must cover all axis indices exactly
      once, and all groups must be the same size.
    axis: a positional axis into which the chunks along ``axis_name`` will be
      concatenated.
    tiled: when ``False``, the chunks will be stacked into a fresh positional
      axis at index ``axis`` in the output. When ``True``, ``axis`` has to
      refer to an existing positional dimension and the chunks will be
      concatenated into that dimension.

  Returns:
    Array(s) representing the result of an all-gather along the axis
    ``axis_name``. Shapes are the same as ``x.shape``, but:

    - when ``tiled`` is ``False``, there is a new dimension equal to the
      size of axis ``axis_name`` in position ``axis``,
    - when ``tiled`` is ``True``, the size of dimension in position ``axis``
      is multiplied by the size of axis ``axis_name``.

  For example, with 4 XLA devices available:

  >>> x = np.arange(4)
  >>> y = jax.pmap(lambda x: jax.lax.all_gather(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [[0 1 2 3]
   [0 1 2 3]
   [0 1 2 3]
   [0 1 2 3]]

  An example of using axis_index_groups, groups split by even & odd device ids:

  >>> x = np.arange(16).reshape(4, 4)
  >>> print(x)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]]
  >>> def f(x):
  ...   return jax.lax.all_gather(
  ...       x, 'i', axis_index_groups=[[0, 2], [3, 1]])
  >>> y = jax.pmap(f, axis_name='i')(x)
  >>> print(y)
  [[[ 0  1  2  3]
    [ 8  9 10 11]]
   [[12 13 14 15]
    [ 4  5  6  7]]
   [[ 0  1  2  3]
    [ 8  9 10 11]]
   [[12 13 14 15]
    [ 4  5  6  7]]]
  """
  if not isinstance(axis_name, tuple):
    axis_name = axis_name,
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  axis_size = psum(1, axis_name, axis_index_groups=axis_index_groups)
  def bind(leaf):
    return all_gather_p.bind(
        leaf,
        all_gather_dimension=canonicalize_axis(
            axis, np.ndim(leaf) if tiled else np.ndim(leaf) + 1),
        axis_name=axis_name, axis_index_groups=axis_index_groups,
        axis_size=int(axis_size), tiled=tiled)
  return tree_util.tree_map(bind, x)

def _all_gather_impl(x, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled):
  raise AssertionError("Unexpected call to _all_gather_impl")

def _all_gather_lowering(ctx, x, *, all_gather_dimension, axis_name,
                         axis_index_groups, axis_size, tiled,
                         platform=None):
  x_aval, = ctx.avals_in
  out_aval, = ctx.avals_out
  axis_context = ctx.module_context.axis_context
  is_spmd = isinstance(axis_context, (SPMDAxisContext, ShardingContext))
  if not tiled:
    new_shape = list(x_aval.shape)
    new_shape.insert(all_gather_dimension, 1)
    broadcast_dimensions = [i for i in range(len(new_shape)) if i != all_gather_dimension]
    x = hlo.broadcast_in_dim(
        mlir.aval_to_ir_type(x_aval.update(shape=new_shape)), x,
        mlir.dense_int_array(broadcast_dimensions))
  replica_groups = _replica_groups(ctx.module_context.axis_env, axis_name,
                                    axis_index_groups)
  if is_spmd:
    # We want to emit the all-gather with global device IDs and a unique
    # channel ID, as otherwise it interprets the devices as replicas instead
    # of partitions - and XLA is configured with only a single replica.
    channel = ctx.module_context.new_channel()
    other_args = dict(
        channel_handle=hlo.ChannelHandle.get(
            channel, mlir.DEVICE_TO_DEVICE_TYPE),
        use_global_device_ids=ir.BoolAttr.get(True))
  else:
    other_args = {}

  if hlo.get_api_version() < 8:
    return hlo.AllGatherOp(
        mlir.aval_to_ir_type(out_aval),
        x, all_gather_dim=mlir.i64_attr(all_gather_dimension),
        replica_groups=_replica_groups_hlo(replica_groups),
        **other_args).results
  return hlo.AllGatherOp(
      [mlir.aval_to_ir_type(out_aval)],
      [x], all_gather_dim=mlir.i64_attr(all_gather_dimension),
      replica_groups=_replica_groups_hlo(replica_groups),
      **other_args).results


def _all_gather_effectful_abstract_eval(
    x_aval, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled
):
  if not isinstance(axis_name, (list, tuple)):
    axis_name = (axis_name,)
  _check_axis_names(axis_name)
  new_shape = list(x_aval.shape)
  if tiled:
    new_shape[all_gather_dimension] *= axis_size
  else:
    new_shape.insert(all_gather_dimension, axis_size)
  return x_aval.update(shape=new_shape), {*map(core.NamedAxisEffect, axis_name)}

def _all_gather_transpose_rule(cts, x, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled):
  return (psum_scatter(cts, axis_name=axis_name,
                       scatter_dimension=all_gather_dimension,
                       axis_index_groups=axis_index_groups,
                       tiled=tiled),)
  # TODO(sharadmv,apaszke): re-enable this when we can properly detect replication.
  # return (lax.dynamic_index_in_dim(cts, idx, axis=all_gather_dimension, keepdims=False) * axis_size,)

def _all_gather_batcher(vals_in, dims_in, *, all_gather_dimension, axis_name, axis_index_groups, axis_size, tiled):
  (x,), (d,) = vals_in, dims_in
  if d is not batching.not_mapped:
    if d <= all_gather_dimension:
      all_gather_dimension += 1
    elif not tiled:  # Tiled all-gather doesn't modify the set of dimensions
      d += 1
  result = all_gather_p.bind(
      x,
      all_gather_dimension=all_gather_dimension,
      axis_name=axis_name,
      axis_index_groups=axis_index_groups,
      axis_size=axis_size,
      tiled=tiled)
  return result, d

def _all_gather_batched_collective(axis_data, vals_in, dims_in,
                                   all_gather_dimension, axis_name,
                                   axis_index_groups, axis_size, tiled):
  frame_size, frame_name = axis_data.size, axis_data.name
  if frame_name not in axis_name:
    return _all_gather_batcher(
        vals_in, dims_in, all_gather_dimension=all_gather_dimension,
        axis_name=axis_name, axis_index_groups=axis_index_groups,
        axis_size=axis_size, tiled=tiled)
  if axis_index_groups is not None:
    raise NotImplementedError("axis_index_groups not supported in vmap")
  assert axis_size == frame_size, "axis size doesn't match"
  if not isinstance(axis_name, tuple):
    axis_name = (axis_name,)
  if len(axis_name) > 1:
    raise NotImplementedError("Please open a feature request!")
  assert axis_name == (frame_name,), "batcher called with wrong axis name"
  (x,), (d,) = vals_in, dims_in
  if d is batching.not_mapped:
    out_shape = list(np.shape(x))
    out_shape.insert(all_gather_dimension, axis_size)
    broadcast_dims = [i for i in range(len(out_shape)) if i != all_gather_dimension]
    y = lax.broadcast_in_dim(x, out_shape, broadcast_dims)
  else:
    y = _moveaxis(d, all_gather_dimension, x)
  if tiled:
    y = _foldaxis(all_gather_dimension, y)
  return y, batching.not_mapped

all_gather_p = core.Primitive('all_gather')
all_gather_p.def_effectful_abstract_eval(_all_gather_effectful_abstract_eval)
all_gather_p.def_impl(_all_gather_impl)
mlir.register_lowering(all_gather_p, _all_gather_lowering)
for p in ("cuda", "rocm", "tpu"):
  mlir.register_lowering(all_gather_p,
                         partial(_all_gather_lowering, platform=p),
                         platform=p)
ad.deflinear2(all_gather_p, _all_gather_transpose_rule)
batching.fancy_primitive_batchers[all_gather_p] = _all_gather_batched_collective
batching.skippable_batchers[all_gather_p] = partial(_names_in_param, 'axis_name')


def _reduce_scatter_lowering(
    prim, ctx, x,
    *, scatter_dimension, axis_name,
    axis_index_groups, axis_size, tiled):
  x_aval, = ctx.avals_in
  aval_out, = ctx.avals_out
  scalar_aval = x_aval.update(shape=())
  replica_groups = _replica_groups(ctx.module_context.axis_env, axis_name,
                                   axis_index_groups)
  scatter_out_shape = list(x_aval.shape)
  scatter_out_shape[scatter_dimension] //= axis_size
  axis_context = ctx.module_context.axis_context
  is_spmd = isinstance(
      axis_context,
      (SPMDAxisContext, ShardingContext),
  )
  if is_spmd:
    # We want to emit the all-gather with global device IDs and a unique
    # channel ID, as otherwise it interprets the devices as replicas instead
    # of partitions - and XLA is configured with only a single replica.
    channel = ctx.module_context.new_channel()
    other_args = dict(
        channel_handle=hlo.ChannelHandle.get(
            channel, mlir.DEVICE_TO_DEVICE_TYPE),
        use_global_device_ids=ir.BoolAttr.get(True))
  else:
    other_args = {}
  op = hlo.ReduceScatterOp(
      mlir.aval_to_ir_type(x_aval.update(shape=scatter_out_shape)),
      x,
      scatter_dimension=mlir.i64_attr(scatter_dimension),
      replica_groups=_replica_groups_hlo(replica_groups),
      **other_args)
  scalar_type = mlir.aval_to_ir_type(scalar_aval)
  reducer_block = op.regions[0].blocks.append(scalar_type, scalar_type)
  with ir.InsertionPoint(reducer_block):
    lower_reducer = mlir.lower_fun(prim.bind, multiple_results=False)
    reducer_ctx = ctx.replace(primitive=None,
                              avals_in=[scalar_aval] * 2,
                              avals_out=[scalar_aval])
    out_nodes = lower_reducer(reducer_ctx, *reducer_block.arguments)
    hlo.return_(mlir.flatten_ir_values(out_nodes))

  if tiled:
    return op.results
  else:
    return [hlo.reshape(mlir.aval_to_ir_type(aval_out), op.result)]


def _reduce_scatter_effectful_abstract_eval(
    x_aval, *, axis_name, scatter_dimension, axis_index_groups, axis_size, tiled
):
  if not isinstance(axis_name, (list, tuple)):
    axis_name = (axis_name,)
  _check_axis_names(axis_name)
  new_shape = list(x_aval.shape)
  scatter_dim_input_size = x_aval.shape[scatter_dimension]
  if tiled:
    if scatter_dim_input_size % axis_size != 0:
      raise ValueError(f"tiled reduce_scatter operand scatter dimension size "
                       f"{scatter_dim_input_size} must be divisible by "
                       f"shard_count {axis_size}")
    new_shape[scatter_dimension] = scatter_dim_input_size // axis_size
  else:
    if scatter_dim_input_size != axis_size:
      raise ValueError(f"reduce_scatter operand scatter dimension size "
                       f"{scatter_dim_input_size} must match shard count "
                       f"{axis_size}")
    del new_shape[scatter_dimension]
  return x_aval.update(shape=new_shape), {*map(core.NamedAxisEffect, axis_name)}


def _reduce_scatter_transpose_rule(cts, x, *, axis_name, scatter_dimension,
                                   axis_index_groups, axis_size, tiled):
  return (all_gather(cts, axis_name=axis_name,
                     axis_index_groups=axis_index_groups,
                     axis=scatter_dimension, tiled=tiled),)


def _reduce_scatter_batcher(vals_in, dims_in, *, scatter_dimension, axis_name,
                            axis_index_groups, axis_size, tiled):
  (x,), (d,) = vals_in, dims_in
  if d <= scatter_dimension:
    scatter_dimension += 1
  elif not tiled:  # Tiled all-scatter doesn't change the rank
    d += 1
  result = reduce_scatter_p.bind(
      x,
      scatter_dimension=scatter_dimension,
      axis_name=axis_name,
      axis_index_groups=axis_index_groups,
      axis_size=axis_size,
      tiled=tiled)
  return result, d

def _reduce_scatter_collective(axis_data, vals_in, dims_in,
                               scatter_dimension, axis_name,
                               axis_index_groups, axis_size, tiled):
  frame_size, frame_name = axis_data.size, axis_data.name
  if frame_name not in axis_name:
    return _reduce_scatter_batcher(
        vals_in, dims_in, scatter_dimension=scatter_dimension,
        axis_name=axis_name, axis_index_groups=axis_index_groups,
        axis_size=axis_size, tiled=tiled)
  if axis_index_groups is not None:
    raise NotImplementedError("axis_index_groups not supported in vmap")
  assert axis_size == frame_size, "axis size doesn't match"
  if not isinstance(axis_name, tuple):
    axis_name = (axis_name,)
  if len(axis_name) > 1:
    raise NotImplementedError("Please open a feature request!")
  assert axis_name == (frame_name,), "batcher called with wrong axis name"
  (x,), (d,) = vals_in, dims_in
  if d is batching.not_mapped:
    y, dy = x * axis_size, scatter_dimension
  else:
    y, dy = lax.reduce(x, 0., lax.add, (d,)), scatter_dimension
  if tiled:
    y = _splitaxis(dy, axis_size, y)
  return y, dy


reduce_scatter_p = core.Primitive("reduce_scatter")
reduce_scatter_p.def_effectful_abstract_eval(
    _reduce_scatter_effectful_abstract_eval
)
ad.deflinear2(reduce_scatter_p, _reduce_scatter_transpose_rule)
batching.fancy_primitive_batchers[reduce_scatter_p] = _reduce_scatter_collective
batching.skippable_batchers[reduce_scatter_p] = partial(_names_in_param, 'axis_name')

mlir.register_lowering(reduce_scatter_p,
                       partial(_reduce_scatter_lowering, lax.add_p))

def psum_scatter(x, axis_name, *, scatter_dimension=0, axis_index_groups=None,
                 tiled=False):
  """
  Like ``psum(x, axis_name)`` but each device retains only part of the result.

  For example, ``psum_scatter(x, axis_name, scatter_dimension=0, tiled=False)``
  computes the same value as ``psum(x, axis_name)[axis_index(axis_name)]``, but
  it is more efficient. Thus the ``psum`` result is left scattered along the
  mapped axis.

  One efficient algorithm for computing ``psum(x, axis_name)`` is to perform a
  ``psum_scatter`` followed by an ``all_gather``, essentially evaluating
  ``all_gather(psum_scatter(x, axis_name))``. So we can think of
  ``psum_scatter`` as "the first half" of a ``psum``.

  Args:
    x: array(s) with a mapped axis named ``axis_name``.
    axis_name: hashable Python object used to name a mapped axis (see the
      :func:`jax.pmap` documentation for more details).
    scatter_dimension: a positional axis into which the all-reduce result along
      ``axis_name`` will be scattered.
    axis_index_groups: optional list of lists of integers containing axis
      indices. For example, for an axis of size 4,
      ``axis_index_groups=[[0, 1], [2, 3]]`` would run reduce-scatter over the
      first two and the last two axis indices. Groups must cover all axis
      indices exactly once, and all groups must be the same size.
    tiled: boolean representing whether to use rank-preserving 'tiled' behavior.
      When ``False`` (the default value), the size of dimension in
      ``scatter_dimension`` must match the size of axis ``axis_name`` (or the
      group size if ``axis_index_groups`` is given). After scattering the
      all-reduce result along ``scatter_dimension``, the output is squeezed by
      removing ``scatter_dimension``, so the result has lower rank than the
      input. When ``True``, the size of dimension in ``scatter_dimension`` must
      be divisible by the size of axis ``axis_name`` (or the group size if
      ``axis_index_groups`` is given), and the ``scatter_dimension`` axis is
      preserved (so the result has the same rank as the input).

  Returns:
    Array(s) with the similar shape as ``x``, except the size of dimension in
    position ``scatter_dimension`` is divided by the size of axis ``axis_name``
    (when ``tiled=True``), or the dimension in position ``scatter_dimension`` is
    eliminated (when ``tiled=False``).

  For example, with 4 XLA devices available:

  >>> x = np.arange(16).reshape(4, 4)
  >>> print(x)
  [[ 0  1  2  3]
   [ 4  5  6  7]
   [ 8  9 10 11]
   [12 13 14 15]]
  >>> y = jax.pmap(lambda x: jax.lax.psum_scatter(x, 'i'), axis_name='i')(x)
  >>> print(y)
  [24 28 32 36]

  if using tiled:

  >>> y = jax.pmap(lambda x: jax.lax.psum_scatter(x, 'i', tiled=True), axis_name='i')(x)
  >>> print(y)
  [[24]
   [28]
   [32]
   [36]]

  An example of using axis_index_groups:

  >>> def f(x):
  ...   return jax.lax.psum_scatter(
  ...       x, 'i', axis_index_groups=[[0, 2], [3, 1]], tiled=True)
  >>> y = jax.pmap(f, axis_name='i')(x)
  >>> print(y)
  [[ 8 10]
   [20 22]
   [12 14]
   [16 18]]
  """
  if not isinstance(axis_name, tuple):
    axis_name = axis_name,
  axis_size = psum(1, axis_name, axis_index_groups=axis_index_groups)
  axis_index_groups = _canonicalize_axis_index_groups(axis_index_groups)
  bind = partial(
      reduce_scatter_p.bind,
      axis_name=axis_name,
      scatter_dimension=scatter_dimension,
      axis_index_groups=axis_index_groups,
      axis_size=axis_size,
      tiled=tiled)
  return tree_util.tree_map(bind, x)


def _build_axis_index_lowering_hlo(ctx, axis_name, axis_env):
  if isinstance(axis_name, tuple):
    assert axis_name, 'empty axis name'
    if len(axis_name) > 1:
      raise NotImplementedError(
          '`axis_index` translation rule does not support multiple axis names.')
    axis_name, = axis_name
  if axis_name not in axis_env.names:
    raise NameError(f"unbound axis name: {axis_name}")
  axis_context = ctx.module_context.axis_context
  axis_pos = list(axis_env.names).index(axis_name)

  # For partial auto, lower using iota.
  if (isinstance(axis_context, SPMDAxisContext) and
      axis_context.manual_axes and
      axis_context.manual_axes != frozenset(axis_context.mesh.axis_names)):
    if axis_env.sizes[axis_pos] == 1:
      return hlo.constant(ir.DenseElementsAttr.get(np.asarray(0, dtype=np.int32)))
    x = hlo.iota(ir.RankedTensorType.get(
        [axis_env.sizes[axis_pos]], ir.IntegerType.get_signless(32)), mlir.i64_attr(0))
    sharding_proto = (
        NamedSharding(axis_context.mesh, P(axis_name))
        ._to_xla_hlo_sharding(1).to_proto())
    aval_in = ShapedArray((axis_env.sizes[axis_pos],), np.int32)
    aval_out = ShapedArray((1,), np.int32)
    sx = mlir.wrap_with_sharding_op(ctx, x, aval_in, sharding_proto)
    proto = pxla.manual_proto(aval_in, axis_context.manual_axes, axis_context.mesh)
    x = mlir.wrap_with_full_to_shard_op(ctx, sx, aval_out, proto)
    return hlo.reshape(ir.RankedTensorType.get([], ir.IntegerType.get_signless(32)), x)

  nreplicas = axis_env.nreps // math.prod(axis_env.sizes)
  div = mlir.ir_constant(
      np.array(
          nreplicas * math.prod(axis_env.sizes[axis_pos + 1 :]), dtype=np.uint32
      )
  )
  mod = mlir.ir_constant(np.array(axis_env.sizes[axis_pos], dtype=np.uint32))
  if isinstance(axis_context, (ShardingContext, SPMDAxisContext)):
    device_id = hlo.partition_id()
  else:
    device_id = hlo.replica_id()
  unsigned_index = hlo.remainder(hlo.divide(device_id, div), mod)
  return hlo.convert(
      ir.RankedTensorType.get([], ir.IntegerType.get_signless(32)),
      unsigned_index)

def _axis_index_lowering(ctx, *, axis_name):
  return [_build_axis_index_lowering_hlo(ctx, axis_name,
                                         ctx.module_context.axis_env)]

def _axis_index_effectful_abstract_eval(*, axis_name):
  _check_axis_names([axis_name])
  return ShapedArray((), np.int32), {core.NamedAxisEffect(axis_name)}

def _axis_index_batcher(axis_data, vals_in, dims_in, *, axis_name):
  return lax.iota(np.int32, axis_data.size), 0

axis_index_p = core.Primitive('axis_index')
axis_index_p.def_impl(partial(dispatch.apply_primitive, axis_index_p))
mlir.register_lowering(axis_index_p, _axis_index_lowering)
axis_index_p.def_effectful_abstract_eval(_axis_index_effectful_abstract_eval)
batching.fancy_primitive_batchers[axis_index_p] = _axis_index_batcher
batching.skippable_batchers[axis_index_p] = partial(_names_in_param, 'axis_name')

def _pgather_impl(src, idx, *, axes):
  assert all(isinstance(axis, int) for axis in axes)
  src_axes_front = moveaxis(src, axes, range(len(axes)))
  non_axes_shape = src_axes_front.shape[len(axes):]
  src_one_axis_front = src_axes_front.reshape((-1,) + non_axes_shape)
  slice_sizes = (1,) + non_axes_shape
  idx = lax.expand_dims(idx, (-1,))
  offset_dims = tuple(range(idx.ndim - 1, idx.ndim + src_one_axis_front.ndim - 2))
  dnums = slicing.GatherDimensionNumbers(
      offset_dims=offset_dims,
      collapsed_slice_dims=(0,),
      start_index_map=(0,),
  )
  return slicing.gather(src_one_axis_front, idx, dimension_numbers=dnums,
                        slice_sizes=tuple(slice_sizes))

def _pgather_abstract_eval(src, idx, *, axes):
  # TODO: Avals with names rule: remove all axes from src, insert those from idx
  #       The order is important, because it is ok to re-insert one of the deleted axes!
  _check_axis_names(axes)
  shape = list(src.shape)
  for axis in sorted((a for a in axes if isinstance(a, int)), reverse=True):
    del shape[axis]
  shape = idx.shape + tuple(shape)
  return ShapedArray(shape, src.dtype)

def _pgather_parallel_lowering(ctx, src, idx, *, axes):
  if any(not isinstance(axis, int) for axis in axes):
    raise NotImplementedError("pgather only supported in the SPMD lowering."
                              "Please open a feature request!")
  return mlir.lower_fun(_pgather_impl, multiple_results=False)(
      ctx, src, idx, axes=axes)

def _pgather_collective_batcher(axis_size, frame_name, _, vals_in, dims_in, *, axes):
  src, idx = vals_in
  dsrc, didx = dims_in
  if dsrc is batching.not_mapped:
    raise ValueError("pgather axis {frame.name} is missing from the indexed value")
  if didx is not batching.not_mapped:
    # NOTE: This is allowed and the output would be mapped along this axis!
    raise NotImplementedError("Please open a feature request!")
  # Now source is mapped, idx is not
  new_axes = tuple(dsrc if axis == frame_name else
                   axis + (dsrc <= axis) if isinstance(axis, int) else
                   axis
                   for axis in axes)
  # The result is not mapped, because we eliminate all axes, and those include
  # the batched axis.
  if all(isinstance(axis, int) for axis in axes):
    # We rewrite a purely positional pgather as a gather, because that one
    # is more fully featured (e.g. supports AD).
    return _pgather_impl(src, idx, axes=new_axes), batching.not_mapped
  else:
    return pgather_p.bind(src, idx, axes=new_axes), batching.not_mapped

pgather_p = core.Primitive('pgather')
pgather_p.def_impl(_pgather_impl)
pgather_p.def_abstract_eval(_pgather_abstract_eval)
mlir.register_lowering(pgather_p, _pgather_parallel_lowering)
# TODO: Transpose? That requires adding pscatter...
batching.fancy_primitive_batchers[pgather_p] = _pgather_collective_batcher
batching.skippable_batchers[pgather_p] = partial(_names_in_param, 'axes')
