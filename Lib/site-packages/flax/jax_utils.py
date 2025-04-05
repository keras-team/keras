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

"""Utilities we could consider upstreaming to Jax."""

import collections
import itertools
import warnings
from collections.abc import Iterable  # pylint: disable=g-importing-member

import jax
import jax.numpy as jnp
import numpy as np
from jax import core, lax
from jax.extend import linear_util as lu
from jax.interpreters import partial_eval as pe


def _pmap_device_order():
  return jax.local_devices()


def replicate(tree, devices=None):
  """Replicates arrays to multiple devices.

  Args:
    tree: a pytree containing the arrays that should be replicated.
    devices: the devices the data is replicated to
      (default: same order as expected by ``jax.pmap()``).
  Returns:
    A new pytree containing the replicated arrays.
  """
  devices = devices or _pmap_device_order()
  return jax.device_put_replicated(tree, devices)


def unreplicate(tree):
  """Returns a single instance of a replicated array."""
  return jax.tree_util.tree_map(lambda x: x[0], tree)


def pmean(xs, axis_name):
  warnings.warn('use jax.lax.pmean instead', DeprecationWarning)
  return lax.pmean(xs, axis_name)


def partial_eval_by_shape(fn, input_spec, *args, **kwargs):
  """Lazily evaluate a function by using the shapes of the inputs.

  This function is similar to ``jax.eval_shape`` with the key difference that
  function outputs that can be computed without a concrete value of the
  inputs are returned as is instead of only the shape. See for example
  ``module.init_by_shape`` where this functionality is used to initialize a
  model without using input data lr computation.

  Args:
    fn: the function to be lazily evaluated.
    input_spec: an iterable of shapes or (shape, dtype) tuples specifying the
      shape and type of the inputs. If unspecified the dtype is float32.
    *args: other arguments passed to the module's apply function
    **kwargs: keyword arguments passed to the module's apply function
  Returns:
    A pair consisting of the model output and an instance of Model
  """
  # output cannot be returned in lazy_create because jax.eval_shape will only
  # return the shape and dtype.
  # TODO(mattjj,jheek): use a public JAX API
  f = lambda *inputs: fn(*inputs, *args, **kwargs)
  input_structs = [_parse_spec(spec) for spec in input_spec]
  inputs_flat, in_tree = jax.tree_util.tree_flatten(input_structs)

  debug_info = jax.api_util.debug_info("flax partial_eval_by_shape", f,
                                        (in_tree,), {})
  f_flat, out_tree = jax.api_util.flatten_fun_nokwargs(
    lu.wrap_init(f, debug_info=debug_info), in_tree)
  in_pvals = [
    pe.PartialVal.unknown(core.ShapedArray(x.shape, x.dtype))
    for x in inputs_flat
  ]
  _, out_pvals, _ = pe.trace_to_jaxpr_nounits(f_flat, in_pvals)
  out_flat = [
    const if pv is None else jax.ShapeDtypeStruct(pv.shape, pv.dtype)
    for pv, const in out_pvals
  ]
  return jax.tree_util.tree_unflatten(out_tree(), out_flat)


def _parse_spec(spec):
  """Parse an input spec of the form (shape, dtype) or shape into a jax.ShapeDtypeStruct."""
  spec = tuple(spec)
  if len(spec) == 2 and isinstance(spec[0], Iterable):
    return jax.ShapeDtypeStruct(tuple(spec[0]), spec[1])
  else:
    return jax.ShapeDtypeStruct(spec, jnp.float32)


def prefetch_to_device(iterator, size, devices=None):
  """Shard and prefetch batches on device.

  This utility takes an iterator and returns a new iterator which fills an on
  device prefetch buffer. Eager prefetching can improve the performance of
  training loops significantly by overlapping compute and data transfer.

  This utility is mostly useful for GPUs, for TPUs and CPUs it should not be
  necessary -- the TPU & CPU memory allocators (normally) don't pick a memory
  location that isn't free yet so they don't block. Instead those allocators OOM.

  Args:
    iterator: an iterator that yields a pytree of ndarrays where the first
      dimension is sharded across devices.

    size: the size of the prefetch buffer.

      If you're training on GPUs, 2 is generally the best choice because this
      guarantees that you can overlap a training step on GPU with a data
      prefetch step on CPU.

    devices: the list of devices to which the arrays should be prefetched.

      Defaults to the order of devices expected by ``jax.pmap``.

  Yields:
    The original items from the iterator where each ndarray is now sharded to
    the specified devices.
  """
  queue = collections.deque()
  devices = _pmap_device_order() if devices is None else devices

  def _prefetch(xs):
    return jax.device_put_sharded(list(xs), devices)

  def enqueue(n):  # Enqueues *up to* `n` elements from the iterator.
    for data in itertools.islice(iterator, n):
      queue.append(jax.tree_util.tree_map(_prefetch, data))

  enqueue(size)  # Fill up the buffer.
  while queue:
    yield queue.popleft()
    enqueue(1)


def _scan_nd(body_fn, init, xs, n=1, unroll=(1,)):
  """Utility for performing an n-dimensional `lax.scan`.

  The n-d scan is simply recursive call of 1-d scan.
  Args:
    body_fn: the body of the loop of type (c, x) -> (c, y).
    init: initial value for the carry.
    xs: a pytree of tensors to scan over.
    n: number of dimensions to scan over (default: 1)
  Returns:
    A tuple of the final carry and the values returned by the body.
  """
  if n == 1:
    return lax.scan(body_fn, init, xs, unroll=unroll[0])
  else:

    def scan_body(c, x):
      return _scan_nd(body_fn, c, x, n=n - 1, unroll=unroll[1:])

    return lax.scan(scan_body, init, xs, unroll=unroll[0])


def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)


def scan_in_dim(body_fn, init, xs, axis=(0,), unroll=(1,), keepdims=False):
  """utility for doing a scan along arbitrary dimensions.

  See `lax.scan` for details on how the scan operation works.

  Note on `unroll`: This argument gets left padded with ones to match the size
  of `axis`. Doing so allows unrolls to performed from the innermost loop first.
  For example, `scan_in_dim(..., axis=(1, 2, 3), unroll=5)` is equivalent to
  `scan_in_dim(..., axis=(1, 2, 3), unroll=(1, 1, 5))`.

  Args:
    body_fn: the body of the loop of type (c, x) -> (c, y).
    init: initial value for the carry.
    xs: a pytree of tensors to scan over.
    axis: the axis to scan over.
    keepdims: keep the dimensions that are scanned over.
    unroll: an optional positive integer, or tuple of positive integers
      showing how many iterations of the loop to be unrolled into a single
      iteration for each axis.
  Returns:
    A tuple of the final carry and the values returned by the body.
  """
  if not isinstance(axis, Iterable):
    axis = (axis,)

  if not isinstance(unroll, Iterable):
    unroll = (unroll,)

  # Pad unroll with ones so we start unrolling from the innermost loop
  len_diff = len(axis) - len(unroll)
  unroll = (1,) * len_diff + unroll

  def transpose_in(x):
    perm = axis + tuple(np.delete(np.arange(x.ndim), axis))
    return x.transpose(perm)

  def transpose_out(x):
    perm = axis + tuple(np.delete(np.arange(x.ndim), axis))
    return x.transpose(_invert_perm(perm))

  def body_wrapper(c, xs):
    if keepdims:
      xs = jax.tree_util.tree_map(
        lambda x: x.reshape((1,) * len(axis) + x.shape), xs
      )
      xs = jax.tree_util.tree_map(transpose_out, xs)
    c, ys = body_fn(c, xs)
    if keepdims:
      ys = jax.tree_util.tree_map(transpose_in, ys)
      ys = jax.tree_util.tree_map(lambda x: x.reshape(x.shape[len(axis) :]), ys)
    return c, ys

  xs = jax.tree_util.tree_map(transpose_in, xs)
  c, ys = _scan_nd(body_wrapper, init, xs, n=len(axis), unroll=unroll)
  ys = jax.tree_util.tree_map(transpose_out, ys)
  return c, ys


# Copied from https://github.com/google-research/big_vision
def pad_shard_unpad(
  wrapped, static_argnums=(0,), static_argnames=(), static_return=False
):
  """Wraps a function with code that pads, shards, then un-shards, un-pads.

  Args:
    wrapped: the function to be wrapped. Signature is ``params, *args, *kwargs``.
    static_argnums: indices of arguments to ``wrapped`` that should _not_ be
      padded and sharded, but instead be forwarded as-is. The default is (0,)
      because by far the most common use-case is to pass ``params`` first.
    static_argnames: names of kwargs to ``wrapped`` that should _not_ be padded
      and sharded, but instead be forwarded as-is.
    static_return: whether not to un-shard, and un-pad the return value; static
      return values are typically used with eval steps that compute metrics

  Returns:
    A new function that pads and shards its arguments before passing them to
    the wrapped function, and un-shards and un-pads the returned pytree.

    This is useful for calling a pmap'ed function with inputs that aren't
    divisible by the number of devices. A typical use is:
      @pad_shard_unpad
      @jax.pmap
      def forward(params, x): ...

  Notes:
    The padding is done in host-memory before being passed to the function, and
    the values returned by the function are transferred back to host memory.

    The returned function is augmented with a new keyword-only argument
    ``min_device_batch`` that, if specified, forces padding inputs to at least
    this size per device. This can be useful to avoid recompiles for the last
    batch and reduce memory fragmentation.

    For more information refer to https://flax.readthedocs.io/en/latest/guides/data_preprocessing/full_eval.html
  """

  def pad_shard_unpad_wrapper(*args, min_device_batch=None, **kw):
    d = jax.local_device_count()  # d = devices, b = batch
    batch_sizes = set()
    for i, a in enumerate(args):
      if i not in static_argnums:
        batch_sizes |= {t.shape[0] for t in jax.tree_util.tree_leaves(a)}
    for k, v in kw.items():
      if k not in static_argnames:
        batch_sizes |= {t.shape[0] for t in jax.tree_util.tree_leaves(v)}
    assert len(batch_sizes) == 1, f'Inconsistent batch-sizes: {batch_sizes}'
    b = batch_sizes.pop()

    def pad(x):
      _, *shape = x.shape
      db, rest = divmod(b, d)
      if rest:
        x = np.concatenate([x, np.zeros((d - rest, *shape), x.dtype)], axis=0)
        db += 1
      if min_device_batch and db < min_device_batch:
        x = np.concatenate(
          [x, np.zeros((d * (min_device_batch - db), *shape), x.dtype)]
        )
        db = min_device_batch
      return x.reshape(d, db, *shape)

    def maybe_pad(tree, actually_pad=True):
      if not actually_pad:
        return tree  # For call-site convenience below.
      return jax.tree_util.tree_map(pad, tree)

    args = [maybe_pad(a, i not in static_argnums) for i, a in enumerate(args)]
    kw = {k: maybe_pad(v, k not in static_argnames) for k, v in kw.items()}
    out = wrapped(*args, **kw)

    def unpad(x):
      # Transfer back before cutting, to reduce on-device shape diversity.
      return jax.device_get(x).reshape([np.prod(x.shape[:2]), *x.shape[2:]])[:b]

    return out if static_return else jax.tree_util.tree_map(unpad, out)

  return pad_shard_unpad_wrapper
