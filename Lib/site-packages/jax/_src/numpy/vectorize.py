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

from collections.abc import Callable, Collection, Sequence
import functools
import re
from typing import Any
import warnings

from jax._src import api
from jax._src import config
from jax import lax
from jax._src.numpy import lax_numpy as jnp
from jax._src.util import set_module, safe_map as map, safe_zip as zip


export = set_module('jax.numpy')

# See http://docs.scipy.org/doc/numpy/reference/c-api.generalized-ufuncs.html
_DIMENSION_NAME = r'\w+'
_CORE_DIMENSION_LIST = '(?:{0:}(?:,{0:})*)?'.format(_DIMENSION_NAME)
_ARGUMENT = fr'\({_CORE_DIMENSION_LIST}\)'
_ARGUMENT_LIST = '{0:}(?:,{0:})*'.format(_ARGUMENT)
_SIGNATURE = '^{0:}->{0:}$'.format(_ARGUMENT_LIST)


CoreDims = tuple[str, ...]
NDArray = Any


def _parse_gufunc_signature(
    signature: str,
) -> tuple[list[CoreDims], list[CoreDims]]:
  """Parse string signatures for a generalized universal function.

  Args:
    signature: generalized universal function signature, e.g.,
      ``(m,n),(n,p)->(m,p)`` for ``jnp.matmul``.

  Returns:
    Input and output core dimensions parsed from the signature.
  """
  if not re.match(_SIGNATURE, signature):
    raise ValueError(
        f'not a valid gufunc signature: {signature}')
  args, retvals = ([tuple(re.findall(_DIMENSION_NAME, arg))
                   for arg in re.findall(_ARGUMENT, arg_list)]
                   for arg_list in signature.split('->'))
  return args, retvals


def _update_dim_sizes(
    dim_sizes: dict[str, int],
    shape: tuple[int, ...],
    core_dims: CoreDims,
    error_context: str = "",
    *,
    is_input: bool):
  """Incrementally check and update core dimension sizes for a single argument.

  Args:
    dim_sizes: sizes of existing core dimensions. Will be updated in-place.
    shape: shape of this argument.
    core_dims: core dimensions for this argument.
    error_context: string context for error messages.
    is_input: are we parsing input or output arguments?
  """
  num_core_dims = len(core_dims)
  if is_input:
    if len(shape) < num_core_dims:
      raise ValueError(
          'input with shape %r does not have enough dimensions for all core '
          'dimensions %r %s' % (shape, core_dims, error_context))
  else:
    if len(shape) != num_core_dims:
      raise ValueError(
          'output shape %r does not match core dimensions %r %s'
          % (shape, core_dims, error_context))

  core_shape = shape[-num_core_dims:] if core_dims else ()
  for dim, size in zip(core_dims, core_shape):
    if dim not in dim_sizes:
      dim_sizes[dim] = size
    elif size != dim_sizes[dim]:
      raise ValueError(
          'inconsistent size for core dimension %r: %r vs %r %s'
          % (dim, size, dim_sizes[dim], error_context))


def _parse_input_dimensions(
    args: tuple[NDArray, ...],
    input_core_dims: list[CoreDims],
    error_context: str = "",
) -> tuple[tuple[int, ...], dict[str, int]]:
  """Parse broadcast and core dimensions for vectorize with a signature.

  Args:
    args: tuple of input arguments to examine.
    input_core_dims: list of core dimensions corresponding to each input.
    error_context: string context for error messages.

  Returns:
    broadcast_shape: common shape to broadcast all non-core dimensions to.
    dim_sizes: common sizes for named core dimensions.
  """
  if len(args) != len(input_core_dims):
    raise TypeError(
        'wrong number of positional arguments: expected %r, got %r %s'
        % (len(input_core_dims), len(args), error_context))
  shapes = []
  dim_sizes: dict[str, int] = {}
  for arg, core_dims in zip(args, input_core_dims):
    _update_dim_sizes(dim_sizes, arg.shape, core_dims, error_context,
                      is_input=True)
    ndim = arg.ndim - len(core_dims)
    shapes.append(arg.shape[:ndim])
  broadcast_shape = lax.broadcast_shapes(*shapes)
  # TODO(mattjj): this code needs updating for dynamic shapes (hence ignore)
  return broadcast_shape, dim_sizes


def _check_output_dims(
    func: Callable,
    dim_sizes: dict[str, int],
    expected_output_core_dims: list[CoreDims],
    error_context: str = "",
) -> Callable:
  """Check that output core dimensions match the signature."""
  def wrapped(*args):
    out = func(*args)
    out_shapes = map(jnp.shape, out if isinstance(out, tuple) else [out])

    if expected_output_core_dims is None:
      output_core_dims = [()] * len(out_shapes)
    else:
      output_core_dims = expected_output_core_dims
      if len(output_core_dims) > 1 and not isinstance(out, tuple):
        raise TypeError(
            "output must be a tuple when multiple outputs are expected, "
            "got: {!r}\n{}".format(out, error_context))
      if len(out_shapes) != len(output_core_dims):
        raise TypeError(
            'wrong number of output arguments: expected %r, got %r %s'
            % (len(output_core_dims), len(out_shapes), error_context))

    sizes = dict(dim_sizes)
    for shape, core_dims in zip(out_shapes, output_core_dims):
      _update_dim_sizes(sizes, shape, core_dims, error_context,
                        is_input=False)

    return out
  return wrapped


def _apply_excluded(func: Callable[..., Any],
                    excluded: Collection[int | str],
                    args: Sequence[Any],
                    kwargs: dict[str, Any]) -> tuple[Callable[..., Any], Sequence[Any], dict[str, Any]]:
  """Partially apply positional arguments in `excluded` to a function."""
  if not excluded:
    return func, args, kwargs

  dynamic_args = [arg for i, arg in enumerate(args) if i not in excluded]
  dynamic_kwargs = {key: val for key, val in kwargs.items() if key not in excluded}
  static_args = [(i, args[i]) for i in sorted(e for e in excluded if isinstance(e, int))
                 if i < len(args)]
  static_kwargs = {key: val for key, val in kwargs.items() if key in excluded}

  def new_func(*args, **kwargs):
    args = list(args)
    for i, arg in static_args:
      args.insert(i, arg)
    return func(*args, **kwargs, **static_kwargs)

  return new_func, dynamic_args, dynamic_kwargs


@export
def vectorize(pyfunc, *, excluded=frozenset(), signature=None):
  """Define a vectorized function with broadcasting.

  :func:`vectorize` is a convenience wrapper for defining vectorized
  functions with broadcasting, in the style of NumPy's
  `generalized universal functions <https://numpy.org/doc/stable/reference/c-api/generalized-ufuncs.html>`_.
  It allows for defining functions that are automatically repeated across
  any leading dimensions, without the implementation of the function needing to
  be concerned about how to handle higher dimensional inputs.

  :func:`jax.numpy.vectorize` has the same interface as
  :class:`numpy.vectorize`, but it is syntactic sugar for an auto-batching
  transformation (:func:`vmap`) rather than a Python loop. This should be
  considerably more efficient, but the implementation must be written in terms
  of functions that act on JAX arrays.

  Args:
    pyfunc: function to vectorize.
    excluded: optional set of integers representing positional arguments for
      which the function will not be vectorized. These will be passed directly
      to ``pyfunc`` unmodified.
    signature: optional generalized universal function signature, e.g.,
      ``(m,n),(n)->(m)`` for vectorized matrix-vector multiplication. If
      provided, ``pyfunc`` will be called with (and expected to return) arrays
      with shapes given by the size of corresponding core dimensions. By
      default, pyfunc is assumed to take scalars arrays as input and output.

  Returns:
    Vectorized version of the given function.

  Examples:
    Here are a few examples of how one could write vectorized linear algebra
    routines using :func:`vectorize`:

    >>> from functools import partial

    >>> @partial(jnp.vectorize, signature='(k),(k)->(k)')
    ... def cross_product(a, b):
    ...   assert a.shape == b.shape and a.ndim == b.ndim == 1
    ...   return jnp.array([a[1] * b[2] - a[2] * b[1],
    ...                     a[2] * b[0] - a[0] * b[2],
    ...                     a[0] * b[1] - a[1] * b[0]])

    >>> @partial(jnp.vectorize, signature='(n,m),(m)->(n)')
    ... def matrix_vector_product(matrix, vector):
    ...   assert matrix.ndim == 2 and matrix.shape[1:] == vector.shape
    ...   return matrix @ vector

    These functions are only written to handle 1D or 2D arrays (the ``assert``
    statements will never be violated), but with vectorize they support
    arbitrary dimensional inputs with NumPy style broadcasting, e.g.,

    >>> cross_product(jnp.ones(3), jnp.ones(3)).shape
    (3,)
    >>> cross_product(jnp.ones((2, 3)), jnp.ones(3)).shape
    (2, 3)
    >>> cross_product(jnp.ones((1, 2, 3)), jnp.ones((2, 1, 3))).shape
    (2, 2, 3)
    >>> matrix_vector_product(jnp.ones(3), jnp.ones(3))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: input with shape (3,) does not have enough dimensions for all
    core dimensions ('n', 'k') on vectorized function with excluded=frozenset()
    and signature='(n,k),(k)->(k)'
    >>> matrix_vector_product(jnp.ones((2, 3)), jnp.ones(3)).shape
    (2,)
    >>> matrix_vector_product(jnp.ones((2, 3)), jnp.ones((4, 3))).shape
    (4, 2)

    Note that this has different semantics than `jnp.matmul`:

    >>> jnp.matmul(jnp.ones((2, 3)), jnp.ones((4, 3)))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    TypeError: dot_general requires contracting dimensions to have the same shape, got [3] and [4].
  """
  if any(not isinstance(exclude, (str, int)) for exclude in excluded):
    raise TypeError("jax.numpy.vectorize can only exclude integer or string arguments, "
                    "but excluded={!r}".format(excluded))
  if any(isinstance(e, int) and e < 0 for e in excluded):
    raise ValueError(f"excluded={excluded!r} contains negative numbers")

  @functools.wraps(pyfunc)
  def wrapped(*args, **kwargs):
    error_context = ("on vectorized function with excluded={!r} and "
                     "signature={!r}".format(excluded, signature))
    excluded_func, args, kwargs = _apply_excluded(pyfunc, excluded, args, kwargs)

    if signature is not None:
      input_core_dims, output_core_dims = _parse_gufunc_signature(signature)
    else:
      input_core_dims = [()] * len(args)
      output_core_dims = None

    none_args = {i for i, arg in enumerate(args) if arg is None}
    if any(none_args):
      if any(input_core_dims[i] != () for i in none_args):
        raise ValueError(f"Cannot pass None at locations {none_args} with {signature=}")
      excluded_func, args, _ = _apply_excluded(excluded_func, none_args, args, {})
      input_core_dims = [dim for i, dim in enumerate(input_core_dims) if i not in none_args]

    args = tuple(map(jnp.asarray, args))

    broadcast_shape, dim_sizes = _parse_input_dimensions(
        args, input_core_dims, error_context)

    checked_func = _check_output_dims(
        excluded_func, dim_sizes, output_core_dims, error_context)

    # Detect implicit rank promotion:
    if config.numpy_rank_promotion.value != "allow":
      ranks = [arg.ndim - len(core_dims)
               for arg, core_dims in zip(args, input_core_dims)
               if arg.ndim != 0]
      if len(set(ranks)) > 1:
        msg = (f"operands with shapes {[arg.shape for arg in args]} require rank"
               f" promotion for jnp.vectorize function with signature {signature}."
               " Set the jax_numpy_rank_promotion config option to 'allow' to"
               " disable this message; for more information, see"
               " https://jax.readthedocs.io/en/latest/rank_promotion_warning.html.")
        if config.numpy_rank_promotion.value == "warn":
          warnings.warn(msg)
        elif config.numpy_rank_promotion.value == "raise":
          raise ValueError(msg)


    # Rather than broadcasting all arguments to full broadcast shapes, prefer
    # expanding dimensions using vmap. By pushing broadcasting
    # into vmap, we can make use of more efficient batching rules for
    # primitives where only some arguments are batched (e.g., for
    # lax_linalg.triangular_solve), and avoid instantiating large broadcasted
    # arrays.

    squeezed_args = []
    rev_filled_shapes = []

    for arg, core_dims in zip(args, input_core_dims):
      noncore_shape = arg.shape[:arg.ndim - len(core_dims)]

      pad_ndim = len(broadcast_shape) - len(noncore_shape)
      filled_shape = pad_ndim * (1,) + noncore_shape
      rev_filled_shapes.append(filled_shape[::-1])

      squeeze_indices = tuple(i for i, size in enumerate(noncore_shape) if size == 1)
      squeezed_arg = jnp.squeeze(arg, axis=squeeze_indices)
      squeezed_args.append(squeezed_arg)

    vectorized_func = checked_func
    dims_to_expand = []
    for negdim, axis_sizes in enumerate(zip(*rev_filled_shapes)):
      in_axes = tuple(None if size == 1 else 0 for size in axis_sizes)
      if all(axis is None for axis in in_axes):
        dims_to_expand.append(len(broadcast_shape) - 1 - negdim)
      else:
        vectorized_func = api.vmap(vectorized_func, in_axes)
    result = vectorized_func(*squeezed_args)

    if not dims_to_expand:
      return result
    elif isinstance(result, tuple):
      return tuple(jnp.expand_dims(r, axis=dims_to_expand) for r in result)
    else:
      return jnp.expand_dims(result, axis=dims_to_expand)

  return wrapped
