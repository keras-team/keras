# Copyright 2018 The JAX Authors.
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

"""JAX user-facing transformations and utilities.

The transformations here mostly wrap internal transformations, providing
convenience flags to control behavior and handling Python containers of
arguments and outputs. The Python containers handled are pytrees (see
tree_util.py), which include nested tuples/lists/dicts, where the leaves are
arrays.
"""
from __future__ import annotations

import atexit
import collections
from collections.abc import Callable, Generator, Hashable, Iterable, Sequence
from functools import partial, lru_cache
import inspect
import math
import typing
from typing import (Any, Literal, NamedTuple, TypeVar, overload,
                    cast)
import weakref

import numpy as np
from contextlib import contextmanager

from jax._src import linear_util as lu
from jax._src import stages
from jax._src.tree_util import (
    tree_map, tree_flatten, tree_unflatten, tree_structure, tree_transpose,
    tree_leaves, Partial, PyTreeDef, all_leaves, keystr, broadcast_prefix,
    prefix_errors, generate_key_paths, tree_flatten_with_path)
from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import array
from jax._src import basearray
from jax._src import distributed
from jax._src import dtypes
from jax._src import sharding_impls
from jax._src import sharding_specs
from jax._src import source_info_util
from jax._src import traceback_util
from jax._src import pjit
from jax._src import xla_bridge as xb
from jax._src.core import eval_jaxpr, shaped_abstractify, ShapedArray
from jax._src.api_util import (
    flatten_fun, flatten_fun_nokwargs, flatten_fun_nokwargs2, argnums_partial,
    flatten_axes, donation_vector,
    rebase_donate_argnums, _ensure_index, _ensure_index_tuple,
    apply_flat_fun_nokwargs, check_callable, debug_info,
    result_paths, flat_out_axes, debug_info_final, fun_sourceinfo)
from jax._src.lax import lax as lax_internal
from jax._src.lib import jax_jit
from jax._src.lib import xla_client as xc
from jax._src.lib import pmap_lib
from jax._src.sharding import Sharding
from jax._src.sharding_impls import (PmapSharding, TransferToMemoryKind,
                                     NamedSharding)
from jax._src.layout import Layout, AutoLayout
from jax._src.traceback_util import api_boundary
from jax._src import tree_util
from jax._src.util import unzip2, safe_map, safe_zip, wraps, split_list
from jax._src import util

from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import partial_eval as pe
from jax._src.interpreters import pxla
from jax._src.interpreters import xla


traceback_util.register_exclusion(__file__)

_dtype = partial(dtypes.dtype, canonicalize=True)

AxisName = Hashable

Device = xc.Device

# These TypeVars are used below to express the fact that function types
# (i.e. call signatures) are invariant under the vmap transformation.
F = TypeVar("F", bound=Callable)
T = TypeVar("T")
U = TypeVar("U")

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


def _nan_check_posthook(fun, args, kwargs, output):
  """Hook function called by the C++ jit/pmap to perform NaN checking."""
  buffers = []
  for leaf in tree_leaves(output):
    if hasattr(leaf, "addressable_shards"):
      buffers.extend([shard.data for shard in leaf.addressable_shards])

  try:
    dispatch.check_special(pjit.pjit_p.name, buffers)
  except FloatingPointError:
    # compiled_fun can only raise in this case
    assert config.debug_nans.value or config.debug_infs.value
    print("Invalid nan value encountered in the output of a C++-jit/pmap "
          "function. Calling the de-optimized version.")
    fun._cache_miss(*args, **kwargs)[0]  # probably won't return

def _update_debug_special_global(_):
  if config._read("jax_debug_nans") or config._read("jax_debug_infs"):
    jax_jit.global_state().post_hook = _nan_check_posthook
  else:
    jax_jit.global_state().post_hook = None

def _update_debug_special_thread_local(_):
  if (config.debug_nans.get_local() == True or
      config.debug_infs.get_local() == True):
    jax_jit.thread_local_state().post_hook = _nan_check_posthook
  else:
    jax_jit.thread_local_state().post_hook = None

config.debug_nans._add_hooks(_update_debug_special_global,
                             _update_debug_special_thread_local)
config.debug_infs._add_hooks(_update_debug_special_global,
                             _update_debug_special_thread_local)


float0 = dtypes.float0


def jit(
  fun: Callable,
  in_shardings=sharding_impls.UNSPECIFIED,
  out_shardings=sharding_impls.UNSPECIFIED,
  static_argnums: int | Sequence[int] | None = None,
  static_argnames: str | Iterable[str] | None = None,
  donate_argnums: int | Sequence[int] | None = None,
  donate_argnames: str | Iterable[str] | None = None,
  keep_unused: bool = False,
  device: xc.Device | None = None,
  backend: str | None = None,
  inline: bool = False,
  abstracted_axes: Any | None = None,
  compiler_options: dict[str, Any] | None = None,
) -> pjit.JitWrapped:
  """Sets up ``fun`` for just-in-time compilation with XLA.

  Args:
    fun: Function to be jitted. ``fun`` should be a pure function.

      The arguments and return value of ``fun`` should be arrays, scalar, or
      (nested) standard Python containers (tuple/list/dict) thereof. Positional
      arguments indicated by ``static_argnums`` can be any hashable type. Static
      arguments are included as part of a compilation cache key, which is why
      hash and equality operators must be defined. JAX keeps a weak reference to
      ``fun`` for use as a compilation cache key, so the object ``fun`` must be
      weakly-referenceable.
    in_shardings: optional, a :py:class:`Sharding` or pytree with
      :py:class:`Sharding` leaves and structure that is a tree prefix of the
      positional arguments tuple to ``fun``. If provided, the positional
      arguments passed to ``fun`` must have shardings that are compatible with
      ``in_shardings`` or an error is raised, and the compiled computation has
      input shardings corresponding to ``in_shardings``. If not provided, the
      compiled computation's input shardings are inferred from argument
      shardings.
    out_shardings: optional, a :py:class:`Sharding` or pytree with
      :py:class:`Sharding` leaves and structure that is a tree prefix of the
      output of ``fun``. If provided, it has the same effect as applying
      corresponding :py:func:`jax.lax.with_sharding_constraint`s to the output
      of ``fun``.
    static_argnums: optional, an int or collection of ints that specify which
      positional arguments to treat as static (trace- and compile-time
      constant).

      Static arguments should be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and immutable. Otherwise they can be arbitrary
      Python objects. Calling the jitted function with different values for
      these constants will trigger recompilation. Arguments that are not
      array-like or containers thereof must be marked as static.

      If neither ``static_argnums`` nor ``static_argnames`` is provided, no
      arguments are treated as static. If ``static_argnums`` is not provided but
      ``static_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``static_argnames``
      (or vice versa). If both ``static_argnums`` and ``static_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``static_argnums`` or ``static_argnames`` will
      be treated as static.
    static_argnames: optional, a string or collection of strings specifying
      which named arguments to treat as static (compile-time constant). See the
      comment on ``static_argnums`` for details. If not
      provided but ``static_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    donate_argnums: optional, collection of integers to specify which positional
      argument buffers can be overwritten by the computation and marked deleted
      in the caller. It is safe to donate argument buffers if you no longer need
      them once the computation has started. In some cases XLA can make use of
      donated buffers to reduce the amount of memory needed to perform a
      computation, for example recycling one of your input buffers to store a
      result. You should not reuse buffers that you donate to a computation; JAX
      will raise an error if you try to. By default, no argument buffers are
      donated.

      If neither ``donate_argnums`` nor ``donate_argnames`` is provided, no
      arguments are donated. If ``donate_argnums`` is not provided but
      ``donate_argnames`` is, or vice versa, JAX uses
      :code:`inspect.signature(fun)` to find any positional arguments that
      correspond to ``donate_argnames``
      (or vice versa). If both ``donate_argnums`` and ``donate_argnames`` are
      provided, ``inspect.signature`` is not used, and only actual
      parameters listed in either ``donate_argnums`` or ``donate_argnames`` will
      be donated.

      For more details on buffer donation see the
      `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.
    donate_argnames: optional, a string or collection of strings specifying
      which named arguments are donated to the computation. See the
      comment on ``donate_argnums`` for details. If not
      provided but ``donate_argnums`` is set, the default is based on calling
      ``inspect.signature(fun)`` to find corresponding named arguments.
    keep_unused: optional boolean. If `False` (the default), arguments that JAX
      determines to be unused by `fun` *may* be dropped from resulting compiled
      XLA executables. Such arguments will not be transferred to the device nor
      provided to the underlying executable. If `True`, unused arguments will
      not be pruned.
    device: This is an experimental feature and the API is likely to change.
      Optional, the Device the jitted function will run on. (Available devices
      can be retrieved via :py:func:`jax.devices`.) The default is inherited
      from XLA's DeviceAssignment logic and is usually to use
      ``jax.devices()[0]``.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.
    inline: Optional boolean. Specify whether this function should be inlined
      into enclosing jaxprs. Default False.

  Returns:
    A wrapped version of ``fun``, set up for just-in-time compilation.

  Examples:
    In the following example, ``selu`` can be compiled into a single fused kernel
    by XLA:

    >>> import jax
    >>>
    >>> @jax.jit
    ... def selu(x, alpha=1.67, lmbda=1.05):
    ...   return lmbda * jax.numpy.where(x > 0, x, alpha * jax.numpy.exp(x) - alpha)
    >>>
    >>> key = jax.random.key(0)
    >>> x = jax.random.normal(key, (10,))
    >>> print(selu(x))  # doctest: +SKIP
    [-0.54485  0.27744 -0.29255 -0.91421 -0.62452 -0.24748
    -0.85743 -0.78232  0.76827  0.59566 ]

    To pass arguments such as ``static_argnames`` when decorating a function, a
    common pattern is to use :func:`functools.partial`:

    >>> from functools import partial
    >>>
    >>> @partial(jax.jit, static_argnames=['n'])
    ... def g(x, n):
    ...   for i in range(n):
    ...     x = x ** 2
    ...   return x
    >>>
    >>> g(jnp.arange(4), 3)
    Array([   0,    1,  256, 6561], dtype=int32)
  """
  return pjit.make_jit(
        fun, in_shardings, out_shardings, donate_argnums, donate_argnames,
        static_argnums, static_argnames, device, backend, abstracted_axes,
        keep_unused, inline, compiler_options, use_resource_env=False)


@contextmanager
def disable_jit(disable: bool = True):
  """Context manager that disables :py:func:`jit` behavior under its dynamic context.

  For debugging it is useful to have a mechanism that disables :py:func:`jit`
  everywhere in a dynamic context. Note that this not only disables explicit
  uses of :func:`jit` by the user, but will also remove any implicit JIT compilation
  used by the JAX library: this includes implicit JIT computation of `body` and
  `cond` functions passed to higher-level primitives like :func:`~jax.lax.scan` and
  :func:`~jax.lax.while_loop`, JIT used in implementations of :mod:`jax.numpy` functions,
  and any other case where :func:`jit` is used within an API's implementation.
  Note however that even under `disable_jit`, individual primitive operations
  will still be compiled by XLA as in normal eager op-by-op execution.

  Values that have a data dependence on the arguments to a jitted function are
  traced and abstracted. For example, an abstract value may be a
  :py:class:`ShapedArray` instance, representing the set of all possible arrays
  with a given shape and dtype, but not representing one concrete array with
  specific values. You might notice those if you use a benign side-effecting
  operation in a jitted function, like a print:

  >>> import jax
  >>>
  >>> @jax.jit
  ... def f(x):
  ...   y = x * 2
  ...   print("Value of y is", y)
  ...   return y + 3
  ...
  >>> print(f(jax.numpy.array([1, 2, 3])))  # doctest:+ELLIPSIS
  Value of y is Traced<ShapedArray(int32[3])>with<DynamicJaxprTrace...>
  [5 7 9]

  Here ``y`` has been abstracted by :py:func:`jit` to a :py:class:`ShapedArray`,
  which represents an array with a fixed shape and type but an arbitrary value.
  The value of ``y`` is also traced. If we want to see a concrete value while
  debugging, and avoid the tracer too, we can use the :py:func:`disable_jit`
  context manager:

  >>> import jax
  >>>
  >>> with jax.disable_jit():
  ...   print(f(jax.numpy.array([1, 2, 3])))
  ...
  Value of y is [2 4 6]
  [5 7 9]
  """
  with config.disable_jit(disable):
    yield


def grad(fun: Callable, argnums: int | Sequence[int] = 0,
         has_aux: bool = False, holomorphic: bool = False,
         allow_int: bool = False,
         reduce_axes: Sequence[AxisName] = ()) -> Callable:
  """Creates a function that evaluates the gradient of ``fun``.

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      ``argnums`` should be arrays, scalars, or standard Python containers.
      Argument arrays in the positions specified by ``argnums`` must be of
      inexact (i.e., floating-point or complex) type. It
      should return a scalar (which includes arrays with shape ``()`` but not
      arrays with shape ``(1,)`` etc.)
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. If True, inputs and outputs must be complex. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the gradient
    of ``fun``. If ``argnums`` is an integer then the gradient has the same
    shape and type as the positional argument indicated by that integer. If
    argnums is a tuple of integers, the gradient is a tuple of values with the
    same shapes and types as the corresponding arguments. If ``has_aux`` is True
    then a pair of (gradient, auxiliary_data) is returned.

  For example:

  >>> import jax
  >>>
  >>> grad_tanh = jax.grad(jax.numpy.tanh)
  >>> print(grad_tanh(0.2))
  0.961043
  """
  if reduce_axes:
    raise NotImplementedError("reduce_axes argument to grad is deprecated")
  del reduce_axes
  value_and_grad_f = value_and_grad(fun, argnums, has_aux=has_aux,
                                    holomorphic=holomorphic,
                                    allow_int=allow_int)

  docstr = ("Gradient of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "gradient, which has the same shape as the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def grad_f(*args, **kwargs):
    _, g = value_and_grad_f(*args, **kwargs)
    return g

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def grad_f_aux(*args, **kwargs):
    (_, aux), g = value_and_grad_f(*args, **kwargs)
    return g, aux

  return grad_f_aux if has_aux else grad_f

def value_and_grad(fun: Callable, argnums: int | Sequence[int] = 0,
                   has_aux: bool = False, holomorphic: bool = False,
                   allow_int: bool = False, reduce_axes: Sequence[AxisName] = ()
  ) -> Callable[..., tuple[Any, Any]]:
  """Create a function that evaluates both ``fun`` and the gradient of ``fun``.

  Args:
    fun: Function to be differentiated. Its arguments at positions specified by
      ``argnums`` should be arrays, scalars, or standard Python containers. It
      should return a scalar (which includes arrays with shape ``()`` but not
      arrays with shape ``(1,)`` etc.)
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. If True, inputs and outputs must be complex. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.

  Returns:
    A function with the same arguments as ``fun`` that evaluates both ``fun``
    and the gradient of ``fun`` and returns them as a pair (a two-element
    tuple). If ``argnums`` is an integer then the gradient has the same shape
    and type as the positional argument indicated by that integer. If argnums is
    a sequence of integers, the gradient is a tuple of values with the same
    shapes and types as the corresponding arguments. If ``has_aux`` is True
    then a tuple of ((value, auxiliary_data), gradient) is returned.
  """
  if reduce_axes:
    raise NotImplementedError("reduce_axes argument to grad is deprecated")
  del reduce_axes

  docstr = ("Value and gradient of {fun} with respect to positional "
            "argument(s) {argnums}. Takes the same arguments as {fun} but "
            "returns a two-element tuple where the first element is the value "
            "of {fun} and the second element is the gradient, which has the "
            "same shape as the arguments at positions {argnums}.")

  check_callable(fun)
  argnums = core.concrete_or_error(_ensure_index, argnums)

  @wraps(fun, docstr=docstr, argnums=argnums)
  @api_boundary
  def value_and_grad_f(*args, **kwargs):
    max_argnum = argnums if isinstance(argnums, int) else max(argnums)
    if max_argnum >= len(args):
      raise TypeError(f"differentiating with respect to {argnums=} requires at least "
                      f"{max_argnum + 1} positional arguments to be passed by the caller, "
                      f"but got only {len(args)} positional arguments.")
    fun_src_info = fun_sourceinfo(fun)
    fun_signature = api_util.fun_signature(fun)
    dbg = debug_info('value_and_grad', fun_src_info, fun_signature,
                     args, kwargs, (), ())

    f = lu.wrap_init(fun, params=kwargs, debug_info=dbg)
    f_partial, dyn_args = argnums_partial(f, argnums, args,
                                          require_static_args_hashable=False)
    for leaf in tree_leaves(dyn_args):
      _check_input_dtype_grad(holomorphic, allow_int, leaf)
    if not has_aux:
      ans, vjp_py = _vjp(f_partial, *dyn_args)
    else:
      ans, vjp_py, aux = _vjp(
          f_partial, *dyn_args, has_aux=True)
    _check_scalar(ans)
    tree_map(partial(_check_output_dtype_grad, holomorphic), ans)
    g = vjp_py(lax_internal._one(ans))
    g = g[0] if isinstance(argnums, int) else g
    if not has_aux:
      return ans, g
    else:
      return (ans, aux), g

  return value_and_grad_f

def _check_scalar(x):
  msg = "Gradient only defined for scalar-output functions. Output {}.".format
  try:
    aval = core.get_aval(x)
  except TypeError as e:
    raise TypeError(msg(f"was {x}")) from e
  else:
    if isinstance(aval, ShapedArray):
      if aval.shape != ():
        raise TypeError(msg(f"had shape: {aval.shape}"))
    else:
      raise TypeError(msg(f"had abstract value {aval}"))

def _check_input_dtype_revderiv(name, holomorphic, allow_int, x):
  dispatch.check_arg(x)
  aval = core.get_aval(x)
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError(f"{name} with holomorphic=True requires inputs with complex dtype, "
                      f"but got {aval.dtype.name}.")
  if (dtypes.issubdtype(aval.dtype, dtypes.extended) or
      dtypes.issubdtype(aval.dtype, np.integer) or
      dtypes.issubdtype(aval.dtype, np.bool_)):
    if not allow_int:
      raise TypeError(f"{name} requires real- or complex-valued inputs (input dtype "
                      f"that is a sub-dtype of np.inexact), but got {aval.dtype.name}. "
                      "If you want to use Boolean- or integer-valued inputs, use vjp "
                      "or set allow_int to True.")
  elif not dtypes.issubdtype(aval.dtype, np.inexact):
    raise TypeError(f"{name} requires numerical-valued inputs (input dtype that is a "
                    f"sub-dtype of np.bool_ or np.number), but got {aval.dtype.name}.")
_check_input_dtype_grad = partial(_check_input_dtype_revderiv, "grad")

def _check_output_dtype_revderiv(name, holomorphic, x):
  aval = core.get_aval(x)
  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    raise TypeError(
        f"{name} with output element type {aval.dtype.name}")
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError(f"{name} with holomorphic=True requires outputs with complex dtype, "
                      f"but got {aval.dtype.name}.")
  elif dtypes.issubdtype(aval.dtype, np.complexfloating):
    raise TypeError(f"{name} requires real-valued outputs (output dtype that is "
                    f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
                    "For holomorphic differentiation, pass holomorphic=True. "
                    "For differentiation of non-holomorphic functions involving complex "
                    "outputs, use jax.vjp directly.")
  elif not dtypes.issubdtype(aval.dtype, np.floating):
    raise TypeError(f"{name} requires real-valued outputs (output dtype that is "
                    f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
                    "For differentiation of functions with integer outputs, use "
                    "jax.vjp directly.")
_check_output_dtype_grad = partial(_check_output_dtype_revderiv, "grad")


def jacfwd(fun: Callable, argnums: int | Sequence[int] = 0,
           has_aux: bool = False, holomorphic: bool = False) -> Callable:
  """Jacobian of ``fun`` evaluated column-by-column using forward-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using forward-mode automatic differentiation. If ``has_aux`` is True
    then a pair of (jacobian, auxiliary_data) is returned.

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x):
  ...   return jnp.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
  ...
  >>> print(jax.jacfwd(f)(jnp.array([1., 2., 3.])))
  [[ 1.       0.       0.     ]
   [ 0.       0.       5.     ]
   [ 0.      16.      -2.     ]
   [ 1.6209   0.       0.84147]]
  """
  check_callable(fun)
  argnums = _ensure_index(argnums)

  docstr = ("Jacobian of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "jacobian of the output with respect to the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args,
                                          require_static_args_hashable=False)
    tree_map(partial(_check_input_dtype_jacfwd, holomorphic), dyn_args)
    if not has_aux:
      pushfwd: Callable = partial(_jvp, f_partial, dyn_args)
      y, jac = vmap(pushfwd, out_axes=(None, -1))(_std_basis(dyn_args))
    else:
      pushfwd: Callable = partial(_jvp, f_partial, dyn_args, has_aux=True)
      y, jac, aux = vmap(pushfwd, out_axes=(None, -1, None))(_std_basis(dyn_args))
    tree_map(partial(_check_output_dtype_jacfwd, holomorphic), y)
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac_tree = tree_map(partial(_jacfwd_unravel, example_args), y, jac)
    if not has_aux:
      return jac_tree
    else:
      return jac_tree, aux

  return jacfun

def _check_input_dtype_jacfwd(holomorphic: bool, x: Any) -> None:
  dispatch.check_arg(x)
  aval = core.get_aval(x)
  if dtypes.issubdtype(aval.dtype, dtypes.extended):
    raise TypeError(
        f"jacfwd with input element type {aval.dtype.name}")
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError("jacfwd with holomorphic=True requires inputs with complex "
                      f"dtype, but got {aval.dtype.name}.")
  elif not dtypes.issubdtype(aval.dtype, np.floating):
    raise TypeError("jacfwd requires real-valued inputs (input dtype that is "
                    f"a sub-dtype of np.floating), but got {aval.dtype.name}. "
                    "For holomorphic differentiation, pass holomorphic=True. "
                    "For differentiation of non-holomorphic functions involving "
                    "complex inputs or integer inputs, use jax.jvp directly.")

def _check_output_dtype_jacfwd(holomorphic, x):
  aval = core.get_aval(x)
  if holomorphic:
    if not dtypes.issubdtype(aval.dtype, np.complexfloating):
      raise TypeError("jacfwd with holomorphic=True requires outputs with complex dtype, "
                      f"but got {aval.dtype.name}.")

def jacrev(fun: Callable, argnums: int | Sequence[int] = 0,
           has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> Callable:
  """Jacobian of ``fun`` evaluated row-by-row using reverse-mode AD.

  Args:
    fun: Function whose Jacobian is to be computed.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.
    allow_int: Optional, bool. Whether to allow differentiating with
      respect to integer valued inputs. The gradient of an integer input will
      have a trivial vector-space dtype (float0). Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Jacobian of
    ``fun`` using reverse-mode automatic differentiation. If ``has_aux`` is True
    then a pair of (jacobian, auxiliary_data) is returned.

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x):
  ...   return jnp.asarray(
  ...     [x[0], 5*x[2], 4*x[1]**2 - 2*x[2], x[2] * jnp.sin(x[0])])
  ...
  >>> print(jax.jacrev(f)(jnp.array([1., 2., 3.])))
  [[ 1.       0.       0.     ]
   [ 0.       0.       5.     ]
   [ 0.      16.      -2.     ]
   [ 1.6209   0.       0.84147]]
  """
  check_callable(fun)

  docstr = ("Jacobian of {fun} with respect to positional argument(s) "
            "{argnums}. Takes the same arguments as {fun} but returns the "
            "jacobian of the output with respect to the arguments at "
            "positions {argnums}.")

  @wraps(fun, docstr=docstr, argnums=argnums)
  def jacfun(*args, **kwargs):
    f = lu.wrap_init(fun, kwargs)
    f_partial, dyn_args = argnums_partial(f, argnums, args,
                                          require_static_args_hashable=False)
    tree_map(partial(_check_input_dtype_jacrev, holomorphic, allow_int), dyn_args)
    if not has_aux:
      y, pullback = _vjp(f_partial, *dyn_args)
    else:
      y, pullback, aux = _vjp(f_partial, *dyn_args, has_aux=True)
    tree_map(partial(_check_output_dtype_jacrev, holomorphic), y)
    jac = vmap(pullback)(_std_basis(y))
    jac = jac[0] if isinstance(argnums, int) else jac
    example_args = dyn_args[0] if isinstance(argnums, int) else dyn_args
    jac_tree = tree_map(partial(_jacrev_unravel, y), example_args, jac)
    jac_tree = tree_transpose(tree_structure(example_args), tree_structure(y), jac_tree)
    if not has_aux:
      return jac_tree
    else:
      return jac_tree, aux

  return jacfun


def jacobian(fun: Callable, argnums: int | Sequence[int] = 0,
             has_aux: bool = False, holomorphic: bool = False, allow_int: bool = False) -> Callable:
  """Alias of :func:`jax.jacrev`."""
  return jacrev(fun, argnums=argnums, has_aux=has_aux, holomorphic=holomorphic, allow_int=allow_int)


_check_input_dtype_jacrev = partial(_check_input_dtype_revderiv, "jacrev")
_check_output_dtype_jacrev = partial(_check_output_dtype_revderiv, "jacrev")


def hessian(fun: Callable, argnums: int | Sequence[int] = 0,
            has_aux: bool = False, holomorphic: bool = False) -> Callable:
  """Hessian of ``fun`` as a dense array.

  Args:
    fun: Function whose Hessian is to be computed.  Its arguments at positions
      specified by ``argnums`` should be arrays, scalars, or standard Python
      containers thereof. It should return arrays, scalars, or standard Python
      containers thereof.
    argnums: Optional, integer or sequence of integers. Specifies which
      positional argument(s) to differentiate with respect to (default ``0``).
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
      first element is considered the output of the mathematical function to be
      differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether ``fun`` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as ``fun``, that evaluates the Hessian of
    ``fun``.

  >>> import jax
  >>>
  >>> g = lambda x: x[0]**3 - 2*x[0]*x[1] - x[1]**6
  >>> print(jax.hessian(g)(jax.numpy.array([1., 2.])))
  [[   6.   -2.]
   [  -2. -480.]]

  :py:func:`hessian` is a generalization of the usual definition of the Hessian
  that supports nested Python containers (i.e. pytrees) as inputs and outputs.
  The tree structure of ``jax.hessian(fun)(x)`` is given by forming a tree
  product of the structure of ``fun(x)`` with a tree product of two copies of
  the structure of ``x``. A tree product of two tree structures is formed by
  replacing each leaf of the first tree with a copy of the second. For example:

  >>> import jax.numpy as jnp
  >>> f = lambda dct: {"c": jnp.power(dct["a"], dct["b"])}
  >>> print(jax.hessian(f)({"a": jnp.arange(2.) + 1., "b": jnp.arange(2.) + 2.}))
  {'c': {'a': {'a': Array([[[ 2.,  0.], [ 0.,  0.]],
                           [[ 0.,  0.], [ 0., 12.]]], dtype=float32),
               'b': Array([[[ 1.      ,  0.      ], [ 0.      ,  0.      ]],
                           [[ 0.      ,  0.      ], [ 0.      , 12.317766]]], dtype=float32)},
         'b': {'a': Array([[[ 1.      ,  0.      ], [ 0.      ,  0.      ]],
                           [[ 0.      ,  0.      ], [ 0.      , 12.317766]]], dtype=float32),
               'b': Array([[[0.      , 0.      ], [0.      , 0.      ]],
                           [[0.      , 0.      ], [0.      , 3.843624]]], dtype=float32)}}}

  Thus each leaf in the tree structure of ``jax.hessian(fun)(x)`` corresponds to
  a leaf of ``fun(x)`` and a pair of leaves of ``x``. For each leaf in
  ``jax.hessian(fun)(x)``, if the corresponding array leaf of ``fun(x)`` has
  shape ``(out_1, out_2, ...)`` and the corresponding array leaves of ``x`` have
  shape ``(in_1_1, in_1_2, ...)`` and ``(in_2_1, in_2_2, ...)`` respectively,
  then the Hessian leaf has shape ``(out_1, out_2, ..., in_1_1, in_1_2, ...,
  in_2_1, in_2_2, ...)``. In other words, the Python tree structure represents
  the block structure of the Hessian, with blocks determined by the input and
  output pytrees.

  In particular, an array is produced (with no pytrees involved) when the
  function input ``x`` and output ``fun(x)`` are each a single array, as in the
  ``g`` example above. If ``fun(x)`` has shape ``(out1, out2, ...)`` and ``x``
  has shape ``(in1, in2, ...)`` then ``jax.hessian(fun)(x)`` has shape
  ``(out1, out2, ..., in1, in2, ..., in1, in2, ...)``. To flatten pytrees into
  1D vectors, consider using :py:func:`jax.flatten_util.flatten_pytree`.
  """
  return jacfwd(jacrev(fun, argnums, has_aux=has_aux, holomorphic=holomorphic),
                argnums, has_aux=has_aux, holomorphic=holomorphic)

def _std_basis(pytree):
  import jax.numpy as jnp
  leaves, _ = tree_flatten(pytree)
  ndim = sum(map(np.size, leaves))
  dtype = dtypes.result_type(*leaves)
  flat_basis = jnp.eye(ndim, dtype=dtype)
  return _unravel_array_into_pytree(pytree, 1, None, flat_basis)

def _jacfwd_unravel(input_pytree, output_pytree_leaf, arr):
  return _unravel_array_into_pytree(
    input_pytree, -1, output_pytree_leaf, arr)

def _jacrev_unravel(output_pytree, input_pytree_leaf, arr):
  return _unravel_array_into_pytree(
    output_pytree, 0, input_pytree_leaf, arr)

def _possible_downcast(x, example):
  if (dtypes.issubdtype(x.dtype, np.complexfloating) and
      not dtypes.issubdtype(_dtype(example), np.complexfloating)):
    x = x.real
  dtype = None if example is None else _dtype(example)
  weak_type = None if example is None else dtypes.is_weakly_typed(example)
  return lax_internal._convert_element_type(x, dtype, weak_type)

def _unravel_array_into_pytree(pytree, axis, example, arr):
  """Unravel an array into a PyTree with a given structure.
  Args:
      pytree: The pytree that provides the structure.
      axis: The parameter axis is either -1, 0, or 1.  It controls the
        resulting shapes.
      example: If specified, cast the components to the matching dtype/weak_type,
        or else use the pytree leaf type if example is None.
      arr: The array to be unraveled.
  """
  leaves, treedef = tree_flatten(pytree)
  axis = axis % arr.ndim
  shapes = [arr.shape[:axis] + np.shape(l) + arr.shape[axis+1:] for l in leaves]
  parts = _split(arr, np.cumsum(map(np.size, leaves[:-1])), axis)
  reshaped_parts = [
      _possible_downcast(np.reshape(x, shape), leaf if example is None else example)
      for x, shape, leaf in zip(parts, shapes, leaves)]
  return tree_unflatten(treedef, reshaped_parts)

def _split(x, indices, axis):
  if isinstance(x, np.ndarray):
    return np.split(x, indices, axis)
  else:
    return x._split(indices, axis)


def vmap(fun: F,
         in_axes: int | None | Sequence[Any] = 0,
         out_axes: Any = 0,
         axis_name: AxisName | None = None,
         axis_size: int | None = None,
         spmd_axis_name: AxisName | tuple[AxisName, ...] | None = None
         ) -> F:
  """Vectorizing map. Creates a function which maps ``fun`` over argument axes.

  Args:
    fun: Function to be mapped over additional axes.
    in_axes: An integer, None, or sequence of values specifying which input
      array axes to map over.

      If each positional argument to ``fun`` is an array, then ``in_axes`` can
      be an integer, a None, or a tuple of integers and Nones with length equal
      to the number of positional arguments to ``fun``. An integer or ``None``
      indicates which array axis to map over for all arguments (with ``None``
      indicating not to map any axis), and a tuple indicates which axis to map
      for each corresponding positional argument. Axis integers must be in the
      range ``[-ndim, ndim)`` for each array, where ``ndim`` is the number of
      dimensions (axes) of the corresponding input array.

      If the positional arguments to ``fun`` are container (pytree) types, ``in_axes``
      must be a sequence with length equal to the number of positional arguments to
      ``fun``, and for each argument the corresponding element of ``in_axes`` can
      be a container with a matching pytree structure specifying the mapping of its
      container elements. In other words, ``in_axes`` must be a container tree prefix
      of the positional argument tuple passed to ``fun``. See this link for more detail:
      https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees

      Either ``axis_size`` must be provided explicitly, or at least one
      positional argument must have ``in_axes`` not None. The sizes of the
      mapped input axes for all mapped positional arguments must all be equal.

      Arguments passed as keywords are always mapped over their leading axis
      (i.e. axis index 0).

      See below for examples.

    out_axes: An integer, None, or (nested) standard Python container
      (tuple/list/dict) thereof indicating where the mapped axis should appear
      in the output. All outputs with a mapped axis must have a non-None
      ``out_axes`` specification. Axis integers must be in the range ``[-ndim,
      ndim)`` for each output array, where ``ndim`` is the number of dimensions
      (axes) of the array returned by the :func:`vmap`-ed function, which is one
      more than the number of dimensions (axes) of the corresponding array
      returned by ``fun``.
    axis_name: Optional, a hashable Python object used to identify the mapped
      axis so that parallel collectives can be applied.
    axis_size: Optional, an integer indicating the size of the axis to be
      mapped. If not provided, the mapped axis size is inferred from arguments.

  Returns:
    Batched/vectorized version of ``fun`` with arguments that correspond to
    those of ``fun``, but with extra array axes at positions indicated by
    ``in_axes``, and a return value that corresponds to that of ``fun``, but
    with extra array axes at positions indicated by ``out_axes``.

  For example, we can implement a matrix-matrix product using a vector dot
  product:

  >>> import jax.numpy as jnp
  >>>
  >>> vv = lambda x, y: jnp.vdot(x, y)  #  ([a], [a]) -> []
  >>> mv = vmap(vv, (0, None), 0)      #  ([b,a], [a]) -> [b]      (b is the mapped axis)
  >>> mm = vmap(mv, (None, 1), 1)      #  ([b,a], [a,c]) -> [b,c]  (c is the mapped axis)

  Here we use ``[a,b]`` to indicate an array with shape (a,b). Here are some
  variants:

  >>> mv1 = vmap(vv, (0, 0), 0)   #  ([b,a], [b,a]) -> [b]        (b is the mapped axis)
  >>> mv2 = vmap(vv, (0, 1), 0)   #  ([b,a], [a,b]) -> [b]        (b is the mapped axis)
  >>> mm2 = vmap(mv2, (1, 1), 0)  #  ([b,c,a], [a,c,b]) -> [c,b]  (c is the mapped axis)

  Here's an example of using container types in ``in_axes`` to specify which
  axes of the container elements to map over:

  >>> A, B, C, D = 2, 3, 4, 5
  >>> x = jnp.ones((A, B))
  >>> y = jnp.ones((B, C))
  >>> z = jnp.ones((C, D))
  >>> def foo(tree_arg):
  ...   x, (y, z) = tree_arg
  ...   return jnp.dot(x, jnp.dot(y, z))
  >>> tree = (x, (y, z))
  >>> print(foo(tree))
  [[12. 12. 12. 12. 12.]
   [12. 12. 12. 12. 12.]]
  >>> from jax import vmap
  >>> K = 6  # batch size
  >>> x = jnp.ones((K, A, B))  # batch axis in different locations
  >>> y = jnp.ones((B, K, C))
  >>> z = jnp.ones((C, D, K))
  >>> tree = (x, (y, z))
  >>> vfoo = vmap(foo, in_axes=((0, (1, 2)),))
  >>> print(vfoo(tree).shape)
  (6, 2, 5)

  Here's another example using container types in ``in_axes``, this time a
  dictionary, to specify the elements of the container to map over:

  >>> dct = {'a': 0., 'b': jnp.arange(5.)}
  >>> x = 1.
  >>> def foo(dct, x):
  ...  return dct['a'] + dct['b'] + x
  >>> out = vmap(foo, in_axes=({'a': None, 'b': 0}, None))(dct, x)
  >>> print(out)
  [1. 2. 3. 4. 5.]

  The results of a vectorized function can be mapped or unmapped. For example,
  the function below returns a pair with the first element mapped and the second
  unmapped. Only for unmapped results we can specify ``out_axes`` to be ``None``
  (to keep it unmapped).

  >>> print(vmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None), out_axes=(0, None))(jnp.arange(2.), 4.))
  (Array([4., 5.], dtype=float32), 8.0)

  If the ``out_axes`` is specified for an unmapped result, the result is
  broadcast across the mapped axis:

  >>> print(vmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None), out_axes=0)(jnp.arange(2.), 4.))
  (Array([4., 5.], dtype=float32), Array([8., 8.], dtype=float32, weak_type=True))

  If the ``out_axes`` is specified for a mapped result, the result is transposed
  accordingly.

  Finally, here's an example using ``axis_name`` together with collectives:

  >>> xs = jnp.arange(3. * 4.).reshape(3, 4)
  >>> print(vmap(lambda x: lax.psum(x, 'i'), axis_name='i')(xs))
  [[12. 15. 18. 21.]
   [12. 15. 18. 21.]
   [12. 15. 18. 21.]]

  See the :py:func:`jax.pmap` docstring for more examples involving collectives.
  """
  check_callable(fun)
  docstr = ("Vectorized version of {fun}. Takes similar arguments as {fun} "
            "but with additional array axes over which {fun} is mapped.")
  if fun.__doc__:
    docstr += "\n\nOriginal documentation:\n\n"
    docstr += fun.__doc__

  axis_name = core.no_axis_name if axis_name is None else axis_name
  if spmd_axis_name is not None and type(spmd_axis_name) is not tuple:
    spmd_axis_name = (spmd_axis_name,)

  if isinstance(in_axes, list):
    # To be a tree prefix of the positional args tuple, in_axes can never be a
    # list: if in_axes is not a leaf, it must be a tuple of trees. However,
    # in cases like these users expect tuples and lists to be treated
    # essentially interchangeably, so we canonicalize lists to tuples here
    # rather than raising an error. https://github.com/jax-ml/jax/issues/2367
    in_axes = tuple(in_axes)

  if not (in_axes is None or type(in_axes) in {int, tuple, *batching.spec_types}):
    raise TypeError("vmap in_axes must be an int, None, or a tuple of entries corresponding "
                    f"to the positional arguments passed to the function, but got {in_axes}.")
  if not all(type(l) in {int, *batching.spec_types} for l in tree_leaves(in_axes)):
    raise TypeError("vmap in_axes must be an int, None, or (nested) container "
                    f"with those types as leaves, but got {in_axes}.")
  if not all(type(l) in {int, *batching.spec_types} for l in tree_leaves(out_axes)):
    raise TypeError("vmap out_axes must be an int, None, or (nested) container "
                    f"with those types as leaves, but got {out_axes}.")

  @wraps(fun, docstr=docstr)
  @api_boundary
  def vmap_f(*args, **kwargs):
    if isinstance(in_axes, tuple) and len(in_axes) != len(args):
      raise ValueError("vmap in_axes must be an int, None, or a tuple of entries corresponding "
                       "to the positional arguments passed to the function, "
                       f"but got {len(in_axes)=}, {len(args)=}")
    args_flat, in_tree  = tree_flatten((args, kwargs), is_leaf=batching.is_vmappable)
    f = lu.wrap_init(fun)
    flat_fun, out_tree = batching.flatten_fun_for_vmap(f, in_tree)
    in_axes_flat = flatten_axes("vmap in_axes", in_tree, (in_axes, 0), kws=True)
    axis_size_ = (axis_size if axis_size is not None else
                  _mapped_axis_size(fun, in_tree, args_flat, in_axes_flat, "vmap"))
    try:
      axis_data = batching.AxisData(axis_name, axis_size_, spmd_axis_name)
      out_flat = batching.batch(
          flat_fun, axis_data, in_axes_flat,
          lambda: flatten_axes("vmap out_axes", out_tree(), out_axes)
      ).call_wrapped(*args_flat)
    except batching.SpecMatchError as e:
      out_axes_flat = flatten_axes("vmap out_axes", out_tree(), out_axes)
      out_axes_full = tree_unflatten(out_tree(), out_axes_flat)
      pairs, _ = tree_flatten_with_path(out_axes_full, is_leaf=lambda x: x is None)

      path, _ = pairs[e.leaf_idx]
      raise ValueError(f'at vmap out_axes{keystr(path)}, got axis spec {e.dst} '
                       f'but output was batched on axis {e.src}') from None
    return tree_unflatten(out_tree(), out_flat)

  return cast(F, vmap_f)

def _mapped_axis_size(fn, tree, vals, dims, name):
  if not vals:
    args, kwargs = tree_unflatten(tree, vals)
    raise ValueError(
        f"{name} wrapped function must be passed at least one argument "
        f"containing an array, got empty *args={args} and **kwargs={kwargs}"
    )

  def _get_axis_size(name: str, shape: tuple[core.AxisSize, ...], axis: int
                     ) -> core.AxisSize:
    try:
      return shape[axis]
    except (IndexError, TypeError) as e:
      min_rank = axis + 1 if axis >= 0 else -axis
      # TODO(mattjj): better error message here
      raise ValueError(
          f"{name} was requested to map its argument along axis {axis}, "
          f"which implies that its rank should be at least {min_rank}, "
          f"but is only {len(shape)} (its shape is {shape})") from e

  sizes = core.dedup_referents(_get_axis_size(name, np.shape(x), d)
                               for x, d in zip(vals, dims) if d is not None)
  if len(sizes) == 1:
    sz, = sizes
    return sz
  if not sizes:
    msg = f"{name} must have at least one non-None value in in_axes"
    raise ValueError(msg)

  def _get_argument_type(x):
    try:
      return shaped_abstractify(x).str_short()
    except TypeError: # Catch all for user specified objects that can't be interpreted as a data type
      return "unknown"
  msg = [f"{name} got inconsistent sizes for array axes to be mapped:\n"]
  args, kwargs = tree_unflatten(tree, vals)
  try:
    ba = inspect.signature(fn).bind(*args, **kwargs)
    signature_parameters: list[str] = list(ba.signature.parameters.keys())
  except (TypeError, ValueError):
    signature_parameters = None

  def arg_name(key_path):
    if signature_parameters is None:
      return f"args{keystr(key_path)}"
    # args is a tuple, so key_path[0].idx is the index into args.
    i = key_path[0].idx
    res = f"argument {signature_parameters[i]}"
    if len(key_path) > 1:
      res += keystr(key_path[1:])
    return res

  args_paths = [
    f"{arg_name(p)} of type {_get_argument_type(x)}"
    for (p, x) in generate_key_paths(args)
  ]
  kwargs_paths = [
    f"kwargs{keystr(p)} of type {_get_argument_type(x)}"
    for p, x in generate_key_paths(kwargs)
  ]
  key_paths = [*args_paths, *kwargs_paths]
  all_sizes = [_get_axis_size(name, np.shape(x), d) if d is not None else None
               for x, d in zip(vals, dims)]
  size_counts = collections.Counter(s for s in all_sizes if s is not None)
  (sz, ct), *other_counts = counts = size_counts.most_common()
  def _all_sizes_index(sz):
    for i, isz in enumerate(all_sizes):
      if core.definitely_equal(isz, sz): return i
    assert False, (sz, all_sizes)

  ex, *examples = (key_paths[_all_sizes_index(sz)] for sz, _ in counts)
  ax, *axs = (dims[_all_sizes_index(sz)] for sz, _ in counts)
  if ct == 1:
    msg.append(f"  * one axis had size {sz}: axis {ax} of {ex};\n")
  else:
    msg.append(f"  * most axes ({ct} of them) had size {sz}, e.g. axis {ax} of {ex};\n")
  for ex, ax, (sz, ct) in zip(examples, axs, other_counts):
    if ct == 1:
      msg.append(f"  * one axis had size {sz}: axis {ax} of {ex};\n")
    else:
      msg.append(f"  * some axes ({ct} of them) had size {sz}, e.g. axis {ax} of {ex};\n")
  raise ValueError(''.join(msg)[:-2])  # remove last semicolon and newline


def pmap(
    fun: Callable,
    axis_name: AxisName | None = None,
    *,
    in_axes=0,
    out_axes=0,
    static_broadcasted_argnums: int | Iterable[int] = (),
    devices: Sequence[xc.Device] | None = None,  # noqa: F811
    backend: str | None = None,
    axis_size: int | None = None,
    donate_argnums: int | Iterable[int] = (),
    global_arg_shapes: tuple[tuple[int, ...], ...] | None = None,
  ) -> Any:
  """Parallel map with support for collective operations.

  The purpose of :py:func:`pmap` is to express single-program multiple-data
  (SPMD) programs. Applying :py:func:`pmap` to a function will compile the
  function with XLA (similarly to :py:func:`jit`), then execute it in parallel
  on XLA devices, such as multiple GPUs or multiple TPU cores. Semantically it
  is comparable to :py:func:`vmap` because both transformations map a function
  over array axes, but where :py:func:`vmap` vectorizes functions by pushing the
  mapped axis down into primitive operations, :py:func:`pmap` instead replicates
  the function and executes each replica on its own XLA device in parallel.

  The mapped axis size must be less than or equal to the number of local XLA
  devices available, as returned by :py:func:`jax.local_device_count()` (unless
  ``devices`` is specified, see below). For nested :py:func:`pmap` calls, the
  product of the mapped axis sizes must be less than or equal to the number of
  XLA devices.

  .. note::
    :py:func:`pmap` compiles ``fun``, so while it can be combined with
    :py:func:`jit`, it's usually unnecessary.

  :py:func:`pmap` requires that all of the participating devices are identical.
  For example, it is not possible to use :py:func:`pmap` to parallelize a
  computation across two different models of GPU. It is currently an error for
  the same device to participate twice in the same `pmap`.

  **Multi-process platforms:** On multi-process platforms such as TPU pods,
  :py:func:`pmap` is designed to be used in SPMD Python programs, where every
  process is running the same Python code such that all processes run the same
  pmapped function in the same order. Each process should still call the pmapped
  function with mapped axis size equal to the number of *local* devices (unless
  ``devices`` is specified, see below), and an array of the same leading axis
  size will be returned as usual. However, any collective operations in ``fun``
  will be computed over *all* participating devices, including those on other
  processes, via device-to-device communication.  Conceptually, this can be
  thought of as running a pmap over a single array sharded across processes,
  where each process "sees" only its local shard of the input and output. The
  SPMD model requires that the same multi-process pmaps must be run in the same
  order on all devices, but they can be interspersed with arbitrary operations
  running in a single process.

  Args:
    fun: Function to be mapped over argument axes. Its arguments and return
      value should be arrays, scalars, or (nested) standard Python containers
      (tuple/list/dict) thereof. Positional arguments indicated by
      ``static_broadcasted_argnums`` can be anything at all, provided they are
      hashable and have an equality operation defined.
    axis_name: Optional, a hashable Python object used to identify the mapped
      axis so that parallel collectives can be applied.
    in_axes: A non-negative integer, None, or nested Python container thereof
      that specifies which axes of positional arguments to map over. Arguments
      passed as keywords are always mapped over their leading axis (i.e. axis
      index 0). See :py:func:`vmap` for details.
    out_axes: A non-negative integer, None, or nested Python container thereof
      indicating where the mapped axis should appear in the output. All outputs
      with a mapped axis must have a non-None ``out_axes`` specification
      (see :py:func:`vmap`).
    static_broadcasted_argnums: An int or collection of ints specifying which
      positional arguments to treat as static (compile-time constant).
      Operations that only depend on static arguments will be constant-folded.
      Calling the pmapped function with different values for these constants
      will trigger recompilation. If the pmapped function is called with fewer
      positional arguments than indicated by ``static_broadcasted_argnums`` then
      an error is raised. Each of the static arguments will be broadcasted to
      all devices. Arguments that are not arrays or containers thereof must be
      marked as static. Defaults to ().

      Static arguments must be hashable, meaning both ``__hash__`` and
      ``__eq__`` are implemented, and should be immutable.

    devices: This is an experimental feature and the API is likely to change.
      Optional, a sequence of Devices to map over. (Available devices can be
      retrieved via jax.devices()). Must be given identically for each process
      in multi-process settings (and will therefore include devices across
      processes). If specified, the size of the mapped axis must be equal to
      the number of devices in the sequence local to the given process. Nested
      :py:func:`pmap` s with ``devices`` specified in either the inner or outer
      :py:func:`pmap` are not yet supported.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the XLA backend. 'cpu', 'gpu', or 'tpu'.
    axis_size: Optional; the size of the mapped axis.
    donate_argnums: Specify which positional argument buffers are "donated" to
      the computation. It is safe to donate argument buffers if you no longer need
      them once the computation has finished. In some cases XLA can make use of
      donated buffers to reduce the amount of memory needed to perform a
      computation, for example recycling one of your input buffers to store a
      result. You should not reuse buffers that you donate to a computation, JAX
      will raise an error if you try to.
      Note that donate_argnums only work for positional arguments, and keyword
      arguments will not be donated.

      For more details on buffer donation see the
      `FAQ <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>`_.

  Returns:
    A parallelized version of ``fun`` with arguments that correspond to those of
    ``fun`` but with extra array axes at positions indicated by ``in_axes`` and
    with output that has an additional leading array axis (with the same size).

  For example, assuming 8 XLA devices are available, :py:func:`pmap` can be used
  as a map along a leading array axis:

  >>> import jax.numpy as jnp
  >>>
  >>> out = pmap(lambda x: x ** 2)(jnp.arange(8))  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [0, 1, 4, 9, 16, 25, 36, 49]

  When the leading dimension is smaller than the number of available devices JAX
  will simply run on a subset of devices:

  >>> x = jnp.arange(3 * 2 * 2.).reshape((3, 2, 2))
  >>> y = jnp.arange(3 * 2 * 2.).reshape((3, 2, 2)) ** 2
  >>> out = pmap(jnp.dot)(x, y)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [[[    4.     9.]
    [   12.    29.]]
   [[  244.   345.]
    [  348.   493.]]
   [[ 1412.  1737.]
    [ 1740.  2141.]]]

  If your leading dimension is larger than the number of available devices you
  will get an error:

  >>> pmap(lambda x: x ** 2)(jnp.arange(9))  # doctest: +SKIP
  ValueError: ... requires 9 replicas, but only 8 XLA devices are available

  As with :py:func:`vmap`, using ``None`` in ``in_axes`` indicates that an
  argument doesn't have an extra axis and should be broadcasted, rather than
  mapped, across the replicas:

  >>> x, y = jnp.arange(2.), 4.
  >>> out = pmap(lambda x, y: (x + y, y * 2.), in_axes=(0, None))(x, y)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  ([4., 5.], [8., 8.])

  Note that :py:func:`pmap` always returns values mapped over their leading axis,
  equivalent to using ``out_axes=0`` in :py:func:`vmap`.

  In addition to expressing pure maps, :py:func:`pmap` can also be used to express
  parallel single-program multiple-data (SPMD) programs that communicate via
  collective operations. For example:

  >>> f = lambda x: x / jax.lax.psum(x, axis_name='i')
  >>> out = pmap(f, axis_name='i')(jnp.arange(4.))  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [ 0.          0.16666667  0.33333334  0.5       ]
  >>> print(out.sum())  # doctest: +SKIP
  1.0

  In this example, ``axis_name`` is a string, but it can be any Python object
  with ``__hash__`` and ``__eq__`` defined.

  The argument ``axis_name`` to :py:func:`pmap` names the mapped axis so that
  collective operations, like :func:`jax.lax.psum`, can refer to it. Axis names
  are important particularly in the case of nested :py:func:`pmap` functions,
  where collective operations can operate over distinct axes:

  >>> from functools import partial
  >>> import jax
  >>>
  >>> @partial(pmap, axis_name='rows')
  ... @partial(pmap, axis_name='cols')
  ... def normalize(x):
  ...   row_normed = x / jax.lax.psum(x, 'rows')
  ...   col_normed = x / jax.lax.psum(x, 'cols')
  ...   doubly_normed = x / jax.lax.psum(x, ('rows', 'cols'))
  ...   return row_normed, col_normed, doubly_normed
  >>>
  >>> x = jnp.arange(8.).reshape((4, 2))
  >>> row_normed, col_normed, doubly_normed = normalize(x)  # doctest: +SKIP
  >>> print(row_normed.sum(0))  # doctest: +SKIP
  [ 1.  1.]
  >>> print(col_normed.sum(1))  # doctest: +SKIP
  [ 1.  1.  1.  1.]
  >>> print(doubly_normed.sum((0, 1)))  # doctest: +SKIP
  1.0

  On multi-process platforms, collective operations operate over all devices,
  including those on other processes. For example, assuming the following code
  runs on two processes with 4 XLA devices each:

  >>> f = lambda x: x + jax.lax.psum(x, axis_name='i')
  >>> data = jnp.arange(4) if jax.process_index() == 0 else jnp.arange(4, 8)
  >>> out = pmap(f, axis_name='i')(data)  # doctest: +SKIP
  >>> print(out)  # doctest: +SKIP
  [28 29 30 31] # on process 0
  [32 33 34 35] # on process 1

  Each process passes in a different length-4 array, corresponding to its 4
  local devices, and the psum operates over all 8 values. Conceptually, the two
  length-4 arrays can be thought of as a sharded length-8 array (in this example
  equivalent to jnp.arange(8)) that is mapped over, with the length-8 mapped
  axis given name 'i'. The pmap call on each process then returns the
  corresponding length-4 output shard.

  The ``devices`` argument can be used to specify exactly which devices are used
  to run the parallel computation. For example, again assuming a single process
  with 8 devices, the following code defines two parallel computations, one
  which runs on the first six devices and one on the remaining two:

  >>> from functools import partial
  >>> @partial(pmap, axis_name='i', devices=jax.devices()[:6])
  ... def f1(x):
  ...   return x / jax.lax.psum(x, axis_name='i')
  >>>
  >>> @partial(pmap, axis_name='i', devices=jax.devices()[-2:])
  ... def f2(x):
  ...   return jax.lax.psum(x ** 2, axis_name='i')
  >>>
  >>> print(f1(jnp.arange(6.)))  # doctest: +SKIP
  [0.         0.06666667 0.13333333 0.2        0.26666667 0.33333333]
  >>> print(f2(jnp.array([2., 3.])))  # doctest: +SKIP
  [ 13.  13.]
  """
  if global_arg_shapes is not None:
    raise ValueError(
        "global_arg_shapes only worked with sharded_jit which has long been"
        " removed from JAX. Please migrate to pjit and remove global_arg_shapes"
        " from pmap.")

  # TODO(yashkatariya): Move this out after shard_map is out of experimental and
  # in _src
  if config.pmap_shmap_merge.value:
    from jax.experimental.shard_map import pmap
    return pmap(fun, axis_name, in_axes=in_axes, out_axes=out_axes,
                static_broadcasted_argnums=static_broadcasted_argnums,
                devices=devices, backend=backend,
                axis_size=axis_size,
                donate_argnums=donate_argnums)

  return _cpp_pmap(
      fun,
      axis_name,
      in_axes=in_axes,
      out_axes=out_axes,
      static_broadcasted_argnums=static_broadcasted_argnums,
      devices=devices,
      backend=backend,
      axis_size=axis_size,
      donate_argnums=donate_argnums)


class PmapCallInfo(NamedTuple):
  flat_fun: lu.WrappedFun
  in_tree: PyTreeDef
  out_tree: Callable[[], PyTreeDef]
  flat_args: Sequence[Any]
  donated_invars: Sequence[bool]
  in_axes_flat: Sequence[int | None]
  local_axis_size: int
  out_axes_thunk: Callable
  devices: Sequence[xc.Device] | None
  global_axis_size: int
  is_explicit_global_axis_size: bool


def _get_global_axis_size(local_axis_size: int, in_devices, backend_name: str,
                          global_axis_size: int | None):
  """Determine global_axis_size for multi-host pmap."""
  # TODO(mattjj,skyewm): revive this check (inner_pmap always False now)
  # if xb.process_count() > 1 and global_axis_size is None and inner_pmap:
  #   raise ValueError("'axis_size' must be specified for nested multi-host pmaps")
  if (xb.process_count() == 1 and global_axis_size is not None and
      global_axis_size != local_axis_size):
    raise ValueError(
        f"Specified axis_size {global_axis_size} doesn't match received "
        f"axis_size {local_axis_size}.")

  if in_devices is not None and backend_name is None:
    backend = xb.get_device_backend(in_devices[0])
  else:
    backend = xb.get_backend(backend_name)

  if global_axis_size is None:
    if xb.process_count(backend) == 1:
      global_axis_size = local_axis_size
    elif in_devices:
      global_axis_size = len(in_devices)
    else:
      global_axis_size = local_axis_size * xb.process_count(backend)
      assert all(
          len(xb.local_devices(pi, backend)) == xb.local_device_count(backend)
          for pi in range(xb.process_count(backend)))
  return global_axis_size


def _prepare_pmap(fun, in_axes, out_axes, static_broadcasted_tuple,
                  donate_tuple, in_devices, backend_name,
                  axis_size, args, kwargs):
  if in_devices is not None and len(in_devices) == 0:
    raise ValueError("'devices' argument to pmap must be non-empty, or None.")

  src = fun_sourceinfo(fun)
  signature = api_util.fun_signature(fun)

  dbg = debug_info('pmap', src, signature, args, kwargs,
                   static_broadcasted_tuple, ())

  f = lu.wrap_init(fun)
  if static_broadcasted_tuple:
    if max(static_broadcasted_tuple) >= len(args):
      raise ValueError(
          f"pmapped function has static_broadcasted_argnums={static_broadcasted_tuple}"
          f" but was called with only {len(args)} positional "
          f"argument{'s' if len(args) > 1 else ''}. "
          "All static broadcasted arguments must be passed positionally.")
    dyn_argnums = [i for i in range(len(args))
                   if i not in static_broadcasted_tuple]
    f, dyn_args = argnums_partial(f, dyn_argnums, args)

    if isinstance(in_axes, tuple):
      dyn_in_axes = tuple(in_axes[i] for i in dyn_argnums)
    else:
      dyn_in_axes = in_axes
  else:
    dyn_args, dyn_in_axes = args, in_axes
  args, in_tree = tree_flatten((dyn_args, kwargs))

  if donate_tuple and not config.debug_nans.value:
    donated_invars = donation_vector(donate_tuple, (), in_tree)
  else:
    donated_invars = (False,) * len(args)
  try:
    in_axes_flat = tuple(broadcast_prefix((dyn_in_axes, 0), (dyn_args, kwargs),
                                          is_leaf=lambda x: x is None))
  except ValueError:
    e, *_ = prefix_errors((dyn_in_axes, 0), (dyn_args, kwargs))
    ex = e('pmap in_axes')
    msg, = ex.args
    msg += ("\n\nThe 'full pytree' here is the tuple of arguments passed "
            "positionally to the pmapped function, and the value of `in_axes` "
            "must be a tree prefix of that tuple. But it was not a prefix.")
    if kwargs:
      msg += ("\n\nWhen some arguments are passed by keyword to the pmapped "
              "function, they are not included in the comparison to `in_axes`. "
              "Instead, each argument passed by keyword is mapped over its "
              "leading axis. See the description of `in_axes` in the `pmap` "
              "docstring: "
              "https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html#jax.pmap")
    msg += ("\n\nCheck that the value of the `in_axes` argument to `pmap` "
            "is a tree prefix of the tuple of arguments passed positionally to "
            "the pmapped function.")
    raise ValueError(msg) from None
  local_axis_size = _mapped_axis_size(fun, in_tree, args, in_axes_flat, "pmap")

  f, res_paths = result_paths(f)
  f, out_axes_thunk = flat_out_axes(f, out_axes)
  flat_fun, out_tree = flatten_fun(f, in_tree)
  flat_fun = debug_info_final(flat_fun, dbg, res_paths)

  is_explicit_global_axis_size = axis_size is not None
  global_axis_size = _get_global_axis_size(local_axis_size, in_devices,
                                           backend_name, axis_size)
  return PmapCallInfo(flat_fun=flat_fun,
                      in_tree=in_tree,
                      out_tree=out_tree,
                      flat_args=args,
                      donated_invars=donated_invars,
                      in_axes_flat=in_axes_flat,
                      local_axis_size=local_axis_size,
                      out_axes_thunk=out_axes_thunk,
                      devices=None if in_devices is None else tuple(in_devices),
                      global_axis_size=global_axis_size,
                      is_explicit_global_axis_size=is_explicit_global_axis_size)


def _shared_code_pmap(fun, axis_name, static_broadcasted_argnums,
                      donate_argnums, in_axes, out_axes):
  # axis_size is an optional integer representing the global axis size.  The
  # aggregate size (across all processes) size of the mapped axis must match the
  # given value.
  check_callable(fun)
  axis_name = core._TempAxisName(fun) if axis_name is None else axis_name
  static_broadcasted_tuple = _ensure_index_tuple(static_broadcasted_argnums)
  donate_tuple = rebase_donate_argnums(
      _ensure_index_tuple(donate_argnums), static_broadcasted_tuple)

  if not all(type(l) is int for l in tree_leaves(in_axes)):
    raise TypeError("pmap in_axes must be an int, None, or (nested) container "
                    f"with those types as leaves, but got {in_axes}.")
  if not all(type(l) is int for l in tree_leaves(out_axes)):
    raise TypeError("pmap out_axes must be an int, None, or (nested) container "
                    f"with those types as leaves, but got {out_axes}.")

  return axis_name, static_broadcasted_tuple, donate_tuple


class _PmapFastpathData(NamedTuple):
  version: int  # For forward and backward compatibility
  xla_executable: xc.LoadedExecutable
  in_handler: Any
  out_handler: Any
  out_pytree_def: Any
  # Data needed to handle the inputs.
  input_devices: Sequence[xc.Device]
  input_indices: Sequence[sharding_specs.Index]
  input_array_shardings: Sequence[Any]
  # Data needed to build the Array from C++.
  out_avals: Sequence[Any]
  out_array_shardings: Sequence[Any]
  out_committed: Sequence[Any]


def _cpp_pmap(
    fun: Callable,
    axis_name: AxisName | None = None,
    *,
    in_axes=0,
    out_axes=0,
    static_broadcasted_argnums: int | Iterable[int] = (),
    devices: Sequence[xc.Device] | None = None,  # noqa: F811
    backend: str | None = None,
    axis_size: int | None = None,
    donate_argnums: int | Iterable[int] = (),
  ) -> Any:
  axis_name, static_broadcasted_tuple, donate_tuple = _shared_code_pmap(
      fun, axis_name, static_broadcasted_argnums, donate_argnums, in_axes,
      out_axes)
  del static_broadcasted_argnums, donate_argnums

  @api_boundary
  def cache_miss(*args, **kwargs):
    p = _prepare_pmap(fun, in_axes, out_axes, static_broadcasted_tuple,
                      donate_tuple, devices, backend,
                      axis_size, args, kwargs)
    for arg in p.flat_args:
      dispatch.check_arg(arg)

    params = dict(
        backend=backend,
        axis_name=axis_name,
        axis_size=p.local_axis_size,
        global_axis_size=p.global_axis_size,
        devices=p.devices,
        in_axes=p.in_axes_flat,
        out_axes_thunk=p.out_axes_thunk,
        name=p.flat_fun.__name__,
        donated_invars=p.donated_invars,
        is_explicit_global_axis_size=p.is_explicit_global_axis_size,
    )

    execute: Callable | None = None
    with core.take_current_trace() as trace:
      if isinstance(trace, core.EvalTrace):
        execute = pxla.xla_pmap_impl_lazy(p.flat_fun, *p.flat_args, **params)
        out = execute(*p.flat_args)
      else:
        out = pxla.xla_pmap_p.bind_with_trace(trace, (p.flat_fun, *p.flat_args), params)

    out_tree, out_flat = p.out_tree, out
    out_pytree_def = out_tree()
    out = tree_unflatten(out_pytree_def, out_flat)

    ### Decide whether we can support the C++ fast path
    use_fastpath = False
    if execute is not None and isinstance(execute, pxla.ExecuteReplicated):
      execute_replicated = typing.cast(pxla.ExecuteReplicated, execute)
      use_fastpath = (
        # TODO(sharadmv): Enable effects in replicated computation
        not execute_replicated.has_unordered_effects
        and not execute_replicated.has_host_callbacks and
        # No tracers in the outputs.
        all(isinstance(x, xc.ArrayImpl) for x in out_flat))

    ### If we can use the fastpath, we return required info to the caller.
    if use_fastpath:
      execute_replicated = typing.cast(pxla.ExecuteReplicated, execute)
      out_handler = execute_replicated.out_handler
      in_handler = execute_replicated.in_handler

      out_array_shardings = [out.sharding for out in out_flat]
      out_committed = [out._committed for out in out_flat]
      fastpath_data = _PmapFastpathData(
          version=1,
          xla_executable=execute_replicated.xla_executable,
          in_handler=in_handler,
          out_handler=out_handler,
          out_pytree_def=out_pytree_def,
          input_devices=in_handler.local_devices,
          input_indices=in_handler.input_indices,
          input_array_shardings=in_handler.in_shardings,
          out_avals=out_handler.out_avals,
          out_array_shardings=out_array_shardings,
          out_committed=out_committed,
      )

    else:
      fastpath_data = None

    return out, fastpath_data

  cpp_mapped_f = pmap_lib.pmap(
      fun, cache_miss, static_broadcasted_tuple,
      lambda x, s: pxla.shard_args([s], [None], [None], [x])[0],
      pytree_registry=tree_util.default_registry)
  _pmap_cache_clears.add(cpp_mapped_f)

  pmap_f = wraps(fun)(cpp_mapped_f)

  @api_boundary
  def lower(*args, **kwargs):
    return trace(*args, **kwargs).lower()

  @api_boundary
  def trace(*args, **kwargs):
    p = _prepare_pmap(
        fun, in_axes, out_axes, static_broadcasted_tuple, donate_tuple,
        devices, backend, axis_size, args, kwargs)
    abstract_args = list(map(shaped_abstractify, p.flat_args))
    closed_jaxpr, xc_backend, replicas, shards, pci = pxla.get_pmap_jaxpr(
        p.flat_fun, backend, axis_name,
        axis_size=p.local_axis_size, global_axis_size=p.global_axis_size,
        devices=p.devices,
        name=p.flat_fun.__name__,
        in_axes=p.in_axes_flat,
        out_axes_thunk=p.out_axes_thunk,
        avals=abstract_args)
    lower_callable = partial(
        pxla.lower_parallel_callable, p.flat_fun, axis_name,
        axis_size=p.local_axis_size, global_axis_size=p.global_axis_size,
        devices=p.devices,
        name=p.flat_fun.__name__,
        in_axes=p.in_axes_flat,
        donated_invars=p.donated_invars,
        is_explicit_global_axis_size=p.is_explicit_global_axis_size,
        avals=abstract_args,
        closed_jaxpr=closed_jaxpr,
        backend=xc_backend,
        replicas=replicas,
        shards=shards,
        pci=pci)
    args_info = stages.make_args_info(p.in_tree, abstract_args, donate_tuple)
    return stages.Traced(closed_jaxpr, args_info, p.flat_fun.__name__,
                         p.out_tree(), lower_callable)

  pmap_f.lower = lower
  pmap_f.trace = trace

  return pmap_f

_pmap_cache_clears = weakref.WeakSet()  # type: ignore


def jvp(
    fun: Callable, primals, tangents, has_aux: bool = False
  ) -> tuple[Any, ...]:
  """Computes a (forward-mode) Jacobian-vector product of ``fun``.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard Python container of arrays or scalars.
    primals: The primal values at which the Jacobian of ``fun`` should be
      evaluated. Should be either a tuple or a list of arguments,
      and its length should be equal to the number of positional parameters of
      ``fun``.
    tangents: The tangent vector for which the Jacobian-vector product should be
      evaluated. Should be either a tuple or a list of tangents, with the same
      tree structure and array shapes as ``primals``.
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.

  Returns:
    If ``has_aux`` is ``False``, returns a ``(primals_out, tangents_out)`` pair,
    where ``primals_out`` is ``fun(*primals)``,
    and ``tangents_out`` is the Jacobian-vector product of
    ``function`` evaluated at ``primals`` with ``tangents``. The
    ``tangents_out`` value has the same Python tree structure and shapes as
    ``primals_out``. If ``has_aux`` is ``True``, returns a
    ``(primals_out, tangents_out, aux)`` tuple where ``aux``
    is the auxiliary data returned by ``fun``.

  For example:

  >>> import jax
  >>>
  >>> primals, tangents = jax.jvp(jax.numpy.sin, (0.1,), (0.2,))
  >>> print(primals)
  0.09983342
  >>> print(tangents)
  0.19900084
  """
  check_callable(fun)
  return _jvp(lu.wrap_init(fun), primals, tangents, has_aux=has_aux)

def _jvp(fun: lu.WrappedFun, primals, tangents, has_aux=False):
  """Variant of jvp() that takes an lu.WrappedFun."""
  if (not isinstance(primals, (tuple, list)) or
      not isinstance(tangents, (tuple, list))):
    raise TypeError("primal and tangent arguments to jax.jvp must be tuples or lists; "
                    f"found {type(primals).__name__} and {type(tangents).__name__}.")

  ps_flat, tree_def = tree_flatten(primals)
  ts_flat, tree_def_2 = tree_flatten(tangents)
  if tree_def != tree_def_2:
    raise TypeError("primal and tangent arguments to jax.jvp must have the same tree "
                    f"structure; primals have tree structure {tree_def} whereas tangents have "
                    f"tree structure {tree_def_2}.")
  for p, t in zip(ps_flat, ts_flat):
    if core.primal_dtype_to_tangent_dtype(_dtype(p)) != _dtype(t):
      raise TypeError("primal and tangent arguments to jax.jvp do not match; "
                      "dtypes must be equal, or in case of int/bool primal dtype "
                      "the tangent dtype must be float0."
                      f"Got primal dtype {_dtype(p)} and so expected tangent dtype "
                      f"{core.primal_dtype_to_tangent_dtype(_dtype(p))}, but got "
                      f"tangent dtype {_dtype(t)} instead.")
    if np.shape(p) != np.shape(t):
      raise ValueError("jvp called with different primal and tangent shapes;"
                       f"Got primal shape {np.shape(p)} and tangent shape as {np.shape(t)}")

  if not has_aux:
    flat_fun, out_tree = flatten_fun_nokwargs(fun, tree_def)
    out_primals, out_tangents = ad.jvp(flat_fun).call_wrapped(ps_flat, ts_flat)
    out_tree = out_tree()
    return (tree_unflatten(out_tree, out_primals),
            tree_unflatten(out_tree, out_tangents))
  else:
    flat_fun, out_aux_trees = flatten_fun_nokwargs2(fun, tree_def)
    jvp_fun, aux = ad.jvp(flat_fun, has_aux=True)
    out_primals, out_tangents = jvp_fun.call_wrapped(ps_flat, ts_flat)
    out_tree, aux_tree = out_aux_trees()
    return (tree_unflatten(out_tree, out_primals),
            tree_unflatten(out_tree, out_tangents),
            tree_unflatten(aux_tree, aux()))

@overload
def linearize(fun: Callable, *primals, has_aux: Literal[False] = False
              ) -> tuple[Any, Callable]:
  ...

@overload
def linearize(fun: Callable, *primals, has_aux: Literal[True]
              ) -> tuple[Any, Callable, Any]:
  ...

def linearize(fun: Callable, *primals, has_aux: bool = False
              ) -> tuple[Any, Callable] | tuple[Any, Callable, Any]:
  """Produces a linear approximation to ``fun`` using :py:func:`jvp` and partial eval.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard python container of arrays or scalars.
    primals: The primal values at which the Jacobian of ``fun`` should be
      evaluated. Should be a tuple of arrays, scalar, or standard Python
      container thereof. The length of the tuple is equal to the number of
      positional parameters of ``fun``.
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the first
      element is considered the output of the mathematical function to be linearized,
      and the second is auxiliary data. Default False.

  Returns:
    If ``has_aux`` is ``False``, returns a pair where the first element is the value of
    ``f(*primals)`` and the second element is a function that evaluates the
    (forward-mode) Jacobian-vector product of ``fun`` evaluated at ``primals`` without
    re-doing the linearization work. If ``has_aux`` is ``True``, returns a
    ``(primals_out, lin_fn, aux)`` tuple where ``aux`` is the auxiliary data returned by
    ``fun``.

  In terms of values computed, :py:func:`linearize` behaves much like a curried
  :py:func:`jvp`, where these two code blocks compute the same values::

    y, out_tangent = jax.jvp(f, (x,), (in_tangent,))

    y, f_jvp = jax.linearize(f, x)
    out_tangent = f_jvp(in_tangent)

  However, the difference is that :py:func:`linearize` uses partial evaluation
  so that the function ``f`` is not re-linearized on calls to ``f_jvp``. In
  general that means the memory usage scales with the size of the computation,
  much like in reverse-mode. (Indeed, :py:func:`linearize` has a similar
  signature to :py:func:`vjp`!)

  This function is mainly useful if you want to apply ``f_jvp`` multiple times,
  i.e. to evaluate a pushforward for many different input tangent vectors at the
  same linearization point. Moreover if all the input tangent vectors are known
  at once, it can be more efficient to vectorize using :py:func:`vmap`, as in::

    pushfwd = partial(jvp, f, (x,))
    y, out_tangents = vmap(pushfwd, out_axes=(None, 0))((in_tangents,))

  By using :py:func:`vmap` and :py:func:`jvp` together like this we avoid the stored-linearization
  memory cost that scales with the depth of the computation, which is incurred
  by both :py:func:`linearize` and :py:func:`vjp`.

  Here's a more complete example of using :py:func:`linearize`:

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> def f(x): return 3. * jnp.sin(x) + jnp.cos(x / 2.)
  ...
  >>> jax.jvp(f, (2.,), (3.,))
  (Array(3.2681944, dtype=float32, weak_type=True), Array(-5.007528, dtype=float32, weak_type=True))
  >>> y, f_jvp = jax.linearize(f, 2.)
  >>> print(y)
  3.2681944
  >>> print(f_jvp(3.))
  -5.007528
  >>> print(f_jvp(4.))
  -6.676704
  """
  check_callable(fun)
  f = lu.wrap_init(fun)
  primals_flat, in_tree = tree_flatten(primals)
  if has_aux:
    jaxtree_fun, out_tree = flatten_fun_nokwargs2(f, in_tree)
  else:
    jaxtree_fun, out_tree = flatten_fun_nokwargs(f, in_tree)
  out_primals, out_pvals, jaxpr, consts, *maybe_aux = ad.linearize(
      jaxtree_fun, *primals_flat, has_aux=has_aux)
  if has_aux:
    out_tree, aux_tree = out_tree()
  else:
    out_tree = out_tree()
  out_primal_py = tree_unflatten(out_tree, out_primals)
  primal_avals = list(map(core.get_aval, primals_flat))
  # Ensure that lifted_jvp is a PyTree
  lifted_jvp = Partial(partial(_lift_linearized, jaxpr, primal_avals,
                               (in_tree, out_tree), out_pvals), consts)
  if has_aux:
    [aux] = maybe_aux
    return out_primal_py, lifted_jvp, tree_unflatten(aux_tree, aux)
  else:
    [] = maybe_aux
    return out_primal_py, lifted_jvp

def _lift_linearized(jaxpr, primal_avals, io_tree, out_pvals, consts, *py_args):
  def fun(*tangents):
    tangent_avals = list(map(core.get_aval, tangents))
    for primal_aval, tangent_aval in zip(primal_avals, tangent_avals):
      expected_tangent_aval  = primal_aval.to_tangent_aval()
      if not core.typecompat(expected_tangent_aval, tangent_aval):
        raise ValueError("linearized function called on tangent values inconsistent with "
                         "the original primal values: "
                         f"got tangent aval {tangent_aval} for primal aval {primal_aval} "
                         f"but expected {expected_tangent_aval}")
    tangents_out = eval_jaxpr(jaxpr, consts, *tangents)
    tangents_out_ = iter(tangents_out)
    full_out = [pval.get_known() if pval.is_known() else next(tangents_out_)
                for pval in out_pvals]
    assert next(tangents_out_, None) is None
    return full_out

  return apply_flat_fun_nokwargs(fun, io_tree, py_args)

def _vjp_pullback_wrapper(name, out_primal_avals, io_tree, fun, *py_args_):
  if len(py_args_) != 1:
    msg = (f"The function returned by `jax.vjp` applied to {name} was called "
           f"with {len(py_args_)} arguments, but functions returned by "
           "`jax.vjp` must be called with a single argument corresponding to "
           f"the single value returned by {name} (even if that returned "
           "value is a tuple or other container).\n"
           "\n"
           "For example, if we have:\n"
           "\n"
           "  def f(x):\n"
           "    return (x, x)\n"
           "  _, f_vjp = jax.vjp(f, 1.0)\n"
           "\n"
           "the function `f` returns a single tuple as output, and so we call "
           "`f_vjp` with a single tuple as its argument:\n"
           "\n"
           "  x_bar, = f_vjp((2.0, 2.0))\n"
           "\n"
           "If we instead call `f_vjp(2.0, 2.0)`, with the values 'splatted "
           "out' as arguments rather than in a tuple, this error can arise.")
    raise TypeError(msg)
  py_args, = py_args_
  in_tree_expected, out_tree = io_tree
  args, in_tree = tree_flatten(py_args)
  if in_tree != in_tree_expected:
    raise ValueError(f"unexpected tree structure of argument to vjp function: "
                     f"got {in_tree}, but expected to match {in_tree_expected}")
  for arg, aval in zip(args, out_primal_avals):
    ct_aval = shaped_abstractify(arg)
    ct_aval_expected = aval.to_tangent_aval()
    if (not core.typecompat(ct_aval, ct_aval_expected) and
        not _temporary_dtype_exception(ct_aval, ct_aval_expected)):
      raise ValueError(
          "unexpected JAX type (e.g. shape/dtype) for argument to vjp function: "
          f"got {ct_aval.str_short()}, but expected {ct_aval_expected.str_short()} "
          f"because the corresponding output of the function {name} had JAX type "
          f"{aval.str_short()}")
  ans = fun(*args)
  return tree_unflatten(out_tree, ans)

# TODO(mattjj): see similar function in custom_derivatives.py
def _temporary_dtype_exception(a, a_) -> bool:
  if isinstance(a, core.ShapedArray) and isinstance(a_, core.ShapedArray):
    return a.shape == a_.shape and a_.dtype == float0
  return False

@overload
def vjp(fun: Callable[..., T],
        *primals: Any,
        has_aux: Literal[False] = False,
        reduce_axes: Sequence[AxisName] = ()) -> tuple[T, Callable]:
  ...

@overload
def vjp(fun: Callable[..., tuple[T, U]], *primals: Any,
        has_aux: Literal[True],
        reduce_axes: Sequence[AxisName] = ()) -> tuple[T, Callable, U]:
  ...
def vjp(
    fun: Callable, *primals, has_aux: bool = False, reduce_axes=()
  ) -> tuple[Any, Callable] | tuple[Any, Callable, Any]:
  """Compute a (reverse-mode) vector-Jacobian product of ``fun``.

  :py:func:`grad` is implemented as a special case of :py:func:`vjp`.

  Args:
    fun: Function to be differentiated. Its arguments should be arrays, scalars,
      or standard Python containers of arrays or scalars. It should return an
      array, scalar, or standard Python container of arrays or scalars.
    primals: A sequence of primal values at which the Jacobian of ``fun``
      should be evaluated. The number of ``primals`` should be equal to the
      number of positional parameters of ``fun``. Each primal value should be
      an array, a scalar, or a pytree (standard Python containers) thereof.
    has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.

  Returns:
    If ``has_aux`` is ``False``, returns a ``(primals_out, vjpfun)`` pair, where
    ``primals_out`` is ``fun(*primals)``. If ``has_aux`` is ``True``, returns a
    ``(primals_out, vjpfun, aux)`` tuple where ``aux`` is the auxiliary data
    returned by ``fun``.

    ``vjpfun`` is a function from a cotangent vector with the same shape as
    ``primals_out`` to a tuple of cotangent vectors with the same number and
    shapes as ``primals``, representing the vector-Jacobian product of ``fun``
    evaluated at ``primals``.

  >>> import jax
  >>>
  >>> def f(x, y):
  ...   return jax.numpy.sin(x), jax.numpy.cos(y)
  ...
  >>> primals, f_vjp = jax.vjp(f, 0.5, 1.0)
  >>> xbar, ybar = f_vjp((-0.7, 0.3))
  >>> print(xbar)
  -0.61430776
  >>> print(ybar)
  -0.2524413
  """
  if reduce_axes:
    raise NotImplementedError("reduce_axes argument to vjp is deprecated")
  del reduce_axes
  check_callable(fun)
  return _vjp(
      lu.wrap_init(fun), *primals, has_aux=has_aux)

def _vjp(fun: lu.WrappedFun, *primals, has_aux=False):
  """Variant of vjp() that takes an lu.WrappedFun."""
  primals_flat, in_tree = tree_flatten(primals)
  for arg in primals_flat: dispatch.check_arg(arg)
  if not has_aux:
    flat_fun, out_tree = flatten_fun_nokwargs(fun, in_tree)
    out_primals, vjp = ad.vjp(flat_fun, primals_flat)
    out_tree = out_tree()
  else:
    flat_fun, out_aux_trees = flatten_fun_nokwargs2(fun, in_tree)
    out_primals, vjp, aux = ad.vjp(flat_fun, primals_flat, has_aux=True)
    out_tree, aux_tree = out_aux_trees()
  out_primal_avals = map(shaped_abstractify, out_primals)
  out_primal_py = tree_unflatten(out_tree, out_primals)
  vjp_py = Partial(partial(_vjp_pullback_wrapper, fun.__name__,
                           out_primal_avals, (out_tree, in_tree)), vjp)
  if not has_aux:
    return out_primal_py, vjp_py
  else:
    return out_primal_py, vjp_py, tree_unflatten(aux_tree, aux)


def linear_transpose(fun: Callable, *primals, reduce_axes=()) -> Callable:
  """Transpose a function that is promised to be linear.

  For linear functions, this transformation is equivalent to :py:func:`vjp`, but
  avoids the overhead of computing the forward pass.

  The outputs of the transposed function will always have the exact same dtypes
  as ``primals``, even if some values are truncated (e.g., from complex to
  float, or from float64 to float32). To avoid truncation, use dtypes in
  ``primals`` that match the full range of desired outputs from the transposed
  function. Integer dtypes are not supported.

  Args:
    fun: the linear function to be transposed.
    *primals: a positional argument tuple of arrays, scalars, or (nested)
      standard Python containers (tuples, lists, dicts, namedtuples, i.e.,
      pytrees) of those types used for evaluating the shape/dtype of
      ``fun(*primals)``. These arguments may be real scalars/ndarrays, but that
      is not required: only the ``shape`` and ``dtype`` attributes are accessed.
      See below for an example. (Note that the duck-typed objects cannot be
      namedtuples because those are treated as standard Python containers.)

  Returns:
    A callable that calculates the transpose of ``fun``. Valid input into this
    function must have the same shape/dtypes/structure as the result of
    ``fun(*primals)``. Output will be a tuple, with the same
    shape/dtypes/structure as ``primals``.

  >>> import jax
  >>> import types
  >>>
  >>> f = lambda x, y: 0.5 * x - 0.5 * y
  >>> scalar = types.SimpleNamespace(shape=(), dtype=np.dtype(np.float32))
  >>> f_transpose = jax.linear_transpose(f, scalar, scalar)
  >>> f_transpose(1.0)
  (Array(0.5, dtype=float32), Array(-0.5, dtype=float32))
  """
  if reduce_axes:
    raise NotImplementedError("reduce_axes argument to transpose is deprecated")
  del reduce_axes
  primals_flat, in_tree = tree_flatten(primals)
  flat_fun, out_tree = flatten_fun_nokwargs(lu.wrap_init(fun), in_tree)
  in_avals = map(shaped_abstractify, primals_flat)
  in_dtypes = map(dtypes.dtype, in_avals)

  in_pvals = map(pe.PartialVal.unknown, in_avals)
  jaxpr, out_pvals, const = pe.trace_to_jaxpr_nounits(flat_fun, in_pvals,
                                                      instantiate=True)
  jaxpr, _ = pe.dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars), True)
  out_avals, _ = unzip2(out_pvals)
  out_dtypes = map(dtypes.dtype, out_avals)
  if not (all(dtypes.issubdtype(d, np.inexact) for d in in_dtypes + out_dtypes)
          or all(dtypes.issubdtype(d, np.integer)
                 for d in in_dtypes + out_dtypes)):
    raise TypeError("linear_transpose only supports [float or complex] -> "
                    "[float or complex], and integer -> integer functions, "
                    f"but got {in_dtypes} -> {out_dtypes}.")

  @api_boundary
  def transposed_fun(const, out_cotangent):
    out_cts, out_tree2 = tree_flatten(out_cotangent)
    if out_tree() != out_tree2:
      raise TypeError("cotangent tree does not match function output, "
                      f"expected {out_tree()} but got {out_tree2}")
    if not all(map(core.typecheck, out_avals, out_cts)):
      raise TypeError("cotangent type does not match function output, "
                      f"expected {out_avals} but got {out_cts}")
    dummies = [ad.UndefinedPrimal(a) for a in in_avals]
    in_cts = ad.backward_pass(jaxpr, True, const, dummies, out_cts)
    in_cts = map(ad.instantiate_zeros, in_cts)
    return tree_unflatten(in_tree, in_cts)

  # Ensure that transposed_fun is a PyTree
  return Partial(transposed_fun, const)


def _flat_axes_specs(abstracted_axes, *args, **kwargs
                     ) -> list[pe.AbstractedAxesSpec]:
  if kwargs: raise NotImplementedError
  def ax_leaf(l):
    return (isinstance(l, dict) and all_leaves(l.values()) or
            isinstance(l, tuple) and all_leaves(l, lambda x: x is None))
  return broadcast_prefix(abstracted_axes, args, ax_leaf)


@overload
def make_jaxpr(
    fun: Callable,
    static_argnums: int | Iterable[int] = (),
    axis_env: Sequence[tuple[AxisName, int]] | None = None,
    return_shape: Literal[False] = ...,
    abstracted_axes: Any | None = None,
) -> Callable[..., core.ClosedJaxpr]:
  ...

@overload
def make_jaxpr(
    fun: Callable,
    static_argnums: int | Iterable[int] = (),
    axis_env: Sequence[tuple[AxisName, int]] | None = None,
    return_shape: Literal[True] = ...,
    abstracted_axes: Any | None = None,
) -> Callable[..., tuple[core.ClosedJaxpr, Any]]:
  ...

def make_jaxpr(
    fun: Callable,
    static_argnums: int | Iterable[int] = (),
    axis_env: Sequence[tuple[AxisName, int]] | None = None,
    return_shape: bool = False,
    abstracted_axes: Any | None = None,
) -> Callable[..., core.ClosedJaxpr | tuple[core.ClosedJaxpr, Any]]:
  """Creates a function that produces its jaxpr given example args.

  Args:
    fun: The function whose ``jaxpr`` is to be computed. Its positional
      arguments and return value should be arrays, scalars, or standard Python
      containers (tuple/list/dict) thereof.
    static_argnums: See the :py:func:`jax.jit` docstring.
    axis_env: Optional, a sequence of pairs where the first element is an axis
      name and the second element is a positive integer representing the size of
      the mapped axis with that name. This parameter is useful when lowering
      functions that involve parallel communication collectives, and it
      specifies the axis name/size environment that would be set up by
      applications of :py:func:`jax.pmap`.
    return_shape: Optional boolean, defaults to ``False``. If ``True``, the
      wrapped function returns a pair where the first element is the
      ``ClosedJaxpr`` representation of ``fun`` and the second element is a
      pytree with the same structure as the output of ``fun`` and where the
      leaves are objects with ``shape`` and ``dtype`` attributes representing
      the corresponding types of the output leaves.

  Returns:
    A wrapped version of ``fun`` that when applied to example arguments returns
    a ``ClosedJaxpr`` representation of ``fun`` on those arguments. If the
    argument ``return_shape`` is ``True``, then the returned function instead
    returns a pair where the first element is the ``ClosedJaxpr``
    representation of ``fun`` and the second element is a pytree representing
    the structure, shape, dtypes, and named shapes of the output of ``fun``.

  A ``jaxpr`` is JAX's intermediate representation for program traces. The
  ``jaxpr`` language is based on the simply-typed first-order lambda calculus
  with let-bindings. :py:func:`make_jaxpr` adapts a function to return its
  ``jaxpr``, which we can inspect to understand what JAX is doing internally.
  The ``jaxpr`` returned is a trace of ``fun`` abstracted to
  :py:class:`ShapedArray` level. Other levels of abstraction exist internally.

  We do not describe the semantics of the ``jaxpr`` language in detail here, but
  instead give a few examples.

  >>> import jax
  >>>
  >>> def f(x): return jax.numpy.sin(jax.numpy.cos(x))
  >>> print(f(3.0))
  -0.83602
  >>> jax.make_jaxpr(f)(3.0)
  { lambda ; a:f32[]. let b:f32[] = cos a; c:f32[] = sin b in (c,) }
  >>> jax.make_jaxpr(jax.grad(f))(3.0)
  { lambda ; a:f32[]. let
      b:f32[] = cos a
      c:f32[] = sin a
      _:f32[] = sin b
      d:f32[] = cos b
      e:f32[] = mul 1.0 d
      f:f32[] = neg e
      g:f32[] = mul f c
    in (g,) }
  """
  try:
    hash(fun)
    weakref.ref(fun)
  except TypeError:
    fun = partial(fun)

  @wraps(fun)
  @api_boundary
  def make_jaxpr_f(*args, **kwargs):
    with core.extend_axis_env_nd(axis_env or []):
      traced = jit(fun, static_argnums=static_argnums,
                   abstracted_axes=abstracted_axes).trace(*args, **kwargs)
    # `jit` converts tracers in consts to args but that breaks the semantics of
    # `make_jaxpr`. Hence convert the tracers in args back to consts in jaxpr.
    if traced._num_consts:
      consts, _ = split_list(traced._args_flat, [traced._num_consts])
      jaxpr_ = pe.convert_invars_to_constvars(traced.jaxpr.jaxpr,
                                              traced._num_consts)
      jaxpr = core.ClosedJaxpr(jaxpr_, consts)
    else:
      jaxpr = traced.jaxpr
    if return_shape:
      out = [ShapeDtypeStruct(o.shape, o.dtype) for o in jaxpr.out_avals]
      return jaxpr, tree_unflatten(tree_structure(traced.out_info), out)
    return jaxpr

  make_jaxpr_f.__module__ = "jax"
  if hasattr(fun, "__qualname__"):
    make_jaxpr_f.__qualname__ = f"make_jaxpr({fun.__qualname__})"
  if hasattr(fun, "__name__"):
    make_jaxpr_f.__name__ = f"make_jaxpr({fun.__name__})"
  return make_jaxpr_f

def _infer_src_sharding(src, x) -> Sharding | None:
  if src is not None:
    return src  # pytype: disable=bad-return-type
  if isinstance(x, array.ArrayImpl):
    return x.sharding
  if isinstance(x, core.Tracer):
    val = x.to_concrete_value()
    if val is not None and isinstance(val, array.ArrayImpl):
      return val.sharding
  return None


# TODO(yashkatariya): Generalize check_compatible_aval (maybe renamed) and use
# that to check if shardings are compatible with the input.
@lru_cache(maxsize=2048)
def _check_sharding(aval, s):
  if (s is not None and
      not isinstance(s, (xc.Device, Sharding, Layout, TransferToMemoryKind))):
    raise ValueError(
        "`jax.device_put` only accepts `None`, `jax.sharding.Sharding`,"
        " `jax.Device`, `Layout` or a pytree of these values. Received"
        f" invalid value: {s}")
  if isinstance(s, Sharding):
    if isinstance(aval, core.AbstractToken):
      aval = core.token_shaped_array
    if not isinstance(s, PmapSharding):
      pjit.pjit_check_aval_sharding(
          (s,), (aval,), None, "device_put args", allow_uneven_sharding=False)
    s.shard_shape(aval.shape)  # should raise an Error if incompatible


def device_put(
    x,
    device: None | xc.Device | Sharding | Layout | Any | TransferToMemoryKind = None,
    *, src: None | xc.Device | Sharding | Layout | Any | TransferToMemoryKind = None,
    donate: bool | Any = False, may_alias: bool | None | Any = None):
  """Transfers ``x`` to ``device``.

  Args:
    x: An array, scalar, or (nested) standard Python container thereof.
    device: The (optional) :py:class:`Device`, :py:class:`Sharding`, or a
      (nested) :py:class:`Sharding` in standard Python container (must be a tree
      prefix of ``x``), representing the device(s) to which ``x`` should be
      transferred. If given, then the result is committed to the device(s).
    src: The (optional) :py:class:`Device`, :py:class:`Sharding`, or a (nested)
      :py:class:`Sharding` in standard Python container (must be a tree prefix
      of ``x``), representing the device(s) on which ``x`` belongs.
    donate: bool or a (nested) bool in standard Python container (must be a tree
      prefix of ``x``). If True, ``x`` can be overwritten and marked deleted in
      the caller. This is best effort. JAX will donate if possible, otherwise it
      won't. The input buffer (in the future) will always be deleted if donated.
    may_alias: bool or None or a (nested) bool in standard Python container
      (must be a tree prefix of ``x``). If False, `x` will be copied. If true,
      `x` may be aliased depending on the runtime's implementation.

  Returns:
    A copy of ``x`` that resides on ``device``.

  If the ``device`` parameter is ``None``, then this operation behaves like the
  identity function if the operand is on any device already, otherwise it
  transfers the data to the default device, uncommitted.

  For more details on data placement see the
  :ref:`FAQ on data placement <faq-data-placement>`.

  This function is always asynchronous, i.e. returns immediately without
  blocking the calling Python thread until any transfers are completed.
  """
  with config.explicit_device_put_scope():
    x_flat, treedef = tree_flatten(x)
    if (device is None or
        isinstance(device, (xc.Device, Sharding, TransferToMemoryKind))):
      device_flat = [device] * len(x_flat)
    else:
      device_flat = flatten_axes("device_put device", treedef, device)

    if (src is None or
        isinstance(src, (xc.Device, Sharding, TransferToMemoryKind))):
      src_flat = [_infer_src_sharding(src, xf) for xf in x_flat]
    else:
      src_flat = flatten_axes("device_put source", treedef, src)
      src_flat = list(map(_infer_src_sharding, src_flat, x_flat))

    if isinstance(donate, bool):
      donate_flat = [donate] * len(x_flat)
    else:
      donate_flat = flatten_axes("device_put donate", treedef, donate)

    if isinstance(may_alias, bool):
      may_alias_flat = [may_alias] * len(x_flat)
    else:
      may_alias_flat = flatten_axes("device_put may_alias", treedef, may_alias)

    copy_semantics = []
    for m, d in zip(may_alias_flat, donate_flat):
      if m and d:
        raise ValueError('may_alias and donate cannot be True at the same time.')
      if m is None:
        m = not d
      if m and not d:
        copy_semantics.append(dispatch.CopySemantics.ALIAS)
      elif not m and d:
        copy_semantics.append(dispatch.CopySemantics.DONATE)
      else:
        assert not m and not d
        copy_semantics.append(dispatch.CopySemantics.COPY)

    for xf, d in zip(x_flat, device_flat):  # type: ignore
      _check_sharding(shaped_abstractify(xf), d)
    out_flat = dispatch.device_put_p.bind(
        *x_flat, devices=device_flat, srcs=src_flat,
        copy_semantics=copy_semantics)
    return tree_unflatten(treedef, out_flat)


def device_put_sharded(shards: Sequence[Any], devices: Sequence[xc.Device]):  # noqa: F811
  """Transfer array shards to specified devices and form Array(s).

  Args:
    shards: A sequence of arrays, scalars, or (nested) standard Python
      containers thereof representing the shards to be stacked together to form
      the output. The length of ``shards`` must equal the length of ``devices``.
    devices: A sequence of :py:class:`Device` instances representing the devices
      to which corresponding shards in ``shards`` will be transferred.

  This function is always asynchronous, i.e. returns immediately.

  Returns:
    A Array or (nested) Python container thereof representing the
    elements of ``shards`` stacked together, with each shard backed by physical
    device memory specified by the corresponding entry in ``devices``.

  Examples:
    Passing a list of arrays for ``shards`` results in a sharded array
    containing a stacked version of the inputs:

    >>> import jax
    >>> devices = jax.local_devices()
    >>> x = [jax.numpy.ones(5) for device in devices]
    >>> y = jax.device_put_sharded(x, devices)
    >>> np.allclose(y, jax.numpy.stack(x))
    True

    Passing a list of nested container objects with arrays at the leaves for
    ``shards`` corresponds to stacking the shards at each leaf. This requires
    all entries in the list to have the same tree structure:

    >>> x = [(i, jax.numpy.arange(i, i + 4)) for i in range(len(devices))]
    >>> y = jax.device_put_sharded(x, devices)
    >>> type(y)
    <class 'tuple'>
    >>> y0 = jax.device_put_sharded([a for a, b in x], devices)
    >>> y1 = jax.device_put_sharded([b for a, b in x], devices)
    >>> np.allclose(y[0], y0)
    True
    >>> np.allclose(y[1], y1)
    True

  See Also:
    - device_put
    - device_put_replicated
  """
  # TODO(jakevdp): provide a default for devices that considers both local
  # devices and pods
  if not isinstance(shards, Sequence):
    raise TypeError("device_put_sharded `shards` input must be a sequence; "
                     f"got {type(shards)}")
  if len(shards) != len(devices):
    raise ValueError(f"len(shards) = {len(shards)} must equal "
                     f"len(devices) = {len(devices)}.")

  def _device_put_sharded(*xs):
    avals = [core.get_aval(x) for x in xs]
    if not all(a1 == a2 for a1, a2 in zip(avals[:-1], avals[1:])):
      a1, a2 = next((a1, a2) for a1, a2 in zip(avals[:-1], avals[1:])
                    if a1 != a2)
      raise ValueError("the shards passed to device_put_sharded must have "
                       f"consistent shape and dtype, but got {a1} and {a2}.")
    stacked_aval = avals[0].update(shape=(len(devices),) + avals[0].shape)
    sharding_spec = sharding_specs.create_pmap_sharding_spec(stacked_aval.shape)
    sharding = PmapSharding(np.array(devices), sharding_spec)
    if dtypes.issubdtype(stacked_aval.dtype, dtypes.extended):
      return stacked_aval.dtype._rules.device_put_sharded(xs, stacked_aval, sharding, devices)
    if config.pmap_no_rank_reduction.value:
      ys = []
      for x in xs:
        if not isinstance(x, (np.ndarray, basearray.Array)):
          x = np.asarray(x)
        ys.append(x[None])
    else:
      ys = xs
    return pxla.batched_device_put(stacked_aval, sharding, ys, list(devices))


  with config.explicit_device_put_scope():
    return tree_map(_device_put_sharded, *shards)


def device_put_replicated(x: Any, devices: Sequence[xc.Device]):  # noqa: F811
  """Transfer array(s) to each specified device and form Array(s).

  Args:
    x: an array, scalar, or (nested) standard Python container thereof
      representing the array to be replicated to form the output.
    devices: A sequence of :py:class:`Device` instances representing the devices
      to which ``x`` will be transferred.

  This function is always asynchronous, i.e. returns immediately.

  Returns:
    An Array or (nested) Python container thereof representing the
    value of ``x`` broadcasted along a new leading axis of size
    ``len(devices)``, with each slice along that new leading axis backed by
    memory on the device specified by the corresponding entry in ``devices``.

  Examples:
    Passing an array:

    >>> import jax
    >>> devices = jax.local_devices()
    >>> x = jax.numpy.array([1., 2., 3.])
    >>> y = jax.device_put_replicated(x, devices)
    >>> np.allclose(y, jax.numpy.stack([x for _ in devices]))
    True

  See Also:
    - device_put
    - device_put_sharded
  """
  if not isinstance(devices, Sequence) or not devices:
    raise ValueError("`devices` argument to `device_put_replicated must be "
                     "a non-empty sequence.")
  def _device_put_replicated(x):
    aval = core.unmapped_aval(len(devices), core.no_axis_name, 0,
                              core.get_aval(x))
    assert isinstance(aval, ShapedArray)
    sharding_spec = sharding_specs.create_pmap_sharding_spec(aval.shape)
    if config.pmap_no_rank_reduction.value:
      if isinstance(x, (np.ndarray, basearray.Array)):
        buf = device_put(x[None], devices[0])
      else:
        buf = device_put(x, devices[0])[None]
    else:
      buf = device_put(x, devices[0])
    sharding = PmapSharding(np.array(devices), sharding_spec)
    if dtypes.issubdtype(aval.dtype, dtypes.extended):
      return aval.dtype._rules.device_put_replicated(buf, aval, sharding, devices)
    assert len(xla.aval_to_xla_shapes(aval)) == 1
    return pxla.batched_device_put(aval, sharding, [buf] * len(devices), devices)

  with config.explicit_device_put_scope():
    return tree_map(_device_put_replicated, x)


# TODO(mattjj): consider revising
def _device_get(x):
  if isinstance(x, core.Tracer):
    return x

  # Extended dtypes dispatch via their device_get rule.
  if isinstance(x, basearray.Array) and dtypes.issubdtype(x.dtype, dtypes.extended):
    bufs, tree = tree_util.dispatch_registry.flatten(x)
    return tree.unflatten(device_get(bufs))

  # Other types dispatch via their __array__ method.
  try:
    toarray = x.__array__
  except AttributeError:
    return x
  else:
    return toarray()

def device_get(x: Any):
  """Transfer ``x`` to host.

  If ``x`` is a pytree, then the individual buffers are copied in parallel.

  Args:
    x: An array, scalar, Array or (nested) standard Python container thereof
      representing the array to be transferred to host.

  Returns:
    An array or (nested) Python container thereof representing the
    value of ``x``.

  Examples:
    Passing a Array:

    >>> import jax
    >>> x = jax.numpy.array([1., 2., 3.])
    >>> jax.device_get(x)
    array([1., 2., 3.], dtype=float32)

    Passing a scalar (has no effect):

    >>> jax.device_get(1)
    1

  See Also:
    - device_put
    - device_put_sharded
    - device_put_replicated
  """
  with config.explicit_device_get_scope():
    for y in tree_leaves(x):
      try:
        y.copy_to_host_async()
      except AttributeError:
        pass
    return tree_map(_device_get, x)


class ShapeDtypeStruct:
  """A container for the shape, dtype, and other static attributes of an array.

  ``ShapeDtypeStruct`` is often used in conjunction with :func:`jax.eval_shape`.

  Args:
    shape: a sequence of integers representing an array shape
    dtype: a dtype-like object
    sharding: (optional) a :class:`jax.Sharding` object
  """
  __slots__ = ["shape", "dtype", "sharding", "_dll", "weak_type"]

  def __init__(self, shape, dtype, *, sharding=None, weak_type=False):
    self.shape = tuple(shape)
    if dtype is None:
      raise ValueError("ShapeDtypeStruct: dtype must be specified.")
    self.dtype = dtype if dtypes.issubdtype(dtype, dtypes.extended) else np.dtype(dtype)
    if sharding is not None and not isinstance(sharding, (Sharding, Layout)):
      raise ValueError(
          "sharding should be an instance of `jax.sharding.Sharding` or"
          f" `jax.experimental.layout.Layout`. Got {sharding} of type"
          f" {type(sharding)}.")
    if (isinstance(sharding, Layout) and
        isinstance(sharding.device_local_layout, AutoLayout)):
      raise TypeError(
          "`DeviceLocalLayout.AUTO` cannot be used in place of a device-local"
          f" layout in a `ShapeDtypeStruct`. Got {sharding}")
    self.sharding = sharding.sharding if isinstance(sharding, Layout) else sharding
    self._dll = sharding.device_local_layout if isinstance(sharding, Layout) else None
    self.weak_type = weak_type

  size = property(lambda self: math.prod(self.shape))
  ndim = property(lambda self: len(self.shape))

  @property
  def layout(self):
    return Layout(self._dll, self.sharding)

  def __len__(self):
    try:
      return self.shape[0]
    except IndexError as e:
      raise TypeError("len() of unsized object") from e  # same as numpy error

  def __repr__(self):
    sh = f", sharding={self.sharding}" if self.sharding is not None else ""
    l = f", layout={self.layout}" if self._dll is not None else ""
    wt = f", weak_type={self.weak_type}" if self.weak_type else ""
    return (f"{type(self).__name__}(shape={self.shape}, "
            f"dtype={self.dtype.name}{sh}{l}{wt})")

  __str__ = __repr__

  def __eq__(self, other):
    if not isinstance(other, ShapeDtypeStruct):
      return False
    else:
      return ((self.shape, self.dtype, self.sharding, self.layout, self.weak_type) ==
              (other.shape, other.dtype, other.sharding, other.layout, other.weak_type))

  def __hash__(self):
    # TODO(frostig): avoid the conversion from dict by addressing
    # https://github.com/jax-ml/jax/issues/8182
    return hash((self.shape, self.dtype, self.sharding, self.layout, self.weak_type))

def _sds_aval_mapping(x):
  aval = ShapedArray(
      x.shape, dtypes.canonicalize_dtype(x.dtype, allow_extended_dtype=True),
      weak_type=x.weak_type)
  if config.sharding_in_types.value and isinstance(x.sharding, NamedSharding):
    return aval.update(sharding=NamedSharding(
        x.sharding.mesh.abstract_mesh,
        x.sharding.spec._normalized_spec(x.ndim)))
  return aval
core.pytype_aval_mappings[ShapeDtypeStruct] = _sds_aval_mapping


@api_boundary
def eval_shape(fun: Callable, *args, **kwargs):
  """Compute the shape/dtype of ``fun`` without any FLOPs.

  This utility function is useful for performing shape inference. Its
  input/output behavior is defined by::

    def eval_shape(fun, *args, **kwargs):
      out = fun(*args, **kwargs)
      shape_dtype_struct = lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype)
      return jax.tree_util.tree_map(shape_dtype_struct, out)

  But instead of applying ``fun`` directly, which might be expensive, it uses
  JAX's abstract interpretation machinery to evaluate the shapes without doing
  any FLOPs.

  Using :py:func:`eval_shape` can also catch shape errors, and will raise same
  shape errors as evaluating ``fun(*args, **kwargs)``.

  Args:
    fun: The function whose output shape should be evaluated.
    *args: a positional argument tuple of arrays, scalars, or (nested) standard
      Python containers (tuples, lists, dicts, namedtuples, i.e. pytrees) of
      those types. Since only the ``shape`` and ``dtype`` attributes are
      accessed, one can use :class:`jax.ShapeDtypeStruct` or another container
      that duck-types as ndarrays (note however that duck-typed objects cannot
      be namedtuples because those are treated as standard Python containers).
    **kwargs: a keyword argument dict of arrays, scalars, or (nested) standard
      Python containers (pytrees) of those types. As in ``args``, array values
      need only be duck-typed to have ``shape`` and ``dtype`` attributes.

  Returns:
    out: a nested PyTree containing :class:`jax.ShapeDtypeStruct` objects as leaves.

  For example:

  >>> import jax
  >>> import jax.numpy as jnp
  >>>
  >>> f = lambda A, x: jnp.tanh(jnp.dot(A, x))
  >>> A = jax.ShapeDtypeStruct((2000, 3000), jnp.float32)
  >>> x = jax.ShapeDtypeStruct((3000, 1000), jnp.float32)
  >>> out = jax.eval_shape(f, A, x)  # no FLOPs performed
  >>> print(out.shape)
  (2000, 1000)
  >>> print(out.dtype)
  float32

  All arguments passed via :func:`eval_shape` will be treated as dynamic;
  static arguments can be included via closure, for example using :func:`functools.partial`:

  >>> import jax
  >>> from jax import lax
  >>> from functools import partial
  >>> import jax.numpy as jnp
  >>>
  >>> x = jax.ShapeDtypeStruct((1, 1, 28, 28), jnp.float32)
  >>> kernel = jax.ShapeDtypeStruct((32, 1, 3, 3), jnp.float32)
  >>>
  >>> conv_same = partial(lax.conv_general_dilated, window_strides=(1, 1), padding="SAME")
  >>> out = jax.eval_shape(conv_same, x, kernel)
  >>> print(out.shape)
  (1, 32, 28, 28)
  >>> print(out.dtype)
  float32
  """
  try: hash(fun)
  except TypeError: fun = partial(fun)
  return jit(fun).eval_shape(*args, **kwargs)


def named_call(
    fun: F,
    *,
    name: str | None = None,
) -> F:
  """Adds a user specified name to a function when staging out JAX computations.

  When staging out computations for just-in-time compilation to XLA (or other
  backends such as TensorFlow) JAX runs your Python program but by default does
  not preserve any of the function names or other metadata associated with it.
  This can make debugging the staged out (and/or compiled) representation of
  your program complicated because there is limited context information for each
  operation being executed.

  `named_call` tells JAX to stage the given function out as a subcomputation
  with a specific name. When the staged out program is compiled with XLA these
  named subcomputations are preserved and show up in debugging utilities like
  the TensorFlow Profiler in TensorBoard. Names are also preserved when staging
  out JAX programs to TensorFlow using :func:`experimental.jax2tf.convert`.

  Args:
    fun: Function to be wrapped. This can be any Callable.
    name: Optional. The prefix to use to name all sub computations created
      within the name scope. Use the fun.__name__ if not specified.

  Returns:
    A version of `fun` that is wrapped in a name_scope.
  """
  if name is None:
    name = fun.__name__

  return source_info_util.extend_name_stack(name)(fun)


@contextmanager
def named_scope(
    name: str,
  ) -> Generator[None, None, None]:
  """A context manager that adds a user specified name to the JAX name stack.

  When staging out computations for just-in-time compilation to XLA (or other
  backends such as TensorFlow) JAX does not, by default, preserve the names
  (or other source metadata) of Python functions it encounters.
  This can make debugging the staged out (and/or compiled) representation of
  your program complicated because there is limited context information for each
  operation being executed.

  ``named_scope`` tells JAX to stage the given function with additional
  annotations on the underlying operations. JAX internally keeps track of these
  annotations in a name stack. When the staged out program is compiled with XLA
  these annotations are preserved and show up in debugging utilities like the
  TensorFlow Profiler in TensorBoard. Names are also preserved when staging out
  JAX programs to TensorFlow using :func:`experimental.jax2tf.convert`.


  Args:
    name: The prefix to use to name all operations created within the name
      scope.
  Yields:
    Yields ``None``, but enters a context in which `name` will be appended to
    the active name stack.

  Examples:
    ``named_scope`` can be used as a context manager inside compiled functions:

    >>> import jax
    >>>
    >>> @jax.jit
    ... def layer(w, x):
    ...   with jax.named_scope("dot_product"):
    ...     logits = w.dot(x)
    ...   with jax.named_scope("activation"):
    ...     return jax.nn.relu(logits)

    It can also be used as a decorator:

    >>> @jax.jit
    ... @jax.named_scope("layer")
    ... def layer(w, x):
    ...   logits = w.dot(x)
    ...   return jax.nn.relu(logits)
  """
  if not isinstance(name, str):
    raise TypeError("named_scope name argument must be a string.")
  with source_info_util.extend_name_stack(name):
    yield

def effects_barrier():
  """Waits until existing functions have completed any side-effects."""
  dispatch.runtime_tokens.block_until_ready()

def block_until_ready(x):
  """
  Tries to call a ``block_until_ready`` method on pytree leaves.

  Args:
    x: a pytree, usually with at least some JAX array instances at its leaves.

  Returns:
    A pytree with the same structure and values of the input, where the values
    of all JAX array leaves are ready.
  """
  def try_to_block(x):
    try:
      return x.block_until_ready()
    except AttributeError:
      return x

  arrays = []
  for leaf in tree_leaves(x):
    if isinstance(leaf, array.ArrayImpl):
      arrays.append(leaf)
    else:
      try_to_block(leaf)

  if not arrays:
    # `arrays` will be empty if tree_leaves(x) is empty or all leaves are not
    # jax.Array.
    pass
  elif len(arrays) == 1:
    # Fast path for single array.
    try_to_block(arrays[0])
  else:
    # Optimized for multiple arrays.
    xc.batched_block_until_ready(arrays)

  return x

def clear_backends():
  """
  Clear all backend clients so that new backend clients can be created later.
  """
  xb._clear_backends()
  xb.local_devices.cache_clear()
  xb.process_count.cache_clear()
  dispatch.xla_primitive_callable.cache_clear()
  util.clear_all_caches()
  pjit._infer_params_cached.cache_clear()
  pjit._create_pjit_jaxpr.cache_clear()  # pytype: disable=attribute-error
  pjit._cpp_pjit_cache_fun_only.clear()
  pjit._cpp_pjit_cache_explicit_attributes.clear()
  xc._xla.PjitFunctionCache.clear_all()
  xc._xla.jax_jit.thread_local_state().extra_jit_context = None

@atexit.register
def clean_up():
  if xb._default_backend is not None:
    clear_backends()
  clear_caches()

  # Shut down distributed system if it exists. Otherwise, this is a no-op.
  distributed.shutdown()

def live_arrays(platform=None):
  """Return all live arrays in the backend for `platform`.

  If platform is None, it is the default backend.
  """
  return xb.get_backend(platform).live_arrays()

def clear_caches():
  """Clear all compilation and staging caches.

  This doesn't clear the persistent cache; to disable it (e.g. for benchmarks),
  set the jax_enable_compilation_cache config option to False.
  """
  # Clear all lu.cache, util.cache and util.weakref_lru_cache instances
  # (used for staging and Python-dispatch compiled executable caches).
  util.clear_all_caches()
  util.clear_all_weakref_lru_caches()

  # Clear all C++ compiled executable caches for pjit
  pjit._cpp_pjit_cache_fun_only.clear()
  pjit._cpp_pjit_cache_explicit_attributes.clear()
  pjit._infer_params_cached.cache_clear()
  xc._xla.PjitFunctionCache.clear_all()

  # Clear all C++ compiled executable caches for pmap
  for fun in _pmap_cache_clears:
    fun._cache_clear()

  # Clear particular util.cache instances.
  dispatch.xla_primitive_callable.cache_clear()
