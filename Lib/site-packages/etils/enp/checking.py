# Copyright 2024 The etils Authors.
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

"""Check util."""

from __future__ import annotations

import collections
import dataclasses
import functools
import inspect
import typing
from typing import Any, Callable, Optional, TypeVar

from etils import epy
from etils.enp import numpy_utils
from etils.enp import type_parsing
from etils.enp.array_types import typing as array_typing
import numpy as np

_TypeForm = Any
_Fn = TypeVar('_Fn')

# TODO(epot): Support:
# * return annotations
# * Tuple,...
# * Union
# * shape checking


@dataclasses.dataclass
class _ArrayParam:
  """Argument matching an array."""

  type: array_typing.ArrayAliasMeta
  is_optional: bool
  name: str

  def asarray(self, v, *, xnp: numpy_utils.NpModule):
    """Convert the value to array of the correct dtype."""
    try:
      return self.type.dtype.asarray(v, xnp=xnp, casting='none')
    except Exception as e:  # pylint: disable=broad-except
      epy.reraise(e, prefix=f'Invalid {self.name}: ')


@dataclasses.dataclass
class _FnSignatureCache:
  """Cache of the function signature."""

  sig: inspect.Signature
  has_xnp_kwargs: bool
  array_params: dict[str, _ArrayParam]


@typing.overload
def check_and_normalize_arrays(
    fn: None = ...,
    *,
    strict: bool = ...,
) -> Callable[[_Fn], _Fn]:
  ...


@typing.overload
def check_and_normalize_arrays(
    fn: _Fn = ...,
    *,
    strict: bool = ...,
) -> _Fn:
  ...


def check_and_normalize_arrays(fn=None, *, strict: bool = True):
  """Check and normalize arrays.

  This function:

  * Validate that the dtype/shape input arrays match the typing annotations
  * Normalize np, jnp, tf types to be consistent
  * Add an optional `xnp` argument to convert input arrays to np/jnp/tnp.

  See doc at: https://github.com/google/etils/blob/main/etils/array_types/README.md

  Example:

  ```python
  @enp.check_and_normalize_arrays(strict=False)
  def add(x: FloatArray[...], y: FloatArray[...]) -> y: FloatArray[...]:
    return x + y

  # Inside the function, `np` normalized to `jnp`
  add(np.array(1.), jnp.array(2.)) == jnp.array(3.)

  # strict=False, so `list` accepted and normalized to `xnp`
  add(jnp.array(1.), [1., 2., 3.]) == jnp.array([2., 3., 4.])
  ```

  Example with an explicit `xnp` argument, which automatically passed to the
  wrapped function:

  ```python
  @enp.check_and_normalize_arrays(strict=False)
  def clip(x: Array, *, xnp: enp.NpModule = ...) -> y: Array:
    return xnp.clip(x, 0.0, 1.0)
  ```

  If present in the function signature, the `xnp` argument must be a
  keyword-only argument.

  Args:
    fn: The function to decorate. Arguments will be automatically infered.
    strict: If `False`, `fn` will also accept list, int,... in which case those
      are automatically converted to `xnp`

  Returns:
    fn: The decorated function, with dynamic shape checking
  """

  if fn is None:
    return functools.partial(check_and_normalize_arrays, strict=strict)

  fn._array_types_state = None  # pylint: disable=protected-access

  @functools.wraps(fn)
  def decorated_fn(*args, **kwargs):
    try:
      kwargs = dict(kwargs)
      xnp = kwargs.pop('xnp', None)

      # First time the function is called, precompute & cache the info
      if fn._array_types_state is None:  # pylint: disable=protected-access
        fn._array_types_state = _parse_signature(fn)  # pylint: disable=protected-access

      state: _FnSignatureCache = fn._array_types_state  # pylint: disable=protected-access

      # In case `xnp` do not have default value
      if state.has_xnp_kwargs:
        kwargs['xnp'] = ...
      bound_args = state.sig.bind(*args, **kwargs)

      # Filter the non-array args
      # TODO(epot): Should raise an error for non-optional when v is None
      array_args = {
          k: v
          for k, v in bound_args.arguments.items()
          if k in state.array_params and v is not None
      }

      # Extract the xnp (either explicitly passed, or auto-infered)
      xnp = xnp or _get_xnp(array_args, strict=strict)
      _maybe_set_tnp_casting(xnp)

      # Normalize all arrays:
      # * Convert to xnp
      # * Check dtype
      array_args = {
          k: state.array_params[k].asarray(v, xnp=xnp)
          for k, v in array_args.items()
      }

      # TODO(epot): Check the shape

      # Update the arguments after normalization
      bound_args.arguments.update(array_args)

      # Eventually add `xnp` kwarg
      if state.has_xnp_kwargs:
        bound_args.arguments['xnp'] = xnp
    except Exception as e:  # pylint: disable=broad-except
      epy.reraise(
          e,
          prefix=(
              f'@enp.check_and_normalize_arrays error for {fn.__qualname__}: '
          ),
      )

    return fn(*bound_args.args, **bound_args.kwargs)

  return decorated_fn


def _get_xnp(
    array_args: dict[str, Any],
    *,
    strict: bool,
) -> numpy_utils.NpModule:
  """Extract the xnp module common to the args."""

  xnps = collections.defaultdict(list)
  for k, v in array_args.items():
    try:
      xnps[numpy_utils.lazy.get_xnp(v, strict=strict)].append(k)
    except Exception as e:  # pylint: disable=broad-except
      epy.reraise(e, prefix=f'Invalid {k}: Expected xnp.ndarray: ')

  return _infer_xnp(xnps)


def _infer_xnp(
    xnps: dict[numpy_utils.NpModule, list[str]]
) -> numpy_utils.NpModule:
  """Extract the `xnp` module."""
  non_np_xnps = set(xnps) - {np}  # jnp, tnp, torch take precedence on `np`

  # Detecting conflicting xnp
  if len(non_np_xnps) > 1:
    xnps = {k.__name__: v for k, v in xnps.items()}
    raise ValueError(f'Conflicting numpy types: {xnps}')

  if not non_np_xnps:
    return np
  else:
    (xnp,) = non_np_xnps
    return xnp


@functools.lru_cache(None)
def _maybe_set_tnp_casting(xnp: numpy_utils.NpModule) -> None:
  """If TF numpy mode is not set, make sure `tnp.asarray(1.)` is `tf.float32`.

  If user uses TF without numpy mode, it will create casting issues (for
  example: `tf.float64 + tf.float32` will raise an error).
  To limit the errors encountered, we set `tnp.asarray(1.)` to `tf.float32`
  instead of `tf.float64`.

  If numpy mode is already activated, then no need to do anything, as
  `tf.float64 + tf.float32` will support auto-casting, like Jax and Numpy.

  Args:
    xnp: numpy module.
  """
  if not numpy_utils.lazy.has_tf or xnp is not numpy_utils.lazy.tnp:
    return  # Not tnp module

  if not numpy_utils.lazy.is_tnp_enabled:
    # When TF numpy mode is not enabled, `tnp.asarray(1.)` returns tf.float64,
    # creating conflict because TF do fail for operations like:
    # `tf.float64 + tf.float32`
    from tensorflow.python.ops.numpy_ops import np_dtypes  # pylint: disable=g-import-not-at-top,g-direct-tensorflow-import  # pytype: disable=import-error

    if not np_dtypes.is_prefer_float32():
      np_dtypes.set_prefer_float32(True)

    msg = epy.dedent(
        """
        WARNING: Using array types for TF but without numpy mode enabled. It
        is recommended to activate numpy mode as:

        import tensorflow.experimental.numpy as tnp
        tnp.experimental_enable_numpy_behavior(prefer_float32=True)
    """
    )
    # Use print otherwise this isn't displayed on Colab
    # Could have a `epy.logging` module which auto-print on Colab.
    print(msg)


def _parse_signature(fn) -> _FnSignatureCache:
  """Parse the function signature."""
  # At this point, `ForwardRef` should have been resolved.
  try:
    hints = typing.get_type_hints(fn)
  except Exception as e:  # pylint: disable=broad-except
    epy.reraise(
        e,
        prefix=(
            f'Could not infer typing annotation of {fn.__qualname__} '
            f'defined in {fn.__module__}'
        ),
    )

  sig = inspect.signature(fn)

  # For each valid params, create the validator
  # TODO(py38): Use :=
  array_params = {}
  for name, param in sig.parameters.items():
    array_param = _get_array_param(param, hints)
    if array_param is not None:
      array_params[name] = array_param

  if not array_params:
    raise ValueError(
        f'Could not detect any array type hints in {fn.__qualname__} with '
        f'signature {sig}.'
    )

  return _FnSignatureCache(
      sig=sig,
      has_xnp_kwargs='xnp' in sig.parameters,
      array_params=array_params,
  )


def _get_array_param(
    param: inspect.Parameter,
    hints: dict[str, _TypeForm],
) -> Optional[_ArrayParam]:
  """Parse the type & hint of the array."""
  name = param.name
  if name not in hints:  # Not an array param
    return None

  hint = hints[name]

  def make_err(msg: str) -> Exception:
    return NotImplementedError(
        f'`enp.check_and_normalize_arrays` does not support {msg}. Please open '
        f'an issue if you need this feature. For `{name}: {hint}`'
    )

  leaf_types = type_parsing.get_leaf_types(hint)
  is_optional = None in leaf_types
  # Filter Optional
  leaf_types = [t for t in leaf_types if t is not None]

  # Currently, only Optional[Array] or Array supported
  are_array = [isinstance(l, array_typing.ArrayAliasMeta) for l in leaf_types]
  count_array = are_array.count(True)
  count_non_array = are_array.count(False)

  if count_array and count_non_array:
    raise make_err('Union of array and non-array')
  if count_array > 1:
    raise make_err('Union of arrays')
  if count_non_array:
    return None  # Not an array param

  (array_type,) = leaf_types

  if param.kind in {
      inspect.Parameter.VAR_POSITIONAL,
      inspect.Parameter.VAR_KEYWORD,
  }:
    raise make_err('*args, **kwargs')

  return _ArrayParam(
      is_optional=is_optional,
      type=array_type,
      name=name,
  )
