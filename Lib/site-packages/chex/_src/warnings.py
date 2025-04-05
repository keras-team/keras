# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
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
# ==============================================================================
"""Utilities to emit warnings."""

import functools
from typing import Any, Callable, Optional
import warnings


def warn_only_n_pos_args_in_future(fun, n):
  """Warns if more than ``n`` positional arguments are passed to ``fun``.

  For instance:
  >>> @functools.partial(chex.warn_only_n_pos_args_in_future, n=1)
  ... def f(a, b, c=1):
  ...   return a + b + c

  Will raise a DeprecationWarning if ``f`` is called with more than one
  positional argument (e.g. both f(1, 2, 3) and f(1, 2, c=3) raise a warning).

  Args:
    fun: the function to wrap.
    n: the number of positional arguments to allow.

  Returns:
    A wrapped function that emits a warning if more than `n` positional
    arguments are passed.
  """

  @functools.wraps(fun)
  def wrapper(*args, **kwargs):
    if len(args) > n:
      warnings.warn(
          f'only the first {n} arguments can be passed positionally '
          'additional args will become keyword-only soon',
          DeprecationWarning,
          stacklevel=2
          )
    return fun(*args, **kwargs)

  return wrapper


warn_keyword_args_only_in_future = functools.partial(
    warn_only_n_pos_args_in_future, n=0
)


def warn_deprecated_function(
    fun: Callable[..., Any], replacement: Optional[str] = None
) -> Callable[..., Any]:
  """A decorator to mark a function definition as deprecated.

  Example usage:
  >>> @functools.partial(chex.warn_deprecated_function, replacement='g')
  ... def f(a, b):
  ...   return a + b

  Args:
    fun: the deprecated function.
    replacement: name of the function to be used instead.

  Returns:
    the wrapped function.
  """
  if hasattr(fun, '__name__'):
    warning_message = f'The function {fun.__name__} is deprecated.'
  else:
    warning_message = 'The function is deprecated.'
  if replacement:
    warning_message += f' Please use {replacement} instead.'

  @functools.wraps(fun)
  def new_fun(*args, **kwargs):
    warnings.warn(warning_message, category=DeprecationWarning, stacklevel=2)
    return fun(*args, **kwargs)

  return new_fun


def create_deprecated_function_alias(fun, new_name, deprecated_alias):
  """Create a deprecated alias for a function.

  Example usage:
  >>> g = create_deprecated_function_alias(f, 'path.f', 'path.g')

  Args:
    fun: the deprecated function.
    new_name: the new name to use (you may include the path for clarity).
    deprecated_alias: the old name (you may include the path for clarity).

  Returns:
    the wrapped function.
  """

  @functools.wraps(fun)
  def new_fun(*args, **kwargs):
    warnings.warn(
        f'The function {deprecated_alias} is deprecated, '
        f'please use {new_name} instead.',
        category=DeprecationWarning,
        stacklevel=2)
    return fun(*args, **kwargs)
  return new_fun
