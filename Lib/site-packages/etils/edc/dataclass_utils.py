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

"""Dataclass utils."""

from __future__ import annotations

import dataclasses
import functools
import typing
from typing import Any, Callable, TypeVar

from etils import epy
from etils.edc import cast_utils
from etils.edc import context
from etils.edc import frozen_utils
from etils.edc import helpers

_Cls = Any
_ClsT = TypeVar('_ClsT')
_T = TypeVar('_T')


@typing.overload
def dataclass(
    cls: None = ...,
    *,
    kw_only: bool = ...,
    replace: bool = ...,  # pylint: disable=redefined-outer-name
    repr: bool = ...,  # pylint: disable=redefined-builtin
    auto_cast: bool = ...,
    contextvars: bool = ...,
    allow_unfrozen: bool = ...,
) -> Callable[[_ClsT], _ClsT]:
  ...


@typing.overload
def dataclass(
    cls: _ClsT,
    *,
    kw_only: bool = ...,
    replace: bool = ...,  # pylint: disable=redefined-outer-name
    repr: bool = ...,  # pylint: disable=redefined-builtin
    auto_cast: bool = ...,
    contextvars: bool = ...,
    allow_unfrozen: bool = ...,
) -> _ClsT:
  ...


def dataclass(
    cls=None,
    *,
    kw_only=False,
    replace=True,  # pylint: disable=redefined-outer-name
    repr=True,  # pylint: disable=redefined-builtin
    auto_cast=True,
    contextvars=True,
    allow_unfrozen=False,
):
  """Augment a dataclass with additional features.

  `auto_cast`: Auto-convert init assignements to the annotated class.

  ```python
  @edc.dataclass
  class A:
    path: edc.AutoCast[epath.Path]
    some_enum: edc.AutoCast[MyEnum]
    x: edc.AutoCast[str]

  a = A(
      path='/some/path',
      some_enum='A',
      x=123
  )
  # Fields annotated with `AutoCast` are automatically casted to their type
  assert a.path == epath.Path('/some/path')
  assert a.some_enum is MyEnum.A
  assert a.x == '123'
  ```

  `allow_unfrozen`: allow nested dataclass to be updated. This add two methods:

   * `.unfrozen()`: Create a lazy deep-copy of the current dataclass. Updates
     to nested attributes will be propagated to the top-level dataclass.
   * `.frozen()`: Returns the frozen dataclass, after it was mutated.

  Example:

  ```python
  old_x = X(y=Y(z=123))

  x = old_x.unfrozen()
  x.y.z = 456
  x = x.frozen()

  assert x == X(y=Y(z=123))  # Only new x is mutated
  assert old_x == X(y=Y(z=456))  # Old x is not mutated
  ```

  Note:

  * Only the last `.frozen()` call resolve the dataclass by calling `.replace`
    recursivelly.
  * Dataclass returned by `.unfrozen()` and nested attributes are not the
    original dataclass but proxy objects which track the mutations. As such,
    those object are not compatible with `isinstance()`, `jax.tree.map`,...
  * Only the top-level dataclass need to be `allow_unfrozen=True`
  * Avoid using `unfrozen` if 2 attributes of the dataclass point to the
    same nested dataclass. Updates on one attribute might not be reflected on
    the other.

    ```python
    y = Y(y=123)
    x = X(x0=y, x1=y)  # Same instance assigned twice in `x0` and `x1`
    x = x.unfrozen()
    x.x0.y = 456  # Changes in `x0` not reflected in `x1`
    x = x.frozen()

    assert x == X(x0=Y(y=456), x1=Y(y=123))
    ```

    This is because only attributes which are accessed are tracked, so `etils`
    do not know the object exist somewhere else in the attribute tree.

  * After `.frozen()` has been called, any of the temporary sub-attribute
    become invalid:

    ```python
    a = a.unfrozen()
    y = a.y
    a = a.frozen()

    y.x  # Raise error (created between the unfrozen/frozen call)
    a.y.x  # Work
    ```

  `contextvars`: Fields annotated as `edc.ContextVar` are wrapped in
  a `contextvars.ContextVar`. Afterward each thread / asyncio coroutine will
  have its own version of the fields (similarly to `threading.local`).

  The contextvars are lazily initialized at first usage.

  Example:

  ```python
  @edc.dataclass
  @dataclasses.dataclass
  class Context:
    thread_id: edc.ContextVar[int] = dataclasses.field(
        default_factory=threading.get_native_id
    )
    stack: edc.ContextVar[list[str]] = dataclasses.field(default_factory=list)

  # Global context object
  context = Context(thread_id=0)

  def worker():
    # Inside each thread, the worker use its own context
    assert context.thread_id != 0
    context.stack.append(1)

  with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    for _ in range(10):
      executor.submit(worker)
  ```

  Args:
    cls: The dataclass to decorate
    kw_only: If True, make the dataclass `__init__` keyword-only.
    replace: If `True`, add a `.replace(` alias of `dataclasses.replace`.
    repr: If `True`, the class `__repr__` will return a pretty-printed `str`
      (one attribute per line)
    auto_cast: If `True`, fields annotated as `x: edc.AutoCast[Cls]` will be
      converted to `x: Cls = edc.field(validator=Cls)`.
    contextvars: It `True`, fields annotated as `x: edc.AutoCast[T]` are
      converted to `contextvars`. This allow to have a `threading.local`-like
      API for contextvars.
    allow_unfrozen: If `True`, add `.frozen`, `.unfrozen` methods.

  Returns:
    Decorated class
  """
  # Return decorator
  if cls is None:
    return functools.partial(
        dataclass,
        kw_only=kw_only,
        replace=replace,
        repr=repr,
        auto_cast=auto_cast,
        allow_unfrozen=allow_unfrozen,
    )

  if kw_only:
    cls = _make_kw_only(cls)

  if repr:
    cls = add_repr(cls)

  if replace:
    cls = _add_replace(cls)

  if allow_unfrozen:
    cls = frozen_utils.add_unfrozen(cls)

  descriptor_fns = []
  if auto_cast:
    descriptor_fns.append(
        helpers.DescriptorInfo(
            annotation=cast_utils.AutoCast,
            descriptor_fn=cast_utils.make_auto_cast_descriptor,
        )
    )

  if contextvars:
    descriptor_fns.append(
        helpers.DescriptorInfo(
            annotation=context.ContextVar,
            descriptor_fn=context.make_contextvar_descriptor,
        )
    )

  cls = helpers.wrap_new(cls, descriptor_fns)

  return cls


def _make_kw_only(cls: _ClsT) -> _ClsT:
  """Replace the `__init__` by a keyword-only version."""
  # Use `cls.__dict__` and not `hasattr` to ignore parent classes
  if '__init__' not in cls.__dict__:
    return cls  # Do not mutate the class if __init__ isn't present

  old_init = cls.__init__

  # Despite `@functools.wraps`, the function has to be called `__init__` (
  # see: https://stackoverflow.com/q/29919804/4172685)
  @functools.wraps(old_init)
  def __init__(self, *args, **kwargs):  # pylint: disable=invalid-name
    if args:
      raise TypeError(
          f'{self.__class__.__name__} contructor is keyword-only. '
          f'Got {len(args)} positional arguments.'
      )
    return old_init(self, **kwargs)

  cls.__init__ = __init__

  return cls


def _add_replace(cls: _ClsT) -> _ClsT:
  """Add a `.replace` method to the class, if not already present."""
  # Only add replace if not present
  if not hasattr(cls, 'replace'):
    cls.replace = replace
  return cls


def replace(self: _T, **kwargs: Any) -> _T:
  """Similar to `dataclasses.replace`."""
  return dataclasses.replace(self, **kwargs)


def add_repr(cls: _ClsT) -> _ClsT:
  """Add a `.__repr__` method to the class, if not already present."""
  # Use `cls.__dict__` and not `hasattr` to ignore parent classes
  if '__repr__' not in cls.__dict__:
    return cls
  if epy.text_utils.has_default_repr(cls):
    cls.__repr__ = __repr__
  return cls


__repr__ = epy.pretty_repr
