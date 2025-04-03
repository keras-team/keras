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

"""Support for keyword-only fields in dataclasses for Python versions <3.10.

This module provides wrappers for `dataclasses.dataclass` and
`dataclasses.field` that simulate support for keyword-only fields for Python
versions before 3.10 (which is the version where dataclasses added keyword-only
field support).  If this module is imported in Python 3.10+, then
`kw_only_dataclasses.dataclass` and `kw_only_dataclasses.field` will simply be
aliases for `dataclasses.dataclass` and `dataclasses.field`.

For earlier Python versions, when constructing a dataclass, any fields that have
been marked as keyword-only (including inherited fields) will be moved to the
end of the constuctor's argument list. This makes it possible to have a base
class that defines a field with a default, and a subclass that defines a field
without a default. E.g.:

>>> from flax.linen import kw_only_dataclasses
>>> @kw_only_dataclasses.dataclass
... class Parent:
...   name: str = kw_only_dataclasses.field(default='', kw_only=True)

>>> @kw_only_dataclasses.dataclass
... class Child(Parent):
...   size: float  # required.

>>> import inspect
>>> print(inspect.signature(Child.__init__))
(self, size: float, name: str = '') -> None


(If we used `dataclasses` rather than `kw_only_dataclasses` for the above
example, then it would have failed with TypeError "non-default argument
'size' follows default argument.")

WARNING: fields marked as keyword-only will not *actually* be turned into
keyword-only parameters in the constructor; they will only be moved to the
end of the parameter list (after all non-keyword-only parameters).
"""

import dataclasses
import functools
import inspect
from types import MappingProxyType
from typing import Any, TypeVar

import typing_extensions as tpe

import flax

M = TypeVar('M', bound='flax.linen.Module')
FieldName = str
Annotation = Any
Default = Any


class _KwOnlyType:
  """Metadata tag used to tag keyword-only fields."""

  def __repr__(self):
    return 'KW_ONLY'


KW_ONLY = _KwOnlyType()


def field(*, metadata=None, kw_only=dataclasses.MISSING, **kwargs):
  """Wrapper for dataclassess.field that adds support for kw_only fields.

  Args:
    metadata: A mapping or None, containing metadata for the field.
    kw_only: If true, the field will be moved to the end of `__init__`'s
      parameter list.
    **kwargs: Keyword arguments forwarded to `dataclasses.field`

  Returns:
    A `dataclasses.Field` object.
  """
  if kw_only is not dataclasses.MISSING and kw_only:
    if (
      kwargs.get('default', dataclasses.MISSING) is dataclasses.MISSING
      and kwargs.get('default_factory', dataclasses.MISSING)
      is dataclasses.MISSING
    ):
      raise ValueError('Keyword-only fields with no default are not supported.')
    if metadata is None:
      metadata = {}
    metadata[KW_ONLY] = True
  return dataclasses.field(metadata=metadata, **kwargs)


@tpe.dataclass_transform(field_specifiers=(field,))  # type: ignore[literal-required]
def dataclass(cls=None, extra_fields=None, **kwargs):
  """Wrapper for dataclasses.dataclass that adds support for kw_only fields.

  Args:
    cls: The class to transform (or none to return a decorator).
    extra_fields: A list of `(name, type, Field)` tuples describing extra fields
      that should be added to the dataclass.  This is necessary for linen's
      use-case of this module, since the base class (linen.Module) is *not* a
      dataclass.  In particular, linen.Module class is used as the base for both
      frozen and non-frozen dataclass subclasses; but the frozen status of a
      dataclass must match the frozen status of any base dataclasses.
    **kwargs: Additional arguments for `dataclasses.dataclass`.

  Returns:
    `cls`.
  """

  def wrap(cls):
    return _process_class(cls, extra_fields=extra_fields, **kwargs)

  return wrap if cls is None else wrap(cls)


def _process_class(cls: type[M], extra_fields=None, **kwargs):
  """Transforms `cls` into a dataclass that supports kw_only fields."""
  if '__annotations__' not in cls.__dict__:
    cls.__annotations__ = {}

  # The original __dataclass_fields__ dicts for all base classes.  We will
  # modify these in-place before turning `cls` into a dataclass, and then
  # restore them to their original values.
  base_dataclass_fields = {}  # dict[cls, cls.__dataclass_fields__.copy()]

  # The keyword only fields from `cls` or any of its base classes.
  kw_only_fields: dict[FieldName, tuple[Annotation, Default]] = {}

  # Scan for KW_ONLY marker.
  kw_only_name = None
  for name, annotation in cls.__annotations__.items():
    if annotation is KW_ONLY:
      if kw_only_name is not None:
        raise TypeError('Multiple KW_ONLY markers')
      kw_only_name = name
    elif kw_only_name is not None:
      if not hasattr(cls, name):
        raise ValueError(
          'Keyword-only fields with no default are not supported.'
        )
      default = getattr(cls, name)
      if isinstance(default, dataclasses.Field):
        default.metadata = MappingProxyType({**default.metadata, KW_ONLY: True})
      else:
        default = field(default=default, kw_only=True)
      setattr(cls, name, default)
  if kw_only_name:
    del cls.__annotations__[kw_only_name]

  # Inject extra fields.
  if extra_fields:
    for name, annotation, default in extra_fields:
      if not (isinstance(name, str) and isinstance(default, dataclasses.Field)):
        raise ValueError(
          'Expected extra_fields to a be a list of '
          '(name, type, Field) tuples.'
        )
      setattr(cls, name, default)
      cls.__annotations__[name] = annotation

  # Extract kw_only fields from base classes' __dataclass_fields__.
  for base in reversed(cls.__mro__[1:]):
    if not dataclasses.is_dataclass(base):
      continue
    base_annotations = base.__dict__.get('__annotations__', {})
    base_dataclass_fields[base] = dict(
      getattr(base, '__dataclass_fields__', {})
    )
    for base_field in list(dataclasses.fields(base)):
      field_name = base_field.name
      if base_field.metadata.get(KW_ONLY) or field_name in kw_only_fields:
        kw_only_fields[field_name] = (
          base_annotations.get(field_name),
          base_field,
        )
        del base.__dataclass_fields__[field_name]

  # Remove any keyword-only fields from this class.
  cls_annotations = cls.__dict__['__annotations__']
  for name, annotation in list(cls_annotations.items()):
    value = getattr(cls, name, None)
    if (
      isinstance(value, dataclasses.Field) and value.metadata.get(KW_ONLY)
    ) or name in kw_only_fields:
      del cls_annotations[name]
      kw_only_fields[name] = (annotation, value)

  # Add keyword-only fields at the end of __annotations__, in the order they
  # were found in the base classes and in this class.
  for name, (annotation, default) in kw_only_fields.items():
    setattr(cls, name, default)
    cls_annotations.pop(name, None)
    cls_annotations[name] = annotation

  create_init = '__init__' not in vars(cls) and kwargs.get('init', True)

  # Apply the dataclass transform.
  transformed_cls: type[M] = dataclasses.dataclass(cls, **kwargs)

  # Restore the base classes' __dataclass_fields__.
  for _cls, fields in base_dataclass_fields.items():
    _cls.__dataclass_fields__ = fields

  if create_init:
    dataclass_init = transformed_cls.__init__
    # use sum to count the number of init fields that are not keyword-only
    expected_num_args = sum(
      f.init and not f.metadata.get(KW_ONLY, False)
      for f in dataclasses.fields(transformed_cls)
    )

    @functools.wraps(dataclass_init)
    def init_wrapper(self, *args, **kwargs):
      num_args = len(args)
      if num_args > expected_num_args:
        # we add + 1 to each to account for `self`, matching python's
        # default error message
        raise TypeError(
          f'__init__() takes {expected_num_args + 1} positional '
          f'arguments but {num_args + 1} were given'
        )

      dataclass_init(self, *args, **kwargs)

    init_wrapper.__signature__ = inspect.signature(dataclass_init)  # type: ignore
    transformed_cls.__init__ = init_wrapper  # type: ignore[method-assign]

  # Return the transformed dataclass
  return transformed_cls
