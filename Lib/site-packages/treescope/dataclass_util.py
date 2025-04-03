# Copyright 2024 The Treescope Authors.
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

"""Utilities for working with dataclasses."""

import dataclasses
import functools
import inspect
from typing import Any, TypeVar

T = TypeVar("T")


def dataclass_from_attributes(cls: type[T], **field_values) -> T:
  """Directly instantiates a dataclass given all of its fields.

  Dataclasses can override ``__init__`` to have arbitrary custom behavior, but
  this may make it difficult to construct new instances of dataclasses with
  particular field values. This function makes it possible to directly
  instantiate an instance of a dataclass with given attributes.

  Callers of this method are responsible for maintaining any invariants
  expected by the class. The intended use of this function is to restore a
  dataclass from field values extracted from another instance of that exact
  dataclass type.

  Args:
    cls: Class to instantiate.
    **field_values: Values for each of the dataclass's fields

  Returns:
    A new instance of the class.
  """
  # Make sure our fields are correct.
  expected_fields = dataclasses.fields(cls)
  expected_names = set(field.name for field in expected_fields)
  given_names = set(field_values.keys())
  if expected_names != given_names:
    raise ValueError(
        "Incorrect fields provided to `dataclass_from_attributes`; expected"
        f" {expected_names}, got {given_names}"
    )
  # Make a new object, bypassing the class's initializer.
  value = object.__new__(cls)
  # Set all the attributes, bypassing the class's __setattr__.
  for k, v in field_values.items():
    object.__setattr__(value, k, v)
  return value


@functools.cache
def init_takes_fields(cls: type[Any]) -> bool:
  """Returns True if ``cls.__init__`` takes exactly one argument per field.

  This is a heuristic for determining whether this dataclass can be rebuilt
  from its attributes using a simple repr-like format (e.g.
  ``Foo(bar=1, baz=2)``) or whether safely rebuilding it requires using
  :func:`dataclass_from_attributes` above. This is used during pretty-printing
  to determine whether to switch to a more verbose form when a round-trippable
  representation is requested.

  Note that it's technically possible to override ``__init__`` so that it takes
  the fields as attributes and then modifies them; it's not really possible to
  check for this, so we just check that the signature looks correct.

  Args:
    cls: The dataclass to check.
  """
  assert dataclasses.is_dataclass(cls)
  fields = dataclasses.fields(cls)
  remaining_field_set = set(field.name for field in fields)
  signature = inspect.signature(cls.__init__)

  # Skip `self` argument.
  parameters = list(signature.parameters.values())
  for parameter in parameters[1:]:
    if parameter.kind not in {
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    }:
      # It might not be safe to pass keyword arguments.
      return False
    if parameter.name in remaining_field_set:
      remaining_field_set.remove(parameter.name)
    else:
      # Unexpected parameter; this means __init__ was overridden with extra
      # parameters.
      return False

  if remaining_field_set:
    # Some fields were not present in __init__!
    return False
  else:
    return True
