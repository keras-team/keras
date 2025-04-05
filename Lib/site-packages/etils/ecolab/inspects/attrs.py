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

"""Attributes parser."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class ExceptionWrapper:
  """Wrapper around an exception to signal an error was raised.

  While parsing the attribute.
  """
  e: Exception


def get_attrs(obj: object) -> dict[str, object]:
  """Parse all attributes from an object.

  Limitation:

  * Descriptor will be resolved, so all properties are executed (some can
    have side effects, or take a lot of time to compute)

  Args:
    obj: Object to inspect

  Returns:
    Dict mapping attribute name to values.
  """
  attrs = {}
  # Merge `dir(obj)` with `object.__dir__(obj)` to bypass custom object
  # `__dir__`
  for k in dir(obj) + object.__dir__(obj):
    if k in attrs:
      continue
    try:
      v = getattr(obj, k)
    except Exception as e:  # pylint: disable=broad-except
      v = ExceptionWrapper(e)
    attrs[k] = v

  return attrs
