# Copyright 2024 The Orbax Authors.
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

"""Handles empty values in the checkpoint PyTree."""

import collections
from typing import Any, Mapping
from orbax.checkpoint._src.metadata import pytree_metadata_options as pytree_metadata_options_lib
from orbax.checkpoint._src.tree import utils as tree_utils

PyTreeMetadataOptions = pytree_metadata_options_lib.PyTreeMetadataOptions

RESTORE_TYPE_NONE = 'None'
RESTORE_TYPE_DICT = 'Dict'
RESTORE_TYPE_LIST = 'List'
RESTORE_TYPE_TUPLE = 'Tuple'
RESTORE_TYPE_NAMED_TUPLE = 'NamedTuple'
RESTORE_TYPE_UNKNOWN = 'Unknown'


def is_supported_empty_value(
    value: Any,
    pytree_metadata_options: PyTreeMetadataOptions = (
        pytree_metadata_options_lib.PYTREE_METADATA_OPTIONS
    ),
) -> bool:
  """Determines if the *empty* `value` is supported without custom TypeHandler."""
  # Check isinstance first to avoid `not` checks on jax.Arrays (raises error).
  if tree_utils.isinstance_of_namedtuple(value):
    if pytree_metadata_options.support_rich_types and not value:
      return True
    return False
  return (
      isinstance(value, (dict, list, tuple, type(None), Mapping)) and not value
  )


def get_empty_value_typestr(
    value: Any, pytree_metadata_options: PyTreeMetadataOptions
) -> str:
  """Returns the typestr constant for the empty value."""
  if not is_supported_empty_value(value, pytree_metadata_options):
    raise ValueError(
        f'{value} is not a supported empty type with pytree_metadata_options:'
        f' {pytree_metadata_options}.'
    )
  if isinstance(value, list):
    return RESTORE_TYPE_LIST
  if tree_utils.isinstance_of_namedtuple(value):  # Call before tuple check.
    return RESTORE_TYPE_NAMED_TUPLE
  if isinstance(value, tuple):
    return RESTORE_TYPE_TUPLE
  if isinstance(value, (dict, Mapping)):
    return RESTORE_TYPE_DICT
  if value is None:
    return RESTORE_TYPE_NONE
  raise ValueError(
      f'Unrecognized empty type: {value} with pytree_metadata_options:'
      f' {pytree_metadata_options}.'
  )


def override_empty_value_typestr(
    typestr: str, pytree_metadata_options: PyTreeMetadataOptions
) -> str:
  """Returns updated typestr based on pytree_metadata_options."""
  if not pytree_metadata_options.support_rich_types:
    if typestr == RESTORE_TYPE_NAMED_TUPLE:
      return RESTORE_TYPE_NONE
  return typestr


def is_empty_typestr(typestr: str) -> bool:
  return (
      typestr == RESTORE_TYPE_LIST
      or typestr == RESTORE_TYPE_NAMED_TUPLE
      or typestr == RESTORE_TYPE_TUPLE
      or typestr == RESTORE_TYPE_DICT
      or typestr == RESTORE_TYPE_NONE
  )


class OrbaxEmptyNamedTuple(collections.namedtuple('OrbaxEmptyNamedTuple', ())):
  pass


def get_empty_value_from_typestr(
    typestr: str, pytree_metadata_options: PyTreeMetadataOptions
) -> Any:
  """Returns the empty value for the given typestr.

  Args:
    typestr: The typestr constant for the empty value.
    pytree_metadata_options: The pytree metadata options.

  Raises:
    ValueError: If the typestr is not supported.
  """
  if typestr == RESTORE_TYPE_LIST:
    return []
  if typestr == RESTORE_TYPE_NAMED_TUPLE:
    if pytree_metadata_options.support_rich_types:
      return OrbaxEmptyNamedTuple()
    else:
      return None
  if typestr == RESTORE_TYPE_TUPLE:
    return tuple()
  if typestr == RESTORE_TYPE_DICT:
    return {}
  if typestr == RESTORE_TYPE_NONE:
    return None
  raise ValueError(
      f'Unrecognized typestr: {typestr} with pytree_metadata_options:'
      f' {pytree_metadata_options}.'
  )
