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

"""Storage for PyTree checkpoint metadata with rich types."""

from __future__ import annotations

import collections
import functools
from typing import Any, Iterable, Mapping, Sequence, Type, TypeAlias

import jax
from orbax.checkpoint._src.metadata import pytree_metadata_options as pytree_metadata_options_lib
from orbax.checkpoint._src.metadata import value_metadata_entry
from orbax.checkpoint._src.tree import utils as tree_utils
import simplejson

PyTree: TypeAlias = Any


@functools.lru_cache()
def _new_namedtuple_type(
    module_name: str,
    class_name: str,
    fields: Sequence[str],
) -> Type[tuple[Any, ...]]:
  """Returns a namedtuple type created in the current module.

  NOTE: `module_name` and `class_name` are concatenated to create a unique
  class name to avoid name collisions.

  Args:
    module_name: Module name of original namedtuple saved in metadata.
    class_name: Class name of original namedtuple saved in metadata.
    fields: The fields of the namedtuple.
  """
  # TODO: b/365169723 - Return concrete NamedTuple if available in given module.
  arity = len(fields)
  unique_class_name = f'{module_name}_{class_name}_{arity}'
  # Valid class name must not contain dots.
  valid_class_name = unique_class_name.replace('.', '_')
  return collections.namedtuple(valid_class_name, fields)


def _create_namedtuple(
    *, module_name: str, class_name: str, attrs: Iterable[tuple[str, Any]]
) -> tuple[Any, ...]:
  """Returns a namedtuple instance with the given attributes and values, `attrs`.

  The namedtuple type is created in the current module on the fly using the
  given `module_name` and `class_name`. The two names are combined to create a
  unique class name to avoid name collisions. See `_new_namedtuple_type()` for
  more details.

  Args:
    module_name: Module name of original namedtuple saved in metadata.
    class_name: Class name of original namedtuple saved in metadata.
    attrs: The attributes of the namedtuple.
  """
  ks, vs = [*zip(*attrs)] or ((), ())
  result = _new_namedtuple_type(module_name, class_name, ks)(*vs)
  return result


def _module_and_class_name(cls) -> tuple[str, str]:
  """Returns the module and class name of the given class instance."""
  return cls.__module__, cls.__qualname__


_VALUE_METADATA_ENTRY_CLAZZ = 'ValueMetadataEntry'
_VALUE_METADATA_ENTRY_MODULE_AND_CLASS = _module_and_class_name(
    value_metadata_entry.ValueMetadataEntry
)


def _value_metadata_tree_for_json_dumps(obj: Any) -> Any:
  """Callback for `simplejson.dumps` to convert a PyTree to JSON object."""
  # Handle ValueMetadataEntry instances.
  if tree_utils.is_empty_or_leaf(obj):
    if (
        _module_and_class_name(obj.__class__)
        == _VALUE_METADATA_ENTRY_MODULE_AND_CLASS
    ):
      return dict(
          category='custom',
          clazz=_VALUE_METADATA_ENTRY_CLAZZ,
          data=obj.to_json(),
      )
    raise ValueError(
        f'Expected ValueMetadataEntry, got metadata pytree leaf: {obj}'
    )

  # Check namedtuple first and then tuple.
  if tree_utils.isinstance_of_namedtuple(obj):
    module_name, class_name = _module_and_class_name(obj.__class__)
    return dict(
        category='namedtuple',
        module=module_name,
        clazz=class_name,
        entries=[
            dict(key=k, value=_value_metadata_tree_for_json_dumps(v))
            for k, v in zip(obj._fields, obj)
        ],
    )
  # Check namedtuple first and then tuple.
  if isinstance(obj, tuple):
    return dict(
        category='custom',
        clazz='tuple',
        entries=[_value_metadata_tree_for_json_dumps(e) for e in obj],
    )

  if isinstance(obj, Mapping):
    return {k: _value_metadata_tree_for_json_dumps(v) for k, v in obj.items()}

  if isinstance(obj, list):
    return [_value_metadata_tree_for_json_dumps(e) for e in obj]

  # Handle objects that are registered as Jax container nodes.
  key_leafs, _ = jax.tree_util.tree_flatten_with_path(
      obj,
      is_leaf=lambda x: x is not obj,  # flatten just one level.
  )
  return {
      tree_utils.get_key_name(keypath[0]): _value_metadata_tree_for_json_dumps(
          leaf
      )
      for keypath, leaf in key_leafs
  }


def _value_metadata_tree_for_json_loads(obj):
  """Callback for `simplejson.loads` to convert JSON object to a PyTree."""
  if not isinstance(obj, Mapping):
    return obj

  if 'category' in obj:
    if obj['category'] == 'custom':
      if obj['clazz'] == _VALUE_METADATA_ENTRY_CLAZZ:
        return value_metadata_entry.ValueMetadataEntry.from_json(
            obj['data'],
            pytree_metadata_options_lib.PyTreeMetadataOptions(
                support_rich_types=True,  # Always in rich types mode.
            ),
        )
      if obj['clazz'] == 'tuple':
        return tuple(
            [(_value_metadata_tree_for_json_loads(v)) for v in obj['entries']]
        )
      raise ValueError(
          f'Unsupported "custom" object in JSON deserialization: {obj}'
      )

    if obj['category'] == 'namedtuple':
      return _create_namedtuple(
          module_name=obj['module'],
          class_name=obj['clazz'],
          attrs=[
              (
                  e['key'],
                  _value_metadata_tree_for_json_loads(e['value']),
              )
              for e in obj['entries']
          ],
      )

  return {k: _value_metadata_tree_for_json_loads(v) for k, v in obj.items()}


def value_metadata_tree_to_json_str(tree: PyTree) -> str:
  """Returns a JSON string representation of the given PyTree.

  Sample JSON::
  ```
  '{
    "mu_nu": {
      "category": "namedtuple",
      "module": "orbax.checkpoint._src.testing.test_tree_utils",
      "clazz": "MuNu",
      "entries": [
        {
          "key": "mu",
          "value": {
            "category": "custom",
            "clazz": "ValueMetadataEntry",
            "data": {
              "value_type": "jax.Array",
              "skip_deserialize": false
            }
          }
        },
        {
          "key": "nu",
          "value": {
            "category": "custom",
            "clazz": "ValueMetadataEntry",
            "data": {
              "value_type": "np.ndarray",
              "skip_deserialize": false
            }
          }
        }
      ]
    },
    "my_tuple": {
      "category": "custom",
      "clazz": "tuple",
      "entries": [
          {
            "category": "custom",
            "clazz": "ValueMetadataEntry",
            "data": {
              "value_type": "np.ndarray",
              "skip_deserialize": false
            }
          }
      ]
    }
  }'
  ```

  Args:
    tree: A PyTree to be converted to JSON string.
  """
  return simplejson.dumps(
      tree,
      default=_value_metadata_tree_for_json_dumps,
      tuple_as_array=False,  # Must be False to preserve tuples.
      namedtuple_as_object=False,  # Must be False to preserve namedtuples.
  )


def value_metadata_tree_from_json_str(json_str: str) -> PyTree:
  """Returns a PyTree from the given JSON string."""
  return simplejson.loads(
      json_str,
      object_hook=_value_metadata_tree_for_json_loads,
  )
