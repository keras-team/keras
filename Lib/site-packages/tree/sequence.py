# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
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
"""Contains _sequence_like and helpers for sequence data structures."""
import collections
from collections import abc as collections_abc
import types
from tree import _tree

# pylint: disable=g-import-not-at-top
try:
  import wrapt
  ObjectProxy = wrapt.ObjectProxy
except ImportError:
  class ObjectProxy(object):
    """Stub-class for `wrapt.ObjectProxy``."""


def _sorted(dictionary):
  """Returns a sorted list of the dict keys, with error if keys not sortable."""
  try:
    return sorted(dictionary)
  except TypeError:
    raise TypeError("tree only supports dicts with sortable keys.")


def _is_attrs(instance):
  return _tree.is_attrs(instance)


def _is_namedtuple(instance, strict=False):
  """Returns True iff `instance` is a `namedtuple`.

  Args:
    instance: An instance of a Python object.
    strict: If True, `instance` is considered to be a `namedtuple` only if
        it is a "plain" namedtuple. For instance, a class inheriting
        from a `namedtuple` will be considered to be a `namedtuple`
        iff `strict=False`.

  Returns:
    True if `instance` is a `namedtuple`.
  """
  return _tree.is_namedtuple(instance, strict)


def _sequence_like(instance, args):
  """Converts the sequence `args` to the same type as `instance`.

  Args:
    instance: an instance of `tuple`, `list`, `namedtuple`, `dict`, or
        `collections.OrderedDict`.
    args: elements to be converted to the `instance` type.

  Returns:
    `args` with the type of `instance`.
  """
  if isinstance(instance, (dict, collections_abc.Mapping)):
    # Pack dictionaries in a deterministic order by sorting the keys.
    # Notice this means that we ignore the original order of `OrderedDict`
    # instances. This is intentional, to avoid potential bugs caused by mixing
    # ordered and plain dicts (e.g., flattening a dict but using a
    # corresponding `OrderedDict` to pack it back).
    result = dict(zip(_sorted(instance), args))
    keys_and_values = ((key, result[key]) for key in instance)
    if isinstance(instance, collections.defaultdict):
      # `defaultdict` requires a default factory as the first argument.
      return type(instance)(instance.default_factory, keys_and_values)
    elif isinstance(instance, types.MappingProxyType):
      # MappingProxyType requires a dict to proxy to.
      return type(instance)(dict(keys_and_values))
    else:
      return type(instance)(keys_and_values)
  elif isinstance(instance, collections_abc.MappingView):
    # We can't directly construct mapping views, so we create a list instead
    return list(args)
  elif _is_namedtuple(instance) or _is_attrs(instance):
    if isinstance(instance, ObjectProxy):
      instance_type = type(instance.__wrapped__)
    else:
      instance_type = type(instance)
    try:
      if _is_attrs(instance):
        return instance_type(
            **{
                attr.name: arg
                for attr, arg in zip(instance_type.__attrs_attrs__, args)
            })
      else:
        return instance_type(*args)
    except Exception as e:
      raise TypeError(
          f"Couldn't traverse {instance!r} with arguments {args}") from e
  elif isinstance(instance, ObjectProxy):
    # For object proxies, first create the underlying type and then re-wrap it
    # in the proxy type.
    return type(instance)(_sequence_like(instance.__wrapped__, args))
  else:
    # Not a namedtuple
    return type(instance)(args)
