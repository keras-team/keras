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
"""JAX/dm-tree friendly dataclass implementation reusing Python dataclasses."""

import collections
import dataclasses
import functools
import sys

from absl import logging
import jax
from typing_extensions import dataclass_transform  # pytype: disable=not-supported-yet


FrozenInstanceError = dataclasses.FrozenInstanceError
_RESERVED_DCLS_FIELD_NAMES = frozenset(("from_tuple", "replace", "to_tuple"))


def mappable_dataclass(cls):
  """Exposes dataclass as ``collections.abc.Mapping`` descendent.

  Allows to traverse dataclasses in methods from `dm-tree` library.

  NOTE: changes dataclasses constructor to dict-type
  (i.e. positional args aren't supported; however can use generators/iterables).

  Args:
    cls: A dataclass to mutate.

  Returns:
    Mutated dataclass implementing ``collections.abc.Mapping`` interface.
  """
  if not dataclasses.is_dataclass(cls):
    raise ValueError(f"Expected dataclass, got {cls} (change wrappers order?).")

  # Define methods for compatibility with `collections.abc.Mapping`.
  setattr(cls, "__getitem__", lambda self, x: self.__dict__[x])
  setattr(cls, "__len__", lambda self: len(self.__dict__))
  setattr(cls, "__iter__", lambda self: iter(self.__dict__))
  # Override the default `collections.abc.Mapping` method implementation for
  # cleaner visualization. Without this change x.keys() shows the full repr(x)
  # instead of only the dict_keys present. The same goes for values and items.
  setattr(cls, "keys", lambda self: self.__dict__.keys())
  setattr(cls, "values", lambda self: self.__dict__.values())
  setattr(cls, "items", lambda self: self.__dict__.items())

  # Update constructor.
  orig_init = cls.__init__
  all_fields = set(f.name for f in cls.__dataclass_fields__.values())
  init_fields = [f.name for f in cls.__dataclass_fields__.values() if f.init]

  @functools.wraps(orig_init)
  def new_init(self, *orig_args, **orig_kwargs):
    if (orig_args and orig_kwargs) or len(orig_args) > 1:
      raise ValueError(
          "Mappable dataclass constructor doesn't support positional args."
          "(it has the same constructor as python dict)")
    all_kwargs = dict(*orig_args, **orig_kwargs)
    unknown_kwargs = set(all_kwargs.keys()) - all_fields
    if unknown_kwargs:
      raise ValueError(f"__init__() got unexpected kwargs: {unknown_kwargs}.")

    # Pass only arguments corresponding to fields with `init=True`.
    valid_kwargs = {k: v for k, v in all_kwargs.items() if k in init_fields}
    orig_init(self, **valid_kwargs)

  cls.__init__ = new_init

  # Update base class to derive from Mapping
  dct = dict(cls.__dict__)
  if "__dict__" in dct:
    dct.pop("__dict__")  # Avoid self-references.

  # Remove object from the sequence of base classes. Deriving from both Mapping
  # and object will cause a failure to create a MRO for the updated class
  bases = tuple(b for b in cls.__bases__ if b != object)
  cls = type(cls.__name__, bases + (collections.abc.Mapping,), dct)
  return cls


@dataclass_transform()
def dataclass(
    cls=None,
    *,
    init=True,
    repr=True,  # pylint: disable=redefined-builtin
    eq=True,
    order=False,
    unsafe_hash=False,
    frozen=False,
    kw_only: bool = False,
    mappable_dataclass=True,  # pylint: disable=redefined-outer-name
):
  """JAX-friendly wrapper for :py:func:`dataclasses.dataclass`.

  This wrapper class registers new dataclasses with JAX so that tree utils
  operate correctly. Additionally a replace method is provided making it easy
  to operate on the class when made immutable (frozen=True).

  Args:
    cls: A class to decorate.
    init: See :py:func:`dataclasses.dataclass`.
    repr: See :py:func:`dataclasses.dataclass`.
    eq: See :py:func:`dataclasses.dataclass`.
    order: See :py:func:`dataclasses.dataclass`.
    unsafe_hash: See :py:func:`dataclasses.dataclass`.
    frozen: See :py:func:`dataclasses.dataclass`.
    kw_only: See :py:func:`dataclasses.dataclass`.
    mappable_dataclass: If True (the default), methods to make the class
      implement the :py:class:`collections.abc.Mapping` interface will be
      generated and the class will include :py:class:`collections.abc.Mapping`
      in its base classes.
      `True` is the default, because being an instance of `Mapping` makes
      `chex.dataclass` compatible with e.g. `jax.tree_util.tree_*` methods, the
      `tree` library, or methods related to tensorflow/python/utils/nest.py.
      As a side-effect, e.g. `np.testing.assert_array_equal` will only check
      the field names are equal and not the content. Use `chex.assert_tree_*`
      instead.

  Returns:
    A JAX-friendly dataclass.
  """
  def dcls(cls):
    # Make sure to create a separate _Dataclass instance for each `cls`.
    return _Dataclass(
        init, repr, eq, order, unsafe_hash, frozen, kw_only, mappable_dataclass
    )(cls)

  if cls is None:
    return dcls
  return dcls(cls)


class _Dataclass():
  """JAX-friendly wrapper for `dataclasses.dataclass`."""

  def __init__(
      self,
      init=True,
      repr=True,  # pylint: disable=redefined-builtin
      eq=True,
      order=False,
      unsafe_hash=False,
      frozen=False,
      kw_only=False,
      mappable_dataclass=True,  # pylint: disable=redefined-outer-name
  ):
    self.init = init
    self.repr = repr  # pylint: disable=redefined-builtin
    self.eq = eq
    self.order = order
    self.unsafe_hash = unsafe_hash
    self.frozen = frozen
    self.kw_only = kw_only
    self.mappable_dataclass = mappable_dataclass

  def __call__(self, cls):
    """Forwards class to dataclasses's wrapper and registers it with JAX."""

    # Remove once https://github.com/python/cpython/pull/24484 is merged.
    for base in cls.__bases__:
      if (dataclasses.is_dataclass(base) and
          getattr(base, "__dataclass_params__").frozen and not self.frozen):
        raise TypeError("cannot inherit non-frozen dataclass from a frozen one")

    # `kw_only` is only available starting from 3.10.
    version_dependent_args = {}
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
      version_dependent_args = {"kw_only": self.kw_only}
    # pytype: disable=wrong-keyword-args
    dcls = dataclasses.dataclass(
        cls,
        init=self.init,
        repr=self.repr,
        eq=self.eq,
        order=self.order,
        unsafe_hash=self.unsafe_hash,
        frozen=self.frozen,
        **version_dependent_args,
    )
    # pytype: enable=wrong-keyword-args

    fields_names = set(f.name for f in dataclasses.fields(dcls))
    invalid_fields = fields_names.intersection(_RESERVED_DCLS_FIELD_NAMES)
    if invalid_fields:
      raise ValueError(f"The following dataclass fields are disallowed: "
                       f"{invalid_fields} ({dcls}).")

    if self.mappable_dataclass:
      dcls = mappable_dataclass(dcls)

    def _from_tuple(args):
      return dcls(zip(dcls.__dataclass_fields__.keys(), args))

    def _to_tuple(self):
      return tuple(getattr(self, k) for k in self.__dataclass_fields__.keys())

    def _replace(self, **kwargs):
      return dataclasses.replace(self, **kwargs)

    def _getstate(self):
      return self.__dict__

    # Register the dataclass at definition. As long as the dataclass is defined
    # outside __main__, this is sufficient to make JAX's PyTree registry
    # recognize the dataclass and the dataclass' custom PyTreeDef, especially
    # when unpickling either the dataclass object, its type, or its PyTreeDef,
    # in a different process, because the defining module will be imported.
    #
    # However, if the dataclass is defined in __main__, unpickling in a
    # subprocess does not trigger re-registration. Therefore we also need to
    # register when deserializing the object, or construction (e.g. when the
    # dataclass type is being unpickled). Unfortunately, there is not yet a way
    # to trigger re-registration when the treedef is unpickled as that's handled
    # by JAX.
    #
    # See internal dataclass_test for unit tests demonstrating the problems.
    # The registration below may result in pickling failures of the sort
    # _pickle.PicklingError: Can't pickle <functools._lru_cache_wrapper object>:
    # it's not the same object as register_dataclass_type_with_jax_tree_util
    # for modules defined in __main__ so we disable registration in this case.
    if dcls.__module__ != "__main__":
      register_dataclass_type_with_jax_tree_util(dcls)

    # Patch __setstate__ to register the dataclass on deserialization.
    def _setstate(self, state):
      register_dataclass_type_with_jax_tree_util(dcls)
      self.__dict__.update(state)

    orig_init = dcls.__init__

    # Patch __init__ such that the dataclass is registered on creation if it is
    # not registered on deserialization.
    @functools.wraps(orig_init)
    def _init(self, *args, **kwargs):
      register_dataclass_type_with_jax_tree_util(dcls)
      return orig_init(self, *args, **kwargs)

    setattr(dcls, "from_tuple", _from_tuple)
    setattr(dcls, "to_tuple", _to_tuple)
    setattr(dcls, "replace", _replace)
    setattr(dcls, "__getstate__", _getstate)
    setattr(dcls, "__setstate__", _setstate)
    setattr(dcls, "__init__", _init)

    return dcls


def _dataclass_unflatten(dcls, keys, values):
  """Creates a chex dataclass from a flatten jax.tree_util representation."""
  dcls_object = dcls.__new__(dcls)
  attribute_dict = dict(zip(keys, values))
  # Looping over fields instead of keys & values preserves the field order.
  # Using dataclasses.fields fails because dataclass uids change after
  # serialisation (eg, with cloudpickle).
  for field in dcls.__dataclass_fields__.values():
    if field.name in attribute_dict:  # Filter pseudo-fields.
      object.__setattr__(dcls_object, field.name, attribute_dict[field.name])
  # Need to manual call post_init here as we have avoided calling __init__
  if getattr(dcls_object, "__post_init__", None):
    dcls_object.__post_init__()
  return dcls_object


def _flatten_with_path(dcls):
  path = []
  keys = []
  for k, v in sorted(dcls.__dict__.items()):
    keys.append(k)  # generate same aux data as flatten without path
    k = jax.tree_util.GetAttrKey(k)
    path.append((k, v))
  return path, tuple(keys)


@functools.cache
def register_dataclass_type_with_jax_tree_util(data_class):
  """Register an existing dataclass so JAX knows how to handle it.

  This means that functions in jax.tree_util operate over the fields
  of the dataclass. See
  https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees
  for further information.

  Args:
    data_class: A class created using dataclasses.dataclass. It must be
      constructable from keyword arguments corresponding to the members exposed
      in instance.__dict__.
  """
  flatten = lambda d: jax.util.unzip2(sorted(d.__dict__.items()))[::-1]
  unflatten = functools.partial(_dataclass_unflatten, data_class)
  try:
    jax.tree_util.register_pytree_with_keys(
        nodetype=data_class, flatten_with_keys=_flatten_with_path,
        flatten_func=flatten, unflatten_func=unflatten)
  except ValueError:
    logging.info("%s is already registered as JAX PyTree node.", data_class)
