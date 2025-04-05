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

"""Testing utilities for pytrees."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Sequence, Tuple

import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax.checkpoint._src.metadata import tree as tree_metadata
from orbax.checkpoint._src.metadata import tree_rich_types

PyTree = Any


def create_namedtuple(
    cls,
    field_value_tuples: list[tuple[str, tree_metadata.ValueMetadataEntry]],
) -> type[tuple[Any, ...]]:
  """Returns instance of a new namedtuple type structurally identical to `cls`."""
  fields, values = zip(*field_value_tuples)
  module_name, class_name = tree_rich_types._module_and_class_name(cls)  # pylint: disable=protected-access
  new_type = tree_rich_types._new_namedtuple_type(module_name, class_name, fields)  # pylint: disable=protected-access
  return new_type(*values)


class MuNu(NamedTuple):
  mu: jax.Array | None
  nu: np.ndarray | None


class IntegerNamedTuple(NamedTuple):
  x: int | None
  y: int | None


class EmptyNamedTuple(NamedTuple):
  pass


class NamedTupleWithNestedAttributes(NamedTuple):
  nested_mu_nu: MuNu | None = None
  nested_dict: Dict[str, jax.Array] | None = None
  nested_tuple: Tuple[jax.Array, jax.Array] | None = None
  nested_empty_named_tuple: EmptyNamedTuple | None = None
  my_empty_chex: MyEmptyChex | None = None


@chex.dataclass
class MyEmptyChex:
  pass


@chex.dataclass
class MyChex:
  my_jax_array: jax.Array | None = None
  my_np_array: np.ndarray | None = None


@chex.dataclass
class MyChexWithNestedAttributes:
  my_chex: MyChex | None = None
  my_dict: Dict[str, jax.Array] | None = None
  my_list: List[np.ndarray] | None = None
  my_empty_chex: MyEmptyChex | None = None


@flax.struct.dataclass
class MyEmptyFlax(flax.struct.PyTreeNode):
  pass


@flax.struct.dataclass
class MyFlax(flax.struct.PyTreeNode):
  my_jax_array: jax.Array | None = None
  my_nested_mapping: Mapping[str, Any] | None = None
  my_sequence: Sequence[Any] | None = None


@dataclasses.dataclass
class MyEmptyDataClass:
  pass


@dataclasses.dataclass
class MyDataClass:
  my_jax_array: jax.Array | None = None
  my_np_array: np.ndarray | None = None
  my_empty_dataclass: MyEmptyDataClass | None = None
  my_chex: MyChex | None = None


class MyEmptyClass:
  pass


class MyClass:
  """Test class.

  Attributes:
    a: optional jax.Array. default=None.
    b: optional np.ndarray. default=None.
  """

  def __init__(
      self,
      a: jax.Array | None = None,
      b: np.ndarray | None = None,
  ):
    self._a = a
    self._b = b

  @property
  def a(self) -> jax.Array | None:
    return self._a

  @property
  def b(self) -> np.ndarray | None:
    return self._b


class MyClassWithNestedAttributes:
  """Test class.

  Attributes:
    my_empty_class: optional `MyEmptyClass`. default=None.
    my_class: optional `MyClass`. default=None.
    my_chex: optional `MyChex`. default=None.
    mu_nu: optional `MuNu`. default=None.
  """

  def __init__(
      self,
      my_empty_class: MyEmptyClass | None = None,
      my_class: MyClass | None = None,
      my_chex: MyChex | None = None,
      mu_nu: MuNu | None = None,
  ):
    self._my_empty_class = my_empty_class
    self._my_class = my_class
    self._my_chex = my_chex
    self._mu_nu = mu_nu


@dataclasses.dataclass
class TestPyTree:
  """Test data class for pytree.

  Attributes:
    unique_name: unique name for the test.
    provide_tree: function to provide the pytree.
    expected_save_response: expected save response. Can be a BaseException or
      None. None implies that save should succeed.
    expected_restore_response: expected restore response. Can be a
      BaseException, a function to provide the pytree or None. None implies that
      expected restored tree should be the same as the tree returned by
      `provide_tree` function.
    expected_nested_tree_metadata: PyTree of ValueMetadataEntry as returned by
      `InternalTreeMetadata.as_nested_tree` with
      `PyTreeMetadataOptions.support_rich_types=false`.
    expected_nested_tree_metadata_with_rich_types: PyTree of ValueMetadataEntry
      as returned by `InternalTreeMetadata.as_nested_tree` with
      `PyTreeMetadataOptions.support_rich_types=true`. If not provided,
      `expected_nested_tree_metadata` will be used.
  """

  unique_name: str
  # Provide tree lazily via `tree_provider` to avoid error:
  # RuntimeError: Attempted call to JAX before absl.app.run() is called.
  provide_tree: Callable[[], PyTree]
  expected_save_response: BaseException | None = None
  expected_restore_response: BaseException | Callable[[], PyTree] | None = None
  expected_nested_tree_metadata: PyTree | None = None
  expected_nested_tree_metadata_with_rich_types: PyTree | None = None

  def __post_init__(self):
    self.expected_restore_response = (
        self.expected_restore_response or self.provide_tree
    )
    self.expected_nested_tree_metadata_with_rich_types = (
        self.expected_nested_tree_metadata_with_rich_types
        or self.expected_nested_tree_metadata
    )

  def __str__(self):
    return self.unique_name

  def __repr__(self):
    return self.unique_name


TEST_PYTREES = [
    TestPyTree(
        unique_name='empty_pytree',
        provide_tree=lambda: {},
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
        expected_nested_tree_metadata=(
            tree_metadata.ValueMetadataEntry(
                value_type='Dict',
                skip_deserialize=True,
            )
        ),
    ),
    TestPyTree(
        unique_name='empty_dict',
        provide_tree=lambda: {'empty_dict': {}},
        expected_nested_tree_metadata={
            'empty_dict': tree_metadata.ValueMetadataEntry(
                value_type='Dict',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='empty_list',
        provide_tree=lambda: {'empty_list': []},
        expected_nested_tree_metadata={
            'empty_list': tree_metadata.ValueMetadataEntry(
                value_type='List',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='empty_tuple',
        provide_tree=lambda: {'empty_tuple': tuple()},
        expected_nested_tree_metadata={
            'empty_tuple': tree_metadata.ValueMetadataEntry(
                value_type='Tuple',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='empty_named_tuple',
        provide_tree=lambda: {'empty_named_tuple': EmptyNamedTuple()},
        expected_nested_tree_metadata={
            'empty_named_tuple': tree_metadata.ValueMetadataEntry(
                value_type='None',
                skip_deserialize=True,
            )
        },
        expected_nested_tree_metadata_with_rich_types={
            'empty_named_tuple': tree_metadata.ValueMetadataEntry(
                value_type='NamedTuple',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='tuple_of_empty_list',
        provide_tree=lambda: {'tuple_of_empty_list': ([],)},
        expected_nested_tree_metadata={
            'tuple_of_empty_list': [
                tree_metadata.ValueMetadataEntry(
                    value_type='List',
                    skip_deserialize=True,
                ),
            ]
        },
        expected_nested_tree_metadata_with_rich_types={
            'tuple_of_empty_list': tuple([
                tree_metadata.ValueMetadataEntry(
                    value_type='List',
                    skip_deserialize=True,
                ),
            ])
        },
    ),
    TestPyTree(
        unique_name='list_of_empty_tuple',
        provide_tree=lambda: {'list_of_empty_tuple': [tuple()]},
        expected_nested_tree_metadata={
            'list_of_empty_tuple': [
                tree_metadata.ValueMetadataEntry(
                    value_type='Tuple',
                    skip_deserialize=True,
                ),
            ]
        },
    ),
    TestPyTree(
        unique_name='none_param',
        provide_tree=lambda: {'none_param': None},
        expected_nested_tree_metadata={
            'none_param': tree_metadata.ValueMetadataEntry(
                value_type='None',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='scalar_param',
        provide_tree=lambda: {'scalar_param': 1},
        expected_nested_tree_metadata={
            'scalar_param': tree_metadata.ValueMetadataEntry(
                value_type='scalar',
                skip_deserialize=False,
            )
        },
    ),
    TestPyTree(
        unique_name='nested_scalars',
        provide_tree=lambda: {
            'b': {'scalar_param': 1, 'nested_scalar_param': 3.4}
        },
        expected_nested_tree_metadata={
            'b': {
                'scalar_param': tree_metadata.ValueMetadataEntry(
                    value_type='scalar',
                    skip_deserialize=False,
                ),
                'nested_scalar_param': tree_metadata.ValueMetadataEntry(
                    value_type='scalar',
                    skip_deserialize=False,
                ),
            }
        },
    ),
    TestPyTree(
        unique_name='list_of_arrays',
        provide_tree=lambda: {'list_of_arrays': [np.arange(8), jnp.arange(8)]},
        expected_nested_tree_metadata={
            'list_of_arrays': [
                tree_metadata.ValueMetadataEntry(
                    value_type='np.ndarray',
                    skip_deserialize=False,
                ),
                tree_metadata.ValueMetadataEntry(
                    value_type='jax.Array',
                    skip_deserialize=False,
                ),
            ]
        },
    ),
    TestPyTree(
        unique_name='dict_of_nested_data',
        provide_tree=lambda: {
            'x': {'a': np.arange(8), 'b': MyEmptyChex()},
            'y': (np.arange(8), MyChex(my_np_array=np.arange(8))),
            'z': [
                {'c': np.arange(8)},
                [
                    (np.arange(8),),
                ],
            ],
        },
        expected_nested_tree_metadata={
            'x': {
                'a': tree_metadata.ValueMetadataEntry(
                    value_type='np.ndarray',
                    skip_deserialize=False,
                ),
                'b': tree_metadata.ValueMetadataEntry(
                    value_type='Dict',
                    skip_deserialize=True,
                ),
            },
            'y': [
                tree_metadata.ValueMetadataEntry(
                    value_type='np.ndarray',
                    skip_deserialize=False,
                ),
                {
                    'my_jax_array': tree_metadata.ValueMetadataEntry(
                        value_type='None',
                        skip_deserialize=True,
                    ),
                    'my_np_array': tree_metadata.ValueMetadataEntry(
                        value_type='np.ndarray',
                        skip_deserialize=False,
                    ),
                },
            ],
            'z': [
                {
                    'c': tree_metadata.ValueMetadataEntry(
                        value_type='np.ndarray',
                        skip_deserialize=False,
                    )
                },
                [
                    [
                        tree_metadata.ValueMetadataEntry(
                            value_type='np.ndarray',
                            skip_deserialize=False,
                        ),
                    ],
                ],
            ],
        },
        expected_nested_tree_metadata_with_rich_types={
            'x': {
                'a': tree_metadata.ValueMetadataEntry(
                    value_type='np.ndarray',
                    skip_deserialize=False,
                ),
                'b': tree_metadata.ValueMetadataEntry(
                    value_type='Dict',
                    skip_deserialize=True,
                ),
            },
            'y': (
                tree_metadata.ValueMetadataEntry(
                    value_type='np.ndarray',
                    skip_deserialize=False,
                ),
                {
                    'my_jax_array': tree_metadata.ValueMetadataEntry(
                        value_type='None',
                        skip_deserialize=True,
                    ),
                    'my_np_array': tree_metadata.ValueMetadataEntry(
                        value_type='np.ndarray',
                        skip_deserialize=False,
                    ),
                },
            ),
            'z': [
                {
                    'c': tree_metadata.ValueMetadataEntry(
                        value_type='np.ndarray',
                        skip_deserialize=False,
                    )
                },
                [
                    (
                        tree_metadata.ValueMetadataEntry(
                            value_type='np.ndarray',
                            skip_deserialize=False,
                        ),
                    ),
                ],
            ],
        },
    ),
    TestPyTree(
        unique_name='tuple_of_arrays',
        provide_tree=lambda: {'tuple_of_arrays': (np.arange(8), jnp.arange(8))},
        expected_nested_tree_metadata={
            'tuple_of_arrays': [
                tree_metadata.ValueMetadataEntry(
                    value_type='np.ndarray',
                    skip_deserialize=False,
                ),
                tree_metadata.ValueMetadataEntry(
                    value_type='jax.Array',
                    skip_deserialize=False,
                ),
            ]
        },
        expected_nested_tree_metadata_with_rich_types={
            'tuple_of_arrays': tuple([
                tree_metadata.ValueMetadataEntry(
                    value_type='np.ndarray',
                    skip_deserialize=False,
                ),
                tree_metadata.ValueMetadataEntry(
                    value_type='jax.Array',
                    skip_deserialize=False,
                ),
            ])
        },
    ),
    TestPyTree(
        unique_name='mu_nu',
        provide_tree=lambda: {'mu_nu': MuNu(mu=jnp.arange(8), nu=np.arange(8))},
        expected_nested_tree_metadata={
            'mu_nu': {
                'mu': tree_metadata.ValueMetadataEntry(
                    value_type='jax.Array',
                    skip_deserialize=False,
                ),
                'nu': tree_metadata.ValueMetadataEntry(
                    value_type='np.ndarray',
                    skip_deserialize=False,
                ),
            }
        },
        expected_nested_tree_metadata_with_rich_types={
            'mu_nu': create_namedtuple(
                MuNu,
                [
                    (
                        'mu',
                        tree_metadata.ValueMetadataEntry(
                            value_type='jax.Array',
                            skip_deserialize=False,
                        ),
                    ),
                    (
                        'nu',
                        tree_metadata.ValueMetadataEntry(
                            value_type='np.ndarray',
                            skip_deserialize=False,
                        ),
                    ),
                ],
            )
        },
    ),
    TestPyTree(
        unique_name='my_empty_chex',
        provide_tree=lambda: {'my_empty_chex': MyEmptyChex()},
        expected_nested_tree_metadata={
            'my_empty_chex': tree_metadata.ValueMetadataEntry(
                value_type='Dict',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='default_my_chex',
        provide_tree=lambda: {'default_my_chex': MyChex()},
        expected_nested_tree_metadata={
            'default_my_chex': {
                'my_jax_array': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
                'my_np_array': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
            }
        },
    ),
    TestPyTree(
        unique_name='my_chex',
        provide_tree=lambda: {
            'my_chex': MyChex(
                my_jax_array=jnp.arange(8),
                my_np_array=np.arange(8),
            )
        },
        expected_nested_tree_metadata={
            'my_chex': {
                'my_jax_array': tree_metadata.ValueMetadataEntry(
                    value_type='jax.Array',
                    skip_deserialize=False,
                ),
                'my_np_array': tree_metadata.ValueMetadataEntry(
                    value_type='np.ndarray',
                    skip_deserialize=False,
                ),
            }
        },
    ),
    TestPyTree(
        unique_name='default_my_chex_with_nested_attrs',
        provide_tree=lambda: {
            'default_my_chex_with_nested_attrs': MyChexWithNestedAttributes()
        },
        expected_nested_tree_metadata={
            'default_my_chex_with_nested_attrs': {
                'my_chex': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
                'my_dict': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
                'my_empty_chex': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
                'my_list': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
            }
        },
    ),
    TestPyTree(
        unique_name='my_chex_with_nested_attrs',
        provide_tree=lambda: {
            'my_chex_with_nested_attrs': MyChexWithNestedAttributes(
                my_chex=MyChex(
                    my_jax_array=jnp.arange(8), my_np_array=np.arange(8)
                ),
                my_dict={'a': jnp.arange(8), 'b': np.arange(8)},
                my_list=[jnp.arange(8), np.arange(8)],
                my_empty_chex=MyEmptyChex(),
            )
        },
        expected_nested_tree_metadata={
            'my_chex_with_nested_attrs': {
                'my_chex': {
                    'my_jax_array': tree_metadata.ValueMetadataEntry(
                        value_type='jax.Array',
                        skip_deserialize=False,
                    ),
                    'my_np_array': tree_metadata.ValueMetadataEntry(
                        value_type='np.ndarray',
                        skip_deserialize=False,
                    ),
                },
                'my_dict': {
                    'a': tree_metadata.ValueMetadataEntry(
                        value_type='jax.Array',
                        skip_deserialize=False,
                    ),
                    'b': tree_metadata.ValueMetadataEntry(
                        value_type='np.ndarray',
                        skip_deserialize=False,
                    ),
                },
                'my_empty_chex': tree_metadata.ValueMetadataEntry(
                    value_type='Dict',
                    skip_deserialize=True,
                ),
                'my_list': [
                    tree_metadata.ValueMetadataEntry(
                        value_type='jax.Array',
                        skip_deserialize=False,
                    ),
                    tree_metadata.ValueMetadataEntry(
                        value_type='np.ndarray',
                        skip_deserialize=False,
                    ),
                ],
            }
        },
    ),
    TestPyTree(
        unique_name='my_empty_dataclass',
        provide_tree=lambda: {'my_empty_dataclass': MyEmptyDataClass()},
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
        expected_nested_tree_metadata={
            'my_empty_dataclass': tree_metadata.ValueMetadataEntry(
                value_type='None',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='default_my_dataclass',
        provide_tree=lambda: {'default_my_dataclass': MyDataClass()},
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
        expected_nested_tree_metadata={
            'default_my_dataclass': tree_metadata.ValueMetadataEntry(
                value_type='None',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='my_dataclass',
        provide_tree=lambda: {
            'my_dataclass': MyDataClass(
                my_jax_array=jnp.arange(8),
                my_np_array=np.arange(8),
                my_empty_dataclass=MyEmptyDataClass(),
                my_chex=MyChex(
                    my_jax_array=jnp.arange(8),
                    my_np_array=np.arange(8),
                ),
            )
        },
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
        expected_nested_tree_metadata={
            'my_dataclass': tree_metadata.ValueMetadataEntry(
                value_type='None',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='my_empty_class',
        provide_tree=lambda: {'my_empty_class': MyEmptyClass()},
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
        expected_nested_tree_metadata={
            'my_empty_class': tree_metadata.ValueMetadataEntry(
                value_type='None',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='default_my_class',
        provide_tree=lambda: {'default_my_class': MyClass()},
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
        expected_nested_tree_metadata={
            'default_my_class': tree_metadata.ValueMetadataEntry(
                value_type='None',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='my_class',
        provide_tree=lambda: {
            'my_class': MyClass(a=jnp.arange(8), b=np.arange(8))
        },
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
        expected_nested_tree_metadata={
            'my_class': tree_metadata.ValueMetadataEntry(
                value_type='None',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='default_my_class_with_nested_attrs',
        provide_tree=lambda: {
            'default_my_class_with_nested_attrs': MyClassWithNestedAttributes()
        },
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
        expected_nested_tree_metadata={
            'default_my_class_with_nested_attrs': (
                tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                )
            )
        },
    ),
    TestPyTree(
        unique_name='my_class_with_nested_attrs',
        provide_tree=lambda: {
            'my_class_with_nested_attrs': MyClassWithNestedAttributes(
                my_empty_class=MyEmptyClass(),
                my_class=MyClass(a=jnp.arange(8), b=np.arange(8)),
                my_chex=MyChex(
                    my_jax_array=jnp.arange(8), my_np_array=np.arange(8)
                ),
                mu_nu=MuNu(mu=jnp.arange(8), nu=np.arange(8)),
            )
        },
        expected_save_response=ValueError(),
        expected_restore_response=ValueError(),
        expected_nested_tree_metadata={
            'my_class_with_nested_attrs': tree_metadata.ValueMetadataEntry(
                value_type='None',
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='default_named_tuple_with_nested_attrs',
        provide_tree=lambda: {
            'default_named_tuple_with_nested_attrs': (
                NamedTupleWithNestedAttributes()
            )
        },
        expected_nested_tree_metadata={
            'default_named_tuple_with_nested_attrs': {
                'nested_mu_nu': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
                'nested_dict': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
                'nested_tuple': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
                'nested_empty_named_tuple': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
                'my_empty_chex': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
            }
        },
        expected_nested_tree_metadata_with_rich_types={
            'default_named_tuple_with_nested_attrs': create_namedtuple(
                NamedTupleWithNestedAttributes,
                [
                    (
                        'nested_mu_nu',
                        tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                    ),
                    (
                        'nested_dict',
                        tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                    ),
                    (
                        'nested_tuple',
                        tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                    ),
                    (
                        'nested_empty_named_tuple',
                        tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                    ),
                    (
                        'my_empty_chex',
                        tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                    ),
                ],
            )
        },
    ),
    TestPyTree(
        unique_name='named_tuple_with_nested_attrs',
        provide_tree=lambda: {
            'named_tuple_with_nested_attrs': NamedTupleWithNestedAttributes(
                nested_mu_nu=MuNu(mu=jnp.arange(8), nu=np.arange(8)),
                nested_dict={'a': jnp.arange(8), 'b': np.arange(8)},
                nested_tuple=(jnp.arange(8), jnp.arange(8)),
                nested_empty_named_tuple=EmptyNamedTuple(),
                my_empty_chex=MyEmptyChex(),
            )
        },
        expected_nested_tree_metadata={
            'named_tuple_with_nested_attrs': {
                'nested_mu_nu': {
                    'mu': tree_metadata.ValueMetadataEntry(
                        value_type='jax.Array',
                        skip_deserialize=False,
                    ),
                    'nu': tree_metadata.ValueMetadataEntry(
                        value_type='np.ndarray',
                        skip_deserialize=False,
                    ),
                },
                'nested_dict': {
                    'a': tree_metadata.ValueMetadataEntry(
                        value_type='jax.Array',
                        skip_deserialize=False,
                    ),
                    'b': tree_metadata.ValueMetadataEntry(
                        value_type='np.ndarray',
                        skip_deserialize=False,
                    ),
                },
                'nested_tuple': [
                    tree_metadata.ValueMetadataEntry(
                        value_type='jax.Array',
                        skip_deserialize=False,
                    ),
                    tree_metadata.ValueMetadataEntry(
                        value_type='jax.Array',
                        skip_deserialize=False,
                    ),
                ],
                'nested_empty_named_tuple': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
                'my_empty_chex': tree_metadata.ValueMetadataEntry(
                    value_type='Dict',
                    skip_deserialize=True,
                ),
            }
        },
        expected_nested_tree_metadata_with_rich_types={
            'named_tuple_with_nested_attrs': create_namedtuple(
                NamedTupleWithNestedAttributes,
                [
                    (
                        'nested_mu_nu',
                        create_namedtuple(
                            MuNu,
                            [
                                (
                                    'mu',
                                    tree_metadata.ValueMetadataEntry(
                                        value_type='jax.Array',
                                        skip_deserialize=False,
                                    ),
                                ),
                                (
                                    'nu',
                                    tree_metadata.ValueMetadataEntry(
                                        value_type='np.ndarray',
                                        skip_deserialize=False,
                                    ),
                                ),
                            ],
                        ),
                    ),
                    (
                        'nested_dict',
                        {
                            'a': tree_metadata.ValueMetadataEntry(
                                value_type='jax.Array',
                                skip_deserialize=False,
                            ),
                            'b': tree_metadata.ValueMetadataEntry(
                                value_type='np.ndarray',
                                skip_deserialize=False,
                            ),
                        },
                    ),
                    (
                        'nested_tuple',
                        (
                            tree_metadata.ValueMetadataEntry(
                                value_type='jax.Array',
                                skip_deserialize=False,
                            ),
                            tree_metadata.ValueMetadataEntry(
                                value_type='jax.Array',
                                skip_deserialize=False,
                            ),
                        ),
                    ),
                    (
                        'nested_empty_named_tuple',
                        tree_metadata.ValueMetadataEntry(
                            value_type='NamedTuple',
                            skip_deserialize=True,
                        ),
                    ),
                    (
                        'my_empty_chex',
                        tree_metadata.ValueMetadataEntry(
                            value_type='Dict',
                            skip_deserialize=True,
                        ),
                    ),
                ],
            )
        },
    ),
    TestPyTree(
        unique_name='my_empty_flax',
        provide_tree=lambda: {'my_empty_flax': MyEmptyFlax()},
        expected_nested_tree_metadata={
            'my_empty_flax': tree_metadata.ValueMetadataEntry(
                value_type='None',  # TODO: b/378905913 - Should be Dict.
                skip_deserialize=True,
            )
        },
    ),
    TestPyTree(
        unique_name='default_my_flax',
        provide_tree=lambda: {'default_my_flax': MyFlax()},
        expected_nested_tree_metadata={
            'default_my_flax': {
                'my_jax_array': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
                'my_nested_mapping': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
                'my_sequence': tree_metadata.ValueMetadataEntry(
                    value_type='None',
                    skip_deserialize=True,
                ),
            }
        },
    ),
    TestPyTree(
        unique_name='my_flax',
        provide_tree=lambda: {
            'my_flax': MyFlax(
                my_jax_array=jnp.arange(8),
                my_nested_mapping={
                    'a': None,
                    'b': jnp.arange(8),
                    'c': MyFlax(),
                    'd': MyFlax(
                        my_nested_mapping={
                            'e': MyEmptyChex(),
                        },
                        my_sequence=[
                            MyChex(my_jax_array=jnp.arange(8)),
                        ],
                    ),
                },
                my_sequence=(
                    None,
                    optax.EmptyState(),
                    MyEmptyFlax(),
                    MyEmptyChex(),
                    EmptyNamedTuple(),
                ),
            )
        },
        expected_nested_tree_metadata={
            'my_flax': {
                'my_jax_array': tree_metadata.ValueMetadataEntry(
                    value_type='jax.Array',
                    skip_deserialize=False,
                ),
                'my_nested_mapping': {
                    'a': tree_metadata.ValueMetadataEntry(
                        value_type='None',
                        skip_deserialize=True,
                    ),
                    'b': tree_metadata.ValueMetadataEntry(
                        value_type='jax.Array',
                        skip_deserialize=False,
                    ),
                    'c': {
                        'my_jax_array': tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                        'my_nested_mapping': tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                        'my_sequence': tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                    },
                    'd': {
                        'my_jax_array': tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                        'my_nested_mapping': {
                            'e': tree_metadata.ValueMetadataEntry(
                                value_type='Dict',
                                skip_deserialize=True,
                            ),
                        },
                        'my_sequence': [
                            {
                                'my_jax_array': (
                                    tree_metadata.ValueMetadataEntry(
                                        value_type='jax.Array',
                                        skip_deserialize=False,
                                    )
                                ),
                                'my_np_array': tree_metadata.ValueMetadataEntry(
                                    value_type='None',
                                    skip_deserialize=True,
                                ),
                            },
                        ],
                    },
                },
                'my_sequence': [
                    tree_metadata.ValueMetadataEntry(
                        value_type='None',
                        skip_deserialize=True,
                    ),
                    tree_metadata.ValueMetadataEntry(
                        value_type='None',
                        skip_deserialize=True,
                    ),
                    tree_metadata.ValueMetadataEntry(
                        value_type='None',
                        skip_deserialize=True,
                    ),
                    tree_metadata.ValueMetadataEntry(
                        value_type='Dict',
                        skip_deserialize=True,
                    ),
                    tree_metadata.ValueMetadataEntry(
                        value_type='None',
                        skip_deserialize=True,
                    ),
                ],
            }
        },
        expected_nested_tree_metadata_with_rich_types={
            'my_flax': {
                'my_jax_array': tree_metadata.ValueMetadataEntry(
                    value_type='jax.Array',
                    skip_deserialize=False,
                ),
                'my_nested_mapping': {
                    'a': tree_metadata.ValueMetadataEntry(
                        value_type='None',
                        skip_deserialize=True,
                    ),
                    'b': tree_metadata.ValueMetadataEntry(
                        value_type='jax.Array',
                        skip_deserialize=False,
                    ),
                    'c': {
                        'my_jax_array': tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                        'my_nested_mapping': tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                        'my_sequence': tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                    },
                    'd': {
                        'my_jax_array': tree_metadata.ValueMetadataEntry(
                            value_type='None',
                            skip_deserialize=True,
                        ),
                        'my_nested_mapping': {
                            'e': tree_metadata.ValueMetadataEntry(
                                value_type='Dict',
                                skip_deserialize=True,
                            ),
                        },
                        'my_sequence': [
                            {
                                'my_jax_array': (
                                    tree_metadata.ValueMetadataEntry(
                                        value_type='jax.Array',
                                        skip_deserialize=False,
                                    )
                                ),
                                'my_np_array': tree_metadata.ValueMetadataEntry(
                                    value_type='None',
                                    skip_deserialize=True,
                                ),
                            },
                        ],
                    },
                },
                'my_sequence': (
                    tree_metadata.ValueMetadataEntry(
                        value_type='None',
                        skip_deserialize=True,
                    ),
                    tree_metadata.ValueMetadataEntry(
                        value_type='NamedTuple',
                        skip_deserialize=True,
                    ),
                    tree_metadata.ValueMetadataEntry(
                        value_type='None',
                        skip_deserialize=True,
                    ),
                    tree_metadata.ValueMetadataEntry(
                        value_type='Dict',
                        skip_deserialize=True,
                    ),
                    tree_metadata.ValueMetadataEntry(
                        value_type='NamedTuple',
                        skip_deserialize=True,
                    ),
                ),
            }
        },
    ),
]

# Suitable for parameterized.named_parameters.
TEST_PYTREES_FOR_NAMED_PARAMETERS = [
    (test_pytree.unique_name, test_pytree) for test_pytree in TEST_PYTREES
]
