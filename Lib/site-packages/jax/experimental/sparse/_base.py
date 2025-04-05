# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Base JAX Sparse object."""
import abc
from collections.abc import Sequence
import math

import jax
from jax._src import core
from jax._src import util
from jax._src.typing import Array


class JAXSparse(util.StrictABC):
  """Base class for high-level JAX sparse objects."""
  data: jax.Array
  shape: tuple[int, ...]
  nse: property
  dtype: property

  # Ignore type because of https://github.com/python/mypy/issues/4266.
  __hash__ = None  # type: ignore

  def __len__(self):
    return self.shape[0]

  @property
  def size(self) -> int:
    return math.prod(self.shape)

  @property
  def ndim(self) -> int:
    return len(self.shape)

  def __init__(self, args: tuple[Array, ...], *, shape: Sequence[int]):
    self.shape = core.canonicalize_shape(shape)

  def __repr__(self):
    name = self.__class__.__name__
    try:
      nse = self.nse
      dtype = self.dtype
      shape = list(self.shape)
    except:
      repr_ = f"{name}(<invalid>)"
    else:
      repr_ = f"{name}({dtype}{shape}, {nse=})"
    if isinstance(self.data, core.Tracer):
      repr_ = f"{type(self.data).__name__}[{repr_}]"
    return repr_

  @abc.abstractmethod
  def tree_flatten(self):
    ...

  @classmethod
  @abc.abstractmethod
  def tree_unflatten(cls, aux_data, children):
    ...

  @abc.abstractmethod
  def transpose(self, axes=None):
    ...

  @property
  def T(self):
    return self.transpose()

  def block_until_ready(self):
    for arg in self.tree_flatten()[0]:
      arg.block_until_ready()
    return self

  # Not abstract methods because not all sparse classes implement them

  def sum(self, *args, **kwargs):
    raise NotImplementedError(f"{self.__class__}.sum")

  def __neg__(self):
    raise NotImplementedError(f"{self.__class__}.__neg__")

  def __pos__(self):
    raise NotImplementedError(f"{self.__class__}.__pos__")

  def __matmul__(self, other):
    raise NotImplementedError(f"{self.__class__}.__matmul__")

  def __rmatmul__(self, other):
    raise NotImplementedError(f"{self.__class__}.__rmatmul__")

  def __mul__(self, other):
    raise NotImplementedError(f"{self.__class__}.__mul__")

  def __rmul__(self, other):
    raise NotImplementedError(f"{self.__class__}.__rmul__")

  def __add__(self, other):
    raise NotImplementedError(f"{self.__class__}.__add__")

  def __radd__(self, other):
    raise NotImplementedError(f"{self.__class__}.__radd__")

  def __sub__(self, other):
    raise NotImplementedError(f"{self.__class__}.__sub__")

  def __rsub__(self, other):
    raise NotImplementedError(f"{self.__class__}.__rsub__")

  def __getitem__(self, item):
    raise NotImplementedError(f"{self.__class__}.__getitem__")
