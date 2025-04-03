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
from __future__ import annotations

import functools
import typing as tp

import jax
import jax.numpy as jnp

from flax import struct
from flax.nnx import graph
from flax.nnx import statelib
from flax.nnx.statelib import State
from flax.nnx.variablelib import Variable
from flax.nnx import filterlib
from flax.nnx.filterlib import All
from flax.nnx.object import Object
from flax.typing import MISSING, Missing

F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
Counts = list[int]
AxesValue = tp.Union[int, None]
SplitPattern = tp.Union[AxesValue, tuple[AxesValue, ...]]


class RngState(Variable[jax.Array]):
  tag: str


class RngCount(RngState): ...


class RngKey(RngState): ...


NotKey = filterlib.All(RngState, filterlib.Not(RngKey))


class RngStream(Object):
  def __init__(
    self,
    tag: str,
    key: jax.Array,
    count: jax.Array,
  ):
    if not isinstance(key, jax.Array):
      raise TypeError(f'key must be a jax.Array, got {type(key)}')

    self.key = RngKey(key, tag=tag)
    self.count = RngCount(count, tag=tag)

  def __call__(self) -> jax.Array:
    self._check_valid_context(
      lambda: 'Cannot call RngStream from a different trace level'
    )
    key = jax.random.fold_in(self.key.value, self.count.value)
    self.count.value += 1
    return key


RngValue = tp.Union[int, jax.Array]
RngDict = tp.Union[
  tp.Mapping[str, int],
  tp.Mapping[str, jax.Array],
  tp.Mapping[str, RngValue],
]


class Rngs(Object):
  """NNX rng container class. To instantiate the ``Rngs``, pass
  in an integer, specifying the starting seed. ``Rngs`` can have
  different "streams", allowing the user to generate different
  rng keys. For example, to generate a key for the ``params``
  and ``dropout`` stream::

    >>> from flax import nnx
    >>> import jax, jax.numpy as jnp

    >>> rng1 = nnx.Rngs(0, params=1)
    >>> rng2 = nnx.Rngs(0)

    >>> assert rng1.params() != rng2.dropout()

  Because we passed in ``params=1``, the starting seed for
  ``params`` is ``1``, whereas the starting seed for ``dropout``
  defaults to the ``0`` we passed in, since we didn't specify
  a seed for ``dropout``. If we didn't specify a seed for ``params``,
  then both streams will default to using the ``0`` we passed in::

    >>> rng1 = nnx.Rngs(0)
    >>> rng2 = nnx.Rngs(0)

    >>> assert rng1.params() == rng2.dropout()

  The ``Rngs`` container class contains a separate counter for
  each stream. Every time the stream is called to generate a new rng
  key, the counter increments by ``1``. To generate a new rng key,
  we fold in the counter value for the current rng stream into its
  corresponding starting seed. If we try to generate an rng key for
  a stream we did not specify on instantiation, then the ``default``
  stream is used (i.e. the first positional argument passed to ``Rngs``
  during instantiation is the ``default`` starting seed)::

    >>> rng1 = nnx.Rngs(100, params=42)
    >>> # `params` stream starting seed is 42, counter is 0
    >>> assert rng1.params() == jax.random.fold_in(jax.random.key(42), 0)
    >>> # `dropout` stream starting seed is defaulted to 100, counter is 0
    >>> assert rng1.dropout() == jax.random.fold_in(jax.random.key(100), 0)
    >>> # empty stream starting seed is defaulted to 100, counter is 1
    >>> assert rng1() == jax.random.fold_in(jax.random.key(100), 1)
    >>> # `params` stream starting seed is 42, counter is 1
    >>> assert rng1.params() == jax.random.fold_in(jax.random.key(42), 1)

  Let's see an example of using ``Rngs`` in a :class:`Module` and
  verifying the output by manually threading the ``Rngs``::

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     # Linear uses the `params` stream twice for kernel and bias
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...     # Dropout uses the `dropout` stream once
    ...     self.dropout = nnx.Dropout(0.5, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.dropout(self.linear(x))

    >>> def assert_same(x, rng_seed, **rng_kwargs):
    ...   model = Model(rngs=nnx.Rngs(rng_seed, **rng_kwargs))
    ...   out = model(x)
    ...
    ...   # manual forward propagation
    ...   rngs = nnx.Rngs(rng_seed, **rng_kwargs)
    ...   kernel = nnx.initializers.lecun_normal()(rngs.params(), (2, 3))
    ...   assert (model.linear.kernel.value==kernel).all()
    ...   bias = nnx.initializers.zeros_init()(rngs.params(), (3,))
    ...   assert (model.linear.bias.value==bias).all()
    ...   mask = jax.random.bernoulli(rngs.dropout(), p=0.5, shape=(1, 3))
    ...   # dropout scales the output proportional to the dropout rate
    ...   manual_out = mask * (jnp.dot(x, kernel) + bias) / 0.5
    ...   assert (out == manual_out).all()

    >>> x = jnp.ones((1, 2))
    >>> assert_same(x, 0)
    >>> assert_same(x, 0, params=1)
    >>> assert_same(x, 0, params=1, dropout=2)
  """

  def __init__(
    self,
    default: RngValue | RngDict | None = None,
    /,
    **rngs: RngValue,
  ):
    """
    Args:
      default: the starting seed for the ``default`` stream. Any
        key generated from a stream that isn't specified in the
        ``**rngs`` key-word arguments will default to using this
        starting seed.
      **rngs: optional key-word arguments to specify starting
        seeds for different rng streams. The key-word is the
        stream name and its value is the corresponding starting
        seed for that stream.
    """
    if default is not None:
      if isinstance(default, tp.Mapping):
        rngs = {**default, **rngs}
      else:
        rngs['default'] = default

    for name, value in rngs.items():
      if isinstance(value, int):
        key = jax.random.key(value)
      elif isinstance(value, jax.Array):
        if value.dtype == jnp.uint32:
          key = jax.random.wrap_key_data(value)
        else:
          key = value
      else:
        raise ValueError(f'Invalid rng value: {value}')

      stream = RngStream(
        tag=name,
        key=key,
        count=jnp.zeros(key.shape, dtype=jnp.uint32),
      )
      setattr(self, name, stream)

  def _get_stream(self, name: str, error_type: type[Exception]) -> RngStream:
    rngs_vars = vars(self)
    if name not in rngs_vars:
      if 'default' not in rngs_vars:
        raise error_type(f"No RNG named {name!r} or 'default' found in Rngs.")
      stream = rngs_vars['default']
    else:
      stream = rngs_vars[name]

    return stream

  def __getitem__(self, name: str):
    return self._get_stream(name, KeyError)

  def __getattr__(self, name: str):
    return self._get_stream(name, AttributeError)

  def __call__(self):
    return self.default()

  def __iter__(self) -> tp.Iterator[str]:
    for name in vars(self):
      if name != '_object__state':
        yield name

  def __len__(self) -> int:
    return len(vars(self)) - 1

  def __contains__(self, name: tp.Any) -> bool:
    return name in vars(self)

  # pickle support
  def __getstate__(self):
    return vars(self).copy()

  def __setstate__(self, state):
    vars(self).update(state)

  def items(self):
    for name in self:
      yield name, self[name]


class ForkStates(tp.NamedTuple):
  split_keys: State
  split_counts: State
  broadcast_keys: State
  broadcast_counts: State


def fork(
  state: State,
  split_filter: filterlib.Filter,
  split_pattern: SplitPattern,
) -> ForkStates:
  if split_pattern is None:
    raise RuntimeError('Split pattern cannot be None, this is a bug.')

  num_splits: int | tuple[int, ...]
  if isinstance(split_pattern, int):
    num_splits = split_pattern
  else:
    num_splits = tuple(x if x is not None else 1 for x in split_pattern)

  split_keys, split_counts, broadcast_keys, broadcast_counts = (
    statelib.split_state(
      state,
      All(split_filter, RngKey),
      All(split_filter, RngCount),
      RngKey,
      RngCount,
    )
  )

  def split_key(key: tp.Any) -> jax.Array:
    if not isinstance(key, jax.Array):
      raise TypeError(f'key must be a jax.Array, got {type(key)}')

    return jax.random.split(key, num_splits)

  split_keys = jax.tree.map(split_key, split_keys)

  return ForkStates(split_keys, split_counts, broadcast_keys, broadcast_counts)


StreamBackup = (
  tuple[RngStream, jax.Array, jax.Array] | tuple[RngStream, jax.Array]
)


class SplitBackups(struct.PyTreeNode, tp.Iterable[StreamBackup]):
  backups: list[StreamBackup]

  def __iter__(self) -> tp.Iterator[StreamBackup]:
    return iter(self.backups)

  def __enter__(self):
    return self

  def __exit__(self, *args):
    restore_rngs(self)


@tp.overload
def split_rngs(
  node: tp.Any,
  /,
  *,
  splits: int | tuple[int, ...],
  only: filterlib.Filter = ...,
  squeeze: bool = False,
) -> SplitBackups: ...
@tp.overload
def split_rngs(
  *,
  splits: int | tuple[int, ...],
  only: filterlib.Filter = ...,
  squeeze: bool = False,
) -> tp.Callable[[F], F]: ...
def split_rngs(
  node: tp.Any = MISSING,
  /,
  *,
  splits: int | tuple[int, ...],
  only: filterlib.Filter = ...,
  squeeze: bool = False,
) -> SplitBackups | tp.Callable[[F], F]:
  """Splits the (nested) Rng states of the given node.

  Args:
    node: the base node containing the rng states to split.
    splits: an integer or tuple of integers specifying the
      shape of the split rng keys.
    only: a Filter selecting which rng states to split.

  Returns:
    A SplitBackups iterable if ``node`` is provided, otherwise a
    decorator that splits the rng states of the inputs to the
    decorated function.

  Example::

    >>> from flax import nnx
    ...
    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> _ = nnx.split_rngs(rngs, splits=5)
    >>> rngs.params.key.shape, rngs.dropout.key.shape
    ((5,), (5,))

    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> _ = nnx.split_rngs(rngs, splits=(2, 5))
    >>> rngs.params.key.shape, rngs.dropout.key.shape
    ((2, 5), (2, 5))


    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> _ = nnx.split_rngs(rngs, splits=5, only='params')
    >>> rngs.params.key.shape, rngs.dropout.key.shape
    ((5,), ())

  Once split, random state can be used with transforms like :func:`nnx.vmap`::

    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...     self.dropout = nnx.Dropout(0.5, rngs=rngs)
    ...
    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> _ = nnx.split_rngs(rngs, splits=5, only='params')
    ...
    >>> state_axes = nnx.StateAxes({(nnx.Param, 'params'): 0, ...: None})
    ...
    >>> @nnx.vmap(in_axes=(state_axes,), out_axes=state_axes)
    ... def create_model(rngs):
    ...   return Model(rngs)
    ...
    >>> model = create_model(rngs)
    >>> model.dropout.rngs.params.key.shape
    (5,)

  ``split_rngs`` returns a SplitBackups object that can be used to restore the
  original unsplit rng states using :func:`nnx.restore_rngs`, this is useful
  when you only want to split the rng states temporarily::

    >>> rngs = nnx.Rngs(params=0, dropout=1)
    ...
    >>> backups = nnx.split_rngs(rngs, splits=5, only='params')
    >>> model = create_model(rngs)
    >>> nnx.restore_rngs(backups)
    ...
    >>> model.dropout.rngs.params.key.shape
    ()

  SplitBackups can also be used as a context manager to automatically restore
  the rng states when exiting the context::

    >>> rngs = nnx.Rngs(params=0, dropout=1)
    ...
    >>> with nnx.split_rngs(rngs, splits=5, only='params'):
    ...   model = create_model(rngs)
    ...
    >>> model.dropout.rngs.params.key.shape
    ()

    >>> state_axes = nnx.StateAxes({(nnx.Param, 'params'): 0, ...: None})
    ...
    >>> @nnx.split_rngs(splits=5, only='params')
    ... @nnx.vmap(in_axes=(state_axes,), out_axes=state_axes)
    ... def create_model(rngs):
    ...   return Model(rngs)
    ...
    >>> rngs = nnx.Rngs(params=0, dropout=1)
    >>> model = create_model(rngs)
    >>> model.dropout.rngs.params.key.shape
    ()


  """
  if isinstance(node, Missing):

    def split_rngs_decorator(f: F) -> F:
      @functools.wraps(f)
      def split_rngs_wrapper(*args, **kwargs):
        with split_rngs(
          (args, kwargs), splits=splits, only=only, squeeze=squeeze
        ):
          return f(*args, **kwargs)

      return tp.cast(F, split_rngs_wrapper)

    return split_rngs_decorator  # type: ignore[bad-return-type]

  if squeeze and splits != 1:
    raise ValueError('squeeze=True is only supported for splits=1')

  predicate = filterlib.to_predicate(only)
  backups: list[StreamBackup] = []
  for path, stream in graph.iter_graph(node):
    if (
      isinstance(stream, RngStream)
      and predicate((*path, 'key'), stream.key)
      and predicate((*path, 'count'), stream.count)
    ):
      key = stream()
      backups.append((stream, stream.key.value, stream.count.value))
      key = jax.random.split(key, splits)
      if squeeze:
        key = key[0]
      stream.key.value = key
      if squeeze:
        counts_shape = stream.count.shape
      elif isinstance(splits, int):
        counts_shape = (splits, *stream.count.shape)
      else:
        counts_shape = (*splits, *stream.count.shape)
      stream.count.value = jnp.zeros(counts_shape, dtype=jnp.uint32)

  return SplitBackups(backups)


def backup_keys(node: tp.Any, /):
  backups: list[StreamBackup] = []
  for _, stream in graph.iter_graph(node):
    if isinstance(stream, RngStream):
      backups.append((stream, stream.key.value))
  return backups


def reseed(node, /, **stream_keys: RngValue):
  """Update the keys of the specified RNG streams with new keys.

  Args:
    node: the node to reseed the RNG streams in.
    **stream_keys: a mapping of stream names to new keys. The keys can be
      either integers or jax arrays. If an integer is passed in, then the
      key will be generated using ``jax.random.key``.

  Raises:
    ValueError: if an existing stream key is not a scalar.

  Example::

    >>> from flax import nnx
    >>> import jax.numpy as jnp
    ...
    >>> class Model(nnx.Module):
    ...   def __init__(self, rngs):
    ...     self.linear = nnx.Linear(2, 3, rngs=rngs)
    ...     self.dropout = nnx.Dropout(0.5, rngs=rngs)
    ...   def __call__(self, x):
    ...     return self.dropout(self.linear(x))
    ...
    >>> model = Model(nnx.Rngs(params=0, dropout=42))
    >>> x = jnp.ones((1, 2))
    ...
    >>> y1 = model(x)
    ...
    >>> # reset the ``dropout`` stream key to 42
    >>> nnx.reseed(model, dropout=42)
    >>> y2 = model(x)
    ...
    >>> jnp.allclose(y1, y2)
    Array(True, dtype=bool)
  """
  for _, stream in graph.iter_graph(node):
    if isinstance(stream, RngStream):
      if stream.key.tag in stream_keys:
        if stream.key.shape != ():
          raise ValueError(
            f'Cannot reseed stream {stream.key.tag!r} with a non-scalar key, '
            f' found key with shape {stream.key.shape}.'
          )
        key = stream_keys[stream.key.tag]
        if isinstance(key, int):
          key = jax.random.key(key)
        stream.key.value = key
        stream.count.value = jnp.array(0, dtype=jnp.uint32)


def restore_rngs(backups: tp.Iterable[StreamBackup], /):
  for backup in backups:
    stream = backup[0]
    stream.key.value = backup[1]  # key
    if len(backup) == 3:
      stream.count.value = backup[2]  # count
