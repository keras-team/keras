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
# pytype: skip-file
from __future__ import annotations

from abc import abstractmethod
import dataclasses
import functools
import inspect
import typing as tp

from jax._src import checkify as checkify_lib

from flax.nnx import (
  extract,
  graph,
)
from flax.nnx.module import Module
from flax.nnx.proxy_caller import (
  CallableProxy,
  DelayedAccessor,
)
from flax.nnx.transforms import general
from flax.typing import MISSING, Leaf, Missing
import jax
import jax.core
import jax.stages

A = tp.TypeVar('A')
C = tp.TypeVar('C')
B = tp.TypeVar('B')
F = tp.TypeVar('F', bound=tp.Callable[..., tp.Any])
G = tp.TypeVar('G', bound=tp.Callable[..., tp.Any])
M = tp.TypeVar('M', bound=Module)
MA = tp.TypeVar('MA', bound=Module)
N = tp.TypeVar('N', bound=Module)
StrInt = tp.TypeVar('StrInt', str, int)
AxisName = tp.Hashable
Leaves = tp.List[Leaf]
Index = int


@tp.overload
def resolve_kwargs(
  fun: tp.Callable[..., tp.Any],
  args: tuple,
  kwargs: dict[str, tp.Any],
) -> tuple: ...
@tp.overload
def resolve_kwargs() -> tp.Callable[[F], F]: ...
def resolve_kwargs(
  fun: tp.Callable[..., tp.Any] | Missing = MISSING,
  args: tuple | Missing = MISSING,
  kwargs: dict[str, tp.Any] | Missing = MISSING,
) -> tuple | tp.Callable[[F], F]:
  if isinstance(fun, Missing):

    def resolve_kwargs_decorator(f):
      @functools.wraps(f)
      def resolve_kwargs_wrapper(*args, **kwargs):
        args = resolve_kwargs(f, args, kwargs)
        return f(*args)

      return resolve_kwargs_wrapper

    return resolve_kwargs_decorator  # type: ignore

  if isinstance(args, Missing):
    raise ValueError('args must be provided')
  if isinstance(kwargs, Missing):
    raise ValueError('kwargs must be provided')

  if isinstance(fun, functools.partial):
    # functools.partial should have an opaque signature.
    fun = lambda *args, **kwargs: None
  ba = inspect.signature(fun).bind(*args, **kwargs)
  ba.apply_defaults()
  if ba.kwargs:
    raise TypeError('keyword arguments could not be resolved to positions')
  else:
    return ba.args



class LiftedModule(tp.Generic[M], Module):  # type: ignore[ignored-abstractmethod]
  @abstractmethod
  def _call(self, accessor: DelayedAccessor, *args, **kwargs) -> tp.Any:
    pass

  @property
  @abstractmethod
  def _submodule(self) -> M:
    pass  # type: ignore[bad-return-type] # why pytype?

  def __call__(self, *args, **kwargs) -> tp.Any:
    return self.call(*args, **kwargs)  # type: ignore

  @property
  def call(self) -> tp.Any:
    module = self

    def check_and_call(accessor: DelayedAccessor, *args, **kwargs):
      return self._call(accessor, *args, **kwargs)

    proxy = CallableProxy(check_and_call)  # type: ignore[arg-type]

    while isinstance(module._submodule, LiftedModule):
      module = module._submodule
      proxy = proxy.call

    return proxy  # type: ignore


# -------------------------------
# simple transforms
# -------------------------------


def eval_shape(
  f: tp.Callable[..., A],
  *args: tp.Any,
  **kwargs: tp.Any,
) -> A:
  args, kwargs = extract.to_tree((args, kwargs))

  @functools.wraps(f)
  def _eval_shape_fn(*args, **kwargs):
    args, kwargs = extract.from_tree((args, kwargs))
    out = f(*args, **kwargs)
    return extract.to_tree(out)

  out = jax.eval_shape(_eval_shape_fn, *args, **kwargs)
  return extract.from_tree(out)
  """A "lifted" version of `jax.eval_shape <https://jax.readthedocs.io/en/latest/_autosummary/jax.eval_shape.html#jax.eval_shape>`_
    that can handle `flax.nnx.Module <https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html#flax.nnx.Module>`_
    / graph nodes as arguments.

  Similar to ``jax.eval_shape``, it computes the shape/dtype of a function `f` without
    performing any floating point operations (FLOPs) which can be expensive. This can be
    useful for performing shape inference, for example.
  """

@dataclasses.dataclass(eq=False)
class CheckifyFn:
  f: tp.Callable[..., tp.Any]

  def __post_init__(self):
    functools.update_wrapper(self, self.f)

  def __call__(self, *pure_args, **pure_kwargs):
    args, kwargs = extract.from_tree(
      (pure_args, pure_kwargs), ctxtag='checkify', is_inner=True
    )
    out = self.f(*args, **kwargs)

    args_out, kwargs_out = extract.clear_non_graph_nodes((args, kwargs))
    pure_args_out, pure_kwargs_out, pure_out = extract.to_tree(
      (args, kwargs, out), ctxtag='checkify'
    )
    return pure_args_out, pure_kwargs_out, pure_out

def checkify(
  f: tp.Callable[..., checkify_lib.Out],
  errors: frozenset[type[checkify_lib.JaxException]] = checkify_lib.user_checks,  # type: ignore
) -> tp.Callable[..., tuple[checkify_lib.Error, checkify_lib.Out]]:
  """Reference-aware version of `jax.experimental.checkify
  <https://flax.readthedocs.io/en/latest/nnx_basics.html#the-flax-functional-api>`_.

  Example::

    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jax.experimental import checkify
    >>> import dataclasses
    >>> from flax import nnx
    ...
    >>> class Foo(nnx.Module):
    ...   def __init__(self, a):
    ...     self.a = nnx.Param(a)
    ...
    >>> @nnx.jit
    ... def f(m):
    ...   y = jnp.sin(m.a.value) # error
    ...   return m.a + y
    ...
    >>> m = Foo(a=jnp.inf)
    >>> err, out = nnx.checkify(f, errors=checkify.float_checks)(m)
    >>> # err.throw()
    >>> print(err)
    Error(nan generated by primitive: sin.)
  """
  checkify_fn = checkify_lib.checkify(CheckifyFn(f), errors)

  @functools.wraps(f)
  @graph.update_context('checkify')
  def jit_wrapper(*args, **kwargs):
    pure_args, pure_kwargs = extract.to_tree(
      (args, kwargs),
      ctxtag='checkify',
    )
    error, (pure_args_out, pure_kwargs_out, pure_out) = checkify_fn(
      *pure_args, **pure_kwargs
    )

    args_out, kwargs_out, out = extract.from_tree(
      (pure_args_out, pure_kwargs_out, pure_out),
      ctxtag='checkify',
      is_inner=False,
    )

    return error, out

  return jit_wrapper  # type: ignore


@general.split_inputs(ctxtag='cond')
def cond(
  pred,
  true_fun: tp.Callable[..., A],
  false_fun: tp.Callable[..., A],
  *operands,
  **kwargs,
) -> A:
  return jax.lax.cond(
    pred,
    general.merge_inputs(true_fun, ctxtag='cond'),
    general.merge_inputs(false_fun, ctxtag='cond'),
    *operands,
    **kwargs,
  )


@general.split_inputs(ctxtag='switch')
def switch(
  index,
  branches: tp.Sequence[tp.Callable[..., A]],
  *operands,
) -> A:
  return jax.lax.switch(
    index,
    [general.merge_inputs(f, ctxtag='switch') for f in branches],
    *operands,
  )

