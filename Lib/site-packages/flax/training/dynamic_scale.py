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

"""Dynamic loss scaling for mixed precision gradients."""

import functools
from typing import Any, NamedTuple
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax import lax

from flax import struct
from flax.typing import Array


class DynamicScaleResult(NamedTuple):
  dynamic_scale: 'DynamicScale'
  finite: Array
  aux: Any
  grad: Any


class DynamicScale(struct.PyTreeNode):
  """Dynamic loss scaling for mixed precision gradients.

  For many models gradient computations in float16 will result in numerical
  issues because small/large gradients being flushed to zero/infinity.
  Dynamic loss scaling is an algorithm that aims to find the largest scalar
  multiple for which the gradient does not overflow. This way the risk of
  underflow is minimized.

  the `value_and_grad` method mimicks `jax.value_and_grad`. Beside the loss
  and gradients it also ouputs and updated `DynamicScale` instance with the
  current loss scale factor. This method also returns a boolean value indicating
  whether the gradients are finite.

  Example::

    from flax.training.dynamic_scale import DynamicScale

    def loss_fn(p):
      return jnp.asarray(p, jnp.float16) ** 2
    p = jnp.array(1., jnp.float32)

    dyn_scale = DynamicScale(growth_interval=10)
    compute_grad = jax.jit(lambda ds, p: ds.value_and_grad(loss_fn)(p))
    for _ in range(100):
      dyn_scale, is_fin, loss, grad = compute_grad(dyn_scale, p)
      p += jnp.where(is_fin, 0.01 * grad, 0.)
      print(loss)

  Jax currently cannot execute conditionals efficiently on GPUs therefore we
  selectively ignore the gradient update using `jax.numpy.where` in case of
  non-finite gradients.

  Attributes:
    growth_factor: how much to grow the scalar after a period of finite
      gradients (default: 2.).
    backoff_factor: how much to shrink the scalar after a non-finite gradient
      (default: 0.5).
    growth_interval: after how many steps of finite gradients the scale should
      be increased (default: 2000).
    fin_steps: indicates how many gradient steps in a row have been finite.
    scale: the current scale by which the loss is multiplied.
    minimum_scale: the minimum value that the scale can take (default: the
      smallest positive number representable in floating point).
  """

  growth_factor: float = struct.field(pytree_node=False, default=2.0)
  backoff_factor: float = struct.field(pytree_node=False, default=0.5)
  growth_interval: int = struct.field(pytree_node=False, default=2000)
  fin_steps: int = 0
  scale: float = 65536.0
  minimum_scale: float | None = struct.field(
    pytree_node=False, default=jnp.finfo(jnp.float32).tiny
  )

  def value_and_grad(
    self,
    fun: Callable[..., Any],
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    axis_name: str | None = None,
  ) -> Callable[..., DynamicScaleResult]:
    """Wrapper around `jax.value_and_grad`.

    Args:
      fun: Function to be differentiated. Its arguments at positions specified
        by ``argnums`` should be arrays, scalars, or standard Python containers.
        It should return a scalar (which includes arrays with shape ``()`` but
        not arrays with shape ``(1,)`` etc.)
      argnums: Optional, integer or sequence of integers. Specifies which
        positional argument(s) to differentiate with respect to (default 0).
      has_aux: Optional, bool. Indicates whether ``fun`` returns a pair where
        the first element is considered the output of the mathematical function
        to be differentiated and the second element is auxiliary data. Default
        False.
      axis_name: If an axis is given the gradients will be averaged across
        replicas (default: None). Note, this is only used for pmap and shard
        map. For SPMD jit, you do not need to manually synchronize. Just make
        sure that the axes are correctly annotated and XLA:SPMD will insert the
        necessary collectives.

    Returns:
      A function that takes the same arguments as `fun` and
      returns a DynamicScaleResult
    """

    @functools.wraps(fun)
    def loss_wrapper(*args):
      aux = fun(*args)
      if has_aux:
        return (self.scale * aux[0], aux[1])
      else:
        return self.scale * aux

    grad_fn = jax.value_and_grad(loss_wrapper, argnums, has_aux)

    def grad_fn_wrapper(*args):
      aux, grad = grad_fn(*args)
      aux = (aux[0] / self.scale, aux[1]) if has_aux else aux / self.scale

      grad = jax.tree_util.tree_map(
        lambda g: jnp.asarray(g, jnp.float32) / self.scale, grad
      )
      if axis_name is not None:
        grad = lax.pmean(grad, axis_name)

      finite = jnp.array(True)
      for g in jax.tree_util.tree_leaves(grad):
        finite &= jnp.all(lax.is_finite(g))

      grow = self.fin_steps == self.growth_interval
      fin_scale = jnp.where(
        grow & finite,
        jnp.minimum(
          self.scale * self.growth_factor, jnp.finfo(jnp.float32).max
        ),
        self.scale,
      )
      inf_scale = self.scale * self.backoff_factor
      if self.minimum_scale is not None:
        inf_scale = jnp.maximum(inf_scale, self.minimum_scale)
      new_scale = jnp.where(finite, fin_scale, inf_scale)
      new_fin_steps = jnp.where(grow | (~finite), 0, self.fin_steps + 1)

      new_self = self.replace(fin_steps=new_fin_steps, scale=new_scale)
      return DynamicScaleResult(new_self, finite, aux, grad)

    return grad_fn_wrapper
