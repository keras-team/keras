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

"""Common utilty functions used in data-parallel Flax examples.

This module is a historical grab-bag of utility functions primarily concerned
with helping write pmap-based data-parallel training loops.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax


def shard(xs):
  """Helper for pmap to shard a pytree of arrays by local_device_count.

  Args:
    xs: a pytree of arrays.
  Returns:
    A matching pytree with arrays' leading dimensions sharded by the
    local device count.
  """
  local_device_count = jax.local_device_count()
  return jax.tree_util.tree_map(
    lambda x: x.reshape((local_device_count, -1) + x.shape[1:]), xs
  )


def shard_prng_key(prng_key):
  """Helper to shard (aka split) a PRNGKey for use with pmap'd functions.

  PRNG keys can be used at train time to drive stochastic modules
  e.g. Dropout. We would like a different PRNG key for each local
  device so that we end up with different random numbers on each one,
  hence we split our PRNG key.

  Args:
    prng_key: JAX PRNGKey
  Returns:
    A new array of PRNGKeys with leading dimension equal to local device count.
  """
  return jax.random.split(prng_key, num=jax.local_device_count())


def stack_forest(forest):
  """Helper function to stack the leaves of a sequence of pytrees.

  Args:
    forest: a sequence of pytrees (e.g tuple or list) of matching structure
      whose leaves are arrays with individually matching shapes.
  Returns:
    A single pytree of the same structure whose leaves are individually
      stacked arrays.
  """
  stack_args = lambda *args: np.stack(args)
  return jax.tree_util.tree_map(stack_args, *forest)


def get_metrics(device_metrics):
  """Helper utility for pmap, gathering replicated timeseries metric data.

  Args:
   device_metrics: replicated, device-resident pytree of metric data,
     whose leaves are presumed to be a sequence of arrays recorded over time.
  Returns:
   A pytree of unreplicated, host-resident, stacked-over-time arrays useful for
   computing host-local statistics and logging.
  """
  # We select the first element of x in order to get a single copy of a
  # device-replicated metric.
  device_metrics = jax.tree_util.tree_map(lambda x: x[0], device_metrics)
  metrics_np = jax.device_get(device_metrics)
  return stack_forest(metrics_np)


def onehot(labels, num_classes, on_value=1.0, off_value=0.0):
  """Create a dense one-hot version of an indexed array.

  NB: consider using the more standard ``jax.nn.one_hot`` instead.

  Args:
    labels: an n-dim JAX array whose last dimension contains integer indices.
    num_classes: the maximum possible index.
    on_value: the "on" value for the one-hot array, defaults to 1.0.
    off_value: the "off" value for the one-hot array, defaults to 0.0.
  Returns:
    A (n+1)-dim array whose last dimension contains one-hot vectors of length
    num_classes.
  """
  x = labels[..., None] == jnp.arange(num_classes).reshape(
    (1,) * labels.ndim + (-1,)
  )
  x = lax.select(x, jnp.full(x.shape, on_value), jnp.full(x.shape, off_value))
  return x.astype(jnp.float32)
