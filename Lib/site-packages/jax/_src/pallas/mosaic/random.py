# Copyright 2024 The JAX Authors.
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

from collections.abc import Callable

import functools
import jax
from jax import numpy as jnp
from jax import random as jax_api_random
from jax._src import blocked_sampler
from jax._src import typing
from jax._src.pallas.mosaic.primitives import prng_seed
from jax._src.pallas.mosaic.primitives import prng_random_bits
from jax._src.pallas import primitives
from jax._src import prng as jax_prng


Shape = jax_prng.Shape
SampleFnType = blocked_sampler.SampleFn
KeylessSampleFnType = Callable[..., jax.Array]

set_seed = prng_seed

FOLD_IN_ROUNDS = 128


def to_pallas_key(key: jax_prng.PRNGKeyArray) -> jax_prng.PRNGKeyArray:
  """Helper function for converting non-Pallas PRNG keys into Pallas keys."""
  generate_key = functools.partial(
      jax.random.bits, shape=tpu_key_impl.key_shape, dtype=jnp.uint32
  )
  if len(key.shape) == 0:
    pallas_key_data = generate_key(key)
  else:
    pallas_key_data = (jax.vmap(generate_key))(key)
  return jax_api_random.wrap_key_data(pallas_key_data, impl="pallas_tpu")


def is_pallas_impl(impl: jax_prng.PRNGImpl) -> bool:
  """Returns True if the PRNGImpl is a Pallas-specific implementation."""
  return impl == tpu_key_impl or impl == tpu_internal_stateful_impl


def _seed_func(seed: jnp.int32):
  seed_data = jnp.zeros(tpu_key_impl.key_shape, dtype=jnp.int32)
  return (seed_data + seed).astype(jnp.uint32)

def _random_bits(key: typing.Array, bit_width: int, shape: Shape):
  if bit_width != 32:
    raise ValueError("Bit width must be 32")
  prng_seed(key)
  return prng_random_bits(shape)

def _fold_in(key: jax_prng.PRNGKeyArray, data: typing.Array):
  # Roughly, we compute the new key as follows:
  # new_key = random_bits(data)[..., 127] ^ random_bits(old_key)[..., 127]
  # Because the TPU generates random numbers in (8, 128) blocks at once, we
  # can generate that many values without additional cost which will reduce
  # correlation between the old and new keys.
  key_shape = tpu_key_impl.key_shape

  prng_seed(data)
  data_bits = prng_random_bits(
      key_shape + (FOLD_IN_ROUNDS,)).astype(jnp.uint32)
  prng_seed(key)
  key_bits = prng_random_bits(
      key_shape + (FOLD_IN_ROUNDS,)).astype(jnp.uint32)

  mixed = key_bits[..., FOLD_IN_ROUNDS-1] ^ data_bits[..., FOLD_IN_ROUNDS-1]
  assert mixed.shape == key_shape
  return jax.random.wrap_key_data(mixed, impl="pallas_tpu")

def _split(key: typing.Array, shape: Shape):
  del key, shape
  raise NotImplementedError()

tpu_key_impl = jax_prng.PRNGImpl(
   # Pallas currently only supports 2D+ windows, so set the key_shape
   # to be 2D to have better compatibility with setting BlockSpecs.
   key_shape=(1, 1),
   seed=_seed_func,
   split=_split,
   random_bits=_random_bits,
   fold_in=_fold_in,
   name="pallas_tpu",
   tag="pl"
)
jax_prng.register_prng(tpu_key_impl)

# Implementation of the stateful Pallas PRNG API.
# Users should set the seed using the `set_seed` function,
# and call the appropriate stateful sampling functions.
# The actual key impl should never be used. The impl
# serves as internal boilerplate code because JAX's existing
# random functions expect a key as an argument, and
# the keys are only generated as part of unused arguments.

def _pl_stateful_seed_func(seed: jnp.int32):
  del seed
  # Unused. Return the correct shape and dtype.
  return jnp.empty((), dtype=jnp.int32)

def _pl_stateful_random_bits(key: typing.Array, bit_width: int, shape: Shape):
  del key
  assert bit_width == 32, "Bit width must be 32"
  return prng_random_bits(shape)

def _pl_stateful_fold_in(key: typing.Array, data: typing.Array):
  del key, data
  raise NotImplementedError()

def _pl_stateful_split(key: typing.Array, shape: Shape):
  del key, shape
  raise NotImplementedError()


tpu_internal_stateful_impl = jax_prng.PRNGImpl(
   key_shape=(),
   seed=_pl_stateful_seed_func,
   split=_pl_stateful_split,
   random_bits=_pl_stateful_random_bits,
   fold_in=_pl_stateful_fold_in,
   name="_pallas_internal_stateful",
   tag="_pl_stateful"
)
jax_prng.register_prng(tpu_internal_stateful_impl)

def _make_stateful_sampler(sampler: SampleFnType) -> KeylessSampleFnType:
  """Converts a jax.random sampling function to a stateful version.

  Args:
    sampler: A sampling function that consumes a key and returns
      random samples.

  Returns:
    A stateful sampling function with the key argument removed.
  """
  def new_sampler(*args, **kwargs):
    # Pass in a placeholder key into the sampling function.
    # The key is ignored by the stateful random_bits function, but all jax
    # sampling functions expect a key as input so we must pass one in here.
    placeholder_key = jax_api_random.key(0, impl=tpu_internal_stateful_impl)
    return sampler(placeholder_key, *args, **kwargs)
  # Remove key argument from docstring.
  if sampler.__doc__:
    doc_lines = filter(
        lambda line: "key:" not in line, sampler.__doc__.split("\n"))
    new_sampler.__doc__ = "\n".join(doc_lines)
  return new_sampler

bits = _make_stateful_sampler(jax_api_random.bits)  # type: ignore
uniform = _make_stateful_sampler(jax_api_random.uniform)  # type: ignore
bernoulli = _make_stateful_sampler(jax_api_random.bernoulli)  # type: ignore


def sample_block(sampler_fn: SampleFnType,
                 global_key: jax_prng.PRNGKeyArray,
                 block_size: Shape,
                 tile_size: Shape,
                 total_size: Shape,
                 block_index: tuple[typing.ArrayLike, ...] | None = None,
                 **kwargs) -> jax.Array:
  """Samples a block of random values with invariance guarantees.

  `sample_block` allows the sampling of identical blocks of random values
  across kernels with different block shapes and iteration orders. Each call
  to `sample_block` returns a `block_size`-shaped array of random samples
  corresponding to the `block_index`.

  `tile_size` should be chosen such that it is a divisor to all block sizes
  one needs to be invariant to. The larger the `tile_size`, the more
  efficient the sampling process wil be and therefore the best choice is
  typically the greatest common divisor between all possible block sizes.

  Args:
    sampler_fn: A sampling function that consumes a key and returns
      random samples.
    global_key: The global key to use for sampling.
    block_size: The shape of an individual block.
    tile_size: The shape of a `tile`, which is the smallest unit at
      which samples are generated. This should be selected to be a divisor
      of all block sizes one needs to be invariant to.
    total_size: The total size of the array to sample.
    block_index: The index denoting which block to generate keys for. Defaults
      to the program_id for each block axis.
    **kwargs: Additional arguments to pass to the sampler_fn.

  Returns:
    A `block_size` shaped array of samples for the current block corresponding
    to `block_index`.
  """
  if len(block_size) != len(tile_size):
    raise ValueError(f"block_size ({len(block_size)}) and tile_size "
                     f"({len(tile_size)}) must have the same length.")

  if block_index is None:
    num_axes = len(block_size)
    block_index = tuple(
      primitives.program_id(axis) for axis in range(num_axes))

  keys = blocked_sampler.blocked_fold_in(
      global_key, total_size, block_size, tile_size, block_index)
  return blocked_sampler.sample_block(
      sampler_fn, keys, block_size, tile_size, **kwargs)
