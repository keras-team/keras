# Copyright 2023 The JAX Authors.
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

from collections.abc import Callable, Hashable

from jax import Array

from jax._src import prng
from jax._src import random

Shape = tuple[int, ...]

def define_prng_impl(*,
                     key_shape: Shape,
                     seed: Callable[[Array], Array],
                     split: Callable[[Array, Shape], Array],
                     random_bits: Callable[[Array, int, Shape], Array],
                     fold_in: Callable[[Array, int], Array],
                     name: str = '<unnamed>',
                     tag: str = '?') -> Hashable:
  return random.PRNGSpec(prng.PRNGImpl(
      key_shape, seed, split, random_bits, fold_in,
      name=name, tag=tag))
