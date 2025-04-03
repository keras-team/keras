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

"""Context managers for toggling X64 mode.

**Experimental: please give feedback, and expect changes.**
"""

# This file provides
#  1. a jax.experimental API endpoint;
#  2. the `disable_x64` wrapper.
# TODO(jakevdp): remove this file, and consider removing `disable_x64` for
# uniformity

from contextlib import contextmanager
from jax._src import config

@contextmanager
def enable_x64(new_val: bool = True):
  """Experimental context manager to temporarily enable X64 mode.

  Usage::

    >>> x = np.arange(5, dtype='float64')
    >>> with enable_x64():
    ...   print(jnp.asarray(x).dtype)
    ...
    float64

  See Also
  --------
  jax.experimental.enable_x64 : temporarily enable X64 mode.
  """
  with config.enable_x64(new_val):
    yield

@contextmanager
def disable_x64():
  """Experimental context manager to temporarily disable X64 mode.

  Usage::

    >>> x = np.arange(5, dtype='float64')
    >>> with disable_x64():
    ...   print(jnp.asarray(x).dtype)
    ...
    float32

  See Also
  --------
  jax.experimental.enable_x64 : temporarily enable X64 mode.
  """
  with config.enable_x64(False):
    yield
