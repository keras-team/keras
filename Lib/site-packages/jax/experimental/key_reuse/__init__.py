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

"""
Experimental Key Reuse Checking
-------------------------------

This module contains **experimental** functionality for detecting reuse of random
keys within JAX programs. It is under active development and the APIs here are
likely to change. The usage below requires JAX version 0.4.26 or newer.

Key reuse checking can be enabled using the ``jax_debug_key_reuse`` configuration.
This can be set globally using::

  >>> jax.config.update('jax_debug_key_reuse', True)  # doctest: +SKIP

Or it can be enabled locally with the :func:`jax.debug_key_reuse` context manager.
When enabled, using the same key twice will result in a :class:`~jax.errors.KeyReuseError`::

  >>> import jax
  >>> with jax.debug_key_reuse(True):
  ...   key = jax.random.key(0)
  ...   val1 = jax.random.normal(key)
  ...   val2 = jax.random.normal(key)  # doctest: +IGNORE_EXCEPTION_DETAIL
  Traceback (most recent call last):
   ...
  KeyReuseError: Previously-consumed key passed to jit-compiled function at index 0

The key reuse checker is currently experimental, but in the future we will likely
enable it by default.
"""
