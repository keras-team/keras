# Copyright 2020 The JAX Authors.
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
"""Backwards compatibility shims for the deprecated host_callback APIs.

.. warning::
  The host_callback APIs are deprecated as of March 20, 2024.
  The functionality is subsumed by the
  `new JAX external callbacks <https://jax.readthedocs.io/en/latest/notebooks/external_callbacks.html>`_
  See https://github.com/jax-ml/jax/issues/20385.

"""

from __future__ import annotations

def call(*_, **__):
  raise NotImplementedError(
      "jax.experimental.host_callback has been deprecated since March 2024 and "
      "is now no longer supported. "
      "See https://github.com/jax-ml/jax/issues/20385"
  )


id_tap = call
