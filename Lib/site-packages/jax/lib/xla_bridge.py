# Copyright 2018 The JAX Authors.
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

# ruff: noqa: F401
from jax._src.xla_bridge import (
  get_backend as _deprecated_get_backend,
)

from jax._src.compiler import (
  get_compile_options as get_compile_options,
)

_deprecations = {
  # Added July 31, 2024
  "get_backend": (
    "jax.lib.xla_bridge.get_backend is deprecated; use jax.extend.backend.get_backend.",
    _deprecated_get_backend
  ),
  # Finalized 2024-12-11; remove after 2025-3-11
  "xla_client": (
    "jax.lib.xla_bridge.xla_client was removed in JAX v0.4.38; use jax.lib.xla_client directly.",
    None
  ),
  "default_backend": (
    "jax.lib.xla_bridge.default_backend was removed in JAX v0.4.38; use jax.default_backend.",
    None
  ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  from jax._src.xla_bridge import get_backend as get_backend
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
