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

from jax._src import ffi as _ffi

_deprecations = {
    # Added 2024-12-20
    "ffi_call": (
        "jax.extend.ffi.ffi_call is deprecated, use jax.ffi.ffi_call instead.",
        _ffi.ffi_call,
    ),
    "ffi_lowering": (
        "jax.extend.ffi.ffi_lowering is deprecated, use jax.ffi.ffi_lowering instead.",
        _ffi.ffi_lowering,
    ),
    "include_dir": (
        "jax.extend.ffi.include_dir is deprecated, use jax.ffi.include_dir instead.",
        _ffi.include_dir,
    ),
    "pycapsule": (
        "jax.extend.ffi.pycapsule is deprecated, use jax.ffi.pycapsule instead.",
        _ffi.pycapsule,
    ),
    "register_ffi_target": (
        "jax.extend.ffi.register_ffi_target is deprecated, use jax.ffi.register_ffi_target instead.",
        _ffi.register_ffi_target,
    ),
}

import typing
if typing.TYPE_CHECKING:
  ffi_call = _ffi.ffi_call
  ffi_lowering = _ffi.ffi_lowering
  include_dir = _ffi.include_dir
  pycapsule = _ffi.pycapsule
  register_ffi_target = _ffi.register_ffi_target
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
del _ffi
