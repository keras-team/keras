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

from jax._src.lib import xla_extension as _xe

get_distributed_runtime_client = _xe.get_distributed_runtime_client
get_distributed_runtime_service = _xe.get_distributed_runtime_service
hlo_module_cost_analysis = _xe.hlo_module_cost_analysis
hlo_module_to_dot_graph = _xe.hlo_module_to_dot_graph
ifrt_proxy = _xe.ifrt_proxy
jax_jit = _xe.jax_jit
mlir = _xe.mlir
pmap_lib = _xe.pmap_lib
profiler = _xe.profiler
pytree = _xe.pytree
Device = _xe.Device
DistributedRuntimeClient = _xe.DistributedRuntimeClient
HloModule = _xe.HloModule
HloPrintOptions = _xe.HloPrintOptions
OpSharding = _xe.OpSharding
PjitFunctionCache = _xe.PjitFunctionCache
PjitFunction = _xe.PjitFunction
PmapFunction = _xe.PmapFunction

_deprecations = {
    # Added Nov 20 2024
    "ArrayImpl": (
        "jax.lib.xla_extension.ArrayImpl is deprecated; use jax.Array instead.",
        _xe.ArrayImpl,
    ),
    "XlaRuntimeError": (
        "jax.lib.xla_extension.XlaRuntimeError is deprecated; use jax.errors.JaxRuntimeError instead.",
        _xe.XlaRuntimeError,
    ),
}

import typing as _typing

if _typing.TYPE_CHECKING:
  ArrayImpl = _xe.ArrayImpl
  XlaRuntimeError = _xe.XlaRuntimeError
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr

  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _xe
