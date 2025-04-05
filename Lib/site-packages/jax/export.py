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
__all__ = ["DisabledSafetyCheck", "Exported", "export", "deserialize",
           "register_pytree_node_serialization",
           "register_namedtuple_serialization",
           "maximum_supported_calling_convention_version",
           "minimum_supported_calling_convention_version",
           "default_export_platform",
           "SymbolicScope", "is_symbolic_dim",
           "symbolic_shape", "symbolic_args_specs"]

from jax._src.export._export import (
  DisabledSafetyCheck,
  Exported,
  export,
  deserialize,
  register_pytree_node_serialization,
  register_namedtuple_serialization,
  maximum_supported_calling_convention_version,
  minimum_supported_calling_convention_version,
  default_export_platform)

from jax._src.export import shape_poly_decision  # Import only to set the decision procedure
del shape_poly_decision
from jax._src.export.shape_poly import (
  SymbolicScope,
  is_symbolic_dim,
  symbolic_shape,
  symbolic_args_specs)
