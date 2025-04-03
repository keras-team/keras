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

from jax._src.interpreters.pxla import (
  Index as Index,
  MapTracer as MapTracer,
  MeshAxisName as MeshAxisName,
  MeshComputation as MeshComputation,
  MeshExecutable as MeshExecutable,
  PmapExecutable as PmapExecutable,
  global_aval_to_result_handler as global_aval_to_result_handler,
  global_avals_to_results_handler as global_avals_to_results_handler,
  global_result_handlers as global_result_handlers,
  parallel_callable as parallel_callable,
  shard_args as shard_args,
  xla_pmap_p as xla_pmap_p,
)
from jax._src.mesh import (
  thread_resources as thread_resources,
)

from jax._src.op_shardings import (
  are_op_shardings_equal as are_op_shardings_equal,
  is_op_sharding_replicated as is_op_sharding_replicated,
  op_sharding_to_indices as op_sharding_to_indices,
)

from jax._src.sharding_impls import (
  ArrayMapping as ArrayMapping,
  UNSPECIFIED as _UNSPECIFIED,  # noqa: F401
  array_mapping_to_axis_resources as array_mapping_to_axis_resources,
)

from jax._src.sharding_specs import (
  Chunked as Chunked,
  NoSharding as NoSharding,
  Replicated as Replicated,
  ShardedAxis as ShardedAxis,
  ShardingSpec as ShardingSpec,
  Unstacked as Unstacked,
  spec_to_indices as spec_to_indices,
)
