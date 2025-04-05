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

# Set default C++ logging level before any logging happens.
import os as _os
_os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
del _os

# Import version first, because other submodules may reference it.
from jax.version import __version__ as __version__
from jax.version import __version_info__ as __version_info__

# Set Cloud TPU env vars if necessary before transitively loading C++ backend
from jax._src.cloud_tpu_init import cloud_tpu_init as _cloud_tpu_init
try:
  _cloud_tpu_init()
except Exception as exc:
  # Defensively swallow any exceptions to avoid making jax unimportable
  from warnings import warn as _warn
  _warn(f"cloud_tpu_init failed: {exc!r}\n This a JAX bug; please report "
        f"an issue at https://github.com/jax-ml/jax/issues")
  del _warn
del _cloud_tpu_init

# Force early import, allowing use of `jax.core` after importing `jax`.
import jax.core as _core
del _core

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.basearray import Array as Array
from jax import tree as tree
from jax import typing as typing

from jax._src.config import (
  config as config,
  enable_checks as enable_checks,
  debug_key_reuse as debug_key_reuse,
  check_tracer_leaks as check_tracer_leaks,
  checking_leaks as checking_leaks,
  enable_custom_prng as enable_custom_prng,
  softmax_custom_jvp as softmax_custom_jvp,
  enable_custom_vjp_by_custom_transpose as enable_custom_vjp_by_custom_transpose,
  debug_nans as debug_nans,
  debug_infs as debug_infs,
  log_compiles as log_compiles,
  no_tracing as no_tracing,
  explain_cache_misses as explain_cache_misses,
  default_device as default_device,
  default_matmul_precision as default_matmul_precision,
  default_prng_impl as default_prng_impl,
  numpy_dtype_promotion as numpy_dtype_promotion,
  numpy_rank_promotion as numpy_rank_promotion,
  jax2tf_associative_scan_reductions as jax2tf_associative_scan_reductions,
  legacy_prng_key as legacy_prng_key,
  threefry_partitionable as threefry_partitionable,
  transfer_guard as transfer_guard,
  transfer_guard_host_to_device as transfer_guard_host_to_device,
  transfer_guard_device_to_device as transfer_guard_device_to_device,
  transfer_guard_device_to_host as transfer_guard_device_to_host,
  spmd_mode as spmd_mode,
)
from jax._src.core import ensure_compile_time_eval as ensure_compile_time_eval
from jax._src.environment_info import print_environment_info as print_environment_info

from jax._src.lib import xla_client as _xc
Device = _xc.Device
del _xc

from jax._src.api import effects_barrier as effects_barrier
from jax._src.api import block_until_ready as block_until_ready
from jax._src.ad_checkpoint import checkpoint_wrapper as checkpoint  # noqa: F401
from jax._src.ad_checkpoint import checkpoint_policies as checkpoint_policies
from jax._src.api import clear_caches as clear_caches
from jax._src.custom_derivatives import closure_convert as closure_convert
from jax._src.custom_derivatives import custom_gradient as custom_gradient
from jax._src.custom_derivatives import custom_jvp as custom_jvp
from jax._src.custom_derivatives import custom_vjp as custom_vjp
from jax._src.xla_bridge import default_backend as default_backend
from jax._src.xla_bridge import device_count as device_count
from jax._src.api import device_get as device_get
from jax._src.api import device_put as device_put
from jax._src.api import device_put_sharded as device_put_sharded
from jax._src.api import device_put_replicated as device_put_replicated
from jax._src.xla_bridge import devices as devices
from jax._src.api import disable_jit as disable_jit
from jax._src.api import eval_shape as eval_shape
from jax._src.dtypes import float0 as float0
from jax._src.api import grad as grad
from jax._src.api import hessian as hessian
from jax._src.xla_bridge import host_count as host_count
from jax._src.xla_bridge import host_id as host_id
from jax._src.xla_bridge import host_ids as host_ids
from jax._src.api import jacobian as jacobian
from jax._src.api import jacfwd as jacfwd
from jax._src.api import jacrev as jacrev
from jax._src.api import jit as jit
from jax._src.api import jvp as jvp
from jax._src.xla_bridge import local_device_count as local_device_count
from jax._src.xla_bridge import local_devices as local_devices
from jax._src.api import linearize as linearize
from jax._src.api import linear_transpose as linear_transpose
from jax._src.api import live_arrays as live_arrays
from jax._src.api import make_jaxpr as make_jaxpr
from jax._src.api import named_call as named_call
from jax._src.api import named_scope as named_scope
from jax._src.api import pmap as pmap
from jax._src.xla_bridge import process_count as process_count
from jax._src.xla_bridge import process_index as process_index
from jax._src.xla_bridge import process_indices as process_indices
from jax._src.callback import pure_callback as pure_callback
from jax._src.ad_checkpoint import checkpoint_wrapper as remat  # noqa: F401
from jax._src.api import ShapeDtypeStruct as ShapeDtypeStruct
from jax._src.api import value_and_grad as value_and_grad
from jax._src.api import vjp as vjp
from jax._src.api import vmap as vmap
from jax._src.sharding_impls import NamedSharding as NamedSharding
from jax._src.sharding_impls import make_mesh as make_mesh

# Force import, allowing jax.interpreters.* to be used after import jax.
from jax.interpreters import ad, batching, mlir, partial_eval, pxla, xla
del ad, batching, mlir, partial_eval, pxla, xla

from jax._src.array import (
    make_array_from_single_device_arrays as make_array_from_single_device_arrays,
    make_array_from_callback as make_array_from_callback,
    make_array_from_process_local_data as make_array_from_process_local_data,
)

from jax._src.tree_util import (
  tree_map as _deprecated_tree_map,
  treedef_is_leaf as _deprecated_treedef_is_leaf,
  tree_flatten as _deprecated_tree_flatten,
  tree_leaves as _deprecated_tree_leaves,
  tree_structure as _deprecated_tree_structure,
  tree_transpose as _deprecated_tree_transpose,
  tree_unflatten as _deprecated_tree_unflatten,
)

# These submodules are separate because they are in an import cycle with
# jax and rely on the names imported above.
from jax import custom_derivatives as custom_derivatives
from jax import custom_batching as custom_batching
from jax import custom_transpose as custom_transpose
from jax import api_util as api_util
from jax import distributed as distributed
from jax import debug as debug
from jax import dlpack as dlpack
from jax import dtypes as dtypes
from jax import errors as errors
from jax import ffi as ffi
from jax import image as image
from jax import lax as lax
from jax import monitoring as monitoring
from jax import nn as nn
from jax import numpy as numpy
from jax import ops as ops
from jax import profiler as profiler
from jax import random as random
from jax import scipy as scipy
from jax import sharding as sharding
from jax import stages as stages
from jax import tree_util as tree_util
from jax import util as util

# Also circular dependency.
from jax._src.array import Shard as Shard

import jax.experimental.compilation_cache.compilation_cache as _ccache
del _ccache

_deprecations = {
  # Added July 2022
  "treedef_is_leaf": (
    "jax.treedef_is_leaf is deprecated: use jax.tree_util.treedef_is_leaf.",
    _deprecated_treedef_is_leaf
  ),
  "tree_flatten": (
    "jax.tree_flatten is deprecated: use jax.tree.flatten (jax v0.4.25 or newer) "
    "or jax.tree_util.tree_flatten (any JAX version).",
    _deprecated_tree_flatten
  ),
  "tree_leaves": (
    "jax.tree_leaves is deprecated: use jax.tree.leaves (jax v0.4.25 or newer) "
    "or jax.tree_util.tree_leaves (any JAX version).",
    _deprecated_tree_leaves
  ),
  "tree_structure": (
    "jax.tree_structure is deprecated: use jax.tree.structure (jax v0.4.25 or newer) "
    "or jax.tree_util.tree_structure (any JAX version).",
    _deprecated_tree_structure
  ),
  "tree_transpose": (
    "jax.tree_transpose is deprecated: use jax.tree.transpose (jax v0.4.25 or newer) "
    "or jax.tree_util.tree_transpose (any JAX version).",
    _deprecated_tree_transpose
  ),
  "tree_unflatten": (
    "jax.tree_unflatten is deprecated: use jax.tree.unflatten (jax v0.4.25 or newer) "
    "or jax.tree_util.tree_unflatten (any JAX version).",
    _deprecated_tree_unflatten
  ),
  # Added Feb 28, 2024
  "tree_map": (
    "jax.tree_map is deprecated: use jax.tree.map (jax v0.4.25 or newer) "
    "or jax.tree_util.tree_map (any JAX version).",
    _deprecated_tree_map
  ),
  # Finalized Nov 12 2024; remove after Feb 12 2025
  "clear_backends": (
    "jax.clear_backends was removed in JAX v0.4.36",
    None
  ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  from jax._src.tree_util import treedef_is_leaf as treedef_is_leaf
  from jax._src.tree_util import tree_flatten as tree_flatten
  from jax._src.tree_util import tree_leaves as tree_leaves
  from jax._src.tree_util import tree_map as tree_map
  from jax._src.tree_util import tree_structure as tree_structure
  from jax._src.tree_util import tree_transpose as tree_transpose
  from jax._src.tree_util import tree_unflatten as tree_unflatten

else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing

import jax.lib  # TODO(phawkins): remove this export.  # noqa: F401

# trailer
