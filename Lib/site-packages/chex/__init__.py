# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Chex: Testing made fun, in JAX!"""

from chex._src.asserts import assert_axis_dimension
from chex._src.asserts import assert_axis_dimension_comparator
from chex._src.asserts import assert_axis_dimension_gt
from chex._src.asserts import assert_axis_dimension_gteq
from chex._src.asserts import assert_axis_dimension_lt
from chex._src.asserts import assert_axis_dimension_lteq
from chex._src.asserts import assert_devices_available
from chex._src.asserts import assert_equal
from chex._src.asserts import assert_equal_rank
from chex._src.asserts import assert_equal_shape
from chex._src.asserts import assert_equal_shape_prefix
from chex._src.asserts import assert_equal_shape_suffix
from chex._src.asserts import assert_equal_size
from chex._src.asserts import assert_exactly_one_is_none
from chex._src.asserts import assert_gpu_available
from chex._src.asserts import assert_is_broadcastable
from chex._src.asserts import assert_is_divisible
from chex._src.asserts import assert_max_traces
from chex._src.asserts import assert_not_both_none
from chex._src.asserts import assert_numerical_grads
from chex._src.asserts import assert_rank
from chex._src.asserts import assert_scalar
from chex._src.asserts import assert_scalar_in
from chex._src.asserts import assert_scalar_negative
from chex._src.asserts import assert_scalar_non_negative
from chex._src.asserts import assert_scalar_positive
from chex._src.asserts import assert_shape
from chex._src.asserts import assert_size
from chex._src.asserts import assert_tpu_available
from chex._src.asserts import assert_tree_all_finite
from chex._src.asserts import assert_tree_has_only_ndarrays
from chex._src.asserts import assert_tree_is_on_device
from chex._src.asserts import assert_tree_is_on_host
from chex._src.asserts import assert_tree_is_sharded
from chex._src.asserts import assert_tree_no_nones
from chex._src.asserts import assert_tree_shape_prefix
from chex._src.asserts import assert_tree_shape_suffix
from chex._src.asserts import assert_trees_all_close
from chex._src.asserts import assert_trees_all_close_ulp
from chex._src.asserts import assert_trees_all_equal
from chex._src.asserts import assert_trees_all_equal_comparator
from chex._src.asserts import assert_trees_all_equal_dtypes
from chex._src.asserts import assert_trees_all_equal_shapes
from chex._src.asserts import assert_trees_all_equal_shapes_and_dtypes
from chex._src.asserts import assert_trees_all_equal_sizes
from chex._src.asserts import assert_trees_all_equal_structs
from chex._src.asserts import assert_type
from chex._src.asserts import clear_trace_counter
from chex._src.asserts import disable_asserts
from chex._src.asserts import enable_asserts
from chex._src.asserts import if_args_not_none
from chex._src.asserts_chexify import block_until_chexify_assertions_complete
from chex._src.asserts_chexify import chexify
from chex._src.asserts_chexify import ChexifyChecks
from chex._src.asserts_chexify import with_jittable_assertions
from chex._src.dataclass import dataclass
from chex._src.dataclass import mappable_dataclass
from chex._src.dataclass import register_dataclass_type_with_jax_tree_util
from chex._src.dimensions import Dimensions
from chex._src.fake import fake_jit
from chex._src.fake import fake_pmap
from chex._src.fake import fake_pmap_and_jit
from chex._src.fake import set_n_cpu_devices
from chex._src.pytypes import Array
from chex._src.pytypes import ArrayBatched
from chex._src.pytypes import ArrayDevice
from chex._src.pytypes import ArrayDeviceTree
from chex._src.pytypes import ArrayDType
from chex._src.pytypes import ArrayNumpy
from chex._src.pytypes import ArrayNumpyTree
from chex._src.pytypes import ArraySharded
from chex._src.pytypes import ArrayTree
from chex._src.pytypes import Device
from chex._src.pytypes import Numeric
from chex._src.pytypes import PRNGKey
from chex._src.pytypes import PyTreeDef
from chex._src.pytypes import Scalar
from chex._src.pytypes import Shape
from chex._src.restrict_backends import restrict_backends
from chex._src.variants import all_variants
from chex._src.variants import ChexVariantType
from chex._src.variants import params_product
from chex._src.variants import TestCase
from chex._src.variants import variants
from chex._src.warnings import create_deprecated_function_alias
from chex._src.warnings import warn_deprecated_function
from chex._src.warnings import warn_keyword_args_only_in_future
from chex._src.warnings import warn_only_n_pos_args_in_future


__version__ = "0.1.89"

__all__ = (
    "all_variants",
    "Array",
    "ArrayBatched",
    "ArrayDevice",
    "ArrayDeviceTree",
    "ArrayDType",
    "ArrayNumpy",
    "ArrayNumpyTree",
    "ArraySharded",
    "ArrayTree",
    "ChexifyChecks",
    "assert_axis_dimension",
    "assert_axis_dimension_comparator",
    "assert_axis_dimension_gt",
    "assert_axis_dimension_gteq",
    "assert_axis_dimension_lt",
    "assert_axis_dimension_lteq",
    "assert_devices_available",
    "assert_equal",
    "assert_equal_rank",
    "assert_equal_shape",
    "assert_equal_shape_prefix",
    "assert_equal_shape_suffix",
    "assert_equal_size",
    "assert_exactly_one_is_none",
    "assert_gpu_available",
    "assert_is_broadcastable",
    "assert_is_divisible",
    "assert_max_traces",
    "assert_not_both_none",
    "assert_numerical_grads",
    "assert_rank",
    "assert_scalar",
    "assert_scalar_in",
    "assert_scalar_negative",
    "assert_scalar_non_negative",
    "assert_scalar_positive",
    "assert_shape",
    "assert_size",
    "assert_tpu_available",
    "assert_tree_all_finite",
    "assert_tree_has_only_ndarrays",
    "assert_tree_is_on_device",
    "assert_tree_is_on_host",
    "assert_tree_is_sharded",
    "assert_tree_no_nones",
    "assert_tree_shape_prefix",
    "assert_tree_shape_suffix",
    "assert_trees_all_close",
    "assert_trees_all_close_ulp",
    "assert_trees_all_equal",
    "assert_trees_all_equal_comparator",
    "assert_trees_all_equal_dtypes",
    "assert_trees_all_equal_shapes",
    "assert_trees_all_equal_shapes_and_dtypes",
    "assert_trees_all_equal_sizes",
    "assert_trees_all_equal_structs",
    "assert_type",
    "block_until_chexify_assertions_complete",
    "chexify",
    "ChexVariantType",
    "clear_trace_counter",
    "create_deprecated_function_alias",
    "dataclass",
    "Device",
    "Dimensions",
    "disable_asserts",
    "enable_asserts",
    "fake_jit",
    "fake_pmap",
    "fake_pmap_and_jit",
    "if_args_not_none",
    "mappable_dataclass",
    "Numeric",
    "params_product",
    "PRNGKey",
    "PyTreeDef",
    "register_dataclass_type_with_jax_tree_util",
    "restrict_backends",
    "Scalar",
    "set_n_cpu_devices",
    "Shape",
    "TestCase",
    "variants",
    "warn_deprecated_function",
    "warn_keyword_args_only_in_future",
    "warn_only_n_pos_args_in_future",
    "with_jittable_assertions",
)


#  _________________________________________
# / Please don't use symbols in `_src` they \
# \ are not part of the Chex public API.    /
#  -----------------------------------------
#         \   ^__^
#          \  (oo)\_______
#             (__)\       )\/\
#                 ||----w |
#                 ||     ||
#
