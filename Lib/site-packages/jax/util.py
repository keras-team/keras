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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.util import (
  HashableFunction as HashableFunction,
  as_hashable_function as as_hashable_function,
  cache as cache,
  safe_map as safe_map,
  safe_zip as safe_zip,
  split_dict as split_dict,
  split_list as split_list,
  split_list_checked as split_list_checked,
  split_merge as split_merge,
  subvals as subvals,
  toposort as toposort,
  unzip2 as unzip2,
  wrap_name as wrap_name,
  wraps as wraps,
)
