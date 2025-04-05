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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.source_info_util import (
  NameStack as NameStack,
  SourceInfo as SourceInfo,
  current as current,
  current_name_stack as current_name_stack,
  extend_name_stack as extend_name_stack,
  new_name_stack as new_name_stack,
  new_source_info as new_source_info,
  register_exclusion as register_exclusion,
  reset_name_stack as reset_name_stack,
  set_name_stack as set_name_stack,
  summarize as summarize,
  transform_name_stack as transform_name_stack,
  user_context as user_context,
)
