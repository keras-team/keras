# Copyright 2021 The JAX Authors.
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

from jax._src.compilation_cache import (
  is_initialized as is_initialized,  # deprecated
  initialize_cache as initialize_cache,  # deprecated; use set_cache_dir instead
  set_cache_dir as set_cache_dir,
  reset_cache as reset_cache,
)
