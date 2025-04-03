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


from jax._src.core import (
  shaped_abstractify as shaped_abstractify
)

from jax._src.api_util import (
  argnums_partial as argnums_partial,
  donation_vector as donation_vector,
  flatten_axes as flatten_axes,
  flatten_fun as flatten_fun,
  flatten_fun_nokwargs as flatten_fun_nokwargs,
  rebase_donate_argnums as rebase_donate_argnums,
  safe_map as safe_map,
)
