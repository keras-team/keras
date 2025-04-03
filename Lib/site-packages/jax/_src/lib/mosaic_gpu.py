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

# ruff: noqa

try:
  try:
    from jaxlib.mosaic.gpu import _mosaic_gpu_ext  # pytype: disable=import-error
  except ImportError as e:
    print(e)
    from jax_cuda12_plugin import _mosaic_gpu_ext  # pytype: disable=import-error
except ImportError as e:
  print("="*128)
  print(e)
  raise ModuleNotFoundError("Failed to import the Mosaic GPU bindings") from e
else:
  _mosaic_gpu_ext.register_passes()
