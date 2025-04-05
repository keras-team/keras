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

"""Those common utility functions for gpu."""


class GpuLibNotLinkedError(Exception):
  """Raised when the GPU library is not linked."""

  error_msg = (
      'JAX was not built with GPU support. Please use a GPU-enabled JAX to use'
      ' this function.'
  )

  def __init__(self):
    super().__init__(self.error_msg)
