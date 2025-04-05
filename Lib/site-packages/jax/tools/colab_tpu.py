# Copyright 2020 The JAX Authors.
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

"""Utilities for running JAX on Cloud TPUs via Colab."""


def setup_tpu(tpu_driver_version=None):
  """Raises an error. Do not use."""
  raise RuntimeError(
    "jax.tools.colab_tpu.setup_tpu() was required for older JAX versions"
    " running on older generations of TPUs, and should no longer be used.")
