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

from jax._src.lib import xla_client


def get_profiled_instructions_proto(tensorboard_dir: str) -> bytes:
  """Gets the serialized profiled instructions proto from profile.

  Restore the xplane from the tensorboard dir and convert it to profiled
  [ProfiledInstructionsProto](https://github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/protobuf/profiled_instructions.proto).
  The result is non-empty only when running on Nvidia GPU.

  Args:
    tensorboard_dir: The directory contains the profile trace, it is in the
      format of `<tb's log_dir>/plugin/profile/<run_dir>/`.

  Returns:
    Serialized
    [ProfiledInstructionsProto](https://github.com/openxla/xla/blob/main/third_party/tsl/tsl/profiler/protobuf/profiled_instructions.proto).
  """
  return xla_client.profiler.get_profiled_instructions_proto(tensorboard_dir)
