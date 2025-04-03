# Copyright 2022 The JAX Authors.
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

from __future__ import annotations

import platform
import subprocess
import sys
import textwrap

from jax import version
from jax._src import lib
from jax._src import xla_bridge as xb
import numpy as np

def try_nvidia_smi() -> str | None:
  try:
    return subprocess.check_output(['nvidia-smi']).decode()
  except Exception:
    return None


def print_environment_info(return_string: bool = False) -> str | None:
  """Returns a string containing local environment & JAX installation information.

  This is useful information to include when asking a question or filing a bug.

  Args: return_string (bool) : if True, return the string rather than printing
  to stdout.
  """
  # TODO(jakevdp): should we include other info, e.g. jax.config.values?
  python_version = sys.version.replace('\n', ' ')
  info = textwrap.dedent(f"""\
  jax:    {version.__version__}
  jaxlib: {lib.version_str}
  numpy:  {np.__version__}
  python: {python_version}
  device info: {xb.devices()[0].device_kind}-{xb.device_count()}, {xb.local_device_count()} local devices"
  process_count: {xb.process_count()}
  platform: {platform.uname()}
""")
  nvidia_smi = try_nvidia_smi()
  if nvidia_smi:
    info += '\n\n$ nvidia-smi\n' + nvidia_smi
  if return_string:
    return info
  else:
    return print(info)
