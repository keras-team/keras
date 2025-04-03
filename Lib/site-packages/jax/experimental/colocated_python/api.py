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
"""Colocated Python top-level API."""

from __future__ import annotations

import collections
from typing import Any, Callable, Sequence

import jax
from jax._src import api_util
from jax.experimental.colocated_python.func import make_callable


def colocated_cpu_devices(
    devices: Sequence[jax.Device],
) -> Sequence[jax.Device]:
  """Finds CPU devices colocated with the given devices."""
  cpu_devices_by_colocation_id = collections.defaultdict(list)
  for device in devices[0].client._get_all_devices():  # pylint: disable=protected-access
    if device.device_kind == "cpu":
      cpu_devices_by_colocation_id[device.colocation_id].append(device)
  if not cpu_devices_by_colocation_id:
    raise ValueError("No CPU devices found")

  colocated_cpu_devices = []
  for device in devices:
    matches = cpu_devices_by_colocation_id[device.colocation_id]
    if not matches:
      raise ValueError(f"Device {device} has no colocated devices")
    elif len(matches) > 1:
      raise ValueError(
          f"Ambiguous colocated devices; device {device} has"
          f" {len(matches)} colocated devices: f{matches}"
      )
    colocated_cpu_devices.append(matches[0])
  return colocated_cpu_devices


def colocated_python(fun: Callable[..., Any]) -> Any:
  """Executes the given Python function on the same device as the arguments."""
  return make_callable(
      fun, api_util.fun_sourceinfo(fun), api_util.fun_signature(fun)
  )
