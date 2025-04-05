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

from __future__ import annotations

from collections.abc import Sequence

import jax
from jax.experimental import mesh_utils
from jax._src.lib import xla_client as xc
from jax._src.lib import xla_extension
from jax._src import xla_bridge as xb

Device = xc.Device


class TopologyDescription:
  def __init__(self, devices: list[Device]):
    self.devices: list[Device] = devices


def get_attached_topology(platform=None) -> TopologyDescription:
  return TopologyDescription(jax.devices(backend=platform))


def get_topology_desc(
    topology_name: str = "", platform: str | None = None, **kwargs
) -> TopologyDescription:
  if platform == "tpu" or platform is None:
    return TopologyDescription(
        xb.make_pjrt_tpu_topology(
            topology_name, **kwargs
        )._make_compile_only_devices()
    )
  try:
    topology = xb.make_pjrt_topology(platform, topology_name, **kwargs)
    return TopologyDescription(topology._make_compile_only_devices())
  except xla_extension.XlaRuntimeError as e:
    msg, *_ = e.args
    if msg.startswith("UNIMPLEMENTED"):
      raise NotImplementedError(msg) from e
    else:
      raise


# -- future mesh_utils --


def make_mesh(
    topo: TopologyDescription,
    mesh_shape: Sequence[int],
    axis_names: tuple[str, ...],
    *,
    contiguous_submeshes: bool = False
) -> jax.sharding.Mesh:
  devices = mesh_utils.create_device_mesh(
      mesh_shape, list(topo.devices), contiguous_submeshes=contiguous_submeshes)
  return jax.sharding.Mesh(devices, axis_names)
