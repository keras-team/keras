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

import os
import re
from jax._src import clusters

#  OMPI_MCA_orte_hnp_uri exists only when processes are launched via mpirun or mpiexec
_ORTE_URI = 'OMPI_MCA_orte_hnp_uri'
_PROCESS_COUNT = 'OMPI_COMM_WORLD_SIZE'
_PROCESS_ID = 'OMPI_COMM_WORLD_RANK'
_LOCAL_PROCESS_ID = 'OMPI_COMM_WORLD_LOCAL_RANK'

class OmpiCluster(clusters.ClusterEnv):

  name: str = "ompi"

  @classmethod
  def is_env_present(cls) -> bool:
    return _ORTE_URI in os.environ

  @classmethod
  def get_coordinator_address(cls, timeout_secs: int | None) -> str:
    # Examples of orte_uri:
    # 1531576320.0;tcp://10.96.0.1,10.148.0.1,10.108.0.1:34911
    # 1314521088.0;tcp6://[fe80::b9b:ac5d:9cf0:b858,2620:10d:c083:150e::3000:2]:43370
    orte_uri = os.environ[_ORTE_URI]
    job_id_str = orte_uri.split('.', maxsplit=1)[0]
    # The jobid is always a multiple of 2^12, let's divide it by 2^12
    # to reduce likelihood of port conflict between jobs
    job_id = int(job_id_str) // 2**12
    # Pick port in ephemeral range [(65535 - 2^12 + 1), 65535]
    port = job_id % 2**12 + (65535 - 2**12 + 1)
    launcher_ip_match = re.search(r"tcp://(.+?)[,:]|tcp6://\[(.+?)[,\]]", orte_uri)
    if launcher_ip_match is None:
        raise RuntimeError('Could not parse coordinator IP address from Open MPI environment.')
    launcher_ip = next(i for i in launcher_ip_match.groups() if i is not None)
    return f'{launcher_ip}:{port}'

  @classmethod
  def get_process_count(cls) -> int:
    return int(os.environ[_PROCESS_COUNT])

  @classmethod
  def get_process_id(cls) -> int:
    return int(os.environ[_PROCESS_ID])

  @classmethod
  def get_local_process_id(cls) -> int | None:
    return int(os.environ[_LOCAL_PROCESS_ID])
