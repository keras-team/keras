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

import os
from jax._src import clusters

_JOBID_PARAM = 'SLURM_JOB_ID'
_NODE_LIST = 'SLURM_STEP_NODELIST'
_PROCESS_COUNT = 'SLURM_NTASKS'
_PROCESS_ID = 'SLURM_PROCID'
_LOCAL_PROCESS_ID = 'SLURM_LOCALID'
_NUM_NODES = 'SLURM_STEP_NUM_NODES'

class SlurmCluster(clusters.ClusterEnv):

  name: str = "slurm"

  @classmethod
  def is_env_present(cls) -> bool:
    return _JOBID_PARAM in os.environ

  @classmethod
  def get_coordinator_address(cls, timeout_secs: int | None) -> str:
    # Pick port in ephemeral range [(65535 - 2^12 + 1), 65535]
    port = int(os.environ[_JOBID_PARAM]) % 2**12 + (65535 - 2**12 + 1)

    # Parse the first hostname of the job
    # If we are looking for 'node001',
    # node_list potential formats are 'node001', 'node001,host2',
    # 'node[001-0015],host2', and 'node[001,007-015],host2'.
    node_list = os.environ[_NODE_LIST]
    delims = {',', '['}
    ind = next((i for i, ch  in enumerate(node_list) if ch in delims), len(node_list))
    if ind == len(node_list) or node_list[ind] == ',': # Formats: 'node001' or 'node001,host2'
        return f'{node_list[:ind]}:{port}'
    else: # Formats: 'node[001-0015],host2' or 'node[001,007-015],host2'
        prefix = node_list[:ind]
        suffix = node_list[ind+1:]
        delims2 = {',', '-'}
        ind2 = next((i for i, ch  in enumerate(suffix) if ch in delims2), None)
        return f'{prefix}{suffix[:ind2]}:{port}'

  @classmethod
  def get_process_count(cls) -> int:
    return int(os.environ[_PROCESS_COUNT])

  @classmethod
  def get_process_id(cls) -> int:
    return int(os.environ[_PROCESS_ID])

  @classmethod
  def get_local_process_id(cls) -> int | None:
    return int(os.environ[_LOCAL_PROCESS_ID])
