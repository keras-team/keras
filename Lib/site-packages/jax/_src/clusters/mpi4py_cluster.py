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

from __future__ import annotations

from jax._src import clusters
import socket

from importlib.util import find_spec


class Mpi4pyCluster(clusters.ClusterEnv):


  name: str = "mpi4py"
  opt_in_only_method: bool = True

  @classmethod
  def is_env_present(cls) -> bool:

    # Relies on mpi4py:
    return find_spec("mpi4py") is not None

  @classmethod
  def get_coordinator_address(cls, timeout_secs: int | None) -> str:

    # Using mpi4py, figure out rank 0 and it's hostname.
    # Then broadcast the hostname and port.


    from mpi4py import MPI #type: ignore
    # Get the global communicator:
    COMM_WORLD = MPI.COMM_WORLD

    # On rank 0, get the hostname:

    if COMM_WORLD.Get_rank() == 0:
        # Order all the hostnames, and find unique ones
        hostname = socket.gethostname()

        # Apparently, we want to pick a port in an ephemeral range...
        port_id = hash(hostname) % 2**12 + (65535 - 2**12 + 1)

        hostname = f'{hostname}:{port_id}'

    else:
        hostname = "None"



    # Broadcast the host_ip to all ranks:
    hostname = COMM_WORLD.bcast(hostname, root=0)


    return hostname


  @classmethod
  def get_process_count(cls) -> int:
    from mpi4py import MPI  # pytype: disable=import-error
    return int(MPI.COMM_WORLD.Get_size())

  @classmethod
  def get_process_id(cls) -> int:
    from mpi4py import MPI  # pytype: disable=import-error
    return int(MPI.COMM_WORLD.Get_rank())

  @classmethod
  def get_local_process_id(cls) -> int | None:

    # Using mpi4py, split the global communicator into sub communicators
    # based on hostname.  mpi will assign them ranks and that will allow
    # a selection of the local process ID.
    from mpi4py import MPI  # pytype: disable=import-error
    COMM_WORLD = MPI.COMM_WORLD

    # This is the alternative method that is simpler:
    new_comm = COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED)


    # The rank in the new communicator - which is host-local only - IS the local rank:
    return int(new_comm.Get_rank())
