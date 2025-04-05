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

from collections.abc import Sequence
import logging
from jax._src.cloud_tpu_init import running_in_cloud_tpu_vm

logger = logging.getLogger(__name__)

class ClusterEnv:
  """Interface for defining a cluster environment.

  To enable auto bootrapping (aka :func:`jax.distributed.initialize()`),
  cluster environments need to derive from :class:`ClusterEnv` and implement
  :func:`is_env_present`, :func:`get_coordinator_address`,
  :func:`get_process_count`, and :func:`get_process_id`.
  :class:`ClusterEnv` subclasses are automatically detected when imported.
  """

  _cluster_types: list[type[ClusterEnv]] = []
  opt_in_only_method: bool = False # Override this in derived classes if necessary

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls._cluster_types.append(cls)


  @classmethod
  # pytype: disable=bad-return-type
  def auto_detect_unset_distributed_params(cls,
                                           coordinator_address: str | None,
                                           num_processes: int | None,
                                           process_id: int | None,
                                           local_device_ids: Sequence[int] | None,
                                           cluster_detection_method: str | None,
                                           initialization_timeout: int | None,
                                          ) -> tuple[str | None, int | None, int | None,
                                                     Sequence[int] | None]:
    # First, we check the spec detection method because it will ignore submitted values
    # If if succeeds.
    if cluster_detection_method is not None:
      env = next( (env for env in cls._cluster_types if env.name == cluster_detection_method), None )  # pytype: disable=attribute-error
      if env is None:
        logger.error(f"Automatic Distributed initialization can not proceed:"
                     f" {cluster_detection_method} is not supported.")
      elif not env.is_env_present():
        logger.error(f"Automatic Distributed initialization can not proceed:"
                     f" {cluster_detection_method} is supported but not functional in this environment.")
    else:
      env = next((env for env in cls._cluster_types if env.opt_in_only_method == False and env.is_env_present()), None)

    # Above: I have wrapped the env selection in a conditional to go through
    # opt-in methods first (currently only mpi4py) but to check all possible options
    # otherwise.  Passing no cluster_detection_method results in the default, original behavior.

    if env:
      logger.debug('Initializing distributed JAX environment via %s', env.__name__)
      if coordinator_address is None:
        coordinator_address = env.get_coordinator_address(timeout_secs=initialization_timeout)
      if num_processes is None:
        num_processes = env.get_process_count()
      if process_id is None:
        process_id = env.get_process_id()
      # Never automatically set local_device_ids on TPUs
      # Defaults to single process per device if local_process_id is available.
      # This only runs if we're in a managed distributed environment.
      # Otherwise local_device_ids will remain unset,
      # which will default to all devices being visible.
      if (local_device_ids is None and not running_in_cloud_tpu_vm and
          env.get_local_process_id() is not None):
        local_device_ids = [env.get_local_process_id()] # type: ignore[list-item]
    else:
      logger.debug('Could not find a known environment for initializing distributed JAX. '
        'Known environments: %s', ', '.join(e.__name__ for e in cls._cluster_types))
    return (coordinator_address, num_processes, process_id, local_device_ids)
  # pytype: enable=bad-return-type

  @classmethod
  def is_env_present(cls) -> bool:
    """Returns True if process is running in this cluster environment.
    """
    raise NotImplementedError("ClusterEnv subclasses must implement is_env_present")

  @classmethod
  def get_coordinator_address(cls, timeout_secs: int | None) -> str:
    """Returns address and port used by JAX to bootstrap.

    Process id 0 will open a tcp socket at "hostname:port" where
    all the processes will connect to initialize the distributed JAX service.
    The selected port needs to be free.
    :func:`get_coordinator_address` needs to return the same hostname and port on all the processes.

    Returns:
      "hostname:port"
    """
    raise NotImplementedError("ClusterEnv subclasses must implement get_coordinator_address")

  @classmethod
  def get_process_count(cls) -> int:
    raise NotImplementedError("ClusterEnv subclasses must implement get_process_count")

  @classmethod
  def get_process_id(cls) -> int:
    raise NotImplementedError("ClusterEnv subclasses must implement get_process_id")

  @classmethod
  def get_local_process_id(cls) -> int | None:
    """ Get index of current process inside a host.

    The method is only useful to support single device per process.
    In that case, each process will see a local device whose ID is
    the same as its local process ID.
    If None, JAX will not restrict the visible devices.
    """
    return None
