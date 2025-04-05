# Copyright 2021 The JAX Authors.
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
import os
from typing import Any

from jax._src import clusters
from jax._src import config
from jax._src import xla_bridge
from jax._src.lib import xla_extension

logger = logging.getLogger(__name__)


_CHECK_PROXY_ENVS = config.bool_flag(
    name="jax_check_proxy_envs",
    default=True,
    help="Checks proxy vars in user envs and emit warnings.",
)


class State:
  process_id: int = 0
  num_processes: int = 1
  service: Any | None = None
  client: Any | None = None
  preemption_sync_manager: Any | None = None
  coordinator_address: str | None = None

  def initialize(self,
                 coordinator_address: str | None = None,
                 num_processes: int | None = None,
                 process_id: int | None = None,
                 local_device_ids: int | Sequence[int] | None = None,
                 cluster_detection_method: str | None = None,
                 initialization_timeout: int = 300,
                 coordinator_bind_address: str | None = None,
                 service_heartbeat_interval_seconds: int = 10,
                 service_max_missing_heartbeats: int = 10,
                 client_heartbeat_interval_seconds: int = 10,
                 client_max_missing_heartbeats: int = 10):
    coordinator_address = (coordinator_address or
                           os.environ.get('JAX_COORDINATOR_ADDRESS'))
    if isinstance(local_device_ids, int):
      local_device_ids = [local_device_ids]

    if local_device_ids is None and (env_ids := os.environ.get('JAX_LOCAL_DEVICE_IDS')):
      local_device_ids = list(map(int, env_ids.split(",")))

    if (cluster_detection_method != 'deactivate' and
        None in (coordinator_address, num_processes, process_id, local_device_ids)):
      (coordinator_address, num_processes, process_id, local_device_ids) = (
          clusters.ClusterEnv.auto_detect_unset_distributed_params(
              coordinator_address,
              num_processes,
              process_id,
              local_device_ids,
              cluster_detection_method,
              initialization_timeout,
          )
      )

    if coordinator_address is None:
      raise ValueError('coordinator_address should be defined.')
    if num_processes is None:
      raise ValueError('Number of processes must be defined.')
    if process_id is None:
      raise ValueError('The process id of the current process must be defined.')
    if not isinstance(process_id, int):
      raise TypeError("process_id must be a nonnegative int. "
                      f"Got process_id={process_id} of type {type(process_id)}.")
    if not isinstance(num_processes, int):
      raise TypeError("num_processes must be a positive int. "
                      f"Got num_processes={num_processes} of type {type(num_processes)}.")
    if not (0 <= process_id < num_processes):
      raise ValueError("process_id and num_processes must be nonnegative, with process_id < num_processes. "
                       f"Got process_id={process_id}, num_processes={num_processes}.")

    self.coordinator_address = coordinator_address

    # The default value of [::]:port tells the coordinator to bind to all
    # available addresses on the same port as coordinator_address.
    default_coordinator_bind_address = '[::]:' + coordinator_address.rsplit(':', 1)[1]
    coordinator_bind_address = (coordinator_bind_address or
                                os.environ.get('JAX_COORDINATOR_BIND_ADDRESS',
                                               default_coordinator_bind_address))
    if coordinator_bind_address is None:
      raise ValueError('coordinator_bind_address should be defined.')

    if local_device_ids:
      visible_devices = ','.join(str(x) for x in local_device_ids)
      logger.info('JAX distributed initialized with visible devices: %s', visible_devices)
      config.update("jax_cuda_visible_devices", visible_devices)
      config.update("jax_rocm_visible_devices", visible_devices)

    self.process_id = process_id

    proxy_vars = []
    if _CHECK_PROXY_ENVS.value:
      proxy_vars = [key for key in os.environ.keys()
                    if '_proxy' in key.lower()]

    if len(proxy_vars) > 0:
      vars = " ".join(proxy_vars) + ". "
      warning = (
        f'JAX detected proxy variable(s) in the environment as distributed setup: {vars}'
        'On some systems, this may cause a hang of distributed.initialize and '
        'you may need to unset these ENV variable(s)'
      )
      logger.warning(warning)

    if process_id == 0:
      if self.service is not None:
        raise RuntimeError('distributed.initialize should only be called once.')
      logger.info(
          'Starting JAX distributed service on %s', coordinator_bind_address
      )
      self.service = xla_extension.get_distributed_runtime_service(
          coordinator_bind_address, num_processes,
          heartbeat_interval=service_heartbeat_interval_seconds,
          max_missing_heartbeats=service_max_missing_heartbeats)

    self.num_processes = num_processes

    if self.client is not None:
      raise RuntimeError('distributed.initialize should only be called once.')

    self.client = xla_extension.get_distributed_runtime_client(
        coordinator_address, process_id, init_timeout=initialization_timeout,
        heartbeat_interval=client_heartbeat_interval_seconds,
        max_missing_heartbeats=client_max_missing_heartbeats, use_compression=True)
    logger.info('Connecting to JAX distributed service on %s', coordinator_address)
    self.client.connect()

    self.initialize_preemption_sync_manager()

  def shutdown(self):
    if self.client:
      self.client.shutdown()
      self.client = None
    if self.service:
      self.service.shutdown()
      self.service = None
    if self.preemption_sync_manager:
      self.preemption_sync_manager = None

  def initialize_preemption_sync_manager(self):
    if self.preemption_sync_manager is not None:
      raise RuntimeError(
          'Preemption sync manager should only be initialized once.')
    self.preemption_sync_manager = (
        xla_extension.create_preemption_sync_manager())
    self.preemption_sync_manager.initialize(self.client)

global_state = State()


def initialize(coordinator_address: str | None = None,
               num_processes: int | None = None,
               process_id: int | None = None,
               local_device_ids: int | Sequence[int] | None = None,
               cluster_detection_method: str | None = None,
               initialization_timeout: int = 300,
               coordinator_bind_address: str | None = None):
  """Initializes the JAX distributed system.

  Calling :func:`~jax.distributed.initialize` prepares JAX for execution on
  multi-host GPU and Cloud TPU. :func:`~jax.distributed.initialize` must be
  called before performing any JAX computations.

  The JAX distributed system serves a number of roles:

    * It allows JAX processes to discover each other and share topology information,
    * It performs health checking, ensuring that all processes shut down if any process dies, and
    * It is used for distributed checkpointing.

  If you are using TPU, Slurm, or Open MPI, all arguments are optional: if omitted, they
  will be chosen automatically.

  The ``cluster_detection_method`` may be used to choose a specific method for detecting those
  distributed arguments. You may pass any of the automatic ``spec_detect_methods`` to this
  argument though it is not necessary in the TPU, Slurm, or Open MPI cases.  For other MPI
  installations, if you have a functional ``mpi4py`` installed, you may pass
  ``cluster_detection_method="mpi4py"`` to bootstrap the required arguments.

  Otherwise, you must provide the ``coordinator_address``,
  ``num_processes``, ``process_id``, and ``local_device_ids`` arguments
  to :func:`~jax.distributed.initialize`. When all four arguments are provided, cluster
  environment auto detection will be skipped.

  Please note: on some systems, particularly HPC clusters that only access external networks
  through proxy variables such as HTTP_PROXY, HTTPS_PROXY, etc., the call to
  :func:`~jax.distributed.initialize` may timeout.  You may need to unset these variables
  prior to application launch.

  Args:
    coordinator_address: the IP address of process `0` and a port on which that
      process should launch a coordinator service. The choice of
      port does not matter, so long as the port is available on the coordinator
      and all processes agree on the port.
      May be ``None`` only on supported environments, in which case it will be chosen automatically.
      Note that special addresses like ``localhost`` or ``127.0.0.1`` usually mean that the program
      will bind to a local interface and are not suitable when running in a multi-host environment.
    num_processes: Number of processes. May be ``None`` only on supported environments, in
      which case it will be chosen automatically.
    process_id: The ID number of the current process. The ``process_id`` values across
      the cluster must be a dense range ``0``, ``1``, ..., ``num_processes - 1``.
      May be ``None`` only on supported environments; if ``None`` it will be chosen automatically.
    local_device_ids: Restricts the visible devices of the current process to ``local_device_ids``.
      If ``None``, defaults to all local devices being visible to the process except when processes
      are launched via Slurm and Open MPI on GPUs. In that case, it will default to a single device per process.
    cluster_detection_method: An optional string to attempt to autodetect the configuration of the distributed
      run.  Note that "mpi4py" method requires you to have a working ``mpi4py`` install in your environment,
      and launch the applicatoin with an MPI-compatible job launcher such as ``mpiexec`` or ``mpirun``.
      Legacy auto-detect options "ompi" (OMPI) and "slurm" (Slurm) remain enabled. "deactivate" bypasses
      automatic cluster detection.
    initialization_timeout: Time period (in seconds) for which connection will
      be retried. If the initialization takes more than the timeout specified,
      the initialization will error. Defaults to 300 secs i.e. 5 mins.
    coordinator_bind_address: the address and port to which the coordinator service
      on process `0` should bind. If this is not specified, the default is to bind to
      all available addresses on the same port as ``coordinator_address``. On systems
      that have multiple network interfaces per node it may be insufficient to only
      have the coordinator service listen on one address/interface.

  Raises:
    RuntimeError: If :func:`~jax.distributed.initialize` is called more than once
      or if called after the backend is already initialized.

  Examples:

  Suppose there are two GPU processes, and process 0 is the designated coordinator
  with address ``10.0.0.1:1234``. To initialize the GPU cluster, run the
  following commands before anything else.

  On process 0:

  >>> jax.distributed.initialize(coordinator_address='10.0.0.1:1234', num_processes=2, process_id=0)  # doctest: +SKIP

  On process 1:

  >>> jax.distributed.initialize(coordinator_address='10.0.0.1:1234', num_processes=2, process_id=1)  # doctest: +SKIP
  """
  if xla_bridge.backends_are_initialized():
    raise RuntimeError("jax.distributed.initialize() must be called before "
                        "any JAX computations are executed.")
  global_state.initialize(coordinator_address, num_processes, process_id,
                          local_device_ids, cluster_detection_method,
                          initialization_timeout, coordinator_bind_address)


def shutdown():
  """Shuts down the distributed system.

  Does nothing if the distributed system is not running.
  """
  global_state.shutdown()
