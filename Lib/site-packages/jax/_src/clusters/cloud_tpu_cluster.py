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

import logging
import os
import re
import socket
import time
from jax._src import clusters
from jax._src.cloud_tpu_init import running_in_cloud_tpu_vm

logger = logging.getLogger(__name__)

# We use an arbitrarily chosen port for the coordinator since we cannot
# rely on communication to choose one in real time.
coordinator_port = '8476'

metadata_response_code_success = 200

def get_metadata(key):
  import requests  # pytype: disable=import-error
  import time  # pytype: disable=import-error
  # Based on https://github.com/tensorflow/tensorflow/pull/40317
  gce_metadata_endpoint = 'http://' + os.environ.get(
      'GCE_METADATA_IP', 'metadata.google.internal')

  retry_count = 0
  retrySeconds = 0.500
  api_resp = None

  while retry_count < 6:
    api_resp = requests.get(
        f'{gce_metadata_endpoint}/computeMetadata/v1/instance/attributes/{key}',
        headers={'Metadata-Flavor': 'Google'}, timeout=60)
    if api_resp.status_code == 200:
      break
    retry_count += 1
    time.sleep(retrySeconds)

  if api_resp is None:
    raise RuntimeError(f"Getting metadata['{key}'] failed for 6 tries")
  return api_resp.text, api_resp.status_code

def get_tpu_env_value(key):
  def get_tpu_env_value_from_metadata(key):
    tpu_env_data = get_metadata('tpu-env')[0]
    key_value_pairs = tpu_env_data.split('\n')
    for key_value_pair in key_value_pairs:
      # Typical line is MEGASCALE_NUM_SLICES: '2'
      if ':' in key_value_pair:
        row_key, value = re.split(':', key_value_pair, 1)
        row_key = row_key.strip()
        if row_key == key:
          return value.strip().strip("'")
    return None

  value = os.environ.get(key, None)
  return value if value is not None else get_tpu_env_value_from_metadata(key)

def has_megascale_address():
  return get_tpu_env_value('MEGASCALE_COORDINATOR_ADDRESS') is not None

class BaseTpuCluster(clusters.ClusterEnv):

  name: str = "tpu"

  """Abstract cluster supports both single and multislice TPU environments.

  If MEGASCALE_COORDINATOR_ADDRESS is not set, we assume single slice topology.
  Concrete extensions of this class must implement methods for generating a list
    of within-slice workers and a within-slice process ID.
  `get_coordinator_address` must return the address of the host with
  process ID 0 (as returned by `get_process_id`), since the coordinator service
  is started on the host with process ID = 0.
  """

  @classmethod
  def is_env_present(cls) -> bool:
    """Override this method to return True if the environment is present."""
    return False

  @classmethod
  def get_coordinator_address(cls, timeout_secs: int | None) -> str:
    if has_megascale_address():
      # For both GCE via QueuedResources and GKE via JobSet, the
      # Megascale coordinator address is set as the host with process id = 0,
      # so can be used as the jax distributed system coordinator.
      coordinator_address = get_tpu_env_value('MEGASCALE_COORDINATOR_ADDRESS')
    else:
      # For both GCE (QueuedResources and TPUVM create) and GKE via Job API,
      # the workers lists are sorted by process ID so the first one can
      # be used as the jax distributed system coordinator.
      coordinator_address = cls._get_worker_list_in_slice()[0]
    coordinator_address = coordinator_address.split(':')[0]
    logger.debug("TPU Cluster using coordinator address: %s", coordinator_address)
    cls.wait_for_coordinator(coordinator_address, timeout_secs)
    return f'{coordinator_address}:{coordinator_port}'

  @classmethod
  def wait_for_coordinator(cls, coordinator_address, timeout_secs):
    # The coordinator may not be up before the other hosts try to
    # communicate with it. We check for its existence with retries.
    coordinator_found = False
    max_time = time.time() + timeout_secs
    coordinator_retry_secs = 5
    while not coordinator_found and time.time() < max_time:
      try:
        ip_address = socket.gethostbyname(coordinator_address)
        coordinator_found = True
        logger.debug("Found coordinator with address %s", coordinator_address)
      except socket.gaierror:
        logger.debug(
            "Failed to recognize coordinator address %s"
            " retrying...", coordinator_address
        )
        time.sleep(coordinator_retry_secs)
    if not coordinator_found:
      raise RuntimeError(f"Failed to recognize coordinator address {coordinator_address}")

  @classmethod
  def get_process_count(cls) -> int:
    processes_per_slice = len(cls._get_worker_list_in_slice())
    num_slices = cls._get_num_slices()
    total_process_count = processes_per_slice * num_slices
    logger.debug("Total process count of %s = %s processes per slice and %s slices", total_process_count, processes_per_slice, num_slices)
    return total_process_count

  @classmethod
  def get_process_id(cls) -> int:
    process_id_in_slice = cls._get_process_id_in_slice()
    slice_id = cls._get_slice_id()
    processes_per_slice = len(cls._get_worker_list_in_slice())
    process_id = process_id_in_slice + slice_id * processes_per_slice
    logger.debug("Process ID of %s generated by within-slice id %s and slice id %s", process_id, process_id_in_slice, slice_id)
    return process_id

  @staticmethod
  def _get_num_slices() -> int:
    if has_megascale_address():
      return int(get_tpu_env_value('MEGASCALE_NUM_SLICES'))
    else:
      return 1

  @staticmethod
  def _get_slice_id() -> int:
    if has_megascale_address():
      return int(get_tpu_env_value('MEGASCALE_SLICE_ID'))
    else:
      return 0

  @staticmethod
  def _get_process_id_in_slice() -> int:
    """Returns a process ID that is unique within slice."""
    raise NotImplementedError()

  @staticmethod
  def _get_worker_list_in_slice() -> list[str]:
    """Returns a list of worker endpoints/hostnames within slice."""
    raise NotImplementedError()

class GceTpuCluster(BaseTpuCluster):

  name: str = "gcetpu"

  @classmethod
  def is_env_present(cls) -> bool:
    if not running_in_cloud_tpu_vm:
      logger.debug("Did not detect cloud TPU VM")
      return False
    if os.environ.get("TPU_SKIP_MDS_QUERY") is not None:
      logger.debug("TPU_SKIP_MDS_QUERY is set to True, so it's probably not a GCE TPU cluster.")
      return False
    metadata_response, metadata_code = get_metadata('agent-worker-number')
    if metadata_code == metadata_response_code_success:
      logger.debug("Gce Tpu Cluster detected for Jax Distributed System")
      return True
    else:
      logger.debug("Did not detect Gce Tpu Cluster since agent-worker-number is not set in metadata")
      logger.debug("Metadata code: %s", metadata_code)
      logger.debug("Metadata response: %s", metadata_response)
      return False

  @staticmethod
  def _get_process_id_in_slice() -> int:
    return int(get_metadata('agent-worker-number')[0])

  @staticmethod
  def _get_worker_list_in_slice() -> list[str]:
    workers = get_metadata('worker-network-endpoints')[0].split(',')
    return [worker.split(':')[2] for worker in workers]

class GkeTpuCluster(BaseTpuCluster):

  name: str = "gketpu"

  @classmethod
  def is_env_present(cls) -> bool:
    if running_in_cloud_tpu_vm and os.environ.get("TPU_WORKER_HOSTNAMES") is not None:
      logger.debug("Gke Tpu Cluster detected for Jax Distributed System")
      return True
    else:
      if not running_in_cloud_tpu_vm:
        logger.debug("Did not detect cloud TPU VM")
      else:
        logger.debug("Did not detect TPU GKE cluster since TPU_WORKER_HOSTNAMES is not set")
      return False

  @staticmethod
  def _get_process_id_in_slice() -> int:
    return int(str(os.environ.get('TPU_WORKER_ID')))

  @staticmethod
  def _get_worker_list_in_slice() -> list[str]:
    return str(os.environ.get('TPU_WORKER_HOSTNAMES', None)).split(',')
