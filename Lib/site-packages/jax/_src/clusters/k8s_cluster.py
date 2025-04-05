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

from contextlib import contextmanager
from functools import cache
import os
import socket
import textwrap
import warnings
from jax._src import clusters


class K8sCluster(clusters.ClusterEnv):

  # Use an arbitrarily chosen port for the coordinator since we cannot
  # rely on communication to choose one in real time.
  _coordinator_port = '55527'

  @classmethod
  def is_env_present(cls) -> bool:
    if 'KUBERNETES_SERVICE_HOST' in os.environ:
      try:
        import kubernetes as k8s  # pytype: disable=import-error
      except ImportError as e:
        warnings.warn(textwrap.fill(
            "Kubernetes environment detected, but the `kubernetes` package is "
            "not installed to enable automatic bootstrapping in this "
            "environment. To enable automatic boostrapping, please install "
            "jax with the [k8s] extra. For example:"
            "    pip install jax[k8s]"
            " OR"
            "    pip install jax[k8s,<MORE-EXTRAS...>]"
        ))
        return False

      k8s.config.load_incluster_config()
      cls._core_api = k8s.client.CoreV1Api()
      cls._batch_api = k8s.client.BatchV1Api()
      cls._ApiException = k8s.client.exceptions.ApiException
      return True
    else:
      return False

  @classmethod
  @contextmanager
  def _handle_api_exception(cls):
    try:
      yield
    except cls._ApiException as e:
      err_msg = [f"Kubernetes API Error: {e.status} - {e.reason}"]
      if e.status == 403:
        err_msg.append(textwrap.fill(
          "It appears that the Kubernetes service account (SA) associated with "
          "this job does not have the permission for pod introspection. Please "
          "either grant the default SA permission to read pod info, or create a "
          "dedicated service account with the permission and associated with "
          "the job. For more details, see <PLACERHOLDER_LINK>.",
          width=80
        ))
      raise RuntimeError('\n'.join(err_msg)) from e

  @classmethod
  @cache
  def _namespace(cls):
    return open(
      '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    ).read().strip()

  @classmethod
  @cache
  def _pod(cls):
    with cls._handle_api_exception():
      ip = socket.gethostbyname(os.getenv('HOSTNAME'))
      pods = cls._core_api.list_namespaced_pod(
        namespace=cls._namespace(),
        field_selector=f'status.podIP={ip}'
      ).items
      assert len(pods) == 1, \
        f"Exactly 1 Kubernetes pod should have IP {ip}, got {len(pods)}."
      return pods[0]

  @classmethod
  @cache
  def _job(cls):
    with cls._handle_api_exception():
      return cls._batch_api.read_namespaced_job(
        name=cls._pod().metadata.labels['job-name'], namespace=cls._namespace()
      )

  @classmethod
  def get_coordinator_address(cls, timeout_secs: int | None) -> str:
    return '{job_name}-0.{jobset_name}:{port}'.format(
      job_name=cls._pod().metadata.labels['job-name'],
      jobset_name=cls._job().metadata.labels['jobset.sigs.k8s.io/jobset-name'],
      port=cls._coordinator_port
    )

  @classmethod
  def get_process_count(cls) -> int:
    # https://kubernetes.io/docs/concepts/workloads/controllers/job/#controlling-parallelism
    return cls._job().spec.parallelism

  @classmethod
  def get_process_id(cls) -> int:
    # https://kubernetes.io/docs/concepts/workloads/controllers/job/#completion-mode
    try:
      return int(os.environ['JOB_COMPLETION_INDEX'])
    except KeyError:
      raise RuntimeError(
        'K8s job must be run with `completionMode: "Indexed"`.'
      )
