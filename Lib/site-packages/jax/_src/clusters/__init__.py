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

from .cluster import ClusterEnv as ClusterEnv

# Order of declaration of the cluster environments
# will dictate the order in which they will be checked.
# Therefore, if multiple environments are available and
# the user did not explicitly provide the arguments
# to :func:`jax.distributed.initialize`, the first
# available one from the list will be picked.
from .ompi_cluster import OmpiCluster as OmpiCluster
from .slurm_cluster import SlurmCluster as SlurmCluster
from .mpi4py_cluster import Mpi4pyCluster as Mpi4pyCluster
from .cloud_tpu_cluster import GkeTpuCluster as GkeTpuCluster
from .cloud_tpu_cluster import GceTpuCluster as GceTpuCluster
from .k8s_cluster import K8sCluster as K8sCluster
