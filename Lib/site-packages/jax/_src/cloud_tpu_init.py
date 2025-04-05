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

import os
from jax import version
from jax._src import config
from jax._src import hardware_utils

running_in_cloud_tpu_vm: bool = False


def maybe_import_libtpu():
  try:
    # pylint: disable=import-outside-toplevel
    # pytype: disable=import-error
    import libtpu

    # pytype: enable=import-error
    # pylint: enable=import-outside-toplevel
  except ImportError:
    return None
  else:
    return libtpu


def get_tpu_library_path() -> str | None:
  path_from_env = os.getenv("TPU_LIBRARY_PATH")
  if path_from_env is not None and os.path.isfile(path_from_env):
    return path_from_env

  libtpu_module = maybe_import_libtpu()
  if libtpu_module is not None:
    return libtpu_module.get_library_path()

  return None


def jax_force_tpu_init() -> bool:
  return 'JAX_FORCE_TPU_INIT' in os.environ


def cloud_tpu_init() -> None:
  """Automatically sets Cloud TPU topology and other env vars.

  **This must be called before the TPU runtime is loaded, which happens as soon
  as JAX's C++ backend is loaded! I.e. call this before xla_bridge or xla_client
  is imported.**

  Safe to call in non-Cloud TPU environments.

  Some of these environment variables are used to tell the TPU runtime what kind
  of mesh topology to use. It assumes a single-host topology by default, so we
  manually set them here to default to the full pod slice if applicable.

  This will not set any env vars if a single topology-related env var is already
  set.
  """
  global running_in_cloud_tpu_vm

  # Exit early if we're not running on a Cloud TPU VM or libtpu isn't installed.
  libtpu_path = get_tpu_library_path()
  num_tpu_chips = hardware_utils.num_available_tpu_chips_and_device_id()[0]
  if (libtpu_path is None or num_tpu_chips == 0) and not jax_force_tpu_init():
    return

  running_in_cloud_tpu_vm = True

  os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
  os.environ.setdefault('TPU_ML_PLATFORM', 'JAX')
  os.environ.setdefault('TPU_ML_PLATFORM_VERSION', version.__version__)
  os.environ.setdefault('ENABLE_RUNTIME_UPTIME_TELEMETRY', '1')
  if '--xla_tpu_use_enhanced_launch_barrier' not in os.environ.get('LIBTPU_INIT_ARGS', ''):
    os.environ['LIBTPU_INIT_ARGS'] = os.environ.get('LIBTPU_INIT_ARGS','') + ' --xla_tpu_use_enhanced_launch_barrier=true'

  # this makes tensorstore serialization work better on TPU
  os.environ.setdefault('TENSORSTORE_CURL_LOW_SPEED_TIME_SECONDS', '60')
  os.environ.setdefault('TENSORSTORE_CURL_LOW_SPEED_LIMIT_BYTES', '256')

  # If the JAX_PLATFORMS env variable isn't set, config.jax_platforms defaults
  # to None. In this case, we set it to 'tpu,cpu' to ensure that JAX uses the
  # TPU backend.
  if config.jax_platforms.value is None:
    config.update('jax_platforms', 'tpu,cpu')

  if config.jax_pjrt_client_create_options.value is None:
    config.update(
      'jax_pjrt_client_create_options',
      f'ml_framework_name:JAX;ml_framework_version:{version.__version__}'
      )
