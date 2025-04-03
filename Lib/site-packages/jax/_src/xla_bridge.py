# Copyright 2018 The JAX Authors.
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

"""Interface and utility functions to XLA.

This module wraps the XLA client(s) and builders to standardize their interfaces
and provide some automatic type mapping logic for converting between Numpy and
XLA. There are also a handful of related casting utilities.
"""
from __future__ import annotations

import atexit
from collections.abc import Callable, Mapping
import dataclasses
from functools import lru_cache, partial
import importlib
import json
import logging
import os
import pkgutil
import platform as py_platform
import threading
import traceback
from typing import Any, Union
import warnings

from jax._src import config
from jax._src import distributed
from jax._src import hardware_utils
from jax._src import traceback_util
from jax._src import util
from jax._src.cloud_tpu_init import get_tpu_library_path
from jax._src.lib import cuda_versions
from jax._src.lib import xla_client
from jax._src.lib import xla_extension

logger = logging.getLogger(__name__)

jax_plugins: Any | None
try:
  import jax_plugins  # pytype: disable=import-error
except ModuleNotFoundError:
  jax_plugins = None
except ImportError as e:
  logger.error("Failed to import jax_plugins: %s", e)
  jax_plugins = None

traceback_util.register_exclusion(__file__)

XlaBackend = xla_client.Client

MIN_COMPUTE_CAPABILITY = 52

# TODO(phawkins): Remove jax_xla_backend.
_XLA_BACKEND = config.string_flag(
    'jax_xla_backend', '',
    'Deprecated, please use --jax_platforms instead.')
BACKEND_TARGET = config.string_flag(
    'jax_backend_target',
    os.getenv('JAX_BACKEND_TARGET', '').lower(),
    'Either "local" or "rpc:address" to connect to a remote service target.')
# TODO(skye): warn when this is used once we test out --jax_platforms a bit
_PLATFORM_NAME = config.string_flag(
    'jax_platform_name',
    os.getenv('JAX_PLATFORM_NAME', '').lower(),
    'Deprecated, please use --jax_platforms instead.')
CUDA_VISIBLE_DEVICES = config.string_flag(
    'jax_cuda_visible_devices', 'all',
    'Restricts the set of CUDA devices that JAX will use. Either "all", or a '
    'comma-separate list of integer device IDs.')
_ROCM_VISIBLE_DEVICES = config.string_flag(
    'jax_rocm_visible_devices', 'all',
    'Restricts the set of ROCM devices that JAX will use. Either "all", or a '
    'comma-separate list of integer device IDs.')

_MOCK_NUM_GPU_PROCESSES = config.int_flag(
    name="mock_num_gpu_processes",
    default=0,
    help="Mock number of JAX processes in GPU client. Value zero turns "
         "off mocking.",
)
_MOCK_GPU_TOPOLOGY = config.string_flag(
    name="jax_mock_gpu_topology",
    default="",
    help='Mock multi-host GPU topology in GPU client. The value should '
         'be of the form "<number-of-slices> x <number-of-hosts-per-slice> x '
         '<number-of-devices-per-host>". Empty string turns off mocking.',
)

_CPU_ENABLE_GLOO_COLLECTIVES = config.bool_flag(
    name="jax_cpu_enable_gloo_collectives",
    default=False,
    help="Deprecated, please use jax_cpu_collectives_implementation instead.",
)

CPU_COLLECTIVES_IMPLEMENTATIONS = ["none", "gloo", "mpi"]
CPU_COLLECTIVES_IMPLEMENTATION = config.enum_flag(
    name="jax_cpu_collectives_implementation",
    default="none",
    enum_values=CPU_COLLECTIVES_IMPLEMENTATIONS,
    help=(
        "Cross-process collective implementation used on CPU. Must be one of"
        f" {CPU_COLLECTIVES_IMPLEMENTATIONS}"
    ),
)

_CPU_ENABLE_ASYNC_DISPATCH = config.bool_flag(
    name="jax_cpu_enable_async_dispatch",
    default=True,
    help="Only applies to non-parallel computations. If False, run computations"
    "inline without async dispatch.",
)

NUM_CPU_DEVICES = config.int_flag(
    name="jax_num_cpu_devices",
    default=-1,
    help="Number of CPU devices to use. If not provided, the value of "
         "the XLA flag --xla_force_host_platform_device_count is used."
         " Must be set before JAX is initialized.",
)


# Warn the user if they call fork(), because it's not going to go well for them.
def _at_fork():
  warnings.warn(
    "os.fork() was called. os.fork() is incompatible with multithreaded code, "
    "and JAX is multithreaded, so this will likely lead to a deadlock.",
    RuntimeWarning, stacklevel=2)

_at_fork_handler_installed = False

# Backends


def tpu_client_timer_callback(timer_secs: float) -> xla_client.Client | None:
  def _log_warning():
    warnings.warn(
      f'TPU backend initialization is taking more than {timer_secs} seconds. '
      'Did you run your code on all TPU hosts? '
      'See https://jax.readthedocs.io/en/latest/multi_process.html '
      'for more information.')

  # Will log a warning after `timer_secs`.
  t = threading.Timer(timer_secs, _log_warning)
  t.start()

  try:
    client = xla_client.make_tpu_client(
        get_tpu_library_path(),
        _options_from_jax_configs("tpu"))
  finally:
    t.cancel()

  return client


# Backends
#
# We have no particular opinion about how "backends" relate to "devices". For
# example, there could be multiple backends that provide the same kind of
# device.

BackendFactory = Callable[[], Union[xla_client.Client, None]]
TopologyFactory = Callable[..., Union[xla_client.DeviceTopology, None]]

@dataclasses.dataclass
class BackendRegistration:
  factory: BackendFactory

  # Priority of this backend when choosing a default backend. Higher = more
  # preferred.
  priority: int

  # If this backend fails to initialize, should we log a user-visible warning?
  # For plugins (e.g., TPU) we usually want a visible failure, because why
  # install a plugin if you don't intend it to be used?
  fail_quietly: bool = False

  # Is this plugin experimental? If a plugin is deemed experimental, we issue
  # a warning when it is initialized. This is mostly to set user expectations
  # correctly: we don't want users to think that JAX is buggy because of a
  # a buggy plugin.
  experimental: bool = False

  # The C API (`PJRT_Api*`) if this backend is a plugin.
  c_api: Any | None = None

_backend_factories: dict[str, BackendRegistration] = {}
_default_backend: xla_client.Client | None = None
_backends : dict[str, xla_client.Client] = {}
_backend_errors : dict[str, str] = {}
_backend_lock = threading.Lock()
_plugins_registered: bool = False
_plugin_lock = threading.Lock()
_topology_factories: dict[str, TopologyFactory] = {}
_plugin_callbacks: list[Any] = []
_plugin_callback_lock = threading.Lock()

# The set of known non-experimental plugins.
#
# If a plugin passes the JAX test suite, it can be added to the allowlist below.
# Send a PR if you would like to be added.
#
# It is fine for a plugin not to implement every feature that JAX uses, provided
# that a reasonable feature set is implemented and the plugin fails gracefully
# for unimplemented features. Wrong outputs are not acceptable.
_nonexperimental_plugins: set[str] = {'cuda', 'rocm'}

# The set of known experimental plugins that have registrations in JAX codebase.
_experimental_plugins: set[str] = {"METAL"}

def register_backend_factory(name: str, factory: BackendFactory, *,
                             priority: int = 0,
                             fail_quietly: bool = True,
                             experimental: bool = False,
                             make_topology: TopologyFactory | None = None,
                             c_api: Any | None = None) -> None:
  with _backend_lock:
    if name in _backends:
      raise RuntimeError(f"Backend {name} already initialized")
  _backend_factories[name] = BackendRegistration(
    factory, priority, fail_quietly, experimental, c_api)
  if make_topology is not None:
    _topology_factories[name] = make_topology


def make_cpu_client(
    collectives: xla_client._xla.CpuCollectives | None = None,
) -> xla_client.Client:
  """Creates a CPU client with the requested collectives implementation.

  The implementation of CPU collectives used by the client is determined by the
  flag `--jax_cpu_collectives_implementation` - unless `collectives` is
  provided, in which case the flag is overridden and `collectives` is used.

  Args:
    collectives: An optional CPU collectives implementation, used by the client
      if provided.

  Raises:
    RuntimeError: If `--jax_cpu_collectives_implementation` is unknown.

  Returns:
    The created CPU client.
  """
  if collectives is None:
    collectives_impl = CPU_COLLECTIVES_IMPLEMENTATION.value
    if _CPU_ENABLE_GLOO_COLLECTIVES.value:
      collectives_impl = 'gloo'
      warnings.warn('Setting `jax_cpu_enable_gloo_collectives` is '
                      'deprecated. Please use `jax.config.update('
                      '"jax_cpu_collectives_implementation", "gloo")` instead.',
                      DeprecationWarning,
                      )
    if collectives_impl == 'gloo':
      collectives = xla_client._xla.make_gloo_tcp_collectives(
        distributed_client=distributed.global_state.client,
      )
    elif collectives_impl == 'mpi':
      collectives = xla_client._xla.make_mpi_collectives()
      collectives.Init()
      atexit.register(collectives.Finalize)
    elif collectives_impl != 'none':
      raise RuntimeError(f"Unknown collectives implementation "
                        f"{collectives_impl}. Available implementations are "
                        f"{CPU_COLLECTIVES_IMPLEMENTATIONS}.")

  num_devices = NUM_CPU_DEVICES.value if NUM_CPU_DEVICES.value >= 0 else None
  if xla_client._version < 303 and num_devices is not None:
    xla_flags = os.getenv("XLA_FLAGS") or ""
    os.environ["XLA_FLAGS"] = (
        f"{xla_flags} --xla_force_host_platform_device_count={num_devices}"
    )
    num_devices = None
  # TODO(phawkins): pass num_devices directly when version 303 is the minimum.
  kwargs = {} if num_devices is None else {"num_devices": num_devices}
  return xla_client.make_cpu_client(
    asynchronous=_CPU_ENABLE_ASYNC_DISPATCH.value,
    distributed_client=distributed.global_state.client,
    node_id=distributed.global_state.process_id,
    num_nodes=distributed.global_state.num_processes,
    collectives=collectives,
    **kwargs,
  )


register_backend_factory(
    "cpu", make_cpu_client, priority=0, fail_quietly=False
)


def _check_cuda_compute_capability(devices_to_check):
  for idx in devices_to_check:
    compute_cap = cuda_versions.cuda_compute_capability(idx)
    if compute_cap < MIN_COMPUTE_CAPABILITY:
      warnings.warn(
        f"Device {idx} has CUDA compute capability {compute_cap/10} which is "
        "lower than the minimum supported compute capability "
        f"{MIN_COMPUTE_CAPABILITY/10}. See "
        "https://jax.readthedocs.io/en/latest/installation.html#nvidia-gpu for "
        "more details",
        RuntimeWarning
      )


def _check_cuda_versions(raise_on_first_error: bool = False,
                         debug: bool = False):
  assert cuda_versions is not None
  results: list[dict[str, Any]] = []

  def _make_msg(name: str,
                runtime_version: int,
                build_version: int,
                min_supported: int,
                debug_msg: bool = False):
    if debug_msg:
      return (f"Package: {name}\n"
              f"Version JAX was built against: {build_version}\n"
              f"Minimum supported: {min_supported}\n"
              f"Installed version: {runtime_version}")
    if min_supported:
      req_str = (f"The local installation version must be no lower than "
                 f"{min_supported}.")
    else:
      req_str = ("The local installation must be the same version as "
                 "the version against which JAX was built.")
    msg = (f"Outdated {name} installation found.\n"
           f"Version JAX was built against: {build_version}\n"
           f"Minimum supported: {min_supported}\n"
           f"Installed version: {runtime_version}\n"
           f"{req_str}")
    return msg

  def _version_check(name: str,
                     get_version,
                     get_build_version,
                     scale_for_comparison: int = 1,
                     min_supported_version: int = 0):
    """Checks the runtime CUDA component version against the JAX one.

    Args:
      name: Of the CUDA component.
      get_version: A function to get the local runtime version of the component.
      get_build_version: A function to get the build version of the component.
      scale_for_comparison: For rounding down a version to ignore patch/minor.
      min_supported_version: An absolute minimum version required. Must be
        passed without rounding down.

    Raises:
      RuntimeError: If the component is not found, or is of unsupported version,
        and if raising the error is not deferred till later.
    """

    build_version = get_build_version()
    try:
      version = get_version()
    except Exception as e:
      err_msg = f"Unable to load {name}. Is it installed?"
      if raise_on_first_error:
        raise RuntimeError(err_msg) from e
      err_msg += f"\n{traceback.format_exc()}"
      results.append({"name": name, "installed": False, "msg": err_msg})
      return

    if not min_supported_version:
      min_supported_version = build_version // scale_for_comparison
    passed = min_supported_version <= version

    if not passed or debug:
      msg = _make_msg(name=name,
                      runtime_version=version,
                      build_version=build_version,
                      min_supported=min_supported_version,
                      debug_msg=passed)
      if not passed and raise_on_first_error:
        raise RuntimeError(msg)
      else:
        record = {"name": name,
                  "installed": True,
                  "msg": msg,
                  "passed": passed,
                  "build_version": build_version,
                  "version": version,
                  "minimum_supported": min_supported_version}
        results.append(record)

  _version_check("CUDA", cuda_versions.cuda_runtime_get_version,
                 cuda_versions.cuda_runtime_build_version,
                 scale_for_comparison=10,
                 min_supported_version=12010)
  _version_check(
      "cuDNN",
      cuda_versions.cudnn_get_version,
      cuda_versions.cudnn_build_version,
      # NVIDIA promise both backwards and forwards compatibility for cuDNN patch
      # versions:
      # https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html#api-compat
      scale_for_comparison=100,
      min_supported_version=9100
  )
  _version_check("cuFFT", cuda_versions.cufft_get_version,
                 cuda_versions.cufft_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100)
  _version_check("cuSOLVER", cuda_versions.cusolver_get_version,
                 cuda_versions.cusolver_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100,
                 min_supported_version=11400)
  _version_check("cuPTI", cuda_versions.cupti_get_version,
                 cuda_versions.cupti_build_version,
                 min_supported_version=18)
  _version_check("cuBLAS", cuda_versions.cublas_get_version,
                 cuda_versions.cublas_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100,
                 min_supported_version=120100)
  _version_check("cuSPARSE", cuda_versions.cusparse_get_version,
                 cuda_versions.cusparse_build_version,
                 # Ignore patch versions.
                 scale_for_comparison=100,
                 min_supported_version=12100)

  errors = []
  debug_results = []
  for result in results:
    message: str = result['msg']
    if not result['installed'] or not result['passed']:
      errors.append(message)
    else:
      debug_results.append(message)

  join_str = f'\n{"-" * 50}\n'
  if debug_results:
    print(f'CUDA components status (debug):\n'
          f'{join_str.join(debug_results)}')
  if errors:
    raise RuntimeError(f'Unable to use CUDA because of the '
                       f'following issues with CUDA components:\n'
                       f'{join_str.join(errors)}')

def _get_num_nodes_from_gpu_topology(topology: str) -> int:
    try:
      slices_str, hosts_per_slice_str, _ = topology.split("x", 2)
      return int(slices_str) * int(hosts_per_slice_str)
    except (IndexError, ValueError):
      raise ValueError('Mock topology must be of the form '
                       '"<number-of-slices> x <number-of-hosts-per-slice> x '
                       '<number-of-devices-per-host>".')

def make_gpu_client(
    *, platform_name: str, visible_devices_flag: config.Flag[str]
) -> xla_client.Client:
  visible_devices = visible_devices_flag.value
  allowed_devices = None
  if visible_devices != "all":
    allowed_devices = {int(x) for x in visible_devices.split(",")}

  mock_gpu_topology = _MOCK_GPU_TOPOLOGY.value or None
  mock_num_gpu_processes = (_get_num_nodes_from_gpu_topology(mock_gpu_topology) if
      mock_gpu_topology else _MOCK_NUM_GPU_PROCESSES.value)

  use_mock_gpu_client = mock_num_gpu_processes > 0
  num_nodes = (mock_num_gpu_processes if use_mock_gpu_client
      else distributed.global_state.num_processes)

  if platform_name == "cuda":
    if not os.getenv("JAX_SKIP_CUDA_CONSTRAINTS_CHECK"):
      _check_cuda_versions()
    else:
      print('Skipped CUDA versions constraints check due to the '
            'JAX_SKIP_CUDA_CONSTRAINTS_CHECK env var being set.')

    devices_to_check = (
        allowed_devices
        if allowed_devices
        else range(cuda_versions.cuda_device_count())
    )
    _check_cuda_compute_capability(devices_to_check)

  return xla_client.make_gpu_client(
      distributed_client=distributed.global_state.client,
      node_id=distributed.global_state.process_id,
      num_nodes=num_nodes,
      platform_name=platform_name,
      allowed_devices=allowed_devices,
      mock=use_mock_gpu_client,
  )


if hasattr(xla_client, "make_gpu_client"):
  register_backend_factory(
      "cuda",
      partial(
          make_gpu_client,
          platform_name="cuda",
          visible_devices_flag=CUDA_VISIBLE_DEVICES,
      ),
      priority=200,
      fail_quietly=True,
  )
  register_backend_factory(
      "rocm",
      partial(
          make_gpu_client,
          platform_name="rocm",
          visible_devices_flag=_ROCM_VISIBLE_DEVICES,
      ),
      priority=200,
      fail_quietly=True,
  )


if hasattr(xla_client, "make_tpu_client"):
  # TODO(phawkins,skyewm): switch TPU plugin to use the PJRT plugin mechanism,
  # and then fail loudly on initialization failure.
  register_backend_factory(
    'tpu', partial(tpu_client_timer_callback, timer_secs=60.0), priority=300,
    fail_quietly=True)


def _get_pjrt_plugin_names_and_library_paths(
    plugins_from_env: str,
) -> dict[str, str]:
  """Gets the names and library paths of PJRT plugins to load from env var.

  Args:
    plugins_from_env: plugin name and paths from env var. It is in the format
      of 'name1:path1,name2:path2' ('name1;path1,name2;path2' for windows).

  Returns:
    A dict of {plugin_name: library path} for the PJRT plugins to load.
  """
  if not plugins_from_env:
    return {}

  pjrt_plugins = {}
  for plugin in plugins_from_env.split(','):
    try:
      name, library_path = plugin.split(os.path.pathsep)
      pjrt_plugins[name] = library_path
    except ValueError:
      logger.warning(
          'invalid value %s in env var PJRT_NAMES_AND_LIBRARY_PATHS %s',
          plugin,
          plugins_from_env,
      )
  return pjrt_plugins


def _get_pjrt_plugin_config(
    json_path: str,
) -> tuple[
    str, Mapping[str, str | int | list[int] | float | bool] | None
]:
  """Gets PJRT plugin configuration from a json file.

  The json file needs to have a "library_path" field for the plugin library
  path. It can have an optional "create_option" field for the options used when
  creating a PJRT plugin client. The value of "create_option" is key-value
  pairs. Please see xla_client._NameValueMapping for the supported types of
  values.
  """
  with open(json_path) as f:
    config = json.load(f)
  if 'library_path' not in config.keys():
    raise ValueError(
        'PJRT plugin config file should contain "library_path" field.'
    )
  return (config['library_path'], config.get('create_options'))

def discover_pjrt_plugins() -> None:
  """Discovers plugins in the namespace package `jax_plugins` and import them.

  There are two methods used to discover plugin modules. They are intended
  to be used together by implementors in order to cover all packaging and
  development cases:

  1. Define a globally unique module under the `jax_plugins` namespace
     package (i.e. just create a `jax_plugins` directory and define your
     module below it).
  2. If building a package via pyproject.toml or setup.py, advertise your
     plugin module name by including an entry-point under the `jax_plugins`
     group which points to your full module name.

  During Jax startup, Jax will load each module discovered in such a way and
  call its `initialize()` function. It is expected that this function should
  register its concrete plugin name/implementations via call(s) to
  `jax._src.xla_bridge.register_plugin(name, priority=, library_paty=,
  options=)`. Since `initialize()` functions are called for all installed
  plugins, they should avoid doing expensive, non-registration related work.

  TODO: We should provide a variant of `register_plugin` which allows the
  library_path and options to be resolved via a callback. This would enable
  light-weight plugin registration in cases where options need to be derived
  from heavy-weight system initialization.
  """
  plugin_modules = set()
  # Scan installed modules under |jax_plugins|. Note that not all packaging
  # scenarios are amenable to such scanning, so we also use the entry-point
  # method to seed the list.
  if jax_plugins:
    for _, name, _ in pkgutil.iter_modules(
        jax_plugins.__path__, jax_plugins.__name__ + '.'
    ):
      logger.debug("Discovered path based JAX plugin: %s", name)
      plugin_modules.add(name)
  else:
    logger.debug("No jax_plugins namespace packages available")

  # Augment with advertised entrypoints.
  from importlib.metadata import entry_points

  for entry_point in entry_points(group="jax_plugins"):
    logger.debug("Discovered entry-point based JAX plugin: %s",
                 entry_point.value)
    plugin_modules.add(entry_point.value)

  # Now load and initialize them all.
  for plugin_module_name in plugin_modules:
    logger.debug("Loading plugin module %s", plugin_module_name)
    plugin_module = None
    try:
      plugin_module = importlib.import_module(plugin_module_name)
    except ModuleNotFoundError:
      logger.warning("Jax plugin configuration error: Plugin module %s "
                     "does not exist", plugin_module_name)
    except ImportError:
      logger.exception("Jax plugin configuration error: Plugin module %s "
                       "could not be loaded")

    if plugin_module:
      try:
        plugin_module.initialize()
      except:
        logger.exception("Jax plugin configuration error: Exception when "
                         "calling %s.initialize()", plugin_module_name)


def _options_from_jax_configs(plugin_name):
  options = {}

  pjrt_client_options = config.jax_pjrt_client_create_options.value
  pjrt_client_option_list = []
  if pjrt_client_options:
    pjrt_client_option_list = pjrt_client_options.split(";")

  for option in pjrt_client_option_list:
    option_list = option.split(":")
    if (len(option_list) != 2):
      raise RuntimeError(
          "Multiple ':' separators for option in "
          f"jax_pjrt_client_create_options: '{option}'. "
          "Should be in format 'key:value'")
    options[option_list[0]] = option_list[1]

  if plugin_name == "cuda":
    visible_devices = CUDA_VISIBLE_DEVICES.value
    if visible_devices != 'all':
      options['visible_devices'] = [int(x) for x in visible_devices.split(',')]
    mock_gpu_topology = _MOCK_GPU_TOPOLOGY.value or None
    mock_num_processes = (_get_num_nodes_from_gpu_topology(mock_gpu_topology) if
        mock_gpu_topology else _MOCK_NUM_GPU_PROCESSES.value)
    options['enable_mock_nccl'] = mock_num_processes > 0
    if mock_num_processes > 0:
      options['num_nodes'] = mock_num_processes
      if mock_gpu_topology:
        options['mock_gpu_topology'] = mock_gpu_topology

  return options


# TODO(b/261345120): decide on a public name and expose a public method which is
# an alias of this method.
def register_plugin(
    plugin_name: str,
    *,
    priority: int = 400,
    library_path: str | None = None,
    options: Mapping[str, str | int | list[int] | float | bool] | None = None,
    c_api: Any | None = None,
) -> Any:
  """Registers a backend factory for the PJRT plugin.

  Args:
    plugin_name: the name of the plugin.
    priority: the priority this plugin should be registered in jax backends.
      Default to be 400.
    library_path: Optional. The full path to the .so file of the plugin. The
      plugin needs to provide either the library_path or the c_api.
    options: Optional. It is used when creating a PJRT plugin client.
    c_api: Optional. The plugin can provide a PJRT C API to be registered.
  """
  def factory():
    if not xla_client.pjrt_plugin_initialized(plugin_name):
      xla_client.initialize_pjrt_plugin(plugin_name)
    updated_options = {}
    if options is not None:
      updated_options.update(options)
    updated_options.update(_options_from_jax_configs(plugin_name))
    if distributed.global_state.client is None:
      return xla_client.make_c_api_client(plugin_name, updated_options, None)

    distribute_options = {
        'node_id': distributed.global_state.process_id,
        'num_nodes': distributed.global_state.num_processes,
    }
    if options is not None:
      distribute_options.update(updated_options)
    return xla_client.make_c_api_client(
        plugin_name, distribute_options, distributed.global_state.client
    )

  if library_path and c_api:
    logger.error(
        "Both library_path and c_api are provided when registering PJRT plugin"
        " %s",
        plugin_name,
    )
    return
  if not library_path and not c_api:
    logger.error(
        "Neither library_path nor c_api provided when registering PJRT plugin"
        " %s",
        plugin_name,
    )
    return

  logger.debug(
      'registering PJRT plugin %s from %s', plugin_name, library_path
  )
  if library_path is not None:
    c_api = xla_client.load_pjrt_plugin_dynamically(plugin_name, library_path)
    xla_client.profiler.register_plugin_profiler(c_api)
  else:
    assert c_api is not None
    xla_client.load_pjrt_plugin_with_c_api(plugin_name, c_api)

  make_topology = partial(xla_client.make_c_api_device_topology, c_api)
  experimental = plugin_name not in _nonexperimental_plugins
  register_backend_factory(plugin_name, factory, priority=priority,
                           fail_quietly=False, experimental=experimental,
                           make_topology=make_topology, c_api=c_api)
  return c_api


def register_pjrt_plugin_factories_from_env() -> None:
  """Registers backend factories for PJRT plugins.

  A backend factory will be registered for every PJRT plugin in the input
  string, in the format of 'name1:path1,name2:path2' ('name1;path1,name2;path2'
  for windows). The path can be a path to the plugin library or a path to the
  plugin configuration json file. The json file needs to have a "library_path"
  field for the plugin library path. It can have an optional "create_option"
  field for the options used when creating a PJRT plugin client. The value of
  "create_option" is key-value pairs. Please see xla_client._NameValueMapping
  for the supported types of values.

  TPU PJRT plugin will be loaded and registered separately in make_tpu_client.
  """
  pjrt_plugins = _get_pjrt_plugin_names_and_library_paths(
      os.getenv('PJRT_NAMES_AND_LIBRARY_PATHS', '')
  )
  for plugin_name, path in pjrt_plugins.items():
    if path.endswith('.json'):
      library_path, options = _get_pjrt_plugin_config(path)
    else:
      library_path = path
      options = None
    logger.debug(
        'registering PJRT plugin %s from %s', plugin_name, library_path
    )
    register_plugin(plugin_name, library_path=library_path, options=options)


def _discover_and_register_pjrt_plugins():
  global _plugins_registered

  # Needs a separate lock because register_backend_factory (called from
  # register_plugin) requires to hold _backend_lock.
  with _plugin_lock:
    if not _plugins_registered:
      # Plugins in the namespace package `jax_plugins` or have an entry-point
      # under the `jax_plugins` group will be imported.
      discover_pjrt_plugins()
      # Registers plugins names and paths set in env var
      # PJRT_NAMES_AND_LIBRARY_PATHS, in the format of 'name1:path1,name2:path2'
      # ('name1;path1,name2;path2' for windows).
      register_pjrt_plugin_factories_from_env()
      with _plugin_callback_lock:
        for factory in _backend_factories.values():
          if factory.c_api is not None:
            for callback in _plugin_callbacks:
              callback(c_api=factory.c_api)
      _plugins_registered = True


_platform_aliases = {
  "cuda": "gpu",
  "rocm": "gpu",
}

_alias_to_platforms: dict[str, list[str]] = {}
for _platform, _alias in _platform_aliases.items():
  _alias_to_platforms.setdefault(_alias, []).append(_platform)


def known_platforms() -> set[str]:
  platforms = set()
  platforms |= set(_nonexperimental_plugins)
  platforms |= set(_experimental_plugins)
  platforms |= set(_backend_factories.keys())
  platforms |= set(_platform_aliases.values())
  return platforms


def is_known_platform(platform: str) -> bool:
  # A platform is valid if there is a registered factory for it. It does not
  # matter if we were unable to initialize that platform; we only care that
  # we've heard of it and it isn't, e.g., a typo.
  return platform in known_platforms()


def canonicalize_platform(platform: str) -> str:
  """Replaces platform aliases with their concrete equivalent.

  In particular, replaces "gpu" with either "cuda" or "rocm", depending on which
  hardware is actually present. We want to distinguish "cuda" and "rocm" for
  purposes such as MLIR lowering rules, but in many cases we don't want to
  force users to care.
  """
  platforms = _alias_to_platforms.get(platform, None)
  if platforms is None:
    return platform

  b = backends()
  for p in platforms:
    if p in b.keys():
      return p
  raise RuntimeError(f"Unknown backend: '{platform}' requested, but no "
                     f"platforms that are instances of {platform} are present. "
                     "Platforms present are: " + ",".join(b.keys()))


def expand_platform_alias(platform: str) -> list[str]:
  """Expands, e.g., "gpu" to ["cuda", "rocm"].

  This is used for convenience reasons: we expect cuda and rocm to act similarly
  in many respects since they share most of the same code.
  """
  return _alias_to_platforms.get(platform, [platform])

def is_gpu(platform):
  return platform in ("cuda", "rocm")


def backends_are_initialized() -> bool:
  "Returns true if backends have already been initialized."
  with _backend_lock:
    return len(_backends) != 0


def register_plugin_callbacks(callback):
  """Registers a callback to be called with c_api after plugins discovery.

  The callback will be called on all discovered PJRT C API plugins. If
  `register_plugin_callbacks` is called before the plugins are discovered, the
  callback will be called right after the plugins are discovered. Otherwise, the
  callback will be called immediately when `register_plugin_callbacks` is
  called.

  Args:
    callback: the callback to be called with c_api.
  """
  with _plugin_callback_lock:
    if _plugins_registered:
      for factory in _backend_factories.values():
        if factory.c_api is not None:
          callback(c_api=factory.c_api)
    else:
      _plugin_callbacks.append(callback)


def backends() -> dict[str, xla_client.Client]:
  global _backends
  global _backend_errors
  global _default_backend
  global _at_fork_handler_installed

  _discover_and_register_pjrt_plugins()

  with _backend_lock:
    if _backends:
      return _backends

    # os.register_at_fork only exists on Unix.
    if not _at_fork_handler_installed and hasattr(os, "register_at_fork"):
      os.register_at_fork(before=_at_fork)
      _at_fork_handler_installed = True

    if jax_platforms := config.jax_platforms.value:
      platforms = []
      # Allow platform aliases in the list of platforms.
      for platform in jax_platforms.split(","):
        platforms.extend(expand_platform_alias(platform))
      priorities = range(len(platforms), 0, -1)
      # If the user specified a list of platforms explicitly, always fail
      # loudly.
      fail_quietly_list = [False] * len(platforms)
      platform_registrations = list(
        zip(platforms, priorities, fail_quietly_list))
    else:
      platform_registrations = [
          (platform, registration.priority, registration.fail_quietly)
          for platform, registration
          in _backend_factories.items()
      ]
    default_priority = -1000
    for platform, priority, fail_quietly in platform_registrations:
      try:
        if platform == "cuda" and not hardware_utils.has_visible_nvidia_gpu():
          continue

        backend = _init_backend(platform)
        _backends[platform] = backend

        if priority > default_priority:
          _default_backend = backend
          default_priority = priority
      except Exception as err:
        err_msg = f"Unable to initialize backend '{platform}': {err}"
        if fail_quietly:
          _backend_errors[platform] = str(err)
          logger.info(err_msg)
        else:
          if config.jax_platforms.value:
            err_msg += " (set JAX_PLATFORMS='' to automatically choose an available backend)"
          else:
            err_msg += " (you may need to uninstall the failing plugin package, or set JAX_PLATFORMS=cpu to skip this backend.)"
          raise RuntimeError(err_msg)

    assert _default_backend is not None
    if not config.jax_platforms.value:
      _suggest_missing_backends()
    return _backends

# Code to suggest plugins that should be installed.
#
# Plugin vendors are welcome to add code to this list, assuming there's a
# lightweight way to determine if hardware is present without requiring
# the relevant plugin be installed.

def _suggest_missing_backends():
  if py_platform.system() != "Linux":
    # If you're not using Linux (or WSL2), we don't have any suggestions at the
    # moment.
    return

  assert _default_backend is not None
  default_platform = _default_backend.platform
  if "cuda" not in _backends and hardware_utils.has_visible_nvidia_gpu():
    if hasattr(xla_extension, "GpuAllocatorConfig") and "cuda" in _backend_errors:
      err = _backend_errors["cuda"]
      warning_msg = f"CUDA backend failed to initialize: {err}."
      if "no supported devices found for platform CUDA." in err:
        warning_msg += (
          "This may be due to JAX pre-allocating too much device "
          "memory, leaving too little for CUDA library initialization. See "
          "https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html "
          "for more details and potential workarounds."
        )
      warning_msg += "(Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)"

      logger.warning(warning_msg)
    else:
      logger.warning("An NVIDIA GPU may be present on this machine, but a "
                     "CUDA-enabled jaxlib is not installed. Falling back to "
                     f"{default_platform}.")
  elif "tpu" not in _backends and hardware_utils.num_available_tpu_chips_and_device_id()[0] > 0:
    logger.warning("A Google TPU may be present on this machine, but either a "
                    "TPU-enabled jaxlib or libtpu is not installed. Falling "
                    f"back to {default_platform}.")


def _clear_backends() -> None:
  global _backends
  global _backend_errors
  global _default_backend

  logger.debug("Clearing JAX backend caches.")
  with _backend_lock:
    _backends = {}
    _backend_errors = {}
    _default_backend = None

  get_backend.cache_clear()


def _init_backend(platform: str) -> xla_client.Client:
  registration = _backend_factories.get(platform, None)
  if registration is None:
    raise RuntimeError(
        f"Backend '{platform}' is not in the list of known backends: "
        f"{list(_backend_factories.keys())}.")

  if registration.experimental:
    logger.warning(f"Platform '{platform}' is experimental and not all JAX "
                   "functionality may be correctly supported!")
  logger.debug("Initializing backend '%s'", platform)
  backend = registration.factory()
  # TODO(skye): consider raising more descriptive errors directly from backend
  # factories instead of returning None.
  if backend is None:
    raise RuntimeError(f"Could not initialize backend '{platform}'")
  if backend.device_count() == 0:
    raise RuntimeError(f"Backend '{platform}' provides no devices.")
  util.distributed_debug_log(("Initialized backend", backend.platform),
                             ("process_index", backend.process_index()),
                             ("device_count", backend.device_count()),
                             ("local_devices", backend.local_devices()))
  logger.debug("Backend '%s' initialized", platform)
  return backend


def _get_backend_uncached(
    platform: None | str | xla_client.Client = None
) -> xla_client.Client:
  # TODO(mattjj,skyewm): remove this input polymorphism after we clean up how
  # 'backend' values are handled
  if platform is not None and not isinstance(platform, str):
    return platform

  platform = (platform or _XLA_BACKEND.value or _PLATFORM_NAME.value or None)

  bs = backends()
  if platform is not None:
    platform = canonicalize_platform(platform)
    backend = bs.get(platform, None)
    if backend is None:
      if platform in _backend_errors:
        raise RuntimeError(f"Backend '{platform}' failed to initialize: "
                           f"{_backend_errors[platform]}. "
                           f'Available backends are {list(bs)}')
      raise RuntimeError(
          f"Unknown backend {platform}. Available backends are {list(bs)}")
    return backend
  else:
    assert _default_backend is not None
    return _default_backend


@lru_cache(maxsize=None)  # don't use util.memoize because there is no X64 dependence.
def get_backend(
    platform: None | str | xla_client.Client = None
) -> xla_client.Client:
  return _get_backend_uncached(platform)


def get_device_backend(
    device: xla_client.Device | None = None,
) -> xla_client.Client:
  """Returns the Backend associated with `device`, or the default Backend."""
  if device is not None:
    return device.client
  return get_backend()


def device_count(
    backend: str | xla_client.Client | None = None
) -> int:
  """Returns the total number of devices.

  On most platforms, this is the same as :py:func:`jax.local_device_count`.
  However, on multi-process platforms where different devices are associated
  with different processes, this will return the total number of devices across
  all processes.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    Number of devices.

  """
  return int(get_backend(backend).device_count())


def local_device_count(
    backend: str | xla_client.Client | None = None
) -> int:
  """Returns the number of devices addressable by this process."""
  return int(get_backend(backend).local_device_count())


def devices(
    backend: str | xla_client.Client | None = None
) -> list[xla_client.Device]:
  """Returns a list of all devices for a given backend.

  .. currentmodule:: jaxlib.xla_extension

  Each device is represented by a subclass of :class:`Device` (e.g.
  :class:`CpuDevice`, :class:`GpuDevice`). The length of the returned list is
  equal to ``device_count(backend)``. Local devices can be identified by
  comparing :attr:`Device.process_index` to the value returned by
  :py:func:`jax.process_index`.

  If ``backend`` is ``None``, returns all the devices from the default backend.
  The default backend is generally ``'gpu'`` or ``'tpu'`` if available,
  otherwise ``'cpu'``.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    List of Device subclasses.
  """
  return get_backend(backend).devices()


def default_backend() -> str:
  """Returns the platform name of the default XLA backend."""
  return get_backend(None).platform


def backend_pjrt_c_api_version(platform=None) -> tuple[int, int] | None:
  """Returns the PJRT C API version of the backend.

  Returns None if the backend does not use PJRT C API.
  """
  backend = get_backend(platform)
  if hasattr(backend, "pjrt_c_api_major_version") and hasattr(
      backend, "pjrt_c_api_minor_version"
  ):
    return (backend.pjrt_c_api_major_version, backend.pjrt_c_api_minor_version)
  return None


def backend_xla_version(platform=None) -> int | None:
  """Returns the XLA version of the backend.

  Returns None if the backend does not use PJRT C API or does not have
  xla_version in the plugin attributes. This methon can be used to skip features
  that are not available before certain xla_version if the backend is a
  plugin and uses xla_version.
  """
  backend = get_backend(platform)
  return getattr(backend, "xla_version", None)


@lru_cache
def local_devices(process_index: int | None = None,
                  backend: str | xla_client.Client | None = None,
                  host_id: int | None = None) -> list[xla_client.Device]:
  """Like :py:func:`jax.devices`, but only returns devices local to a given process.

  If ``process_index`` is ``None``, returns devices local to this process.

  Args:
    process_index: the integer index of the process. Process indices can be
      retrieved via ``len(jax.process_count())``.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    List of Device subclasses.
  """
  if host_id is not None:
    warnings.warn(
        "The argument to jax.local_devices has been renamed from `host_id` to "
        "`process_index`. This alias will eventually be removed; please update "
        "your code.")
    process_index = host_id
  if process_index is None:
    process_index = get_backend(backend).process_index()
  if not (0 <= process_index < process_count(backend)):
    raise ValueError(f"Unknown process_index {process_index}")
  return [d for d in devices(backend) if d.process_index == process_index]


def process_index(
    backend: str | xla_client.Client | None = None
) -> int:
  """Returns the integer process index of this process.

  On most platforms, this will always be 0. This will vary on multi-process
  platforms though.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    Integer process index.
  """
  return get_backend(backend).process_index()


# TODO: remove this sometime after jax 0.2.13 is released
def host_id(backend: str | xla_client.Client | None = None) -> int:
  warnings.warn(
      "jax.host_id has been renamed to jax.process_index. This alias "
      "will eventually be removed; please update your code.")
  return process_index(backend)


@lru_cache
def process_count(
    backend: str | xla_client.Client | None = None
) -> int:
  """Returns the number of JAX processes associated with the backend."""
  return max(d.process_index for d in devices(backend)) + 1


# TODO: remove this sometime after jax 0.2.13 is released
def host_count(backend: str | xla_client.Client | None = None) -> int:
  warnings.warn(
      "jax.host_count has been renamed to jax.process_count. This alias "
      "will eventually be removed; please update your code.")
  return process_count(backend)


def process_indices(
    backend: str | xla_client.Client | None = None
) -> list[int]:
  """Returns the list of all JAX process indices associated with the backend.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    List of integer process indices.
  """
  return list(range(process_count(backend)))


# TODO: remove this sometime after jax 0.2.13 is released
def host_ids(
    backend: str | xla_client.Client | None = None
) -> list[int]:
  warnings.warn(
      "jax.host_ids has been renamed to jax.process_indices. This alias "
      "will eventually be removed; please update your code.")
  return process_indices(backend)


def using_pjrt_c_api(backend=None):
  return "PJRT C API" in get_backend(backend).platform_version

def make_pjrt_topology(platform: str, topology_name='', **kwargs):
  _discover_and_register_pjrt_plugins()
  actual_platform = canonicalize_platform(platform)
  with _backend_lock:
    if actual_platform in _topology_factories:
      return _topology_factories[actual_platform](topology_name, **kwargs)
  raise NotImplementedError("topology not implemented for %s" % platform)


# TODO(parkers): Get rid of this in favor of a generic way to get topologies.
def make_pjrt_tpu_topology(topology_name='', **kwargs):
  if not xla_client.pjrt_plugin_loaded("tpu"):
    library_path = get_tpu_library_path()
    if library_path is None:
      raise RuntimeError(
          "JAX TPU support not installed; cannot generate TPU topology. See"
          " https://github.com/jax-ml/jax#installation")
    c_api = xla_client.load_pjrt_plugin_dynamically("tpu", library_path)
    xla_client.profiler.register_plugin_profiler(c_api)
  assert xla_client.pjrt_plugin_loaded("tpu")
  if not xla_client.pjrt_plugin_initialized("tpu"):
    xla_client.initialize_pjrt_plugin("tpu")
  return xla_client.make_tfrt_tpu_c_api_device_topology(
      topology_name, **kwargs
  )
