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

import copy
import enum
import hashlib
import io
import logging
import os
import sys
from typing import cast as type_cast

from jax._src import config
from jax._src.lib import version_str as jaxlib_version_str
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.lib.mlir import passmanager as pm
import numpy as np


logger = logging.getLogger(__name__)

_extra_flag_prefixes: list[str] = []

def add_flag_prefixes(flag_prefixes: list[str]) -> None:
  """Add flag prefixes to include in the cache key. Call prior to get().
  """
  global _extra_flag_prefixes
  _extra_flag_prefixes += flag_prefixes


def clear_flag_prefixes() -> None:
  """Clear flag prefixes added by add_flag_prefixes().
  """
  global _extra_flag_prefixes
  _extra_flag_prefixes = []


def get_flag_prefixes() -> list[str]:
  """Return flag prefixes added by add_flag_prefixes().
  """
  return _extra_flag_prefixes


def custom_hook() -> str:
  """Custom hook for any addition to the cache key.

  The custom hook will be called everytime get() is called and can be
  defined to return a string that will be hashed into the cache key.
  """
  return ""


class IgnoreCallbacks(enum.IntEnum):
  # Do not remove any callback pointers from precompiled IR.
  NO = enum.auto()
  # Remove all callback pointers from precompiled IR.
  ALL = enum.auto()
  # Remove only custom_partitioning callback pointer from precompiled IR.
  CUSTOM_PARTITIONING = enum.auto()


def get(
    module: ir.Module,
    devices: np.ndarray,
    compile_options: xla_client.CompileOptions,
    backend: xla_client.Client,
    compression_algorithm: str = "zstandard",
    ignore_callbacks: IgnoreCallbacks = IgnoreCallbacks.NO,
) -> str:
  """Creates a hashed string to use as a key to the compilation cache.

  Creates a cache key that is a hex-encoded string of a unique hash based on
  the arguments. The hex-encoded string is 256 characters long.

  Args:
    module: the input program
    devices: an array of accelerator devices that the program will run on
    compile_options: options passed to the XLA compiler
    backend: description of the platform (e.g., TPU version)
    compression_algorithm: a string representing the compression algorithm used
      for the executable before persisting in the cache
    ignore_callbacks: whether to remove the all callback pointer from the
      computation.

  Typical return value example:
   'jit__psum-14ac577cdb2ef6d986078b4054cc9893a9a14a16dbb0d8f37b89167c1f1aacdf'
  """
  entries = [
      (
          "computation",
          lambda hash_obj: _hash_computation(
              hash_obj, module, ignore_callbacks
          ),
      ),
      (
          "jax_lib version",
          lambda hash_obj: hash_obj.update(
              bytes(jaxlib_version_str.encode("utf-8"))
          ),
      ),
      (
          "XLA flags",
          lambda hash_obj: _hash_xla_flags(hash_obj, get_flag_prefixes()),
      ),
      (
          "compile_options",
          lambda hash_obj: _hash_serialized_compile_options(
              hash_obj,
              compile_options,
              # In case of GPU multi-process tasks we need to strip device
              # assignment to use cache key as invariant between processes.
              strip_device_assignment=(backend.platform == "gpu"),
          ),
      ),
      (
          "accelerator_config",
          lambda hash_obj: _hash_accelerator_config(hash_obj, devices, backend),
      ),
      (
          "compression",
          lambda hash_obj: _hash_string(hash_obj, compression_algorithm),
      ),
      ("custom_hook", lambda hash_obj: _hash_string(hash_obj, custom_hook())),
  ]

  hash_obj = hashlib.sha256()
  for name, hashfn in entries:
    hashfn(hash_obj)
    _log_cache_key_hash(hash_obj, name, hashfn)
  sym_name = module.operation.attributes['sym_name']
  module_name = ir.StringAttr(sym_name).value
  return module_name + "-" + hash_obj.digest().hex()


def _log_cache_key_hash(hash_obj, last_serialized: str, hashfn):
  if logger.isEnabledFor(logging.DEBUG):
    # Log the hash of just this entry
    fresh_hash_obj = hashlib.sha256()
    hashfn(fresh_hash_obj)
    logger.debug(
        "get_cache_key hash of serialized %s: %s",
        last_serialized,
        fresh_hash_obj.digest().hex(),
    )
    # Log the cumulative hash
    logger.debug(
        "get_cache_key hash after serializing %s: %s",
        last_serialized,
        hash_obj.digest().hex(),
    )


def _remove_callbacks(m: ir.Module, ignore_callbacks: IgnoreCallbacks):
  """Removes callback pointers from precompiled IR.

  Python function pointers are not deterministic across executions.
  """
  def _update_bc_attribute(op: ir.Operation) -> ir.WalkResult:
    if op.name == "stablehlo.custom_call" and (
        (
            ignore_callbacks == IgnoreCallbacks.ALL
            and op.attributes["call_target_name"].value.endswith("callback")
        )
        or op.attributes["call_target_name"].value == "CustomSPMDPartitioning"
    ):
      op.attributes["backend_config"] = ir.StringAttr.get("REMOVED")
    return ir.WalkResult.ADVANCE

  if ignore_callbacks == IgnoreCallbacks.NO:
    return m

  m.operation.walk(_update_bc_attribute)
  return m


def _serialize_ir(m: ir.Module, ignore_callbacks: IgnoreCallbacks) -> bytes:
  output = io.BytesIO()
  if ignore_callbacks != IgnoreCallbacks.NO:
    m = _remove_callbacks(
        type_cast(ir.Module, m.operation.clone()), ignore_callbacks
    )
  m.operation.write_bytecode(file=output)
  return output.getvalue()


def _canonicalize_ir(
    m_original: ir.Module, ignore_callbacks: IgnoreCallbacks
) -> bytes:
  with m_original.context:
    m = type_cast(ir.Module, m_original.operation.clone())
    passes = pm.PassManager.parse(
        "builtin.module(strip-debuginfo)"
    )
    passes.run(m.operation)
    return _serialize_ir(m, ignore_callbacks)


def _hash_computation(hash_obj, module, ignore_callbacks: IgnoreCallbacks):
  if config.compilation_cache_include_metadata_in_key.value:
    canonical_ir = _serialize_ir(module, ignore_callbacks)
  else:
    canonical_ir = _canonicalize_ir(module, ignore_callbacks)
  hash_obj.update(canonical_ir)


def _hash_devices(hash_obj, devices: np.ndarray) -> None:
  for device in devices.flat:
    _hash_string(hash_obj, device.device_kind)


def _hash_accelerator_config(hash_obj, accelerators: np.ndarray, backend):
  accelerator_devices = []
  for accelerator in accelerators.flat:
    accelerator_devices.append(accelerator)
  try:
    hash_obj.update(
        xla_client.get_topology_for_devices(accelerator_devices).serialize()
    )
  except xla_client._xla.XlaRuntimeError as ex:
    # Fall back for those backends that do not support serialized
    # PjRtTopologyDescription as yet.
    logger.info("get (_hash_accelerator_config): unable to hash "
                "accelerator config, falling back to hashing "
                "devices + platform: %s (type %s)", ex, type(ex))
    _hash_devices(hash_obj, accelerators)
    _hash_platform(hash_obj, backend)

# LINT.IfChange(xla_flags)
xla_flags_to_exclude_from_cache_key = [
    "--xla_dump_compress_protos",
    "--xla_dump_module_metadata",
    "--xla_dump_max_hlo_modules",
    "--xla_dump_include_timestamp",
    "--xla_dump_hlo_pass_re",
    "--xla_dump_hlo_module_re",
    "--xla_dump_hlo_snapshots",
    "--xla_dump_fusion_visualization",
    "--xla_dump_hlo_as_url",
    "--xla_dump_hlo_as_proto",
    "--xla_dump_hlo_as_text",
    "--xla_dump_hlo_as_long_text",
    "--xla_dump_hlo_as_html",
    "--xla_dump_hlo_as_dot",
    "--xla_dump_to",
    "--xla_force_host_platform_device_count",
    "--xla_dump_disable_metadata",
    "--xla_dump_hlo_pipeline_re",
    "--xla_tpu_sdc_checker_streamz_metric",
    "--xla_tpu_sdc_checker_enable_sdc_event_callbacks",
    "--xla_tpu_sdc_checker_enable_coresweep_ng_callbacks",
    "--xla_tpu_sdc_checker_no_logging_if_callbacks_are_present",
    "--xla_gpu_cuda_data_dir",
    "--xla_gpu_experimental_autotune_cache_mode",
]

env_override_flags_to_exclude_from_cache_key = {
    x.strip("-") for x in xla_flags_to_exclude_from_cache_key
}
# LINT.ThenChange(:debug_options)

def _hash_serialized_compile_options(hash_obj, compile_options_obj,
                                     strip_device_assignment=False):
  # Do not mess with the original CompileOptions object since it is passed to
  # the compiler. Create a deep copy for the purpose of cache key generation.
  compile_options_copy = copy.deepcopy(compile_options_obj)

  # Certain debug options do not affect the compile result and thus, should not
  # be part of the cache key as their inclusion will result in unnecessary cache
  # misses. Clear them here by setting bool values to False, ints to 0, and
  # strings to empty. The exact values used to clear are not relevant as long
  # as the same values are used every time for each field.
  debug_options = compile_options_copy.executable_build_options.debug_options
  # LINT.IfChange(debug_options)
  debug_options.xla_force_host_platform_device_count = 0
  debug_options.xla_dump_to = ""
  debug_options.xla_dump_hlo_module_re = ""
  debug_options.xla_dump_hlo_pass_re = ""
  debug_options.xla_dump_hlo_as_text = False
  debug_options.xla_dump_hlo_as_proto = False
  debug_options.xla_dump_hlo_as_dot = False
  debug_options.xla_dump_hlo_as_url = False
  debug_options.xla_dump_hlo_as_html = False
  debug_options.xla_dump_fusion_visualization = False
  debug_options.xla_dump_hlo_snapshots = False
  debug_options.xla_dump_max_hlo_modules = False
  debug_options.xla_dump_module_metadata = False
  debug_options.xla_dump_compress_protos = False
  debug_options.xla_dump_hlo_as_long_text = False
  debug_options.xla_dump_disable_metadata = False
  debug_options.xla_dump_hlo_pipeline_re = ""
  debug_options.xla_gpu_experimental_autotune_cache_mode = 0

  # Optional way to specify the cuda install path to be used by the compiler.
  # This could possibly affect the cuda version compiled with, but this should
  # already be included in the platform information (and might not be reflected
  # by the cuda path regardless, since this only hashes on the directory name
  # and not the contents). It can also cause spurious cache misses if the cuda
  # path changes across runs despite being the same version, so we clear it
  # here.
  debug_options.xla_gpu_cuda_data_dir = ""
  # LINT.ThenChange(:xla_flags)

  compile_options_copy.env_option_overrides = [
      flag_value
      for flag_value in compile_options_copy.env_option_overrides
      if flag_value[0] not in env_override_flags_to_exclude_from_cache_key
  ]
  if strip_device_assignment and compile_options_copy.device_assignment:
    replica_count = compile_options_copy.device_assignment.replica_count()
    computation_count = compile_options_copy.device_assignment.computation_count()
    compile_options_copy.device_assignment = xla_client.DeviceAssignment.create(
        np.arange(replica_count * computation_count).reshape(
          [replica_count, computation_count])
    )
  return hash_obj.update(compile_options_copy.SerializeAsString())


def _hash_platform(hash_obj, backend):
  _hash_string(hash_obj, backend.platform)
  _hash_string(hash_obj, backend.platform_version)
  _hash_string(hash_obj, backend.runtime_type)


def _hash_xla_flags(hash_obj, extra_flag_prefixes: list[str]):
  xla_flags = []

  xla_flags_env_var = os.getenv("XLA_FLAGS")
  if xla_flags_env_var:
    xla_flags.extend(xla_flags_env_var.split())
  libtpu_init_args_env_var = os.getenv("LIBTPU_INIT_ARGS")
  if libtpu_init_args_env_var:
    xla_flags.extend(libtpu_init_args_env_var.split())

  for arg in sys.argv:
    if arg.startswith("--xla") or any(
        arg.startswith(p) for p in extra_flag_prefixes
    ):
      xla_flags.append(arg)

  # N.B. all XLA flags that take an argument must use '=' and not a space
  # (e.g. --xla_force_host_platform_device_count=8) (I think).
  for flag in sorted(xla_flags):
    if flag.split("=")[0] in xla_flags_to_exclude_from_cache_key:
      logger.debug("Not including XLA flag in cache key: %s", flag)
      continue
    logger.debug("Including XLA flag in cache key: %s", flag)
    _hash_string(hash_obj, flag)


def _hash_string(hash_obj, str_var):
  hash_obj.update(str_var.encode("utf-8").strip())
