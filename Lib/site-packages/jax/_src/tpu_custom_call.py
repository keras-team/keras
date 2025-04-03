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

"""JAX bindings for Mosaic."""

# mypy: ignore-errors
from __future__ import annotations

import base64
import collections.abc
from collections.abc import Callable, Sequence
import dataclasses
import enum
import functools
import io
import os
import time
from typing import Any

import jax
from jax._src import core
from jax._src import config
from jax._src import sharding_impls
from jax._src.interpreters import mlir
from jax._src.lib import tpu
from jax._src.lib import xla_client
from jax.interpreters import xla
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import mhlo
from jaxlib.mlir.passmanager import PassManager

try:
  from absl import flags
  FLAGS = flags.FLAGS
except ImportError:
  FLAGS = {}

_MOSAIC_USE_PYTHON_PIPELINE = config.bool_state(
    name="mosaic_use_python_pipeline",
    default=False,
    help=(
        "Run the initial Mosaic MLIR passes from Python, when as_tpu_kernel"
        " is called (for Pallas, this happens at JAX lowering time), instead of"
        " later within XLA."
    ),
)

_MOSAIC_ALLOW_HLO = config.bool_state(
    name="jax_mosaic_allow_hlo",
    default=False,
    help="Allow hlo dialects in Mosaic",
)


# This tracks the latest Mosaic IR version with a monthly delay.
FWD_COMPAT_IR_VERSION = 3


tpu_custom_call_p = core.Primitive("tpu_custom_call")
tpu_custom_call_p.def_impl(
    functools.partial(xla.apply_primitive, tpu_custom_call_p))
tpu_custom_call_p.multiple_results = True


def get_target_shape(hardware_generation: int) -> tuple[int, int]:
  """Returns the target shape for the given hardware generation."""
  del hardware_generation
  return (8, 128)


class MemorySpace(enum.Enum):
  HBM = enum.auto()
  VMEM = enum.auto()
  SEMAPHORE_MEM = enum.auto()
  SMEM = enum.auto()

  @property
  def color(self) -> int:
    if self == MemorySpace.HBM:
      return 0
    elif self == MemorySpace.VMEM:
      return 1
    elif self == MemorySpace.SEMAPHORE_MEM:
      return 2
    elif self == MemorySpace.SMEM:
      return 4
    else:
      raise ValueError("invalid memory space: " + str(self))


@dataclasses.dataclass(frozen=True)
class CostEstimate:
  flops: int
  transcendentals: int
  bytes_accessed: int

  def to_json(self) -> bytes:
    return (
        f'{{"flops": {self.flops}, "transcendentals": {self.transcendentals},'
        f' "bytes_accessed": {self.bytes_accessed}}}'
    ).encode('ascii')


@dataclasses.dataclass(frozen=True)
class CustomCallBackendConfig:
  """Represents an unserialized backend config for custom calls."""
  lowered_module_asm: bytes
  has_communication: bool
  collective_id: int | None
  device_type: str | None
  cost_estimate: CostEstimate | None
  needs_hlo_passes: bool
  needs_layout_passes: bool
  vmem_limit_bytes: int | None
  flags: dict[str, bool | int | float] | None
  allow_input_fusion: list[bool] | None
  serialization_format: int | None
  internal_scratch_in_bytes: int | None
  output_memory_spaces: tuple[MemorySpace | None, ...] | None

  # We omit the body while printing, because primitive params get embedded
  # in HLO metadata, and the body blows up its size.
  def __repr__(self):
    return "CustomCallBackendConfig(<omitted>)"

  def to_json(self) -> bytes:
    """Serializes the backend config into JSON."""
    # We format the JSON ourselves, because json.dumps seems to be overly slow.
    config = io.BytesIO()
    config.write(b'{"custom_call_config": {"body": "')
    config.write(base64.b64encode(self.lowered_module_asm))
    config.write(b'"')
    if self.has_communication:
      config.write(b', "has_communication": ')
      config.write(str(self.has_communication).lower().encode("ascii"))
    if self.collective_id is not None:
      config.write(b', "collective_id": ')
      config.write(str(self.collective_id).encode("ascii"))
    if self.cost_estimate is not None:
      config.write(b', "cost_estimate": ')
      config.write(self.cost_estimate.to_json())
    if self.needs_hlo_passes:
      config.write(b', "needs_hlo_passes": ')
      config.write(str(self.needs_hlo_passes).lower().encode("ascii"))
    if self.serialization_format is not None:
      config.write(b', "serialization_format": ')
      config.write(str(self.serialization_format).lower().encode("ascii"))
    if self.needs_layout_passes:
      config.write(b', "needs_layout_passes": ')
      config.write(str(self.needs_layout_passes).lower().encode("ascii"))
    if self.allow_input_fusion is not None:
      config.write(b', "allow_input_fusion": [')
      for i, value in enumerate(self.allow_input_fusion):
        config.write(b"true" if value else b"false")
        # config.write(str(value).lower().encode("ascii"))
        if i + 1 != len(self.allow_input_fusion):
          config.write(b",")
      config.write(b"]")
    if self.internal_scratch_in_bytes is not None:
      config.write(b', "internal_scratch_in_bytes": ')
      config.write(str(self.internal_scratch_in_bytes).encode("ascii"))
    if self.output_memory_spaces is not None:
      config.write(b', "output_memory_colors": [')
      for i, memory_space in enumerate(self.output_memory_spaces):
        if i:
          config.write(b",")
        color = memory_space.color if memory_space is not None else -1
        config.write(str(color).encode("ascii"))
      config.write(b"]")
    config.write(b"}")  # End of custom_call_config.
    if self.device_type is not None:
      config.write(b', "device_type": ')
      config.write(
          ('"DEVICE_TYPE_' + self.device_type.upper() + '"').encode("ascii")
      )
    if self.vmem_limit_bytes is not None:
      config.write(
          b', "scoped_memory_configs": [{"memory_space":1, "offset": 0,'
          b' "size": '
      )
      config.write(str(self.vmem_limit_bytes).encode("ascii"))
      config.write(b'}]')
    if self.flags is not None:
      config.write(b', "flag_configs": [')
      for i, (flag, value) in enumerate(self.flags.items()):
        config.write(b'{"flag_type": "')
        config.write(flag.encode("ascii"))
        config.write(b'", value: {')
        if isinstance(value, bool):
          config.write(b'"boolean_value": ')
          config.write(b"true" if value else b"false")
        elif isinstance(value, int):
          config.write(b'"integer_value": ')
          config.write(str(value).encode("ascii"))
        elif isinstance(value, float):
          config.write(b'"double_value": ')
          config.write(str(value).encode("ascii"))
        else:
          raise ValueError("invalid flag value: " + str(value))
        config.write(b"}}")
        if i + 1 != len(self.flags):
          config.write(b",")
      config.write(b"]")
    config.write(b"}")
    return config.getvalue()


@tpu_custom_call_p.def_abstract_eval
def _tpu_custom_call_abstract_eval(*_, out_avals, **__):
  return out_avals


def _avals_to_layouts(avals) -> Sequence[Sequence[int]]:
  return [tuple(range(a.ndim - 1, -1, -1)) for a in avals]


def _tpu_custom_call_lowering(
    ctx: mlir.LoweringRuleContext,
    *in_nodes,  # pylint: disable=missing-function-docstring
    config: CustomCallBackendConfig,
    kernel_name: str | None,
    out_avals: Any,
    input_output_aliases: tuple[tuple[int, int], ...],
) -> ...:
  result_types = [mlir.aval_to_ir_type(aval) for aval in out_avals]
  axis_context = ctx.module_context.axis_context
  if isinstance(axis_context, sharding_impls.SPMDAxisContext):
    if axis_context.manual_axes != frozenset(axis_context.mesh.axis_names):
      raise NotImplementedError(
          "Mosaic kernels cannot be automatically partitioned. Please wrap the"
          " call in a shard_map."
      )
  elif isinstance(axis_context, sharding_impls.ShardingContext):
    if axis_context.num_devices != 1:
      raise NotImplementedError(
          "Mosaic kernels cannot be automatically partitioned. Please wrap the"
          " call in a shard_map."
      )
  elif config.has_communication:
    raise NotImplementedError(
        "Replica lowering for Mosaic kernels not implemented."
    )
  if all(core.is_constant_shape(aval_out.shape) for aval_out in ctx.avals_out):
    result_shapes = None
  else:
    result_shapes = [
        mlir.shape_tensor(mlir.eval_dynamic_shape(ctx, aval_out.shape))
        for aval_out in ctx.avals_out]
  extra_attributes = None
  # Add kernel_name and kernel_metadata as attributes to the custom call op.
  # This is because we do not want to pollute the backend_config with this
  # information.
  if kernel_name is not None:
    extra_attributes = dict(kernel_name=ir.StringAttr.get(kernel_name))
  call = mlir.custom_call(
      "tpu_custom_call",
      result_types=result_types,
      operands=in_nodes,
      backend_config=config.to_json(),
      api_version=1,
      operand_output_aliases=dict(input_output_aliases),
      operand_layouts=_avals_to_layouts(ctx.avals_in),
      result_layouts=_avals_to_layouts(ctx.avals_out),
      result_shapes=result_shapes,
      extra_attributes=extra_attributes)

  return call.results


mlir.register_lowering(tpu_custom_call_p, _tpu_custom_call_lowering,
                       platform="tpu")


def _lower_tpu_kernel(
    module: ir.Module,
    hardware_generation: int,
    target_shape: tuple[int, int],
    kernel_name: str | None = None,
) -> ir.Module:
  """Runs MLIR passes lowering the given module to an MLIR module.

  Uses Python versions of canonicalize-mosaic,infer-memref-layout and
    apply-vector-layout.

  Args:
    module: The MLIR module to lower.
    hardware_generation: The TPU hardware generation to target.
    target_shape: The target shape of (sublane_count, lane_count).

  Returns:
    An MLIR module implementing the kernel.
  """
  try:
    module.operation.verify()
  except ir.MLIRError as e:
    raise ValueError("The compiled module fails MLIR verification") from e

  timestamp = time.time_ns()
  dump_cnt = [0]

  def get_dump_file_prefix() -> str:
    s = f"{timestamp}-{dump_cnt[0]:04}"
    dump_cnt[0] += 1
    return s

  with module.context as ctx, module.operation.location as _:
    ctx.append_dialect_registry(mlir.upstream_dialects)
    ctx.load_all_available_dialects()
    tpu.register_dialect(ctx)
    mhlo.register_mhlo_dialect(ctx)
    mhlo.register_mhlo_passes()
    dump_mlir(module, "original", get_dump_file_prefix(), kernel_name)

    if _MOSAIC_ALLOW_HLO.value:
      # Run hlo dialect conversion: hlo -> linalg -> vector.
      pipeline = [
          "hlo-legalize-to-arithmetic",
          "func.func(hlo-legalize-to-linalg)",
          "func.func(linalg-vectorization)",
      ]
      pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
      pipeline.run(module.operation)
      dump_mlir(module, "post-hlo-conversion", get_dump_file_prefix(), kernel_name)

    sl_cnt, l_cnt = target_shape
    # Note: we don't pass the TpuTilingFlags here, since we don't know the
    # tiling decisions made by the compiler / what flags are enabled at this
    # point, so we assume everything can be tiled up to default tiling.
    pipeline = [
        "func.func(tpu-infer-memref-layout{"
        f" hardware-generation={hardware_generation}"
        f" sublane-count={sl_cnt}"
        f" lane-count={l_cnt}"
        "})"
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(module, "post-infer-memref-layout", get_dump_file_prefix(), kernel_name)

    pipeline = [
        "canonicalize",
        "cse",
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(
        module,
        "post-infer-memref-layout-simplify",
        get_dump_file_prefix(),
        kernel_name,
    )

    try:
      on_device_checks = FLAGS["xla_mosaic_on_device_checks"].value
    except KeyError:
      on_device_checks = False

    if checks := on_device_checks:
      checks = set(checks.split(","))
      if checks == {"bounds"}:  # We only support one kind of checks now.
        pipeline = PassManager.parse(
            "builtin.module(func.func(debug-assert-insertion))"
        )
        pipeline.run(module.operation)
        dump_mlir(module, "post-assert-insertion", get_dump_file_prefix(), kernel_name)
      elif checks:
        checks.discard("bounds")
        raise ValueError(
            f"Unrecognized on-device check categories: {', '.join(checks)}"
        )

    # Legacy pipeline always runs in compatibility mode.
    compatibility_mode = True
    pipeline = [
        (
            f"func.func(tpu-canonicalize-mosaic{{hardware-generation={hardware_generation} compatibility-mode={compatibility_mode}}})"
        ),
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(module, "post-canonicalize-mosaic", get_dump_file_prefix(), kernel_name)

    pipeline = [
        (
            "func.func(tpu-infer-vector-layout{"
            f" hardware-generation={hardware_generation}"
            f" sublane-count={sl_cnt} lane-count={l_cnt}"
            "})"
        ),
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(module, "post-infer-vector-layout", get_dump_file_prefix(), kernel_name)

    pipeline = [
        (
            "func.func(tpu-relayout-insertion{"
            f" sublane-count={sl_cnt} lane-count={l_cnt}"
            "})"
        ),
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(module, "post-relayout-insertion", get_dump_file_prefix(), kernel_name)

    mxu_size = 128 if hardware_generation < 6 else 256
    pipeline = [
        "func.func(tpu-apply-vector-layout{"
        f" sublane-count={sl_cnt} lane-count={l_cnt}"
        f" hardware-generation={hardware_generation}"
        f" mxu-contracting-size={mxu_size} mxu-noncontracting-size={mxu_size}"
        f" max-sublanes-in-scratch={sl_cnt * (sl_cnt + 1)}"
        "})"
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(module, "post-apply-vector-layout", get_dump_file_prefix(), kernel_name)

    pipeline = [
        "canonicalize",
        "cse",
    ]
    pipeline = PassManager.parse(f"builtin.module({','.join(pipeline)})")
    pipeline.run(module.operation)
    dump_mlir(
        module,
        "post-apply-vector-layout-simplify",
        get_dump_file_prefix(),
        kernel_name,
    )

    return module


def _lower_mosaic_module_to_asm(
    module: ir.Module,
    *,
    backend: str,
    device_type: str | None,
    kernel_name: str | None,
    ir_version: int | None = None,
) -> tuple[ir.Module, tuple[bool, bool, bool, bool]]:
  has_communication, has_custom_barrier = tpu.private_has_communication(
      module.operation
  )
  needs_hlo_passes = _MOSAIC_ALLOW_HLO.value
  needs_layout_passes = not device_type
  # We'll mutate the module, so clone it
  with module.context as ctx, module.operation.location as _:
    if needs_layout_passes and _MOSAIC_USE_PYTHON_PIPELINE.value:
      module = ir.Module.parse(
          module.operation.get_asm(binary=True, enable_debug_info=True)
      )
      module_op = module.operation
      some_tpu = jax.devices(backend)[0]
      device_kind = some_tpu.device_kind
      if not device_kind.startswith("TPU v"):
        raise ValueError(
            f"Unrecognized TPU device kind: {device_kind}. "
            "tpu_custom_call cannot be lowered on a machine without TPUs "
            "when mosaic_use_python_pipeline=True.")
      hardware_generation = int(device_kind[len("TPU v")])
      target_shape = get_target_shape(hardware_generation)
      module = _lower_tpu_kernel(
          module, hardware_generation, target_shape=target_shape, kernel_name=kernel_name,
      )
      needs_hlo_passes = False
      needs_layout_passes = False
    else:
      module_op = module.operation.clone()
    prev_allow_unregistered_dialects = ctx.allow_unregistered_dialects
    ctx.allow_unregistered_dialects = True
    # TODO(apaszke): Remove once the minimum jaxlib version is at least 0.4.37.
    if jax.version._version_as_tuple(jax.lib.__version__) < (0, 4, 37):
      target_version = ""
    else:
      target_version = (
          f"target-version={ir_version}" if ir_version is not None else ""
      )
    try:
      pipeline = PassManager.parse(
          "builtin.module(mosaic-serde{serialize=true " + target_version + "})"
      )
      pipeline.run(module_op)
    finally:
      ctx.allow_unregistered_dialects = prev_allow_unregistered_dialects
    bytecode_buffer = io.BytesIO()
    module_op.write_bytecode(bytecode_buffer, desired_version=0)
    asm = bytecode_buffer.getvalue()
    return asm, (
        has_communication,
        has_custom_barrier,
        needs_hlo_passes,
        needs_layout_passes,
    )


def _get_device_type(module: ir.Module) -> str | None:
  """Determines the device type based on the core_type annotations."""
  sparsecore_func_found = False
  tensorcore_func_found = False

  def assign_device_type_based_on_core_type(op: ir.Operation) -> ir.WalkResult:
    nonlocal sparsecore_func_found
    nonlocal tensorcore_func_found
    if op.name == "func.func":
      if "tpu.core_type" in op.attributes:
        core_type = op.attributes["tpu.core_type"]
        if str(core_type) in [
            f"#tpu.core_type<{c}>"
            for c in ["sc_scalar_subcore", "sc_vector_subcore"]
        ]:
          sparsecore_func_found = True
          if tensorcore_func_found:
            return ir.WalkResult.INTERRUPT
          return ir.WalkResult.SKIP
        if str(core_type) == "#tpu.core_type<tc>":
          tensorcore_func_found = True
          return ir.WalkResult.SKIP
        raise ValueError(f"Unknown core type: {core_type}")
    return ir.WalkResult.ADVANCE

  module.operation.walk(
      assign_device_type_based_on_core_type, walk_order=ir.WalkOrder.PRE_ORDER
  )
  if tensorcore_func_found and sparsecore_func_found:
    raise ValueError(
        "A single Mosaic kernel cannot contain both "
        "TensorCore and SparseCore functions."
    )
  if sparsecore_func_found:
    return "sparsecore"
  return None


def _lower_to_custom_call_config(
    module: ir.Module,
    *,
    backend: str,
    device_type: str | None,
    vmem_limit_bytes: int | None,
    cost_estimate: CostEstimate | None,
    flags: dict[str, bool | int | float] | None,
    allow_input_fusion: list[bool] | None,
    internal_scratch_in_bytes: int | None,
    collective_id: int | None,
    serialization_format: int | None,
    output_memory_spaces: tuple[MemorySpace | None, ...] | None = None,
    kernel_name: str | None = None,
    ir_version: int | None = None,
) -> CustomCallBackendConfig:
  lowered_module_asm, (
      has_communication,
      has_custom_barrier,
      needs_hlo_passes,
      needs_layout_passes,
  ) = _lower_mosaic_module_to_asm(
      module,
      backend=backend,
      device_type=device_type,
      kernel_name=kernel_name,
      ir_version=ir_version,
  )
  return _lowered_to_custom_call_config(
      lowered_module_asm,
      vmem_limit_bytes=vmem_limit_bytes,
      cost_estimate=cost_estimate,
      flags=flags,
      allow_input_fusion=allow_input_fusion,
      internal_scratch_in_bytes=internal_scratch_in_bytes,
      collective_id=collective_id,
      device_type=device_type,
      serialization_format=serialization_format,
      has_custom_barrier=has_custom_barrier,
      has_communication=has_communication,
      needs_hlo_passes=needs_hlo_passes,
      needs_layout_passes=needs_layout_passes,
      output_memory_spaces=output_memory_spaces,
  )


def _lowered_to_custom_call_config(
    lowered_module_asm: bytes,
    *,
    vmem_limit_bytes: int | None,
    cost_estimate: CostEstimate | None,
    flags: dict[str, bool | int | float] | None,
    allow_input_fusion: list[bool] | None,
    internal_scratch_in_bytes: int | None,
    collective_id: int | None,
    serialization_format: int | None,
    has_custom_barrier: bool,
    has_communication: bool,
    needs_hlo_passes: bool,
    needs_layout_passes: bool,
    device_type: str | None,
    output_memory_spaces: tuple[MemorySpace | None, ...] | None = None,
):
  if has_custom_barrier:
    if collective_id is None:
      raise ValueError(
          "collective_id has to be specified when using a custom barrier"
      )
  elif collective_id is not None:
    raise ValueError(
        "collective_id has to be unspecified or None when not using a custom"
        " barrier"
    )
  if vmem_limit_bytes is not None and not isinstance(vmem_limit_bytes, int):
    raise ValueError(
        "vmem_limit_bytes must be an int: provided with a"
        f" {type(vmem_limit_bytes)}."
    )
  config = CustomCallBackendConfig(
      lowered_module_asm,
      has_communication,
      collective_id,
      device_type,
      cost_estimate,
      needs_hlo_passes,
      needs_layout_passes,
      vmem_limit_bytes,
      flags,
      allow_input_fusion,
      serialization_format,
      internal_scratch_in_bytes,
      output_memory_spaces,
  )
  return config


def lower_module_to_custom_call(
    ctx: mlir.LoweringRuleContext,
    *in_nodes: ir.Value,
    module: ir.Module,
    out_type: Any,
    backend: str,
    kernel_name: str,
    cost_estimate: CostEstimate | None,
    vmem_limit_bytes: int | None,
    flags: dict[str, bool | int | float] | None,
    allow_input_fusion: list[bool] | None,
    input_output_aliases: tuple[tuple[int, int], ...],
    internal_scratch_in_bytes: int | None,
    collective_id: int | None,
    serialization_format: int | None,
    output_memory_spaces: tuple[MemorySpace | None, ...] | None,
    device_type: str | None,
) -> Sequence[ir.Value]:
  config = _lower_to_custom_call_config(
      module,
      backend=backend,
      vmem_limit_bytes=vmem_limit_bytes,
      cost_estimate=cost_estimate,
      flags=flags,
      allow_input_fusion=allow_input_fusion,
      internal_scratch_in_bytes=internal_scratch_in_bytes,
      collective_id=collective_id,
      device_type=device_type,
      serialization_format=serialization_format,
      output_memory_spaces=output_memory_spaces,
      kernel_name=kernel_name,
      ir_version=FWD_COMPAT_IR_VERSION if ctx.is_forward_compat() else None,
  )
  return _tpu_custom_call_lowering(
      ctx,
      *in_nodes,
      config=config,
      kernel_name=kernel_name,
      out_avals=out_type,
      input_output_aliases=input_output_aliases,
  )


def as_tpu_kernel(
    module: ir.Module,
    out_type: Any,
    *,
    cost_estimate: CostEstimate | None = None,
    backend: str | xla_client.Client = "tpu",
    kernel_name: str | None = None,
    vmem_limit_bytes: int | None = None,
    flags: dict[str, bool | int | float] | None = None,
    allow_input_fusion: list[bool] | None = None,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
    internal_scratch_in_bytes: int | None = None,
    collective_id: int | None = None,
    serialization_format: int | None = 1,
    output_memory_spaces: tuple[MemorySpace | None, ...] | None = None,
) -> Callable[..., Any]:
  """Turns an MLIR Mosaic kernel into a JAX-compatible function."""
  device_type = _get_device_type(module)
  config = _lower_to_custom_call_config(
      module,
      backend=backend,
      device_type=device_type,
      vmem_limit_bytes=vmem_limit_bytes,
      cost_estimate=cost_estimate,
      flags=flags,
      allow_input_fusion=allow_input_fusion,
      internal_scratch_in_bytes=internal_scratch_in_bytes,
      collective_id=collective_id,
      serialization_format=serialization_format,
      output_memory_spaces=output_memory_spaces,
      kernel_name=kernel_name,
  )
  return _as_jax_callable(
      config,
      out_type,
      kernel_name=kernel_name,
      input_output_aliases=input_output_aliases,
  )


def lowered_as_tpu_kernel(
    lowered_module: ir.Module,
    out_type: Any,
    *,
    collective_id: int | None = None,
    cost_estimate: CostEstimate | None = None,
    needs_hlo_passes: bool = False,
    needs_layout_passes: bool = False,
    device_type: str | None = None,
    has_communication: bool = False,
    has_custom_barrier: bool = False,
    kernel_name: str | None = None,
    vmem_limit_bytes: int | None = None,
    flags: dict[str, bool | int | float] | None = None,
    allow_input_fusion: list[bool] | None = None,
    input_output_aliases: tuple[tuple[int, int], ...] = (),
    serialization_format: int | None = None,
    internal_scratch_in_bytes: int | None = None,
) -> Callable[..., Any]:
  lowered_module_asm = lowered_module.operation.get_asm(
      binary=True, enable_debug_info=True
  )
  config = _lowered_to_custom_call_config(
      lowered_module_asm,
      vmem_limit_bytes=vmem_limit_bytes,
      cost_estimate=cost_estimate,
      flags=flags,
      allow_input_fusion=allow_input_fusion,
      internal_scratch_in_bytes=internal_scratch_in_bytes,
      collective_id=collective_id,
      device_type=device_type,
      serialization_format=serialization_format,
      has_custom_barrier=has_custom_barrier,
      has_communication=has_communication,
      needs_hlo_passes=needs_hlo_passes,
      needs_layout_passes=needs_layout_passes,
  )
  return _as_jax_callable(
      config,
      out_type,
      kernel_name=kernel_name,
      input_output_aliases=input_output_aliases,
  )


def _as_jax_callable(
    config: CustomCallBackendConfig,
    out_type: Any,
    *,
    kernel_name: str | None,
    input_output_aliases: tuple[tuple[int, int], ...],
) -> Callable[..., Any]:
  unpack = False
  if not isinstance(out_type, collections.abc.Iterable):
    out_type = (out_type,)
    unpack = True
  out_avals = tuple(core.ShapedArray(ty.shape, ty.dtype) for ty in out_type)

  # We use jax.jit to make sure we hit the fast compilation cache.
  def apply_kernel(*args):
    result = tpu_custom_call_p.bind(
        *args,
        config=config,
        kernel_name=kernel_name,
        out_avals=out_avals,
        input_output_aliases=input_output_aliases,
    )
    return result[0] if unpack else result

  return jax.jit(apply_kernel)


def dump_mlir(
    module: ir.Module, name: str, prefix: str, kernel_name: str | None = None
):
  """A helper function to dump mosaic mlir module"""
  try:
    should_dump = FLAGS["xla_mosaic_dump_to"].value
  except KeyError:
    return
  if should_dump == "sponge":
    outdir = os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", None)
    if outdir:
      if kernel_name:
        name = f"{kernel_name}-{name}"
      path = os.path.join(outdir, f"{prefix}-mosaic-dump-{name}-py.txt")
      with open(path, "w") as f:
        f.write(str(module))
