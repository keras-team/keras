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
"""JAX APIs for exporting JAX functions for interoperation.

"""

from __future__ import annotations

import collections
from collections.abc import Callable, Sequence
import copy
import dataclasses
import functools
import itertools
import json
import re
from typing import Any, Protocol, TypeVar, Union, cast

from absl import logging
import numpy as np

import jax
from jax import sharding

from jax._src import ad_util
from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import effects
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lib import xla_client
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src import pjit
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src import stages
from jax._src import tree_util
from jax._src import util
from jax._src import xla_bridge as xb

from jax._src.export import shape_poly

map = util.safe_map
zip = util.safe_zip

DType = Any
Shape = core.Shape
# The values of input and output sharding from the lowering.
LoweringSharding = Union[sharding.Sharding, pxla.UnspecifiedValue]
HloSharding = xla_client.HloSharding

# The minimum and maximum supported calling convention version.
# See https://jax.readthedocs.io/en/latest/export/export.html#export-calling-convention-version
minimum_supported_calling_convention_version = 9
maximum_supported_calling_convention_version = 9


class DisabledSafetyCheck:
  """A safety check that should be skipped on (de)serialization.

  Most of these checks are performed on serialization, but some are deferred to
  deserialization. The list of disabled checks is attached to the serialization,
  e.g., as a sequence of string attributes to `jax.export.Exported` or of
  `tf.XlaCallModuleOp`.

  When using jax2tf, you can disable more deserialization safety checks
  by passing `TF_XLA_FLAGS=--tf_xla_call_module_disabled_checks=platform`.
  """
  _impl: str

  @classmethod
  def platform(cls) -> DisabledSafetyCheck:
    """Allows the compilation platform to differ from the export platform.

    Has effect only on deserialization.
    """
    return DisabledSafetyCheck("platform")

  @classmethod
  def custom_call(cls, target_name: str) -> DisabledSafetyCheck:
    """Allows the serialization of a call target not known to be stable.

    Has effect only on serialization.
    Args:
      target_name: the name of the custom call target to allow.
    """
    return DisabledSafetyCheck(f"custom_call:{target_name}")

  def is_custom_call(self) -> str | None:
    """Returns the custom call target allowed by this directive."""
    m = re.match(r'custom_call:(.+)$', self._impl)
    return m.group(1) if m else None

  def __init__(self, _impl:str):
    # Do not use directly, use builders `platform`, `custom_call`.
    self._impl = _impl

  def __str__(self):
    return self._impl
  __repr__ = __str__

  def __eq__(self, other) -> bool:
    return isinstance(other, DisabledSafetyCheck) and self._impl == other._impl

  def __hash__(self) -> int:
    return hash(self._impl)


@dataclasses.dataclass(frozen=True)
class Exported:
  """A JAX function lowered to StableHLO.

  Attributes:
    fun_name: the name of the exported function, for error messages.
    in_tree: a PyTreeDef describing the tuple (args, kwargs) of the lowered JAX
        function. The actual lowering does not depend on the `in_tree`, but this
        can be used to invoke the exported function using the same argument
        structure.
    in_avals: the flat tuple of input abstract values. May contain dimension
        expressions in the shapes.
    out_tree: a PyTreeDef describing the result of the lowered JAX function.
    out_avals: the flat tuple of output abstract values. May contain dimension
        expressions in the shapes, with dimension variables among those in
        `in_avals`.
    in_shardings_hlo: the flattened input shardings, a sequence as long
        as `in_avals`. `None` means unspecified sharding.
        Note that these do not include the mesh or the actual devices used in
        the mesh. See `in_shardings_jax` for a way to turn these
        into sharding specification that can be used with JAX APIs.
    out_shardings_hlo: the flattened output shardings, a sequence as long
        as `out_avals`. `None` means unspecified sharding.
        Note that these do not include the mesh or the actual devices used in
        the mesh. See `out_shardings_jax` for a way to turn these
        into sharding specification that can be used with JAX APIs.
    nr_devices: the number of devices that the module has been lowered for.
    platforms: a tuple containing the platforms for which the function should
        be exported. The set of platforms in JAX is open-ended; users can
        add platforms. JAX built-in platforms are: 'tpu', 'cpu', 'cuda', 'rocm'.
        See https://jax.readthedocs.io/en/latest/export/export.html#cross-platform-and-multi-platform-export.
    ordered_effects: the ordered effects present in the serialized module.
        This is present from serialization version 9. See https://jax.readthedocs.io/en/latest/export/export.html#module-calling-convention
        for the calling convention in presence of ordered effects.
    unordered_effects: the unordered effects present in the serialized module.
        This is present from serialization version 9.
    mlir_module_serialized: the serialized lowered VHLO module.
    calling_convention_version: a version number for the calling
        convention of the exported module.
        See more versioning details at https://jax.readthedocs.io/en/latest/export/export.html#calling-convention-versions.
    module_kept_var_idx: the sorted indices of the arguments among `in_avals` that
        must be passed to the module. The other arguments have been dropped
        because they are not used.
    uses_global_constants: whether the `mlir_module_serialized` uses shape
        polymorphism or multi-platform export.
        This may be because `in_avals` contains dimension
        variables, or due to inner calls of Exported modules that have
        dimension variables or platform index arguments. Such modules need
        shape refinement before XLA compilation.
    disabled_safety_checks: a list of descriptors of safety checks that have been
        disabled at export time. See docstring for `DisabledSafetyCheck`.
    _get_vjp: an optional function that takes the current exported function and
        returns the exported VJP function.
        The VJP function takes a flat list of arguments,
        starting with the primal arguments and followed by a cotangent argument
        for each primal output. It returns a tuple with the cotangents
        corresponding to the flattened primal inputs.

  See a [description of the calling convention for the `mlir_module`](https://jax.readthedocs.io/en/latest/export/export.html#module-calling-convention).
  """
  fun_name: str
  in_tree: tree_util.PyTreeDef
  in_avals: tuple[core.ShapedArray, ...]
  out_tree: tree_util.PyTreeDef
  out_avals: tuple[core.ShapedArray, ...]

  in_shardings_hlo: tuple[HloSharding | None, ...]
  out_shardings_hlo: tuple[HloSharding | None, ...]
  nr_devices: int
  platforms: tuple[str, ...]
  ordered_effects: tuple[effects.Effect, ...]
  unordered_effects: tuple[effects.Effect, ...]
  disabled_safety_checks: Sequence[DisabledSafetyCheck]

  mlir_module_serialized: bytes
  calling_convention_version: int
  module_kept_var_idx: tuple[int, ...]
  uses_global_constants: bool

  _get_vjp: Callable[[Exported], Exported] | None

  def mlir_module(self) -> str:
    """A string representation of the `mlir_module_serialized`."""
    return xla_client._xla.mlir.deserialize_portable_artifact(self.mlir_module_serialized)

  def __str__(self):
    # This is called to make a MLIR source location when we call an Exported, and we
    # do not want the entire serialized module to end up in locations.
    return f"Exported(fun_name={self.fun_name}, ...)"

  def in_shardings_jax(
    self,
    mesh: sharding.Mesh) -> Sequence[sharding.Sharding | None]:
    """Creates Shardings corresponding to self.in_shardings_hlo.

    The Exported object stores `in_shardings_hlo` as HloShardings, which are
    independent of a mesh or set of devices. This method constructs
    Sharding that can be used in JAX APIs such as `jax.jit` or
    `jax.device_put`.

    Example usage:

      >>> from jax import export
      >>> # Prepare the exported object:
      >>> exp_mesh = sharding.Mesh(jax.devices(), ("a",))
      >>> exp = export.export(jax.jit(lambda x: jax.numpy.add(x, x),
      ...                             in_shardings=sharding.NamedSharding(exp_mesh, sharding.PartitionSpec("a")))
      ...     )(np.arange(jax.device_count()))
      >>> exp.in_shardings_hlo
      ({devices=[8]<=[8]},)
      >>> # Create a mesh for running the exported object
      >>> run_mesh = sharding.Mesh(jax.devices()[::-1], ("b",))
      >>> # Put the args and kwargs on the appropriate devices
      >>> run_arg = jax.device_put(np.arange(jax.device_count()),
      ...     exp.in_shardings_jax(run_mesh)[0])
      >>> res = exp.call(run_arg)
      >>> res.addressable_shards
      [Shard(device=CpuDevice(id=7), index=(slice(0, 1, None),), replica_id=0, data=[0]),
       Shard(device=CpuDevice(id=6), index=(slice(1, 2, None),), replica_id=0, data=[2]),
       Shard(device=CpuDevice(id=5), index=(slice(2, 3, None),), replica_id=0, data=[4]),
       Shard(device=CpuDevice(id=4), index=(slice(3, 4, None),), replica_id=0, data=[6]),
       Shard(device=CpuDevice(id=3), index=(slice(4, 5, None),), replica_id=0, data=[8]),
       Shard(device=CpuDevice(id=2), index=(slice(5, 6, None),), replica_id=0, data=[10]),
       Shard(device=CpuDevice(id=1), index=(slice(6, 7, None),), replica_id=0, data=[12]),
       Shard(device=CpuDevice(id=0), index=(slice(7, 8, None),), replica_id=0, data=[14])]

    """
    return tuple(_hlo_sharding_to_xla_compatible_sharding(s, mesh)
                 for s in self.in_shardings_hlo)

  def out_shardings_jax(
      self,
      mesh: sharding.Mesh) -> Sequence[sharding.Sharding | None]:
    """Creates Shardings corresponding to `self.out_shardings_hlo`.

    See documentation for in_shardings_jax.
    """
    return tuple(_hlo_sharding_to_xla_compatible_sharding(s, mesh)
                 for s in self.out_shardings_hlo)

  def has_vjp(self) -> bool:
    """Returns if this Exported supports VJP."""
    return self._get_vjp is not None

  def vjp(self) -> Exported:
    """Gets the exported VJP.

    Returns None if not available, which can happen if the Exported has been
    loaded from an external format without a VJP.
    """
    if self._get_vjp is None:
      raise ValueError("No VJP is available")
    return self._get_vjp(self)

  def serialize(self,
                vjp_order: int = 0) -> bytearray:
    """Serializes an Exported.

    Args:
      vjp_order: The maximum vjp order to include. E.g., the value 2 means that we
        serialize the primal functions and two orders of the `vjp` function. This
        should allow 2nd order reverse mode differentiation of the deserialized
        function. i.e., `jax.grad(jax.grad(f)).`
    """
    # Lazy load the serialization module, since flatbuffers is an optional
    # dependency.
    from jax._src.export.serialization import serialize
    return serialize(self, vjp_order=vjp_order)

  def call(self, *args, **kwargs):
    """Call an exported function from a JAX program.

    Args:
      args: the positional arguments to pass to the exported function. This
        should be a pytree of arrays with the same pytree structure as the
        arguments for which the function was exported.
      kwargs: the keyword arguments to pass to the exported function.

    Returns: a pytree of result array, with the same structure as the
      results of the exported function.

    The invocation supports reverse-mode AD, and all the features supported
    by exporting: shape polymorphism, multi-platform, device polymorphism.
    See the examples in the [JAX export documentation](https://jax.readthedocs.io/en/latest/export/export.html).
    """
    return call_exported(self)(*args, **kwargs)


def deserialize(blob: bytearray) -> Exported:
  """Deserializes an Exported.

  Args:
    blob: a bytearray obtained from `Exported.serialize`.
  """
  # Lazy load the serialization module, since flatbuffers is an optional
  # dependency.
  from jax._src.export.serialization import deserialize
  return deserialize(blob)


T = TypeVar("T")
PyTreeAuxData = Any  # alias for tree_util._AuxData


class _SerializeAuxData(Protocol):
  def __call__(self, aux_data: PyTreeAuxData) -> bytes:
    """Serializes the PyTree node AuxData.

    The AuxData is returned by the `flatten_func` registered by
    `tree_util.register_pytree_node`).
    """


class _DeserializeAuxData(Protocol):
  def __call__(self, serialized_aux_data: bytes) -> PyTreeAuxData:
    """Deserializes the PyTree node AuxData.

    The result will be passed to `_BuildFromChildren`.
    """


class _BuildFromChildren(Protocol):
  def __call__(self, aux_data: PyTreeAuxData, children: Sequence[Any]) -> Any:
    """Materializes a T given a deserialized AuxData and children.

    This is similar in scope with the `unflatten_func`.
    """


serialization_registry: dict[type, tuple[str, _SerializeAuxData]] = {}


deserialization_registry: dict[
  str,
  tuple[type, _DeserializeAuxData, _BuildFromChildren]] = {}


def _is_namedtuple(nodetype: type) -> bool:
  return (issubclass(nodetype, tuple) and
          hasattr(nodetype, "_fields") and
          isinstance(nodetype._fields, Sequence) and
          all(isinstance(f, str) for f in nodetype._fields))

def register_pytree_node_serialization(
    nodetype: type[T],
    *,
    serialized_name: str,
    serialize_auxdata: _SerializeAuxData,
    deserialize_auxdata: _DeserializeAuxData,
    from_children: _BuildFromChildren | None = None
) -> type[T]:
  """Registers a custom PyTree node for serialization and deserialization.

  You must use this function before you can serialize and deserialize PyTree
  nodes for the types not supported natively. We serialize PyTree nodes for
  the `in_tree` and `out_tree` fields of `Exported`, which are part of the
  exported function's calling convention.

  This function must be called after calling
  `jax.tree_util.register_pytree_node` (except for `collections.namedtuple`,
  which do not require a call to `register_pytree_node`).

  Args:
    nodetype: the type whose PyTree nodes we want to serialize. It is an
      error to attempt to register multiple serializations for a `nodetype`.
    serialized_name: a string that will be present in the serialization and
      will be used to look up the registration during deserialization. It is an
      error to attempt to register multiple serializations for a
      `serialized_name`.
    serialize_auxdata: serialize the PyTree auxdata (returned by the
      `flatten_func` argument to `jax.tree_util.register_pytree_node`.).
    deserialize_auxdata: deserialize the auxdata that was serialized by the
      `serialize_auxdata`.
    from_children: if present, this is a function that takes that result of
      `deserialize_auxdata` along with some children and creates an instance
      of `nodetype`. This is similar to the `unflatten_func` passed to
      `jax.tree_util.register_pytree_node`. If not present, we look up
      and use the `unflatten_func`. This is needed for `collections.namedtuple`,
      which does not have a `register_pytree_node`, but it can be useful to
      override that function. Note that the result of `from_children` is
      only used with `jax.tree_util.tree_structure` to construct a proper
      PyTree node, it is not used to construct the outputs of the serialized
      function.

  Returns:
    the same type passed as `nodetype`, so that this function can
    be used as a class decorator.
  """
  if nodetype in serialization_registry:
    raise ValueError(
        f"Duplicate serialization registration for type `{nodetype}`. "
        "Previous registration was with serialized_name "
        f"`{serialization_registry[nodetype][0]}`.")
  if serialized_name in deserialization_registry:
    raise ValueError(
        "Duplicate serialization registration for "
        f"serialized_name `{serialized_name}`. "
        "Previous registration was for type "
        f"`{deserialization_registry[serialized_name][0]}`.")
  if from_children is None:
    if nodetype not in tree_util._registry:
      raise ValueError(
          f"If `from_children` is not present, you must call first"
          f"`jax.tree_util.register_pytree_node` for `{nodetype}`")
    from_children = tree_util._registry[nodetype].from_iter

  serialization_registry[nodetype] = (
      serialized_name, serialize_auxdata)
  deserialization_registry[serialized_name] = (
      nodetype, deserialize_auxdata, from_children)
  return nodetype


def register_namedtuple_serialization(
    nodetype: type[T],
    *,
    serialized_name: str) -> type[T]:
  """Registers a namedtuple for serialization and deserialization.

  JAX has native PyTree support for `collections.namedtuple`, and does not
  require a call to `jax.tree_util.register_pytree_node`. However, if you
  want to serialize functions that have inputs of outputs of a
  namedtuple type, you must register that type for serialization.

  Args:
    nodetype: the type whose PyTree nodes we want to serialize. It is an
      error to attempt to register multiple serializations for a `nodetype`.
      On deserialization, this type must have the same set of keys that
      were present during serialization.
    serialized_name: a string that will be present in the serialization and
      will be used to look up the registration during deserialization. It is an
      error to attempt to register multiple serializations for
      a `serialized_name`.

  Returns:
    the same type passed as `nodetype`, so that this function can
    be used as a class decorator.
"""
  if not _is_namedtuple(nodetype):
    raise ValueError("Use `jax.export.register_pytree_node_serialization` for "
                     "types other than `collections.namedtuple`.")

  def serialize_auxdata(aux_data: PyTreeAuxData) -> bytes:
    # Store the serialized keys in the serialized auxdata
    del aux_data
    return json.dumps(nodetype._fields).encode("utf-8")

  def deserialize_auxdata(serialized_aux_data: bytes) -> PyTreeAuxData:
    return json.loads(serialized_aux_data.decode("utf-8"))

  def from_children(aux_data: PyTreeAuxData, children: Sequence[Any]) -> Any:
    # Use our own "from_children" because namedtuples do not have a pytree
    # registration.
    ser_keys = cast(Sequence[str], aux_data)
    assert len(ser_keys) == len(children)
    return nodetype(** dict(zip(ser_keys, children)))

  return register_pytree_node_serialization(
      nodetype,
      serialized_name=serialized_name,
      serialize_auxdata=serialize_auxdata,
      deserialize_auxdata=deserialize_auxdata,
      from_children=from_children)


# collections.OrderedDict is registered as a pytree node with auxdata being
# `tuple(x.keys())`.
def _serialize_ordereddict_keys(keys):
  if isinstance(keys, Sequence) and all(isinstance(k, str) for k in keys):
    return json.dumps(keys).encode("utf-8")
  else:
    raise NotImplementedError(
        "Serialization of collections.OrderedDict is supported only when the "
        f"keys are strings. Found keys: {keys}.")


register_pytree_node_serialization(
    collections.OrderedDict,
    serialized_name="collections.OrderedDict",
    serialize_auxdata=_serialize_ordereddict_keys,
    deserialize_auxdata=lambda b: json.loads(b.decode("utf-8")))


def default_export_platform() -> str:
  """Retrieves the default export platform.

  One of: `tpu`, `cpu`, `cuda`, `rocm`.
  """
  # Canonicalize to turn 'gpu' into 'cuda' or 'rocm'
  return xb.canonicalize_platform(jax.default_backend())

default_lowering_platform = default_export_platform

def shape_and_dtype_jax_array(a) -> tuple[Sequence[int | None], DType]:
  """Returns the shape and dtype of a jax.Array or a j"""
  if isinstance(a, jax.ShapeDtypeStruct):
    return a.shape, a.dtype
  aval = core.get_aval(a)
  return aval.shape, aval.dtype


def export(
    fun_jit: stages.Wrapped,
    *,
    platforms: Sequence[str] | None = None,
    disabled_checks: Sequence[DisabledSafetyCheck] = (),
    ) -> Callable[..., Exported]:
  """Exports a JAX function for persistent serialization.

  Args:
    fun_jit: the function to export. Should be the result of `jax.jit`.
    platforms:
        Optional sequence containing a subset of 'tpu', 'cpu',
        'cuda', 'rocm'. If more than one platform is specified, then
        the exported code takes an argument specifying the platform.
        If None, then use the default JAX backend.
        The calling convention for multiple platforms is explained at
        https://jax.readthedocs.io/en/latest/export/export.html#module-calling-convention.
    disabled_checks: the safety checks to disable. See documentation for
        of `jax.export.DisabledSafetyCheck`.

  Returns:
    a function that takes args and kwargs pytrees of {class}`jax.ShapeDtypeStruct`,
    or values with `.shape` and `.dtype` attributes, and returns an
    `Exported`.

  Usage:

      >>> from jax import export
      >>> exported: export.Exported = export.export(jnp.sin)(
      ...     np.arange(4, dtype=np.float32))
      >>>
      >>> # You can inspect the Exported object
      >>> exported.in_avals
      (ShapedArray(float32[4]),)
      >>> blob: bytearray = exported.serialize()
      >>>
      >>> # The serialized bytes are safe to use in a separate process
      >>> rehydrated: export.Exported = export.deserialize(blob)
      >>> rehydrated.fun_name
      'sin'
      >>> rehydrated.call(np.array([.1, .2, .3, .4], dtype=np.float32))
      Array([0.09983342, 0.19866933, 0.29552022, 0.38941833], dtype=float32)
  """
  return _export_internal(fun_jit, platforms=platforms,
                          disabled_checks=disabled_checks)


# TODO(necula): remove this once we improve the integration with jax2tf.
def _export_internal(
    fun_jit: stages.Wrapped,
    *,
    platforms: Sequence[str] | None = None,
    disabled_checks: Sequence[DisabledSafetyCheck] = (),
    _device_assignment_for_internal_jax2tf_use_only = None,
    ) -> Callable[..., Exported]:
  """Exports native serialization for a JAX function.

  Note: this function exists only for internal usage by jax2tf. Use
    `jax.export` instead.
    See https://jax.readthedocs.io/en/latest/export/export.html

  See docstring of `export` for more details.
  """
  if not isinstance(fun_jit, stages.Wrapped):
    raise ValueError(
        f"Function to be exported must be the result of `jit` but is: {fun_jit}")

  def do_export(*args_specs, **kwargs_specs) -> Exported:
    if platforms is not None:
      actual_lowering_platforms = tuple(platforms)
    else:
      actual_lowering_platforms = (default_export_platform(),)

    # TODO: move to `lower`
    check_symbolic_scope_errors(fun_jit, args_specs, kwargs_specs)

    traced = fun_jit.trace(*args_specs, **kwargs_specs)
    lowered = traced.lower(
        lowering_platforms=actual_lowering_platforms,
        _private_parameters=mlir.LoweringParameters(
            for_export=True,
            export_ignore_forward_compatibility=config.export_ignore_forward_compatibility.value))
    return _export_lowered(
        lowered, traced.jaxpr, traced.fun_name,
        disabled_checks=disabled_checks,
        _device_assignment_for_internal_jax2tf_use_only=_device_assignment_for_internal_jax2tf_use_only)
  return do_export


def check_symbolic_scope_errors(fun_jax, args_specs, kwargs_specs):
  symbolic_scope: tuple[shape_poly.SymbolicScope, tree_util.KeyPath] | None = None  # type: ignore[invalid-annotation,unused-ignore]
  for k_path, aval in tree_util.tree_flatten_with_path((args_specs, kwargs_specs))[0]:
    # Static args may have no `shape` attribute.
    if not hasattr(aval, "shape"):
      continue
    for d in aval.shape:
      if shape_poly.is_symbolic_dim(d):
        if symbolic_scope is None:
          symbolic_scope = (d.scope, k_path)
          continue
        symbolic_scope[0]._check_same_scope(
            d, when=f"when exporting {util.fun_name(fun_jax)}",
            self_descr=f"current (from {shape_poly.args_kwargs_path_to_str(symbolic_scope[1])}) ",
            other_descr=shape_poly.args_kwargs_path_to_str(k_path))


def _export_lowered(
    lowered: stages.Lowered,
    jaxpr: core.ClosedJaxpr,
    fun_name: str,
    disabled_checks: Sequence[DisabledSafetyCheck] = (),
    _device_assignment_for_internal_jax2tf_use_only=None,
  ) -> Exported:
  version = config.jax_export_calling_convention_version.value
  if (version < minimum_supported_calling_convention_version or
      version > maximum_supported_calling_convention_version):
    raise ValueError(
      f"The requested export calling convention version {version} is outside the "
      f"range of supported versions [{minimum_supported_calling_convention_version}"
      f"..{maximum_supported_calling_convention_version}]")

  lowering = lowered._lowering
  _check_lowering(lowering)
  mlir_module = lowering.stablehlo()

  args_avals_flat, _ = tree_util.tree_flatten(lowered.in_avals)
  if "mut" in lowering.compile_args:
    if lowering.compile_args["mut"]: raise NotImplementedError
  if "kept_var_idx" in lowering.compile_args:
    module_kept_var_idx = tuple(sorted(lowering.compile_args["kept_var_idx"]))
  else:
    # For pmap
    module_kept_var_idx = tuple(range(len(args_avals_flat)))
  shape_poly_state = lowering.compile_args["shape_poly_state"]
  if (not all(core.is_constant_shape(a.shape) for a in args_avals_flat)
      or lowering.compile_args.get("ordered_effects", [])):
    mlir_module = _wrap_main_func(
        mlir_module, args_avals_flat, args_kwargs_tree=lowered.in_tree,
        has_platform_index_argument=shape_poly_state.has_platform_index_argument,
        module_kept_var_idx=module_kept_var_idx,
        serialization_version=version)

  with mlir_module.context:
    mlir_module_attrs = mlir_module.operation.attributes
    mlir_module_attrs["jax.uses_shape_polymorphism"] = (
        mlir.ir.BoolAttr.get(shape_poly_state.uses_dim_vars))

  mlir_module_serialized = _module_to_bytecode(mlir_module)

  # Figure out the result types and shapes
  if "global_out_avals" in lowering.compile_args:
    # This is currently the case for pjit
    out_avals_flat = lowering.compile_args["global_out_avals"]
  elif "shards" in lowering.compile_args:  # for PmapComputation
    out_avals_flat = lowering.compile_args["shards"].out_sharded_avals
  else:
    out_avals_flat = lowered.compile_args["out_avals"]  # type: ignore

  # Log and then check the module.
  if logging.vlog_is_on(3):
    logmsg = (f"fun_name={fun_name} version={version} "
              f"lowering_platforms={lowering._platforms} "  # type: ignore[unused-ignore,attribute-error]
              f"disabled_checks={disabled_checks}")
    logging.info("Exported JAX function: %s\n", logmsg)
    logging.info(mlir.dump_module_message(mlir_module, "export"))
    logging.info(
        "Size of mlir_module_serialized: %d byte",
        len(mlir_module_serialized),
    )

  _check_module(mlir_module,
                disabled_checks=disabled_checks)

  ordered_effects = tuple(lowering.compile_args["ordered_effects"])
  unordered_effects = tuple(lowering.compile_args["unordered_effects"])

  nr_devices = lowering.compile_args["num_devices"]
  def export_sharding(s: LoweringSharding,
                      aval: core.ShapedArray) -> HloSharding | None:
    if isinstance(s, sharding_impls.UnspecifiedValue):
      return None
    return s._to_xla_hlo_sharding(aval.ndim)

  all_in_shardings = expand_in_shardings(lowering.compile_args["in_shardings"],
                                         module_kept_var_idx,
                                         len(args_avals_flat))
  in_shardings = tuple(
    export_sharding(s, aval)
    for s, aval in zip(all_in_shardings, args_avals_flat))
  out_shardings = tuple(
    export_sharding(s, aval)
    for s, aval in zip(lowering.compile_args["out_shardings"], out_avals_flat))

  device_assignment = lowering.compile_args["device_assignment"]
  if _device_assignment_for_internal_jax2tf_use_only is not None:
    _device_assignment_for_internal_jax2tf_use_only[0] = device_assignment
  def _get_exported_vjp(exp_primal: Exported) -> Exported:
    # Turn the primal jaxpr into a function, in preparation for exporting
    # the VJP. Note that jaxpr_as_fun produces a function with flat arguments
    assert(jaxpr is not None)  # None only when the lowered was created outside JAX
    fun_jax = core.jaxpr_as_fun(jaxpr)

    fun_vjp_jax, vjp_in_avals = _get_vjp_fun(fun_jax,
                                             in_tree=exp_primal.in_tree,
                                             in_avals=exp_primal.in_avals,
                                             in_shardings_hlo=exp_primal.in_shardings_hlo,
                                             out_avals=exp_primal.out_avals,
                                             out_shardings_hlo=exp_primal.out_shardings_hlo,
                                             device_assignment=device_assignment,
                                             apply_jit=True,
                                             flat_primal_fun=True)
    return export(fun_vjp_jax,  # type: ignore[arg-type]
                  platforms=exp_primal.platforms,
                  disabled_checks=exp_primal.disabled_safety_checks)(*vjp_in_avals)

  return Exported(
      fun_name=fun_name,
      in_tree=lowered.in_tree,
      out_tree=lowered.out_tree,
      in_avals=tuple(args_avals_flat),
      out_avals=tuple(out_avals_flat),
      in_shardings_hlo=in_shardings,
      out_shardings_hlo=out_shardings,
      nr_devices=nr_devices,
      platforms=lowering._platforms,  # type: ignore
      ordered_effects=ordered_effects,
      unordered_effects=unordered_effects,
      disabled_safety_checks=tuple(disabled_checks),
      mlir_module_serialized=mlir_module_serialized,
      module_kept_var_idx=module_kept_var_idx,
      uses_global_constants=shape_poly_state.uses_dim_vars,
      calling_convention_version=version,
      _get_vjp=_get_exported_vjp)

def _module_to_bytecode(module: ir.Module) -> bytes:
  mlir_str = mlir.module_to_bytecode(module)
  # `target_version` is used to manage situations when a StableHLO producer
  # and a StableHLO consumer were built using different versions of StableHLO.
  #
  # Each StableHLO version `producer_version` has a compatibility window,
  # i.e. range of versions [`consumer_version_min`, `consumer_version_max`],
  # where StableHLO portable artifacts serialized by `producer_version`
  # can be deserialized by `consumer_version` within the window.
  # See https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md
  # for the exact extent of these compatibility guarantees.
  #
  # `hlo.get_version_from_compatibility_requirement(WEEK_4)` returns a version
  # of StableHLO >= 4w old. This allows new StableHLO features to be used after
  # ~4w and be compatible with any consumer that is updated on at least a
  # monthly cadence.
  #
  # Note that this does not verify any JAX custom calls, which are only
  # guaranteed 3w of forward compatibility, and only prevents use of new
  # StableHLO features from failing on older hardware.
  if hlo.get_api_version() < 9:
    target_version = hlo.get_minimum_version()
  else:
    target_version = hlo.get_version_from_compatibility_requirement(
      hlo.StablehloCompatibilityRequirement.WEEK_4)
  module_serialized = xla_client._xla.mlir.serialize_portable_artifact(  # type: ignore
      mlir_str, target_version)
  return module_serialized


def _wrap_main_func(
    module: ir.Module,
    args_avals_flat: Sequence[core.ShapedArray],
    *,
    args_kwargs_tree: tree_util.PyTreeDef,
    has_platform_index_argument: bool,
    module_kept_var_idx: tuple[int, ...],
    serialization_version: int
) -> ir.Module:
  """Wraps the lowered module with a new "main" handling dimension arguments.

  See calling convention documentation https://jax.readthedocs.io/en/latest/export/export.html#module-calling-convention.

  Args:
    module: the HLO module as obtained from lowering.
    args_avals_flat: the avals for all the arguments of the lowered function,
      which correspond to the array arguments of the `module`.
    args_kwargs_tree: the PyTreeDef corresponding to `(args, kwargs)`, for error
      messages.
    has_platform_index_argument: whether the `module` has a first platform
      index argument
    module_kept_var_idx: a sorted tuple of integers with the indices of arguments
      in `args_avals_flat` that are kept as `module` arguments.
    serialization_version: the target serialization version

  Returns the wrapped module, without dimension and token arguments.
  """
  dim_vars = shape_poly.all_dim_vars(args_avals_flat)
  context = mlir.make_ir_context()
  with context, ir.Location.unknown(context):
    # Make a copy, do not mutate because it may be cached
    wrapped_module = ir.Module.parse(mlir.module_to_bytecode(module))
    symbol_table = ir.SymbolTable(wrapped_module.operation)
    orig_main = symbol_table["main"]
    orig_main.attributes["sym_visibility"] = ir.StringAttr.get("private")
    symbol_table.set_symbol_name(orig_main, "_wrapped_jax_export_main")
    orig_main_name = ir.StringAttr(symbol_table.insert(orig_main)).value

    def is_token(typ, attrs):
      return (typ == mlir.token_type())

    orig_input_types = orig_main.type.inputs  # type: ignore
    arg_attrs = list(ir.ArrayAttr(orig_main.arg_attrs))  # type: ignore
    # The order of args: platform_index_arg, dim args, token args, array args.
    nr_platform_index_args = 1 if has_platform_index_argument else 0
    nr_dim_args = len(dim_vars)
    token_arg_idxs = [i for i, (typ, attrs) in enumerate(zip(orig_input_types,
                                                             arg_attrs))
                      if is_token(typ, attrs)]
    nr_token_args = len(token_arg_idxs)
    if nr_token_args > 0:
      assert min(token_arg_idxs) == nr_platform_index_args + nr_dim_args
      assert token_arg_idxs == list(
        range(nr_platform_index_args + nr_dim_args,
              nr_platform_index_args + nr_dim_args + nr_token_args))
    nr_array_args = (len(orig_input_types) - nr_platform_index_args
                     - nr_dim_args - nr_token_args)
    assert nr_array_args >= 0

    (platform_input_types, dim_var_input_types,
     token_input_types, array_input_types) = util.split_list(
      orig_input_types, [nr_platform_index_args, nr_dim_args, nr_token_args])

    # The order of results: tokens, array results
    orig_output_types = orig_main.type.results  # type: ignore
    result_attrs = list(ir.ArrayAttr(orig_main.result_attrs))  # type: ignore
    token_result_idxs = [i for i, (typ, attrs) in enumerate(zip(orig_output_types,
                                                                result_attrs))
                         if is_token(typ, attrs)]
    nr_token_results = len(token_result_idxs)
    assert token_result_idxs == list(range(0, nr_token_results))
    nr_array_results = len(orig_output_types) - nr_token_results
    assert nr_array_results >= 0
    new_main_arg_indices = (
        *range(nr_platform_index_args),
        *range(nr_platform_index_args + nr_dim_args, len(orig_input_types)))
    new_main_result_indices = tuple(range(0, len(orig_output_types)))

    new_main_input_types = [orig_input_types[idx] for idx in new_main_arg_indices]
    new_main_output_types = [orig_output_types[idx] for idx in new_main_result_indices]
    new_main_ftype = ir.FunctionType.get(new_main_input_types, new_main_output_types)
    new_main_op = func_dialect.FuncOp(
        "main", new_main_ftype, ip=ir.InsertionPoint.at_block_begin(wrapped_module.body))
    new_main_op.attributes["sym_visibility"] = ir.StringAttr.get("public")
    try:
      new_arg_attrs = []
      for idx in new_main_arg_indices:
        new_arg_attr = {}
        for attr in arg_attrs[idx]:
          if attr.name == "tf.aliasing_output":
            i = new_main_result_indices.index(attr.attr.value)
            new_arg_attr[attr.name] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(32), i
            )
          else:
            new_arg_attr[attr.name] = attr.attr
        new_arg_attrs.append(ir.DictAttr.get(new_arg_attr))
      new_main_op.arg_attrs = ir.ArrayAttr.get(new_arg_attrs)
    except KeyError:
      pass  # TODO: better detection if orig_main.arg_attrs does not exist
    try:
      new_main_op.result_attrs = ir.ArrayAttr.get(
          [result_attrs[idx] for idx in new_main_result_indices])
    except KeyError:
      pass
    symbol_table.insert(new_main_op)
    entry_block = new_main_op.add_entry_block()
    with ir.InsertionPoint(entry_block):
      # Make a context just for lowering the dimension value computations
      module_context = mlir.ModuleContext(
          backend=None, platforms=["cpu"],
          axis_context=sharding_impls.ShardingContext(0),
          keepalives=[], channel_iterator=itertools.count(1),
          host_callbacks=[], module=wrapped_module, context=context,
          lowering_parameters=mlir.LoweringParameters(
              global_constant_computation=True,
              for_export=True,
              export_ignore_forward_compatibility=config.export_ignore_forward_compatibility.value,
          ))
      ctx = mlir.LoweringRuleContext(
        module_context=module_context,
        name_stack=source_info_util.new_name_stack(), primitive=None,
        avals_in=args_avals_flat, avals_out=None,
        tokens_in=mlir.TokenSet(), tokens_out=None)
      # We compute dim_values from the array arguments.
      new_main_op_array_args = new_main_op.arguments[-nr_array_args:]
      if shape_poly.all_dim_vars(args_avals_flat):
        # TODO(necula): handle module_kept_var_idx in presence of shape
        # polymorphism. For now we ensured upstream that we keep all variables.
        assert len(set(module_kept_var_idx)) == len(args_avals_flat)
        dim_values = mlir.lower_fun(
            functools.partial(shape_poly.compute_dim_vars_from_arg_shapes,
                              args_avals_flat, args_kwargs_tree=args_kwargs_tree),
            multiple_results=True)(ctx, *new_main_op_array_args)
      else:
        dim_values = ()
      # The arguments to pass to the call to orig_main
      orig_main_args: list[ir.Value] = []
      # The platform index and the dimension variables
      for arg, arg_type in zip(
          list(new_main_op.arguments[0:nr_platform_index_args]) + mlir.flatten_ir_values(dim_values),
          platform_input_types + dim_var_input_types):
        if arg.type != arg_type:
          orig_main_args.append(hlo.convert(arg_type, arg))
        else:
          orig_main_args.append(arg)
      # Then the token arguments
      orig_main_args.extend(
        new_main_op.arguments[nr_platform_index_args: nr_platform_index_args + nr_token_args])
      # Then the array arguments. We insert a ConvertOp as the only use of
      # an input argument. This helps the downstream shape refinement because
      # it will set the type of input arguments to static shapes, and this
      # can invalidate the module if the argument is used as the result of a
      # function, or if it appears as the input to a custom_call with
      # output_operand_alias attribute. See b/287386268.
      for arg, arg_type in zip(new_main_op_array_args, array_input_types):
        if arg.type != arg_type:
          orig_main_args.append(hlo.convert(arg_type, arg))
        else:
          orig_main_args.append(arg)
      call = func_dialect.CallOp(orig_output_types,
                                 ir.FlatSymbolRefAttr.get(orig_main_name),
                                 orig_main_args)
      func_dialect.ReturnOp([call.results[idx] for idx in new_main_result_indices])
    symbol_table.set_symbol_name(new_main_op, "main")

  return wrapped_module

def _check_lowering(lowering) -> None:
  if not isinstance(lowering, pxla.MeshComputation):
    raise NotImplementedError(f"serialization is supported only for jit. {lowering}")

  if lowering.compile_args["host_callbacks"] or lowering.compile_args["keepalive"]:
    raise NotImplementedError("serialization of host_callbacks is not yet implemented")
  # Check that we do not see new compile_args. When we add a compile_args it is
  # safe to add it to the allowed_compile_args if it does not change the semantics
  # or the calling convention of the lowered module.
  allowed_compile_args = {
      "backend", "platforms", "mesh", "global_in_avals",
      "global_out_avals", "in_shardings", "out_shardings", "kept_var_idx",
      "mut", "spmd_lowering", "auto_spmd_lowering",
      "tuple_args", "ordered_effects", "unordered_effects",
      "keepalive", "host_callbacks", "pmap_nreps", "committed",
      "device_assignment", "jaxpr_debug_info", "shape_poly_state",
      "all_default_mem_kind", "in_layouts", "out_layouts", "all_args_info",
      "pgle_profiler", "intermediate_shardings", "context_mesh",
      "num_devices"}
  for compile_arg in lowering.compile_args.keys():
    if compile_arg not in allowed_compile_args:
      raise NotImplementedError(f"Unrecognized lowered.compile_args[{compile_arg}]")

  # We have not implemented support for some of the compile_args. Check here that
  # the compile_args have the values that have been implemented.
  not_implemented_msgs = []
  for compile_arg, check_value, err_msg in (
      ("spmd_lowering", lambda v: v, "True"),
      ("auto_spmd_lowering", lambda v: not v, "False"),
      # tuple_args is a compilation flag, does not affect lowering.
      ("tuple_args", lambda v: True, "N/A"),
      # unordered_effects do not change the calling convention. Those from
      # jax.debug will also result in keepalive being non-empty and unsupported
      # custom calls. The CallTfEffect is an exception, but we want to allow
      # that one.
      ("unordered_effects", lambda v: True, "N/A"),
      ("ordered_effects", lambda v: True, "N/A"),
      # used for TPU jax.debug, send/recv. Not supported yet.
      ("host_callbacks", lambda v: not v, "empty"),
      # used on all platforms for callbacks. Not supported yet.
      ("keepalive", lambda v: not v, "empty"),
      ("pmap_nreps", lambda v: v == 1, "1"),
      ("shape_poly_state", lambda v: True, "N/A"),
  ):
    if compile_arg in lowering.compile_args:
      if not check_value(lowering.compile_args[compile_arg]):
        not_implemented_msgs.append(
            f"{compile_arg} must be {err_msg} and it is {lowering.compile_args[compile_arg]}")
  if not_implemented_msgs:
    raise NotImplementedError(
        "serialization error, unimplemented lowered.compile_args:\n" +
        "\n".join(not_implemented_msgs))

_CPU_FFI_KERNELS = [
    "lapack_spotrf_ffi", "lapack_dpotrf_ffi", "lapack_cpotrf_ffi", "lapack_zpotrf_ffi",
    "lapack_sgeqrf_ffi", "lapack_dgeqrf_ffi", "lapack_cgeqrf_ffi", "lapack_zgeqrf_ffi",
    "lapack_sorgqr_ffi", "lapack_dorgqr_ffi", "lapack_cungqr_ffi", "lapack_zungqr_ffi",
    "lapack_ssyevd_ffi", "lapack_dsyevd_ffi", "lapack_cheevd_ffi", "lapack_zheevd_ffi",
    "lapack_sgeev_ffi", "lapack_dgeev_ffi", "lapack_cgeev_ffi", "lapack_zgeev_ffi",
    "lapack_sgesdd_ffi", "lapack_dgesdd_ffi", "lapack_cgesdd_ffi", "lapack_zgesdd_ffi",
    "lapack_sgetrf_ffi", "lapack_dgetrf_ffi", "lapack_cgetrf_ffi", "lapack_zgetrf_ffi",
    "lapack_ssytrd_ffi", "lapack_dsytrd_ffi", "lapack_chetrd_ffi", "lapack_zhetrd_ffi",
    "lapack_sgehrd_ffi", "lapack_dgehrd_ffi", "lapack_cgehrd_ffi", "lapack_zgehrd_ffi",
    "lapack_sgees_ffi", "lapack_dgees_ffi", "lapack_cgees_ffi", "lapack_zgees_ffi",
    "lapack_strsm_ffi", "lapack_dtrsm_ffi", "lapack_ctrsm_ffi", "lapack_ztrsm_ffi",
    "lapack_sgtsv_ffi", "lapack_dgtsv_ffi", "lapack_cgtsv_ffi", "lapack_zgtsv_ffi",
]
_GPU_FFI_KERNELS = [
    # lu on GPU
    "cu_lu_pivots_to_permutation", "cusolver_getrf_ffi",
    "hip_lu_pivots_to_permutation", "hipsolver_getrf_ffi",
    # qr on GPU
    "cusolver_geqrf_ffi", "cusolver_orgqr_ffi",
    "hipsolver_geqrf_ffi", "hipsolver_orgqr_ffi",
    # eigh on GPU
    "cusolver_syevd_ffi", "hipsolver_syevd_ffi",
    # svd on GPU
    "cusolver_gesvd_ffi", "cusolver_gesvdj_ffi",
    "hipsolver_gesvd_ffi", "hipsolver_gesvdj_ffi",
    # tridiagonal on GPU
    "cusolver_sytrd_ffi",
]
# These are the JAX custom call target names that are guaranteed to be stable.
# Their backwards compatibility is tested by back_compat_test.py.
_CUSTOM_CALL_TARGETS_GUARANTEED_STABLE = {
    *_CPU_FFI_KERNELS,
    *_GPU_FFI_KERNELS,
    "Sharding", "SPMDFullToShardShape", "SPMDShardToFullShape",
    "cu_threefry2x32", "cu_threefry2x32_ffi",
    # Triton IR does not guarantee stability.
    # "__gpu$xla.gpu.triton",
    # cholesky on CPU
    "lapack_spotrf", "lapack_dpotrf", "lapack_cpotrf", "lapack_zpotrf",
    # eigh on TPU
    "Eigh",
    # eig on CPU
    "lapack_sgeev", "lapack_dgeev", "lapack_cgeev", "lapack_zgeev",
    # svd on CPU
    "lapack_sgesdd", "lapack_dgesdd", "lapack_cgesdd", "lapack_zgesdd",
    # qr and svd on TPU
    "Qr", "ProductOfElementaryHouseholderReflectors",
    # triangular_solve on CPU
    "blas_strsm", "blas_dtrsm", "blas_ctrsm", "blas_ztrsm",
    # schur on CPU
    "lapack_sgees", "lapack_dgees", "lapack_cgees", "lapack_zgees",
    # tridiagonal on CPU
    "lapack_ssytrd", "lapack_dsytrd", "lapack_chetrd", "lapack_zhetrd",
    # hessenberg on CPU
    "lapack_sgehrd", "lapack_dgehrd", "lapack_cgehrd", "lapack_zgehrd",
    # lu on TPU
    "LuDecomposition",
    # ApproxTopK on TPU
    "ApproxTopK", "stablehlo.dynamic_approx_top_k",
    "tf.call_tf_function",  # From jax2tf.call_tf(func, call_tf_graph=True)
    "tpu_custom_call",  # Pallas/TPU kernels
    # TODO(burmako): maintain backwards compatibility for these, until they
    # are upstreamed to StableHLO.
    # See https://github.com/openxla/stablehlo/issues/8.
    "stablehlo.dynamic_reduce_window",
    "stablehlo.dynamic_rng_bit_generator",
    "stablehlo.dynamic_top_k",
    "shape_assertion",  # Used by shape_poly to evaluate assertions
}

check_sharding_pattern = re.compile(r"^({replicated}|{unknown shard_as.*}|"")$")

def _check_module(mod: ir.Module, *,
                  disabled_checks: Sequence[DisabledSafetyCheck]) -> bool:
  """Run a number of checks on the module.

  Args:
    disabled_checks: the safety checks that are disabled.

  Returns True if the module uses non-replicated shardings.
  """
  sharding_attr = ir.StringAttr.get("Sharding", mod.context)
  allowed_custom_call_targets: set[str] = copy.copy(_CUSTOM_CALL_TARGETS_GUARANTEED_STABLE)
  for dc in disabled_checks:
    target = dc.is_custom_call()
    if target is not None:
      allowed_custom_call_targets.add(target)

  allowed_custom_call_targets_attrs = {
      ir.StringAttr.get(target, mod.context)
      for target in allowed_custom_call_targets}
  disallowed_custom_call_ops: list[str] = []
  module_uses_non_replicated_sharding = False
  def check_sharding(op: ir.Operation, loc: ir.Location):
    try:
      sharding = op.attributes["mhlo.sharding"]
    except KeyError:
      pass
    else:
      nonlocal module_uses_non_replicated_sharding
      try:
        sharding_value = ir.StringAttr(sharding).value
      except UnicodeDecodeError:
        # The mhlo.sharding attribute may be in pretty-printed format, or
        # as an encoding of an HloSharding protobuf in some rare situations.
        # We handle the latter by conservatively assuming it is non-replicated.
        module_uses_non_replicated_sharding = True
      else:
        if not re.match(check_sharding_pattern, sharding_value):
          module_uses_non_replicated_sharding = True

  def check_op(op: ir.Operation):
    op_name = op.operation.name
    if op_name == "func.func":
      check_sharding(op.operation, op.location)

    elif op_name == "stablehlo.custom_call":
      call_target_name_attr = op.operation.attributes["call_target_name"]
      if (call_target_name_attr not in allowed_custom_call_targets_attrs):
        disallowed_custom_call_ops.append(f"{op} at {op.location}")
      if call_target_name_attr == sharding_attr:
        check_sharding(op, op.location)

  def walk_operations(op):
    check_op(op)
    for region in op.operation.regions:
      for block in region:
        for op in block:
          walk_operations(op)

  walk_operations(mod)
  if disallowed_custom_call_ops:
    disallowed_custom_call_ops_str = "\n".join(disallowed_custom_call_ops)
    msg = ("Cannot serialize code with custom calls whose targets have no "
           "compatibility guarantees. "
           "See https://jax.readthedocs.io/en/latest/export/export.html#compatibility-guarantees-for-custom-calls. "
           "Examples are:\n"
           f"{disallowed_custom_call_ops_str}.\n")
    raise ValueError(msg)
  return module_uses_non_replicated_sharding

def expand_in_shardings(in_shardings: Sequence[LoweringSharding],
                        module_kept_var_idx: Sequence[int],
                        nr_inputs: int) -> Sequence[LoweringSharding]:
  """Expands in_shardings with unspecified shardings for inputs not kept.

  Assumes in_shardings corresponds to module_kept_var_idx.
  """
  assert len(in_shardings) == len(module_kept_var_idx)
  assert nr_inputs >= len(module_kept_var_idx)
  all_in_shardings: list[LoweringSharding] = [sharding_impls.UNSPECIFIED] * nr_inputs
  for idx, in_s in zip(sorted(module_kept_var_idx), in_shardings):
    all_in_shardings[idx] = in_s
  return tuple(all_in_shardings)

def _hlo_sharding_to_xla_compatible_sharding(
    hlo_sharding: HloSharding | None,
    mesh: sharding.Mesh) -> sharding.Sharding | None:
  if hlo_sharding is None:
    return None
  return sharding_impls._gspmd_to_named_sharding_via_mesh(
      _hlo_sharding_to_gspmd_sharding(hlo_sharding, tuple(mesh.devices.flat)),  # type: ignore[arg-type]
      mesh)

def _hlo_sharding_to_gspmd_sharding(
    hlo_sharding: HloSharding | None,
    device_assignment: Sequence[jax.Device]) -> sharding.GSPMDSharding | None:
  if hlo_sharding is None:
    return None
  return sharding.GSPMDSharding(device_assignment, hlo_sharding)

def _get_vjp_fun(primal_fun: Callable, *,
                 in_tree: tree_util.PyTreeDef,
                 in_avals: Sequence[core.AbstractValue],
                 out_avals: Sequence[core.AbstractValue],
                 in_shardings_hlo: tuple[HloSharding | None, ...],
                 out_shardings_hlo: tuple[HloSharding | None, ...],
                 device_assignment: Sequence[sharding_impls.Device] | None,
                 apply_jit: bool,
                 flat_primal_fun: bool = False,
                 ) -> tuple[Callable, Sequence[core.AbstractValue]]:
  # Since jax.vjp does not handle kwargs, it is easier to do all the work
  # here with flattened functions.
  # apply_jit=False is only used for backwards compatibility with the graph
  # graph serialization. When apply_jit=True, we must pass a device assignment.
  # flat_primal_fun=False is used only from jax2tf, and it means that the
  # `primal_fun` takes PyTree `*args` and `**kwargs`.
  def fun_vjp_jax(*args_and_out_cts_flat_jax):
    # Takes a flat list of primals and output cotangents
    def flattened_primal_fun_jax(*args_flat):
      args, kwargs = in_tree.unflatten(args_flat)
      res = primal_fun(*args, **kwargs)
      res_flat, _ = tree_util.tree_flatten(res)
      return res_flat

    args_flat_jax, out_cts_flat_jax = util.split_list(args_and_out_cts_flat_jax,
                                                      [len(in_avals)])
    _, pullback_jax = jax.vjp(primal_fun if flat_primal_fun else flattened_primal_fun_jax,
                              *args_flat_jax)
    return pullback_jax(out_cts_flat_jax)

  vjp_in_avals = list(
      itertools.chain(in_avals,
                      map(lambda a: a.to_tangent_aval(), out_avals)))

  if apply_jit:
    assert device_assignment is not None
    vjp_in_shardings = tuple(
        _hlo_sharding_to_gspmd_sharding(s, device_assignment)
        for s in itertools.chain(in_shardings_hlo, out_shardings_hlo))
    vjp_out_shardings = tuple(
        _hlo_sharding_to_gspmd_sharding(s, device_assignment)
        for s in in_shardings_hlo)
    return pjit.pjit(fun_vjp_jax,
                     in_shardings=vjp_in_shardings,
                     out_shardings=vjp_out_shardings), vjp_in_avals
  else:
    return fun_vjp_jax, vjp_in_avals

### Calling the exported function

def call(exported: Exported) -> Callable[..., jax.Array]:
  if not isinstance(exported, Exported):
    raise ValueError(
      "The exported argument must be an export.Exported. "
      f"Found {exported}.")
  @jax.custom_vjp
  def f_flat(*args_flat):
    return call_exported_p.bind(*args_flat, exported=exported)

  def f_flat_vjp_fwd(*args_flat):
    # Return the primal arguments as the residual
    # TODO: keep as residuals only the arguments that are needed
    return f_flat(*args_flat), args_flat

  def f_flat_vjp_bwd(residual, ct_res_flat):
    args_flat = residual  # residual is the primal argument flat tuple
    exp_vjp = exported.vjp()
    # ct_res_flat may contain arrays of zeros where exp_vjp expect float0.
    # We make the proper arrays of float0 to invoke exp_vjp.
    def fix_float0_ct(ct_res, expected_aval):
      if expected_aval.dtype != dtypes.float0:
        return ct_res
      return ad_util.zeros_like_jaxval(ct_res)

    ct_res_fixed = map(fix_float0_ct,
                       ct_res_flat, exp_vjp.in_avals[len(args_flat):])
    in_ct_flat = call_exported(exp_vjp)(*args_flat, *ct_res_fixed)
    return in_ct_flat

  f_flat.defvjp(f_flat_vjp_fwd, f_flat_vjp_bwd)

  def f_imported(*args, **kwargs):
    # since custom_vjp does not support kwargs, flatten the function first.
    args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
    if in_tree != exported.in_tree:
      # Give errors with the precise tree difference; use fake leaves so we can
      # use tree_util.equality_errors.
      in_args = in_tree.unflatten([0] * in_tree.num_leaves)
      exp_in_args = exported.in_tree.unflatten([0] * exported.in_tree.num_leaves)

      msg = (
          "The invocation args and kwargs must have the same pytree structure "
          f"as when the function '{exported.fun_name}' was exported, but they "
          "have the following structural differences:\n" +
          ("\n".join(
             f"   - {shape_poly.args_kwargs_path_to_str(path)} is a {thing1} in the invocation and a "
             f"{thing2} when exported, so {explanation}.\n"
             for path, thing1, thing2, explanation
             in tree_util.equality_errors(in_args, exp_in_args))))
      raise ValueError(msg)

    res_flat = f_flat(*args_flat)
    return exported.out_tree.unflatten(res_flat)
  return f_imported

call_exported = call

# A JAX primitive for invoking a serialized JAX function.
call_exported_p = core.Primitive("call_exported")
call_exported_p.multiple_results = True

@util.cache()
def _call_exported_abstract_eval(
    *in_avals: core.AbstractValue,
    exported: Exported
    ) -> tuple[tuple[core.AbstractValue, ...], set[effects.Effect]]:
  exported_dim_vars = shape_poly.all_dim_vars(exported.in_avals)
  assert len(in_avals) == len(exported.in_avals)  # since the pytrees have the same structure
  # Check that the expected shapes match the actual ones
  for arg_idx, (exp_aval, actual_aval) in enumerate(zip(exported.in_avals, in_avals)):
    if not isinstance(actual_aval, core.ShapedArray):
      raise ValueError(f"Expected ShapedArray but got: {actual_aval}")
    def pp_arg_dim(dim_idx: int | None) -> str:
      return shape_poly.pretty_print_dimension_descriptor(exported.in_tree,
                                                          arg_idx, dim_idx)
    if len(exp_aval.shape) != len(actual_aval.shape):
      raise ValueError(
          f"Rank mismatch for {pp_arg_dim(None)}: expected {exp_aval.shape} "
          f"and called with {actual_aval.shape}")
    if exp_aval.dtype != actual_aval.dtype:
      raise ValueError(
          f"Dtype mismatch for {pp_arg_dim(None)}: expected {exp_aval.dtype} "
          f"and called with {actual_aval.dtype}")
    for dim_idx, aval_d in enumerate(exp_aval.shape):
      # If the exp_aval has a constant dimension then the actual argument must have
      # a matching constant dimension.
      if core.is_constant_dim(aval_d):
        if (not core.is_constant_dim(actual_aval.shape[dim_idx]) or
            aval_d != actual_aval.shape[dim_idx]):
          raise ValueError(
              f"Shape mismatch for {pp_arg_dim(dim_idx)} "
              "(expected same constant): "
              f"expected {exp_aval.shape} and called with {actual_aval.shape}")

  # Must express the exported_dim_vars in terms of the shapes in in_avals.
  solution, shape_constraints, synth_dim_vars = shape_poly.solve_dim_vars(
      exported.in_avals, args_kwargs_tree=exported.in_tree)
  synthetic_env: shape_poly.DimVarEnv = {
      vname: in_avals[arg_idx].shape[dim_idx]
      for (vname, arg_idx, dim_idx) in synth_dim_vars}
  synthetic_eval = shape_poly.ShapeEvaluator(synthetic_env)
  # We discharge all the constraints statically. This results in much simpler
  # composability (because we do not have to worry about the constraints of the
  # Exported called recursively; we only need to worry about entry-point
  # constraints). This also makes sense from a composability point of view,
  # because we get the same errors if we invoke the exported module, or if we
  # trace the exported function. Consider for example, an exported module with
  # signature `f32[a, a] -> f32[a]`. If we invoke the module with an argument
  # `f32[c, d]` it is better to fail because `c == d` is inconclusive, than
  # succeed and add a compile-time check that `c == d`. In the latter case,
  # it would be ambiguous whether we should continue tracing with a result
  # of type `f32[c]` or `f32[d]`.
  shape_constraints.check_statically(synthetic_eval)
  exported_dim_values = [synthetic_eval.evaluate(solution[var])
                         for var in exported_dim_vars]
  out_avals = tuple(
      core.ShapedArray(core.evaluate_shape(out_aval.shape, exported_dim_vars,
                                           *exported_dim_values),
                       dtype=out_aval.dtype, weak_type=out_aval.weak_type)
      for out_aval in exported.out_avals)
  return out_avals, set(exported.ordered_effects + exported.unordered_effects)


call_exported_p.def_effectful_abstract_eval(_call_exported_abstract_eval)

def _call_exported_impl(*args, exported: Exported):
  return dispatch.apply_primitive(call_exported_p, *args, exported=exported)

call_exported_p.def_impl(_call_exported_impl)

def _call_exported_lowering(ctx: mlir.LoweringRuleContext, *args,
                            exported: Exported):
  if exported.uses_global_constants:
    ctx.module_context.shape_poly_state.uses_dim_vars = True
  submodule = ir.Module.parse(exported.mlir_module())

  axis_context = ctx.module_context.axis_context
  if isinstance(axis_context, sharding_impls.ShardingContext):
    num_devices = axis_context.num_devices
  elif isinstance(axis_context, sharding_impls.SPMDAxisContext):
    num_devices = axis_context.mesh.size
  elif isinstance(axis_context, sharding_impls.ReplicaAxisContext):
    num_devices = axis_context.axis_env.nreps
  else:
    raise NotImplementedError(type(axis_context))
  if num_devices != exported.nr_devices:
    # In some special cases we allow running with a different number of devices
    # than the function was exported for.
    err_msg = ""
    if exported.nr_devices != 1:
      err_msg = "the function was exported for more than 1 device."
    elif (_check_module(submodule, disabled_checks=()) or
          any(s is not None and not s.is_replicated()
              for s in exported.in_shardings_hlo + exported.out_shardings_hlo)):
      err_msg = "the function contains non-replicated sharding annotations."
    if err_msg:
      raise ValueError(
        f"Function {exported.fun_name} was exported for "
        f"{exported.nr_devices} devices and is called in a context with "
        f"{num_devices} devices. This is disallowed because: {err_msg}"
      )

  # Apply in_shardings
  args = tuple(
    wrap_with_sharding(ctx, x, x_aval, x_sharding)
    for x, x_aval, x_sharding in zip(args, ctx.avals_in, exported.in_shardings_hlo))
  symtab = ir.SymbolTable(submodule.operation)
  # The called function may have been exported with polymorphic shapes and called
  # now with more refined shapes. We insert hlo.ConvertOp to ensure the module
  # is valid.
  def convert_shape(x: ir.Value, x_aval: core.AbstractValue, new_aval: core.AbstractValue) -> ir.Value:
    new_ir_type = mlir.aval_to_ir_type(new_aval)
    if x.type != new_ir_type:
      return hlo.convert(mlir.aval_to_ir_type(new_aval), x)
    else:
      return x

  callee_type = symtab["main"].type
  # TODO: maybe cache multiple calls
  fn = mlir.merge_mlir_modules(ctx.module_context.module,
                               f"call_exported_{exported.fun_name}",
                               submodule,
                               dst_symtab=ctx.module_context.symbol_table)

  submodule_args: list[ir.Value] = []
  # All the platforms for the current lowering must be among the platforms
  # for which the callee was lowered.
  lowering_platforms = ctx.module_context.platforms

  callee_lowering_platform_index: list[int] = []
  for platform in lowering_platforms:
    if platform in exported.platforms:
      callee_lowering_platform_index.append(
        exported.platforms.index(platform))
    elif DisabledSafetyCheck.platform() in exported.disabled_safety_checks:
      callee_lowering_platform_index.append(0)
    else:
      raise ValueError(
          f"Function '{exported.fun_name}' was exported for "
          f"platforms '{exported.platforms}' but it is used "
          f"on '{lowering_platforms}'.")

  if len(exported.platforms) > 1:
    # The exported module takes a platform index argument
    if len(lowering_platforms) > 1:
      current_platform_idx = ctx.dim_var_values[0]
    else:
      current_platform_idx = cast(ir.Value, mlir.ir_constant(np.int32(0)))
    # Compute the rule index based on the current platform
    i32_type = mlir.aval_to_ir_type(core.ShapedArray((), dtype=np.int32))
    if current_platform_idx.type != i32_type:
      current_platform_idx = hlo.ConvertOp(i32_type, current_platform_idx)
    callee_platform_idx = hlo.CaseOp([i32_type],
                                     index=current_platform_idx,
                                     num_branches=len(lowering_platforms))
    for i in range(len(lowering_platforms)):
      branch = callee_platform_idx.regions[i].blocks.append()
      with ir.InsertionPoint(branch):
        hlo.return_([mlir.ir_constant(
          np.int32(callee_lowering_platform_index[i]))])
    if callee_platform_idx.result.type != callee_type.inputs[0]:
      callee_platform_idx = hlo.ConvertOp(callee_type.inputs[0],
                                          callee_platform_idx)

    submodule_args.append(callee_platform_idx)
  else:
    assert len(lowering_platforms) == 1

  ordered_effects = exported.ordered_effects
  for eff in ordered_effects:
    token_in = ctx.tokens_in.get(eff)
    submodule_args.append(token_in)
  kept_args = [
      convert_shape(a, a_aval, exported_in_aval)
      for i, (a, a_aval, exported_in_aval) in enumerate(zip(args, ctx.avals_in, exported.in_avals))
      if i in exported.module_kept_var_idx]
  submodule_args = submodule_args + kept_args

  call = func_dialect.CallOp(callee_type.results,
                             ir.FlatSymbolRefAttr.get(fn),
                             submodule_args)
  if ordered_effects:
    tokens_out = {eff: (call.results[effect_idx],)
                  for effect_idx, eff in enumerate(ordered_effects)}
    ctx.set_tokens_out(mlir.TokenSet(tokens_out))
  # The ctx.avals_out already contain the abstract values refined by
  # _call_exported_abstract_eval.
  results = tuple(
      convert_shape(out, out_aval, refined_out_aval)
      for out, out_aval, refined_out_aval in zip(call.results[len(ordered_effects):],
                                                 exported.out_avals, ctx.avals_out))
  # Apply out_shardings
  results = tuple(
    wrap_with_sharding(ctx, x, x_aval, x_sharding)
    for x, x_aval, x_sharding in zip(results, ctx.avals_out, exported.out_shardings_hlo)
  )
  return results

mlir.register_lowering(call_exported_p, _call_exported_lowering)

def wrap_with_sharding(ctx: mlir.LoweringRuleContext,
                       x: ir.Value,
                       x_aval: core.AbstractValue,
                       x_sharding: HloSharding | None) -> ir.Value:
  if x_sharding is None:
    return x
  return mlir.wrap_with_sharding_op(
    ctx, x, x_aval, x_sharding.to_proto())
