# Copyright 2024 The JAX Authors.
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
"""Colocated Python function API implementation."""

from __future__ import annotations

import dataclasses
import inspect
import random
import threading
from typing import Any, Callable, Sequence

import jax
from jax._src import api
from jax._src import tree_util
from jax._src import util
from jax._src.interpreters import pxla
from jax._src.lib import xla_client as xc
from jax._src.traceback_util import api_boundary
from jax._src.util import wraps
from jax.experimental.colocated_python import func_backend
from jax.experimental.colocated_python.serialization import _deserialize_specs, _make_specs_for_serialized_specs, _serialize, _serialize_specs
from jax.extend.ifrt_programs import ifrt_programs

ShapeDtypeStructTree = Any  # PyTree[api.ShapeDtypeStruct]


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionInfo:
  """User function wrapped by colocated_python."""

  fun: Callable[..., Any]
  fun_sourceinfo: str | None
  fun_signature: inspect.Signature | None


@dataclasses.dataclass(frozen=True, slots=True)
class Specialization:
  """Specialization for a colocated_python function."""

  in_specs_treedef: tree_util.PyTreeDef | None = None
  in_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None
  out_specs_fn: Callable[..., ShapeDtypeStructTree] | None = None
  out_specs_treedef: tree_util.PyTreeDef | None = None
  out_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None
  devices: xc.DeviceList | None = None

  def update(
      self,
      *,
      in_specs_treedef: tree_util.PyTreeDef | None = None,
      in_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None,
      out_specs_fn: Callable[..., ShapeDtypeStructTree] | None = None,
      out_specs_treedef: tree_util.PyTreeDef | None = None,
      out_specs_leaves: tuple[api.ShapeDtypeStruct, ...] | None = None,
      devices: Sequence[jax.Device] | xc.DeviceList | None = None,
  ) -> Any:
    """Creates a new specialization with overrides."""
    if in_specs_treedef is None:
      in_specs_treedef = self.in_specs_treedef
    elif self.in_specs_treedef is not None:
      raise ValueError("in_specs already specified")
    if in_specs_leaves is None:
      in_specs_leaves = self.in_specs_leaves
    elif self.in_specs_leaves is not None:
      raise ValueError("in_specs already specified")

    if out_specs_fn is None:
      out_specs_fn = self.out_specs_fn
    elif self.out_specs_fn is not None:
      raise ValueError("out_specs_fn already specified")

    if out_specs_treedef is None:
      out_specs_treedef = self.out_specs_treedef
    elif self.out_specs_treedef is not None:
      raise ValueError("out_specs already specified")
    if out_specs_leaves is None:
      out_specs_leaves = self.out_specs_leaves
    elif self.out_specs_leaves is not None:
      raise ValueError("out_specs already specified")

    if devices is None:
      devices = self.devices
    elif self.devices is not None:
      raise ValueError("devices already specified")
    elif not isinstance(devices, xc.DeviceList):
      devices = xc.DeviceList(tuple(devices))

    return Specialization(
        in_specs_treedef,
        in_specs_leaves,
        out_specs_fn,
        out_specs_treedef,
        out_specs_leaves,
        devices,
    )


def _get_spec(x: Any) -> api.ShapeDtypeStruct:
  """Extracts a spec for a value, which must be a JAX Array."""
  # TODO(hyeontaek): Allow Python values and automatically apply `shard_arg`
  # with a suitable sharding and layout.
  if not isinstance(x, jax.Array):
    raise ValueError(
        "colocated_python only supports jax.Array as input and output, but got"
        f" {type(x)}."
    )
  return api.ShapeDtypeStruct(shape=x.shape, dtype=x.dtype, sharding=x.sharding)


def _infer_devices_from_args(args: Sequence[Any]) -> xc.DeviceList | None:
  """Returns a representative device list from function call arguments."""
  device_list_set: set[xc.DeviceList] = set()
  for x in args:
    sharding = getattr(x, "sharding", None)
    if sharding is not None:
      device_list_set.add(x.sharding._internal_device_list)
  if not device_list_set:
    return None
  if len(device_list_set) != 1:
    raise ValueError(
        "All arguments must use the same device list, but got"
        f" multiple device lists: {device_list_set}."
    )
  return device_list_set.pop()


def _compile_to_executable(
    name: str,
    fun: Callable[..., Any],
    in_specs_treedef: tree_util.PyTreeDef,
    in_specs_leaves: tuple[api.ShapeDtypeStruct, ...],
    out_specs_treedef: tree_util.PyTreeDef,
    out_specs_leaves: tuple[api.ShapeDtypeStruct, ...],
    devices: xc.DeviceList,
) -> Callable[..., Any]:
  """Compiles a Python function into a runtime executable."""
  fun_and_specialization = (
      fun,
      in_specs_treedef,
      in_specs_leaves,
      out_specs_treedef,
      out_specs_leaves,
      devices,
  )
  pickled_function = _serialize(fun_and_specialization)
  program = ifrt_programs.make_colocated_python_program(
      name, pickled_function, devices, in_specs_leaves, out_specs_leaves
  )
  ifrt_client = devices[0].client
  out_sdss = tuple(
      jax.core.ShapedArray(sds.shape, sds.dtype) for sds in out_specs_leaves
  )
  out_shardings = tuple(sds.sharding for sds in out_specs_leaves)
  try:
    compile_options = ifrt_programs.make_colocated_python_compile_options()
    loaded_executable = ifrt_client.compile_ifrt_program(
        program, compile_options
    )
    out_handlers = pxla.global_avals_to_results_handler(
        out_sdss, out_shardings, committed=True
    ).handlers

    def call(*args, **kwargs):
      args_leaves = tree_util.tree_leaves((args, kwargs))
      execute_result = loaded_executable.execute_sharded(
          args_leaves, with_tokens=False
      )
      results = execute_result.consume_with_handlers(out_handlers)
      return tree_util.tree_unflatten(out_specs_treedef, results)

    return call
  except jax.errors.JaxRuntimeError as e:
    # TODO(hyeontaek): Implement colocated Python support in McJAX and remove
    # this fallback path.
    if "PjRtCompiler requires an HloProgram" in str(e):
      return fun
    raise


def _make_output_specs_and_push_result_fun(
    info: FunctionInfo, specialization: Specialization, uid: int
) -> Callable[..., Any]:
  """Creates a function that computes output specs and pushes the result to the result store."""
  assert specialization.in_specs_treedef is not None
  assert specialization.in_specs_leaves is not None
  assert specialization.out_specs_treedef is None
  assert specialization.out_specs_leaves is None
  assert specialization.devices is not None

  devices = specialization.devices

  def lowered_fun(*args, **kwargs) -> Sequence[jax.Array]:
    result = info.fun(*args, **kwargs)
    result_leaves, out_treedef = tree_util.tree_flatten(result)
    out_spec_leaves = tuple(_get_spec(x) for x in result_leaves)
    func_backend.SINGLETON_RESULT_STORE.push(uid, result_leaves)
    return _serialize_specs(out_treedef, out_spec_leaves, devices)

  out_specs_leaves, out_specs_treedef = tree_util.tree_flatten(
      _make_specs_for_serialized_specs(specialization.devices),
  )
  name = getattr(info.fun, "__name__", "unknown")
  name = f"{name}_output_specs_and_push_result"
  return _compile_to_executable(
      name=name,
      fun=lowered_fun,
      in_specs_treedef=specialization.in_specs_treedef,
      in_specs_leaves=specialization.in_specs_leaves,
      out_specs_treedef=out_specs_treedef,
      out_specs_leaves=tuple(out_specs_leaves),
      devices=specialization.devices,
  )


def _make_pop_result_fun(
    info: FunctionInfo, specialization: Specialization, uid: int
) -> Callable[..., Any]:
  """Makes a function that pops results from the result store."""
  assert specialization.out_specs_treedef is not None
  assert specialization.out_specs_leaves is not None
  assert specialization.devices is not None

  out_specs_treedef = specialization.out_specs_treedef

  def lowered_fun() -> Any:
    result_leaves = func_backend.SINGLETON_RESULT_STORE.pop(uid)
    return tree_util.tree_unflatten(out_specs_treedef, result_leaves)

  in_specs_leaves, in_specs_treedef = tree_util.tree_flatten((
      # args
      (),
      # kwargs
      {},
  ))
  name = getattr(info.fun, "__name__", "unknown")
  name = f"{name}_pop_result"
  return _compile_to_executable(
      name=name,
      fun=lowered_fun,
      in_specs_treedef=in_specs_treedef,
      in_specs_leaves=tuple(in_specs_leaves),
      out_specs_treedef=specialization.out_specs_treedef,
      out_specs_leaves=specialization.out_specs_leaves,
      devices=specialization.devices,
  )


def _make_async_execution_fun(
    info: FunctionInfo, specialization: Specialization
) -> Callable[..., Any]:
  """Makes a function that asynchronously executes the function."""
  assert specialization.in_specs_treedef is not None
  assert specialization.in_specs_leaves is not None
  assert specialization.out_specs_treedef is not None
  assert specialization.out_specs_leaves is not None
  assert specialization.devices is not None

  name = getattr(info.fun, "__name__", "unknown")
  return _compile_to_executable(
      name=name,
      fun=info.fun,
      in_specs_treedef=specialization.in_specs_treedef,
      in_specs_leaves=specialization.in_specs_leaves,
      out_specs_treedef=specialization.out_specs_treedef,
      out_specs_leaves=specialization.out_specs_leaves,
      devices=specialization.devices,
  )


@jax.util.cache(max_size=None)
def _get_specialized_func(
    info: FunctionInfo, specialization: Specialization
) -> Callable[..., Any]:
  """Returns a specialized function for the given specialization."""
  util.test_event("colocated_python_func._get_specialized_func")
  assert specialization.in_specs_treedef is not None
  assert specialization.in_specs_leaves is not None
  assert specialization.devices is not None
  uid = random.getrandbits(63)

  mutex = threading.Lock()
  # Asynchronous execution function that has known output_specs.
  async_execution_func = None

  def specialized_func(*args, **kwargs) -> Any:
    """Specialized function to be executed with given args and kwargs."""
    nonlocal specialization, async_execution_func
    with mutex:
      if async_execution_func is None:
        if specialization.out_specs_treedef is None:
          if specialization.out_specs_fn is None:
            serialized_out_specs = _make_output_specs_and_push_result_fun(
                info, specialization, uid
            )(*args, **kwargs)

            # Waits for the output_specs. This may block.
            out_specs_treedef, out_specs_leaves = _deserialize_specs(
                serialized_out_specs
            )

            # Subsequent calls would use async_execution_func with discovered
            # output_specs.
            specialization = specialization.update(
                out_specs_treedef=out_specs_treedef,
                out_specs_leaves=out_specs_leaves,
            )
            async_execution_func = _make_async_execution_fun(
                info, specialization
            )

            return _make_pop_result_fun(info, specialization, uid)()
          else:
            # Compute out_specs using out_specs_fn and inputs.
            args_specs, kwargs_specs = tree_util.tree_map(
                _get_spec, (args, kwargs)
            )
            out_specs = specialization.out_specs_fn(*args_specs, **kwargs_specs)
            # Type checking is ignored to silence mypy error: Incompatible types
            # in assignment (expression has type "list[Any]", variable has type
            # "tuple[ShapeDtypeStruct, ...]")  [assignment]
            out_specs_leaves, out_specs_treedef = tree_util.tree_flatten(  # type: ignore[assignment]
                out_specs
            )
            specialization = specialization.update(
                out_specs_treedef=out_specs_treedef,
                out_specs_leaves=tuple(out_specs_leaves),
            )
            async_execution_func = _make_async_execution_fun(
                info, specialization
            )
            # Fall-through.
        else:
          async_execution_func = _make_async_execution_fun(info, specialization)
          # Fall-through.

    # Asynchronous execution runs outside of the mutex to allow concurrent
    # execution for inline executors.
    return async_execution_func(*args, **kwargs)

  return specialized_func


def make_callable(
    fun: Callable[..., Any],
    fun_sourceinfo: str | None,
    fun_signature: inspect.Signature | None,
) -> Callable[..., Any]:
  """Makes a colocated Python callable."""
  return _make_callable(
      FunctionInfo(fun, fun_sourceinfo, fun_signature), Specialization()
  )


def _make_callable(
    info: FunctionInfo,
    specialization: Specialization,
) -> Callable[..., Any]:
  """Internal implementation of make_callable."""

  def specialize(
      in_specs: ShapeDtypeStructTree | None = None,
      out_specs_fn: Callable[..., ShapeDtypeStructTree] | None = None,
      devices: Sequence[jax.Device] | None = None,
  ) -> Callable[..., Any]:
    """Returns a colocated Python callable with extra specialization.

    Args:
      in_specs: Optionally specifies the expected input specs. Input specs are
        expressed as a `PyTree[ShapeDtypeStruct]` for `(args, kwargs)` of a
        function call.
      out_specs_fn: Optionally specifies a function that computes the output
        specs from input specs. If unspecified, colocated_python will compute
        the output specs during the very first execution, and this execution
        will be synchronous.
      devices: Optionally specifies the devices to execute the function on. Must
        be provided if in_specs has no leaves because devices cannot be inferred
        from input specs or arguments.

    Returns:
      A colocated Python callable with extra specialization.
    """
    # TODO(hyeontaek): Allow unspecified devices for zero-leaf `in_specs` if
    # `out_specs_fn(in_specs)` returns at least one leaf that we can use for
    # inferring `devices`.
    if in_specs is None:
      in_specs_leaves, in_specs_treedef = None, None
    else:
      in_specs_leaves_list, in_specs_treedef = tree_util.tree_flatten(in_specs)
      in_specs_leaves = tuple(in_specs_leaves_list)
    return _make_callable(
        info,
        specialization.update(
            in_specs_treedef=in_specs_treedef,
            in_specs_leaves=in_specs_leaves,
            out_specs_fn=out_specs_fn,
            devices=devices,
        ),
    )

  @api_boundary
  def __call__(*args, **kwargs) -> Any:
    """Executes the function.

    If the output specs are not known, the very first execution will be
    synchronous.
    """
    args_leaves, in_specs_treedef = tree_util.tree_flatten((args, kwargs))

    in_specs_leaves = tuple(_get_spec(x) for x in args_leaves)
    if specialization.in_specs_treedef is None:
      # Allow input polymorphism by applying input_specs specialization
      # temporarily for this call.
      return _make_callable(
          info,
          specialization.update(
              in_specs_treedef=in_specs_treedef,
              in_specs_leaves=in_specs_leaves,
          ),
      )(*args, **kwargs)

    if specialization.devices is None:
      devices = _infer_devices_from_args(args_leaves)
      if devices is None:
        raise ValueError(
            "No devices found. colocated_python function without input"
            " arguments must be first specialized with devices."
        )
      # Allow device polymorphism by applying devices specialization temporarily
      # for this call.
      return _make_callable(info, specialization.update(devices=devices))(
          *args, **kwargs
      )

    # Assertion is added to silence mypy error: Unsupported operand types for !=
    # ("PyTreeDef" and "None")  [operator]
    assert isinstance(specialization.in_specs_treedef, tree_util.PyTreeDef)

    # If input_specs is known, verify that it matches actual inputs.
    if (specialization.in_specs_treedef != in_specs_treedef
        or specialization.in_specs_leaves != in_specs_leaves):
      raise ValueError(
          "Input specs in specialization and input specs of arguments must have"
          " the same pytree structure, but they have the following structural"
          " differences:\n"
          + ("\n".join(
                f"   - {tree_util.keystr(path)} is a {thing1} in value 1 and"
                f" a {thing2} in  value 2, so {explanation}.\n"
                for path, thing1, thing2, explanation in tree_util.equality_errors_pytreedef(
                    specialization.in_specs_treedef, in_specs_treedef
                ))))

    return _get_specialized_func(info, specialization)(*args, **kwargs)

  __call__ = wraps(info.fun)(__call__)
  __call__.specialize = specialize
  return __call__
