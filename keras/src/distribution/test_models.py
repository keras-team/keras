#!/usr/bin/env python3
"""
Test suite for model verification with Parallax (Keras AutoShardDistribution),
using KerasNLP presets and the Tiny Shakespeare dataset.

This script compares the training performance (loss and perplexity) of a
baseline model against its Parallax-sharded equivalent. The goal is to verify
the correctness and performance of Keras's automatic sharding capabilities.

It is generalized to run tests for multiple model architectures, including:
- Gemma
- GPT-2
- Bloom
- OPT
"""

"""Utilities for distribution strategy with JAX backend."""
class MergeableGraph:
    """A graph that supports merging nodes."""

    def __init__(self):
        self._parent = {}
        self._edges = set()

    def get_root(self, node):
        if node not in self._parent:
            self._parent[node] = node
            return node
        if self._parent[node] == node:
            return node
        self._parent[node] = self.get_root(self._parent[node])
        return self._parent[node]

    def merge_nodes(self, node1, node2):
        root1 = self.get_root(node1)
        root2 = self.get_root(node2)
        if root1 != root2:
            self._parent[root1] = root2

    def add_edge(self, node1, node2):
        root1 = self.get_root(node1)
        root2 = self.get_root(node2)
        if root1 != root2:
            self._edges.add(tuple(sorted((root1, root2))))

    def get_edges(self):
        return self._edges
    
import jax
import numpy as np
import collections
import itertools
from keras.src.backend.common import global_state
from keras.src.random import seed_generator
from keras.src.utils import jax_utils
from keras.src.utils import rng_utils
from jax import tree_util


def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note that this should return the global devices in a distributed setting.

    Args:
        device_type: string of `"cpu"`, `"gpu"` or `"tpu"`. Defaults to `"gpu"`
            or `"tpu"` if available when device_type is not provided. Otherwise
            will return the `"cpu"` devices.

    Return:
        List of devices that are available for distribute computation.
    """
    device_type = device_type.lower() if device_type else None
    jax_devices = jax.devices(backend=device_type)
    return [f"{device.platform}:{device.id}" for device in jax_devices]


def distribute_variable(value, layout):
    """Create a distributed variable for JAX.

    Since JAX doesn't have a variable class, this will just return a `jax.Array`
    with the corresponding layout/sharding specified.

    Note that this function should be used in eager context, not in jitted
    function.

    Args:
        value: the initial value of the variable.
        layout: `TensorLayout` for the created variable, or a
            JAX-supported layout instance (e.g. `jax.sharding.Sharding`).

    Returns:
        jax.Array which is the distributed variable.
    """
    return distribute_tensor(value, layout)


def distribute_tensor(tensor, layout):
    """Distribute the tensor based on the layout.

    Note that this function can be used both in eager context, or within a
    jitted function.

    Args:
        tensor: `jax.Array` that need to be distributed.
        layout: `TensorLayout` for the created variable, or a
            JAX-supported layout instance (e.g. `jax.sharding.Sharding`).

    Returns:
        Distributed value.
    """
    # Avoid circular imports.
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = layout.backend_layout

    # TODO(scottzhu): This might not be a cheap check, we should consider
    # have some proper JAX API for doing this check.
    if jax_utils.is_in_jax_tracing_scope():
        return jax.lax.with_sharding_constraint(tensor, layout)

    # Skip relayout if unnecessary.
    if isinstance(tensor, jax.Array):
        if isinstance(
            layout, jax.sharding.Sharding
        ) and tensor.sharding.is_equivalent_to(layout, ndim=len(tensor.shape)):
            return tensor
        # JAX explicit "layout" support.
        elif hasattr(layout, "layout"):
            current_layout = getattr(tensor, "layout", None)
            if current_layout == layout:
                return tensor
        # JAX explicit "format" support.
        elif hasattr(layout, "format"):
            current_layout = getattr(tensor, "format", None)
            if current_layout == layout:
                return tensor

    return jax.device_put(tensor, layout)


def distribute_data_input(per_process_batch, layout, batch_dim_name):
    """Distribute the input data with the corresponding layout.

    Note that the inputs here is a local worker batch. Within the local worker,
    the data need to be further partitioned to map to each of the devices.

    Args:
        inputs: `jax.Array` that is already sharded to a local process size.
        layout: `TensorLayout` for the distribution information, or a
            `jax.sharding.Sharding` instance.

    Returns:
        A global batch distributed according to `layout`.
    """
    # Avoid circular imports.
    from keras.src.distribution import TensorLayout

    if isinstance(layout, TensorLayout):
        layout = layout.backend_layout

    return jax.make_array_from_process_local_data(layout, per_process_batch)


def initialize_rng():
    """Initializes the global random number generator across processes.

    This is required for consistent initialization in multi-host settings.
    """
    global_seed = rng_utils.get_random_seed()
    # Only set a random seed if not already set
    # via keras.config.set_random_seed()
    if global_seed is None:
        # Generate a random seed on each CPU host and psum them to get a single
        # consistent seed across all processes.
        cpu_devices = jax.devices("cpu")
        num_local_cpu_devices = jax.local_device_count("cpu")
        # Seed must be in range [0, 2^32 - 1], so to ensure proper range and
        # avoid signed integer overflow, we use uint32.
        local_seed = jax.numpy.asarray(
            [seed_generator.make_default_seed()] * num_local_cpu_devices,
            dtype=jax.numpy.uint32,
        )
        # Sum across processes and pull out the first item.
        global_seed = jax.pmap(
            lambda x: jax.lax.psum(x, "all"),
            axis_name="all",
            devices=cpu_devices,
        )(local_seed).item(0)
        # Set the global seed.
        rng_utils.set_random_seed(global_seed)

    # Check if the global seed generator is set and ensure it has an initialized
    # seed.  Otherwise, reset the seed to the global seed.
    global_seed_generator = global_state.get_global_attribute(
        "global_seed_generator"
    )
    if global_seed_generator is not None:
        seed = global_seed_generator.get_config()["seed"]
        if seed is None:
            global_state.set_global_attribute(
                "global_seed_generator",
                seed_generator.SeedGenerator(
                    seed=global_seed,
                    name=global_seed_generator.name,
                    backend=global_seed_generator.backend,
                ),
            )


def initialize(job_addresses, num_processes, process_id):
    if job_addresses and "," in job_addresses:
        # When user provide all the job addresses, we will split and get the
        # first one, which is the coordinator.
        job_addresses = job_addresses.split(",")
        # Do a sanity check to make sure the number of addresses also match
        # the num_processes.
        if num_processes is not None and num_processes != len(job_addresses):
            raise ValueError(
                f"The provided job_addresses {job_addresses} has "
                f"{len(job_addresses)} jobs, but num_processes is "
                f"{num_processes}"
            )
        coordinator_address = job_addresses[0]
    else:
        coordinator_address = job_addresses

    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
    )

    # Ensure the random number generator is initialized across processes.
    initialize_rng()


def num_processes():
    """Return the number of processes for the current distribution setting."""
    return jax.process_count()


def process_id():
    """Return the current process ID for the distribution setting."""
    return jax.process_index()


def _to_backend_device(device_name):
    if isinstance(device_name, jax.Device):
        return device_name
    device_name = str(device_name)
    if ":" not in device_name:
        device_type, device_id = device_name, 0
    else:
        device_type, device_id = device_name.split(":")

    devices = jax.devices(backend=device_type)
    for device in devices:
        if device.platform == device_type and device.id == int(device_id):
            return device
    raise ValueError(f"Device not found: {device_name}")


def _to_backend_mesh(device_mesh):
    """Convert the DeviceMesh to JAX backend specific Mesh.

    Args:
        device_mesh: DeviceMesh instance to convert.

    Returns:
        A `jax.sharding.Mesh` instance.
    """
    shape = device_mesh.devices.shape
    devices = [_to_backend_device(d) for d in device_mesh.devices.flatten()]
    devices = np.array(devices).reshape(shape)
    return jax.sharding.Mesh(devices, device_mesh.axis_names)


def _to_backend_layout(tensor_layout):
    """Convert the TensorLayout to JAX backend specific Sharding.

    Args:
        tensor_layout: TensorLayout instance to convert.

    Returns:
        A `jax.sharding.NamedSharding` instance.
    """
    if tensor_layout.device_mesh is None:
        raise ValueError(
            "Cannot create sharding when device mesh is not set "
            "for TensorLayout."
        )
    partition_spec = jax.sharding.PartitionSpec(*tensor_layout.axes)
    jax_mesh = tensor_layout.device_mesh.backend_mesh
    return jax.sharding.NamedSharding(jax_mesh, partition_spec)


_JAX_CLASSES_DEFINED = False
JaxGraph = None
JaxShardingPlanner = None
JaxShardApplier = None


def _define_and_register_jax_classes():
    global _JAX_CLASSES_DEFINED, JaxGraph, JaxShardingPlanner, JaxShardApplier
    if _JAX_CLASSES_DEFINED:
        return

    # from keras.src.distribution.autoshard_utils import MergeableGraph

    def parse_jaxpr(jaxpr) -> MergeableGraph:
        graph = MergeableGraph()

        def same_axis(node1, node2):
            var1, axis1 = node1
            var2, axis2 = node2
            if var1.aval.shape[axis1] != var2.aval.shape[axis2]:
                return
            graph.merge_nodes(node1, node2)

        def parse_dot_general(eqn):
            lhs, rhs = eqn.invars
            out = eqn.outvars[0]
            (lc, rc), (lb, rb) = eqn.params["dimension_numbers"]
            for l, r in zip(lc, rc):
                same_axis((lhs, l), (rhs, r))
            o_offset = 0
            for l, r in zip(lb, rb):
                same_axis((lhs, l), (rhs, r))
                same_axis((lhs, l), (out, o_offset))
                o_offset += 1
            for i in range(lhs.aval.ndim):
                if i not in lb and i not in lc:
                    same_axis((lhs, i), (out, o_offset))
                    o_offset += 1
            for j in range(rhs.aval.ndim):
                if j not in rb and j not in rc:
                    same_axis((rhs, j), (out, o_offset))
                    o_offset += 1

        def parse_reshape(eqn):
            invar, out = eqn.invars[0], eqn.outvars[0]
            in_idx, out_idx, in_prod, out_prod = 0, 0, 1, 1
            while in_idx < invar.aval.ndim and out_idx < out.aval.ndim:
                if (
                    in_prod == out_prod
                    and invar.aval.shape[in_idx] == out.aval.shape[out_idx]
                ):
                    if invar.aval.shape[in_idx] > 1:
                        same_axis((invar, in_idx), (out, out_idx))
                    in_prod *= invar.aval.shape[in_idx]
                    out_prod *= out.aval.shape[out_idx]
                    in_idx += 1
                    out_idx += 1
                elif in_prod < out_prod:
                    in_prod *= invar.aval.shape[in_idx]
                    in_idx += 1
                else:
                    out_prod *= out.aval.shape[out_idx]
                    out_idx += 1

        def parse_transpose(eqn):
            invar, out = eqn.invars[0], eqn.outvars[0]
            for i, j in enumerate(eqn.params["permutation"]):
                same_axis((invar, j), (out, i))

        def parse_elementwise_with_broadcast(eqn):
            out = eqn.outvars[0]
            for invar in eqn.invars:
                if invar.aval.ndim == 0:
                    continue
                for i in range(1, min(invar.aval.ndim, out.aval.ndim) + 1):
                    in_axis, out_axis = -i, -i
                    if invar.aval.shape[in_axis] == out.aval.shape[out_axis]:
                        same_axis(
                            (invar, invar.aval.ndim + in_axis),
                            (out, out.aval.ndim + out_axis),
                        )

        for var in jaxpr.jaxpr.invars:
            for i, j in itertools.combinations(range(var.aval.ndim), 2):
                graph.add_edge((var, i), (var, j))

        for eqn in jaxpr.eqns:
            for outvar in eqn.outvars:
                for i, j in itertools.combinations(range(outvar.aval.ndim), 2):
                    graph.add_edge((outvar, i), (outvar, j))

            primitive_parsers = {
                "dot_general": parse_dot_general,
                "reshape": parse_reshape,
                "transpose": parse_transpose,
            }
            parser = primitive_parsers.get(
                eqn.primitive.name, parse_elementwise_with_broadcast
            )
            parser(eqn)
        return graph

    def shard_model(
        jaxpr,
        out_avals,
        trainable_params,
        non_trainable_params,
        args,
        kwargs,
        min_shard_size=1,
        data_axis_name="data",
        model_axis_name="model",
    ):
        graph = parse_jaxpr(jaxpr)

        t_params_flat, t_params_treedef = tree_util.tree_flatten(
            trainable_params
        )
        nt_params_flat, nt_params_treedef = tree_util.tree_flatten(
            non_trainable_params
        )
        args_flat, args_treedef = tree_util.tree_flatten(args)
        kwargs_flat, kwargs_treedef = tree_util.tree_flatten(kwargs)
        _, outputs_treedef = tree_util.tree_flatten(out_avals)

        pos = 0
        t_param_invars = jaxpr.jaxpr.invars[pos : pos + len(t_params_flat)]
        pos += len(t_params_flat)
        nt_param_invars = jaxpr.jaxpr.invars[pos : pos + len(nt_params_flat)]
        pos += len(nt_params_flat)
        arg_invars = jaxpr.jaxpr.invars[pos : pos + len(args_flat)]
        pos += len(args_flat)
        kwarg_invars = jaxpr.jaxpr.invars[pos:]

        all_param_invars = t_param_invars + nt_param_invars
        data_invars = arg_invars + kwarg_invars

        seen = collections.Counter()
        for var in all_param_invars:
            for i in range(var.aval.ndim):
                if var.aval.shape[i] >= min_shard_size:
                    seen.update([graph.get_root((var, i))])

        model_axis_root = max(seen, key=seen.get) if seen else None

        data_axes_roots = []
        for var in data_invars:
            for i in range(var.aval.ndim):
                root = graph.get_root((var, i))
                if root not in seen and root not in data_axes_roots:
                    data_axes_roots.append(root)

        def assign_layouts(vars_flat, is_params=False):
            assignments = []
            for var in vars_flat:
                layout = [None] * var.aval.ndim
                for i in range(var.aval.ndim):
                    if var.aval.shape[i] < min_shard_size:
                        continue
                    root = graph.get_root((var, i))
                    if (
                        is_params
                        and model_axis_root
                        and root == model_axis_root
                    ):
                        layout[i] = model_axis_name
                    elif not is_params and root in data_axes_roots:
                        name = data_axis_name
                        if len(data_axes_roots) > 1:
                            name += str(data_axes_roots.index(root))
                        layout[i] = name
                assignments.append(layout)
            return assignments

        params_assignments = tree_util.tree_unflatten(
            t_params_treedef, assign_layouts(t_param_invars, is_params=True)
        )
        return params_assignments

    class _JaxGraph:
        def __init__(
            self,
            jaxpr,
            trainable_variables,
            non_trainable_variables,
            in_treedefs,
            out_avals,
        ):
            self.jaxpr = jaxpr
            self.trainable_variables = trainable_variables
            self.non_trainable_variables = non_trainable_variables
            self.in_treedefs = in_treedefs
            self.out_avals = out_avals

        @classmethod
        def from_model(cls, model, *args, **kwargs):
            def stateless_fn(
                trainable_vars, non_trainable_vars, f_args, f_kwargs
            ):
                return model.stateless_call(
                    trainable_vars, non_trainable_vars, *f_args, **f_kwargs
                )

            trainable_vars = model.trainable_variables
            non_trainable_vars = model.non_trainable_variables
            in_treedefs = tree_util.tree_structure(
                (trainable_vars, non_trainable_vars, args, kwargs)
            )

            closed_jaxpr, out_avals = jax.make_jaxpr(
                stateless_fn, return_shape=True
            )(trainable_vars, non_trainable_vars, args, kwargs)

            return cls(
                closed_jaxpr,
                trainable_vars,
                non_trainable_vars,
                in_treedefs,
                out_avals,
            )

    class _JaxShardingPlanner:
        def plan(self, graph, device_mesh):
            all_in_avals = [var.aval for var in graph.jaxpr.jaxpr.invars]
            all_in_leaves = tree_util.tree_unflatten(
                graph.in_treedefs, all_in_avals
            )
            _, _, args_aval_tree, kwargs_aval_tree = all_in_leaves

            dummy_args = tree_util.tree_map(
                lambda x: np.zeros(x.shape, x.dtype), args_aval_tree
            )
            dummy_kwargs = tree_util.tree_map(
                lambda x: np.zeros(x.shape, x.dtype), kwargs_aval_tree
            )

            param_assignments = shard_model(
                jaxpr=graph.jaxpr,
                out_avals=graph.out_avals,
                trainable_params=graph.trainable_variables,
                non_trainable_params=graph.non_trainable_variables,
                args=dummy_args,
                kwargs=dummy_kwargs,
            )

            param_vars_flat, _ = tree_util.tree_flatten(
                graph.trainable_variables
            )
            param_layouts_flat, _ = tree_util.tree_flatten(param_assignments)

            parameter_layout_dict = {
                var.path: tuple(layout) if layout else None
                for var, layout in zip(param_vars_flat, param_layouts_flat)
            }
            return parameter_layout_dict

    class _JaxShardApplier:
        def apply(self, model, plan):
            for var in model.variables:
                layout = plan.get(var.path)
                if layout:
                    var.layout = layout

    JaxGraph = _JaxGraph
    JaxShardingPlanner = _JaxShardingPlanner
    JaxShardApplier = _JaxShardApplier
    _JAX_CLASSES_DEFINED = True


def get_sharding_planner():
    """Returns an instance of the JAX sharding planner."""
    _define_and_register_jax_classes()
    return JaxShardingPlanner()


def get_shard_applier():
    """Returns an instance of the JAX shard applier."""
    _define_and_register_jax_classes()
    return JaxShardApplier()


def create_graph_from_model(model, *args, **kwargs):
    """Returns a JAX graph representation of the Keras model."""
    _define_and_register_jax_classes()
    return JaxGraph.from_model(model, *args, **kwargs)


import os
import time
import logging
import numpy as np
import keras
import keras_nlp
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
from keras.src.distribution import distribution_lib
# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import collections
import contextlib
import os
import re
import warnings

import numpy as np

from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import distribution_lib
from keras.src.backend.common import global_state
"""Unified high-level distribution APIs across backends.

Currently only the JAX backend is supported. The TensorFlow backend
will be supported in the future (via tf.dtensor API).
"""

import collections
import contextlib
import os
import re
import warnings

import numpy as np

from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend import distribution_lib
from keras.src.backend.common import global_state

DEFAULT_BATCH_DIM_NAME = "batch"
GLOBAL_ATTRIBUTE_NAME = "distribution"


@keras_export("keras.distribution.list_devices")
def list_devices(device_type=None):
    """Return all the available devices based on the device type.

    Note: in a distributed setting, global devices are returned.

    Args:
        device_type: string, one of `"cpu"`, `"gpu"` or `"tpu"`.
            Defaults to `"gpu"` or `"tpu"` if available when
            `device_type` is not provided. Otherwise
            will return the `"cpu"` devices.

    Return:
        List of devices that are available for distribute computation.
    """
    return list_devices(device_type)


@keras_export("keras.distribution.initialize")
def initialize(job_addresses=None, num_processes=None, process_id=None):
    """Initialize the distribution system for multi-host/process setting.

    Calling `initialize` will prepare the backend for execution on multi-host
    GPU or TPUs. It should be called before any computations.

    Note that the parameters can also be injected via environment variables,
    which can be better controlled by the launch script at startup time.
    For certain backend that also rely on the environment variables to
    configure, Keras will properly forward them.

    Args:
        job_addresses: string. Comma separated IP addresses for all the jobs
            that will form the whole computation cluster. Note that for JAX
            backend, only the address for job 0 (coodinator) is needed. For
            certain runtime like cloud TPU, this value can be `None`, and the
            backend will figure it out with the TPU environment variables. You
            can also config this value via environment variable
            `KERAS_DISTRIBUTION_JOB_ADDRESSES`.
        num_processes: int. The number of worker/processes that will form the
            whole computation cluster. For certain runtime like cloud TPU, this
            value can be `None`, and the backend will figure it out with the TPU
            environment variables. You can also configure this value via
            environment variable `KERAS_DISTRIBUTION_NUM_PROCESSES`.
        process_id: int. The ID number of the current worker/process. The value
            should be ranged from `0` to `num_processes - 1`. `0` will indicate
            the current worker/process is the master/coordinate job. You can
            also configure this value via environment variable
            `KERAS_DISTRIBUTION_PROCESS_ID`.

        Example:
            Suppose there are two GPU processes, and process 0 is running at
            address `10.0.0.1:1234`, and process 1 is running at address
            `10.0.0.2:2345`. To configure such cluster, you can run

        On process 0:
        ```python
        keras.distribute.initialize(
            job_addresses="10.0.0.1:1234,10.0.0.2:2345",
            num_processes=2,
            process_id=0)
        ```

        On process 1:
        ```python
        keras.distribute.initialize(
            job_addresses="10.0.0.1:1234,10.0.0.2:2345",
            num_processes=2,
            process_id=1)
        ```

        or via the environment variables:
        On process 0:
        ```python
        os.environ[
            "KERAS_DISTRIBUTION_JOB_ADDRESSES"] = "10.0.0.1:1234,10.0.0.2:2345"
        os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"] = "2"
        os.environ["KERAS_DISTRIBUTION_PROCESS_ID"] = "0"
        keras.distribute.initialize()
        ```

        On process 1:
        ```python
        os.environ[
            "KERAS_DISTRIBUTION_JOB_ADDRESSES"] = "10.0.0.1:1234,10.0.0.2:2345"
        os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"] = "2"
        os.environ["KERAS_DISTRIBUTION_PROCESS_ID"] = "1"
        keras.distribute.initialize()
        ```

        Also note that for JAX backend, the `job_addresses` can be further
        reduced to just the master/coordinator address, which is
        `10.0.0.1:1234`.
    """
    if (
        job_addresses is None
        and "KERAS_DISTRIBUTION_JOB_ADDRESSES" in os.environ
    ):
        job_addresses = os.environ["KERAS_DISTRIBUTION_JOB_ADDRESSES"]
    if (
        num_processes is None
        and "KERAS_DISTRIBUTION_NUM_PROCESSES" in os.environ
    ):
        num_processes = int(os.environ["KERAS_DISTRIBUTION_NUM_PROCESSES"])
    if process_id is None and "KERAS_DISTRIBUTION_PROCESS_ID" in os.environ:
        process_id = int(os.environ["KERAS_DISTRIBUTION_PROCESS_ID"])
    initialize(job_addresses, num_processes, process_id)


@keras_export("keras.distribution.DeviceMesh")
class DeviceMesh:
    """A cluster of computation devices for distributed computation.

    This API is aligned with `jax.sharding.Mesh` and `tf.dtensor.Mesh`, which
    represents the computation devices in the global context.

    See more details in [jax.sharding.Mesh](
        https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.Mesh)
    and [tf.dtensor.Mesh](
        https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Mesh).

    Args:
        shape: tuple of list of integers. The shape of the overall
            `DeviceMesh`, e.g. `(8,)` for a data parallel only distribution,
            or `(4, 2)` for a model+data parallel distribution.
        axis_names: List of string. The logical name of the each axis for
            the `DeviceMesh`. The length of the `axis_names` should match to
            the rank of the `shape`. The `axis_names` will be used to
            match/create the `TensorLayout` when distribute the data and
            variables.
        devices: Optional list of devices. Defaults to all the available
            devices locally from `keras.distribution.list_devices()`.
    """

    def __init__(
        self,
        shape,
        axis_names,
        devices=None,
    ):
        if not shape or not axis_names:
            raise ValueError(
                "Shape and axis_names cannot be empty. Received: "
                f"shape={shape}, axis_names={axis_names}"
            )

        if len(shape) != len(axis_names):
            raise ValueError(
                "Shape and axis_names should have same size. "
                f"Received: shape={shape}, axis_names={axis_names}"
            )
        if devices is None:
            devices = list_devices()
        devices = np.array(devices)
        if np.prod(shape) != np.prod(devices.shape):
            raise ValueError(
                "Shape does not match the number of devices. "
                f"Received: shape={shape}; devices.shape="
                f"{devices.shape}"
            )

        self._shape = shape
        self._axis_names = axis_names
        self._devices = np.reshape(devices, shape)

    @property
    def shape(self):
        return self._shape

    @property
    def axis_names(self):
        return self._axis_names

    @property
    def devices(self):
        return self._devices

    @property
    def backend_mesh(self):
        if not hasattr(self, "_backend_mesh"):
            self._backend_mesh = _to_backend_mesh(self)
        return self._backend_mesh

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"shape={self.shape}, axis_names={self.axis_names}>"
        )

    def __str__(self):
        return self.__repr__()


@keras_export("keras.distribution.TensorLayout")
class TensorLayout:
    """A layout to apply to a tensor.

    This API is aligned with `jax.sharding.NamedSharding`
    and `tf.dtensor.Layout`.

    See more details in [jax.sharding.NamedSharding](
        https://jax.readthedocs.io/en/latest/jax.sharding.html#jax.sharding.NamedSharding)
    and [tf.dtensor.Layout](
        https://www.tensorflow.org/api_docs/python/tf/experimental/dtensor/Layout).

    Args:
        axes: tuple of strings that should map to the `axis_names` in
            a `DeviceMesh`. For any dimensions that doesn't need any sharding,
            A `None` can be used a placeholder.
        device_mesh: Optional `DeviceMesh` that will be used to create
            the layout. The actual mapping of tensor to physical device
            is not known until the mesh is specified.
    """

    def __init__(self, axes, device_mesh=None):
        self._axes = tuple(axes)
        self._device_mesh = device_mesh
        self._validate_axes()

    @property
    def axes(self):
        return self._axes

    @property
    def device_mesh(self):
        return self._device_mesh

    @device_mesh.setter
    def device_mesh(self, device_mesh):
        if self._device_mesh is not None:
            raise ValueError(
                "Cannot override device mesh value. Existing "
                f"value is {self._device_mesh}"
            )
        self._device_mesh = device_mesh
        self._validate_axes()

    @property
    def backend_layout(self):
        if not hasattr(self, "_backend_layout"):
            self._backend_layout = _to_backend_layout(self)
        return self._backend_layout

    def _validate_axes(self):
        if self._device_mesh:
            valid_axis_names = set(self._device_mesh.axis_names)
            axis_names = set(self._axes) - set([None])
            if axis_names - valid_axis_names:
                raise ValueError(
                    "Invalid axis names for Layout. Valid axis "
                    f"names: {valid_axis_names}, Got {axis_names}"
                )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} "
            f"axes={self.axes}, device_mesh={self.device_mesh}>"
        )

    def __str__(self):
        return self.__repr__()


class Distribution:
    """Base class for variable distribution strategies.

    A `Distribution` has following key functionalities:

    1. Distribute the model variables to a `DeviceMesh`.
    2. Distribute the input data to a `DeviceMesh`.
    3. Distribute an intermediate state tensor in the model.

    It can create a context scope so that the framework to properly detect the
    `Distribution` and distribute the variable/data accordingly.

    Args:
        device_mesh: A `DeviceMesh` instance.
        batch_dim_name: Optional string name for the batch dimension.
            Defaults to None.
        auto_shard_dataset: Automatically shard the dataset amongst
            processes in a multi-process setting. Set to `False` if the dataset
            is already sharded across hosts.  Defaults to `True`.
    """

    def __init__(
        self, device_mesh, batch_dim_name=None, auto_shard_dataset=True
    ):
        self._device_mesh = device_mesh
        self._batch_dim_name = batch_dim_name
        self._auto_shard_dataset = auto_shard_dataset

    def get_data_layout(self, data_shape):
        """Retrieve the `TensorLayout` for the input data.

        Args:
            data_shape: shape for the input data in list or tuple format.

        Returns:
            The `TensorLayout` for the data, which can be used by
            `backend.distribute_value()` to redistribute a input data.
        """
        raise NotImplementedError()

    def get_variable_layout(self, variable):
        """Retrieve the `TensorLayout` for the variable.

        Args:
            variable: A `Variable` instance.

        return:
            The `TensorLayout` for the variable, which can be used by
            `backend.distribute_value()` to redistribute a variable.
        """
        raise NotImplementedError()

    def get_tensor_layout(self, path):
        """Retrieve the `TensorLayout` for the intermediate tensor.

        Args:
            path: a string path for the corresponding tensor.

        return:
            The `TensorLayout` for the intermediate tensor, which can be used
            by `backend.relayout()` to reshard the tensor. Could also return
            None.
        """
        raise NotImplementedError()

    @contextlib.contextmanager
    def scope(self):
        """Context manager to make the `Distribution` current."""
        original_scope = distribution()
        set_distribution(self)
        try:
            yield
        finally:
            set_distribution(original_scope)

    @property
    def device_mesh(self):
        return self._device_mesh

    @property
    def batch_dim_name(self):
        return self._batch_dim_name

    @property
    def auto_shard_dataset(self):
        return self._auto_shard_dataset

    @auto_shard_dataset.setter
    def auto_shard_dataset(self, auto_shard_dataset):
        self._auto_shard_dataset = auto_shard_dataset

    def distribute_dataset(self, dataset):
        """Create a distributed dataset from the original global dataset.

        Args:
            dataset: the original global dataset instance.

        Returns:
            If `auto_shard_dataset` is `True`, returns a sharded dataset that
            only produces data for the current local worker/process.  Otherwise,
            returns the original dataset.

        Raises:
            ValueError: if auto-sharding is requested in a multi-process
            setting, but the dataset type is not supported.
        """
        raise NotImplementedError()

    def __repr__(self):
        return f"<{self.__class__.__name__} device_mesh={self.device_mesh}>"

    def __str__(self):
        return self.__repr__()

@keras_export("keras.distribution.AutoShardDistribution")
class AutoShardDistribution(Distribution):
    def __init__(
        self,
        device_mesh=None,
    ):
        if device_mesh is None:
            devices = np.array(list_devices())
            axis_names = [DEFAULT_BATCH_DIM_NAME] + [
                f"model_{i}" for i in range(devices.ndim - 1)
            ]
            device_mesh = DeviceMesh(
                shape=devices.shape,
                axis_names=axis_names,
                devices=devices,
            )
        super().__init__(device_mesh, device_mesh.axis_names[0])
        self._sharding_plan = None
        self._sharding_planner = None
        self._shard_applier = None
        self._num_process = num_processes()
        self._process_id = process_id()
        self._num_process = num_processes()
        self._process_id = process_id()

    def _get_backend_components(self):
        if self._sharding_planner and self._shard_applier:
            return
        self._sharding_planner = get_sharding_planner()
        self._shard_applier = get_shard_applier()

    def shard(self, model, *args, **kwargs):
        self._get_backend_components()
        graph = create_graph_from_model(model, *args, **kwargs)

        plan = self._sharding_planner.plan(graph, self.device_mesh)
        self._sharding_plan = plan
        self._shard_applier.apply(model, self._sharding_plan)

    def get_data_layout(self, data_shape):
        data_shard_spec = [None] * len(data_shape)
        data_shard_spec[0] = self.batch_dim_name
        return TensorLayout(data_shard_spec, self.device_mesh)

    def get_variable_layout(self, variable):
        if getattr(variable, "_layout", None) is not None:
            return variable._layout
        variable_shard_spec = [None] * len(variable.shape)
        return TensorLayout(variable_shard_spec, self.device_mesh)

    def get_tensor_layout(self, path):
        return None

    def distribute_dataset(self, dataset):
        from keras.src.utils.module_utils import tensorflow as tf

        if not tf.available or not isinstance(dataset, tf.data.Dataset):
            raise ValueError(
                "Only `tf.data.Dataset` is supported for auto-sharding, "
                f"got {type(dataset)}"
            )

        from tensorflow.python.data.experimental.ops import (
            distribute as tf_data_distribute,
        )

        batch_size = tf_data_distribute.compute_batch_size(dataset)
        if batch_size.numpy() < 0:
            raise ValueError(
                "The batch size of the input dataset is "
                "unknown. Please config the batch size for "
                "the input dataset, e.g via `dataset.batch(batch_size)`"
            )
        per_worker_batch_size = tf_data_distribute.batch_sizes_for_worker(
            global_batch_size=batch_size,
            num_workers=self._num_process,
            num_replicas_per_worker=1,  # We hard code this for now.
            worker_index=self._process_id,
        )
        distributed_dataset = dataset.rebatch(per_worker_batch_size)
        distributed_dataset = tf_data_distribute._AutoShardDataset(
            distributed_dataset,
            num_workers=self._num_process,
            index=self._process_id,
            num_replicas=self._num_process,
        )
        return distributed_dataset.prefetch(tf.data.AUTOTUNE)

@keras_export("keras.distribution.DataParallel")
class DataParallel(Distribution):
    """Distribution for data parallelism.

    You can choose to create this instance by either specifying
    the `device_mesh` or `devices` arguments (but not both).

    The `device_mesh` argument is expected to be a `DeviceMesh` instance,
    and is expected to be 1D only. In case that the mesh has multiple axes,
    then the first axis will be treated as the data parallel dimension
    (and a warning will be raised).

    When a list of `devices` are provided, they will be used to construct a
    1D mesh.

    When both `mesh` and `devices` are absent, then `list_devices()`
    will be used to detect any available devices and create a 1D mesh from
    them.

    Args:
        device_mesh: Optional `DeviceMesh` instance.
        devices: Optional list of devices.
        auto_shard_dataset: Automatically shard the dataset amongst
            processes in a multi-process setting. Set to `False` if the dataset
            is already sharded across hosts.  Defaults to `True`.
    """

    def __init__(self, device_mesh=None, devices=None, auto_shard_dataset=True):
        if device_mesh:
            self._initialize_with_device_mesh(device_mesh, auto_shard_dataset)
        elif devices:
            self._initialize_mesh_from_devices(devices, auto_shard_dataset)
        else:
            self._initialize_mesh_from_list_devices(auto_shard_dataset)

        # Those following attributes might get convert to public methods.
        self._num_process = num_processes()
        self._process_id = process_id()
        self._is_multi_process = self._num_process > 1

    def _initialize_with_device_mesh(self, device_mesh, auto_shard_dataset):
        if not isinstance(device_mesh, DeviceMesh):
            raise ValueError(
                "Expect `mesh` to be an instance of `DeviceMesh`. "
                f"Received: mesh={device_mesh} (of type {type(device_mesh)})"
            )
        super().__init__(
            device_mesh, device_mesh.axis_names[0], auto_shard_dataset
        )
        if self.device_mesh.devices.ndim != 1:
            warnings.warn(
                "Expect the input mesh to be 1D, but received "
                "mesh.devices.ndim=%d. "
                "The first axis will be used for data-parallel sharding.",
                device_mesh.devices.ndim,
            )

    def _initialize_mesh_from_devices(self, devices, auto_shard_dataset):
        devices = np.array(devices)
        device_mesh = DeviceMesh(
            shape=devices.shape,
            axis_names=[DEFAULT_BATCH_DIM_NAME],
            devices=devices,
        )
        super().__init__(
            device_mesh, DEFAULT_BATCH_DIM_NAME, auto_shard_dataset
        )

    def _initialize_mesh_from_list_devices(self, auto_shard_dataset):
        devices = np.array(list_devices())
        device_mesh = DeviceMesh(
            shape=devices.shape,
            axis_names=[DEFAULT_BATCH_DIM_NAME],
            devices=devices,
        )
        super().__init__(
            device_mesh, DEFAULT_BATCH_DIM_NAME, auto_shard_dataset
        )

    def get_data_layout(self, data_shape):
        data_shard_spec = [None] * len(data_shape)
        data_shard_spec[0] = self.batch_dim_name  # Shard on the first dim
        return TensorLayout(data_shard_spec, self.device_mesh)

    def get_variable_layout(self, variable):
        # First check if the variable already has a layout assigned.
        if getattr(variable, "_layout", None) is not None:
            return variable._layout
        # Otherwise, replicate variable.
        variable_shard_spec = [None] * len(variable.shape)
        return TensorLayout(variable_shard_spec, self.device_mesh)

    def get_tensor_layout(self, path):
        # For data parallel training, the intermediate state is not changed.
        return None

    def distribute_dataset(self, dataset):
        if not self._is_multi_process or not self.auto_shard_dataset:
            return dataset

        # Try to distribute a global tf.data.Dataset.
        from keras.src.utils.module_utils import tensorflow as tf

        if not tf.available or not isinstance(dataset, tf.data.Dataset):
            raise ValueError(
                "Only `tf.data.Dataset` is supported for auto-sharding, "
                f"got {type(dataset)}"
            )

        from tensorflow.python.data.experimental.ops import (
            distribute as tf_data_distribute,
        )

        batch_size = tf_data_distribute.compute_batch_size(dataset)
        if batch_size.numpy() < 0:
            raise ValueError(
                "The batch size of the input dataset is "
                "unknown. Please config the batch size for "
                "the input dataset, e.g via `dataset.batch(batch_size)`"
            )
        per_worker_batch_size = tf_data_distribute.batch_sizes_for_worker(
            global_batch_size=batch_size,
            num_workers=self._num_process,
            num_replicas_per_worker=1,  # We hard code this for now.
            worker_index=self._process_id,
        )
        distributed_dataset = dataset.rebatch(per_worker_batch_size)
        distributed_dataset = tf_data_distribute._AutoShardDataset(
            distributed_dataset,
            num_workers=self._num_process,
            index=self._process_id,
            num_replicas=self._num_process,
        )
        return distributed_dataset.prefetch(tf.data.AUTOTUNE)


@keras_export("keras.distribution.ModelParallel")
class ModelParallel(Distribution):
    """Distribution that shards model variables.

    Compare to `DataParallel` which replicates the variables across all devices,
    `ModelParallel` allows you to shard variables in addition to the input data.

    To construct a `ModelParallel` distribution, you need to provide a
    `DeviceMesh` and a `LayoutMap`.

    1. `DeviceMesh` contains physical device information. The axis names in
        the mesh will be used to map the variable and data layout.
    2. `LayoutMap` contains the mapping between variable paths to their
        corresponding `TensorLayout`.

    Example:

    ```python
    devices = list_devices()    # Assume there are 8 devices.

    # Create a mesh with 2 devices for data parallelism and 4 devices for
    # model parallelism.
    device_mesh = DeviceMesh(shape=(2, 4), axis_names=('batch', 'model'),
                             devices=devices)
    # Create a layout map that shard the `Dense` layer and `Conv2D`
    # layer variables on the last dimension.
    # Based on the `device_mesh`, this means the variables
    # will be split across 4 devices. Any other variable that doesn't
    # match any key in the layout map will be fully replicated.
    layout_map = LayoutMap(device_mesh)
    layout_map['dense.*kernel'] = (None, 'model')
    layout_map['dense.*bias'] = ('model',)
    layout_map['conv2d.*kernel'] = (None, None, None, 'model')
    layout_map['conv2d.*bias'] = ('model',)

    distribution = ModelParallel(
        layout_map=layout_map,
        batch_dim_name='batch',
    )

    # Set the global distribution, or via `with distribution.scope():`
    set_distribution(distribution)

    model = model_creation()
    model.compile()
    model.fit(data)
    ```

    You can quickly update the device mesh shape to change the sharding factor
    of the variables. E.g.

    ```python
    # With only the shape change for the device mesh, the variables will be
    # sharded across 8 devices instead of 4, which further reduces the memory
    # footprint of variables on each of the device.
    device_mesh = DeviceMesh(
        shape=(1, 8),
        axis_names=('batch', 'model'),
        devices=devices,
    )
    ```

    To figure out a proper layout mapping rule for all the model variables, you
    can first list out all the model variable paths, which will be used as the
    key to map the variables to `TensorLayout`.

    e.g.

    ```python
    model = create_model()
    for v in model.variables:
        print(v.path)
    ```

    Args:
        layout_map: `LayoutMap` instance which map the variable path to the
            corresponding tensor layout.
        batch_dim_name: Optional string, the axis name in the device mesh
            (of the `layout_map` object)
            that will be used to distribute data. If unspecified, the
            first axis from the device mesh will be used.
        auto_shard_dataset: Automatically shard the dataset amongst
            processes in a multi-process setting. Set to `False` if the dataset
            is already sharded across hosts.  Defaults to `True`.
    """

    def __init__(
        self,
        *,
        layout_map=None,
        batch_dim_name=None,
        auto_shard_dataset=True,
        **kwargs,
    ):
        kwargs.pop("device_mesh", None)
        if layout_map is None:
            raise ValueError("You must specify a layout_map argument.")
        if not isinstance(layout_map, LayoutMap):
            raise ValueError(
                "Argument `layout_map` must be a `LayoutMap` instance. "
                f"Received: layout_map={layout_map}"
            )
        device_mesh = layout_map.device_mesh
        batch_dim_name = batch_dim_name or device_mesh.axis_names[0]
        super().__init__(device_mesh, batch_dim_name, auto_shard_dataset)
        self._layout_map = layout_map

        # Those following attributes might get convert to public methods.
        self._num_process = num_processes()
        self._process_id = process_id()
        self._is_multi_process = self._num_process > 1

    def get_data_layout(self, data_shape):
        data_shard_spec = [None] * len(data_shape)
        data_shard_spec[0] = self.batch_dim_name  # Shard on the first dim
        return TensorLayout(data_shard_spec, self.device_mesh)

    def get_variable_layout(self, variable):
        # First check if the variable already has a layout assigned.
        if getattr(variable, "_layout", None) is not None:
            return variable._layout
        # Check the layout map.
        variable_layout = self._layout_map[variable.path]
        if variable_layout is not None:
            return variable_layout
        variable_shard_spec = [None] * len(variable.shape)
        return TensorLayout(variable_shard_spec, self.device_mesh)

    def get_tensor_layout(self, path):
        return self._layout_map[path]

    def distribute_dataset(self, dataset):
        if not self._is_multi_process or not self.auto_shard_dataset:
            return dataset

        # Try to distribute a global tf.data.Dataset.
        from keras.src.utils.module_utils import tensorflow as tf

        if not tf.available or not isinstance(dataset, tf.data.Dataset):
            raise ValueError(
                "Only `tf.data.Dataset` is supported for auto-sharding, "
                f"got {type(dataset)}"
            )

        from tensorflow.python.data.experimental.ops import (
            distribute as tf_data_distribute,
        )

        global_batch_size = tf_data_distribute.compute_batch_size(dataset)
        if global_batch_size.numpy() < 0:
            raise ValueError(
                "The batch size of the input dataset is "
                "unknown. Please config the batch size for "
                "the input dataset, e.g via `dataset.batch(batch_size)`"
            )

        # We need to compute the per-process/worker/host batch size.
        # This will depend on how many model replicas we have on each process.
        # Note that this might be smaller than one if model replicas are sharded
        # across multiple processes.
        mesh_batch_dim_index = self.device_mesh.axis_names.index(
            self.batch_dim_name
        )
        num_model_replicas = self.device_mesh.shape[mesh_batch_dim_index]
        if num_model_replicas == 1:
            # No sharding is needed in this case. Each process will have the
            # global batch size, and data from the iterator will need to be
            # replicated across all processes.
            return dataset.prefetch(tf.data.AUTOTUNE)
        num_model_replicas_per_process = num_model_replicas / self._num_process
        if num_model_replicas_per_process >= 1:
            # Each process will have one or more full model replicas. Data will
            # be sharded across all processes without replication.
            if global_batch_size % self._num_process != 0:
                raise ValueError(
                    "Global batch size must be divisible by the number of "
                    f"processes. `global_batch_size`={global_batch_size} and "
                    f"`num_process`={self._num_process}"
                )
            per_process_batch_size = global_batch_size // self._num_process
            distributed_dataset = dataset.rebatch(per_process_batch_size)
            distributed_dataset = distributed_dataset.shard(
                num_shards=self._num_process,
                index=self._process_id,
            )
            return distributed_dataset.prefetch(tf.data.AUTOTUNE)
        else:
            # Model replicas are sharded across multiple processes. Data will be
            # sharded across model replicas, and replicated across processes
            # within the same model replica.
            if global_batch_size % num_model_replicas != 0:
                raise ValueError(
                    "Global batch size must be divisible by the number of "
                    f"replicas. `global_batch_size`={global_batch_size} and "
                    f"`num_model_replicas`={num_model_replicas}"
                )
            per_process_batch_size = global_batch_size // num_model_replicas
            distributed_dataset = dataset.rebatch(per_process_batch_size)
            processes_per_replica = self._num_process // num_model_replicas
            # TODO: Figure out what the convention is for data sharding id.
            data_shard_id = self._process_id % processes_per_replica
            distributed_dataset = distributed_dataset.shard(
                num_shards=num_model_replicas,
                index=data_shard_id,
            )
            return distributed_dataset.prefetch(tf.data.AUTOTUNE)


@keras_export("keras.distribution.LayoutMap")
class LayoutMap(collections.abc.MutableMapping):
    """A dict-like object that maps string to `TensorLayout` instances.

    `LayoutMap` uses a string as key and a `TensorLayout` as value. There is a
    behavior difference between a normal Python dict and this class. The string
    key will be treated as a regex when retrieving the value. See the docstring
    of `get` for more details.

    See below for a usage example. You can define the naming schema
    of the `TensorLayout`, and then retrieve the corresponding
    `TensorLayout` instance.

    In the normal case, the key to query is usually the `variable.path`, which
    is the identifier of the variable.

    As shortcut, tuple or list of axis names are also allowed when inserting
    as value, and will be converted to `TensorLayout`.

    ```python
    layout_map = LayoutMap(device_mesh)
    layout_map['dense.*kernel'] = (None, 'model')
    layout_map['dense.*bias'] = ('model',)
    layout_map['conv2d.*kernel'] = (None, None, None, 'model')
    layout_map['conv2d.*bias'] = ('model',)

    layout_1 = layout_map['dense_1.kernel']             # layout_1 == layout_2d
    layout_2 = layout_map['dense_1.bias']               # layout_2 == layout_1d
    layout_3 = layout_map['dense_2.kernel']             # layout_3 == layout_2d
    layout_4 = layout_map['dense_2.bias']               # layout_4 == layout_1d
    layout_5 = layout_map['my_model/conv2d_123/kernel'] # layout_5 == layout_4d
    layout_6 = layout_map['my_model/conv2d_123/bias']   # layout_6 == layout_1d
    layout_7 = layout_map['my_model/conv3d_1/kernel']   # layout_7 == None
    layout_8 = layout_map['my_model/conv3d_1/bias']     # layout_8 == None
    ```

    Args:
        device_mesh: `keras.distribution.DeviceMesh` instance.
    """

    def __init__(self, device_mesh):
        self._layout_map = collections.OrderedDict()
        self._device_mesh = device_mesh

    def __getitem__(self, key):
        """Retrieves the corresponding layout by the string key.

        When there isn't an exact match, all the existing keys in the layout map
        will be treated as a regex and map against the input key again. When
        there are multiple matches for the regex, an `ValueError` will be
        raised. Returns `None` if there isn't any match found.

        Args:
            key: String key to query a layout.

        Returns:
            Corresponding layout based on the query.
        """
        if key in self._layout_map:
            return self._layout_map[key]

        matching_keys = []
        for k in self._layout_map:
            if re.search(k, key):
                matching_keys.append(k)
        if len(matching_keys) > 1:
            raise ValueError(
                f"Path '{key}' matches multiple layout "
                f"specification keys: {matching_keys}. Please make "
                "sure each tensor/variable path only matches at most "
                "one layout specification key in the LayoutMap."
            )
        elif len(matching_keys) == 1:
            return self._layout_map[matching_keys[0]]
        return None

    def __setitem__(self, key, layout):
        """Insert TensorLayout to the LayoutMap.

        Args:
            key: String key for the `TensorLayout`.
            layout: The `TensorLayout`. As a shortcut, tuple of string and None
                are also acceptable, and will be converted to `TensorLayout`.
        """
        if key in self._layout_map:
            raise ValueError(
                f"{key} already exist in the LayoutMap with "
                f"value {self._layout_map[key]}. Please make sure to "
                "not use duplicated keys."
            )
        if isinstance(layout, tuple):
            layout = TensorLayout(axes=layout, device_mesh=None)

        if not isinstance(layout, TensorLayout):
            raise ValueError(
                f"{layout} should be a TensorLayout type, got {type(layout)}"
            )
        self._maybe_populate_device_mesh(layout)
        self._layout_map[key] = layout

    def __delitem__(self, key):
        # let the dict to handle the key missing error
        return self._layout_map.pop(key)

    def __len__(self):
        return len(self._layout_map)

    def __iter__(self):
        return iter(self._layout_map)

    @property
    def device_mesh(self):
        return self._device_mesh

    def _maybe_populate_device_mesh(self, layout):
        if layout.device_mesh is None and self.device_mesh is not None:
            layout.device_mesh = self.device_mesh


LayoutMap.get.__doc__ = LayoutMap.__getitem__.__doc__


@keras_export("keras.distribution.distribute_tensor")
def distribute_tensor(tensor, layout):
    """Change the layout of a Tensor value in the jit function execution.

    Args:
        tensor: a Tensor to change the layout.
        layout: `TensorLayout` to be applied on the value.

    Returns:
        a new value with the specified tensor layout.
    """
    if isinstance(tensor, KerasTensor):
        return tensor
    return distribute_tensor(tensor, layout)


@keras_export("keras.distribution.distribution")
def distribution():
    """Retrieve the current distribution from global context."""
    return global_state.get_global_attribute(GLOBAL_ATTRIBUTE_NAME)


@keras_export("keras.distribution.set_distribution")
def set_distribution(value):
    """Set the distribution as the global distribution setting.

    Args:
        value: a `Distribution` instance.
    """
    global_state.set_global_attribute(GLOBAL_ATTRIBUTE_NAME, value)


"""
Test suite for model verification with Parallax (Keras AutoShardDistribution),
using KerasNLP presets and the Tiny Shakespeare dataset.

This script compares the training performance (loss and perplexity) of a
baseline model against its Parallax-sharded equivalent. The goal is to verify
the correctness and performance of Keras's automatic sharding capabilities.

It is generalized to run tests for multiple model architectures, including:
- Gemma
- GPT-2
- Bloom
- OPT
"""
import os
import time
import logging
import json

import numpy as np
import keras
import keras_nlp
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
import jax

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


BATCH_SIZE = 8
SEQUENCE_LENGTH = 128
LEARNING_RATE = 3e-5
EPOCHS = 10
STEPS_PER_EPOCH = 50
VALIDATION_STEPS = 10

MODEL_MAPPING = {
    "gpt2_base_en": keras_nlp.models.GPT2CausalLM,
    "bloom_560m_multi": keras_nlp.models.BloomCausalLM,
    "opt_125m_en": keras_nlp.models.OPTCausalLM,
}

def load_shakespeare_dataset(model_preset, model_class):
    """Loads and preprocesses the Tiny Shakespeare dataset for a given model."""
    print(f"   Loading and preprocessing Tiny Shakespeare dataset for {model_preset}...")
    ds = tfds.load("tiny_shakespeare", split="train")
    text = "".join(example["text"].decode("utf-8") for example in ds.as_numpy_iterator())

    tokenizer = model_class.from_preset(model_preset).preprocessor.tokenizer
    token_ids = tokenizer.tokenize(text)

    num_tokens = (len(token_ids) // (SEQUENCE_LENGTH + 1)) * (SEQUENCE_LENGTH + 1)
    sequences = np.array(token_ids[:num_tokens]).reshape(-1, SEQUENCE_LENGTH + 1)

    all_data = tf.data.Dataset.from_tensor_slices(sequences)

    num_sequences = sequences.shape[0]
    num_train_samples = int(0.9 * num_sequences)

    train_ds = all_data.take(num_train_samples)
    val_ds = all_data.skip(num_train_samples)

    print(f"      ✅ Dataset ready with {num_train_samples} training and {num_sequences - num_train_samples} validation sequences.")
    return train_ds, val_ds

def format_for_causal_lm(data):
    """Formats data for KerasNLP's CausalLM, creating features and labels."""
    features = {
        "token_ids": data[:, :-1],
        "padding_mask": tf.ones_like(data[:, :-1], dtype=tf.bool),
    }
    labels = data[:, 1:]
    return features, labels

def get_model_from_preset(preset_name, model_class):
    """Creates a CausalLM model from a KerasNLP preset."""
    print(f"   Creating {preset_name} model from Keras preset...")
    model = model_class.from_preset(preset_name, preprocessor=None)
    print(f"      ✅ Model created with {model.count_params():,} parameters.")
    return model

def plot_training_graphs(history_dict, preset_name):
    """Plots and saves the loss and perplexity graphs for the Parallax model."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f"{preset_name} - Parallax (AutoShard) Training Performance", fontsize=16)

    ax1.plot(history_dict["loss"], label="Parallax - Training Loss", color="green", linestyle="-")
    ax1.plot(history_dict["val_loss"], label="Parallax - Validation Loss", color="green", linestyle="--")
    ax1.set_title("Training and Validation Loss")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history_dict["perplexity"], label="Parallax - Training Perplexity", color="purple", linestyle="-")
    ax2.plot(history_dict["val_perplexity"], label="Parallax - Validation Perplexity", color="purple", linestyle="--")
    ax2.set_title("Training and Validation Perplexity")
    ax2.set_ylabel("Perplexity")
    ax2.set_xlabel("Epoch")
    ax2.legend()
    ax2.grid(True)

    output_filename = f"{preset_name}_parallax_training_performance.png"
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    print(f"\n   ✅ Performance graph saved to {output_filename}")
    plt.close()

def run_model_verification(preset_name, model_class):
    """
    Runs the full training verification test for a given model preset.
    Returns a list of per-epoch history dictionaries on success, or a status string on skip.
    """
    print(f"🔧 TRAINING FOR: {preset_name.upper()}")
    print("=" * 50)
    start_time = time.time()

    train_ds_raw, val_ds_raw = load_shakespeare_dataset(preset_name, model_class)

    train_ds = (
        train_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )
    val_ds = (
        val_ds_raw.batch(BATCH_SIZE, drop_remainder=True)
        .map(format_for_causal_lm, num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
        .repeat()
    )

    print("\n   --- Training Parallax (AutoShardDistribution) Model ---")

    num_devices = jax.device_count()
    if num_devices < 2:
        print(f"      ⚠️ Skipping Parallax test for {preset_name}: requires at least 2 JAX devices, but found {num_devices}.")
        return "SKIPPED"

    mesh = keras.distribution.DeviceMesh(
        shape=(num_devices,),
        axis_names=("model",),
        devices=jax.devices()
    )
    distribution = AutoShardDistribution(mesh)
    print(f"      Initialized Parallax with a '{mesh.axis_names}' mesh across {num_devices} devices.")

    parallax_model = get_model_from_preset(preset_name, model_class)

    print("      Running Parallax auto-sharding analysis...")
    sample_batch_features, _ = next(iter(train_ds))
    sample_batch_np = {
        key: value.numpy() for key, value in sample_batch_features.items()
    }

    with distribution.scope():
        distribution.shard(parallax_model, sample_batch_np)
    print("      ✅ Auto-sharding plan applied.")

    parallax_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=LEARNING_RATE),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras_nlp.metrics.Perplexity(from_logits=True, name="perplexity")],
    )

    parallax_history = parallax_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,
        verbose=1
    )
    print("      ✅ Parallax model training completed.")
    
    history_dict = parallax_history.history

    print("\n   --- Final Validation Metrics ---")
    parallax_final_val_loss = history_dict['val_loss'][-1]
    parallax_final_perplexity = history_dict['val_perplexity'][-1]
    print(f"      Parallax Final Validation Loss: {parallax_final_val_loss:.4f}")
    print(f"      Parallax Final Validation Perplexity: {parallax_final_perplexity:.4f}")

    plot_training_graphs(history_dict, preset_name)

    print(f"✅ Test for {preset_name} completed in {time.time() - start_time:.2f}s")
    
    num_epochs = len(history_dict.get("loss", []))
    epoch_by_epoch_history = []
    metric_keys = list(history_dict.keys())

    for i in range(num_epochs):
        epoch_record = {"epoch": i + 1}
        for key in metric_keys:
            epoch_record[key] = history_dict[key][i]
        epoch_by_epoch_history.append(epoch_record)
        
    return epoch_by_epoch_history


if __name__ == "__main__":
    print("\n🎯 PARALLAX (AUTOSHARD) TRAINING SUITE")
    print("=" * 70)

    results = {}
    total_start_time = time.time()

    for preset, model_class in MODEL_MAPPING.items():
        try:
            history_data = run_model_verification(preset, model_class)
            
            if isinstance(history_data, list):
                results[preset] = {
                    "status": "✅ COMPLETED",
                    "history": history_data
                }
            elif history_data == "SKIPPED":
                results[preset] = {
                    "status": "⚠️ SKIPPED",
                    "history": None
                }
        except Exception as e:
            logger.error(f"Test for {preset} failed with an exception: {e}", exc_info=True)
            results[preset] = {
                "status": "💥 ERROR",
                "history": None
            }
        print("-" * 70)

    print("\n" + "=" * 70)
    print("🎉 TRAINING SUITE COMPLETED!")
    print(f"   Total execution time: {time.time() - total_start_time:.2f}s")
    print("\n   --- SUMMARY ---")
    for preset, data in results.items():
        print(f"   - {preset:<25}: {data['status']}")
    print("=" * 70)

    results_filename = "results.json"
    with open(results_filename, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\n✅ Training summary and detailed epoch metrics saved to {results_filename}")