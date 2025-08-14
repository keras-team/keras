"""Utilities for distribution strategy with JAX backend."""

import collections
import itertools

import jax
import numpy as np

from keras.src.backend.common import global_state
from keras.src.random import seed_generator
from keras.src.utils import jax_utils
from keras.src.utils import rng_utils


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

    from keras.src.distribution.autoshard_utils import MergeableGraph

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
            in_idx, out_idx = 0, 0
            in_prod, out_prod = 1, 1
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
        fn,
        params,
        *inputs,
        min_shard_size=1,
        data_axis_name="data",
        model_axis_name="model",
    ):
        """Analyzes a function via jaxpr and returns sharding assignments."""
        jaxpr, abs_ret = jax.make_jaxpr(fn, return_shape=True)(params, *inputs)
        graph = parse_jaxpr(jaxpr)

        params_flat, params_treedef = jax.tree.flatten(params)
        _, inputs_treedef = jax.tree.flatten(inputs)
        _, outputs_treedef = jax.tree.flatten(abs_ret)

        seen = collections.Counter()
        for var in jaxpr.jaxpr.invars[: len(params_flat)]:
            for i in range(var.aval.ndim):
                if var.aval.shape[i] >= min_shard_size:
                    seen.update([graph.get_root((var, i))])

        model_axis_root = max(seen, key=seen.get) if seen else None

        data_axes_roots = []
        for var in jaxpr.jaxpr.invars[len(params_flat) :]:
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

        params_assignments = params_treedef.unflatten(
            assign_layouts(
                jaxpr.jaxpr.invars[: len(params_flat)], is_params=True
            )
        )
        inputs_assignments = inputs_treedef.unflatten(
            assign_layouts(jaxpr.jaxpr.invars[len(params_flat) :])
        )
        output_assignments = outputs_treedef.unflatten(
            assign_layouts(jaxpr.jaxpr.outvars)
        )

        return (params_assignments, *inputs_assignments), output_assignments

    class _JaxGraph:
        """A wrapper for a JAX computation graph (jaxpr) of a Keras model."""

        def __init__(
            self,
            jaxpr,
            trainable_variables,
            non_trainable_variables,
            in_treedefs,
        ):
            self.jaxpr = jaxpr
            self.trainable_variables = trainable_variables
            self.non_trainable_variables = non_trainable_variables
            self.in_treedefs = in_treedefs

        @classmethod
        def from_model(cls, model, *args, **kwargs):
            """Creates a _JaxGraph instance by tracing the model."""

            def stateless_fn(
                trainable_vars, non_trainable_vars, f_args, f_kwargs
            ):
                return model.stateless_call(
                    trainable_vars, non_trainable_vars, *f_args, **f_kwargs
                )

            trainable_vars = model.trainable_variables
            non_trainable_vars = model.non_trainable_variables

            _, t_vars_treedef = jax.tree.flatten(trainable_vars)
            _, nt_vars_treedef = jax.tree.flatten(non_trainable_vars)
            _, args_treedef = jax.tree.flatten(args)
            _, kwargs_treedef = jax.tree.flatten(kwargs)
            in_treedefs = (
                t_vars_treedef,
                nt_vars_treedef,
                args_treedef,
                kwargs_treedef,
            )

            closed_jaxpr, _ = jax.make_jaxpr(stateless_fn, return_shape=True)(
                trainable_vars, non_trainable_vars, args, kwargs
            )
            return cls(
                closed_jaxpr, trainable_vars, non_trainable_vars, in_treedefs
            )

    class _JaxShardingPlanner:
        """
        Determines the optimal sharding layout for model variables using
        the embedded graph-parsing engine.
        """

        def plan(self, graph, device_mesh):
            t_vars = graph.trainable_variables
            nt_vars = graph.non_trainable_variables

            all_in_avals = [var.aval for var in graph.jaxpr.jaxpr.invars]
            t_vars_leaves, _ = jax.tree.flatten(t_vars)
            nt_vars_leaves, _ = jax.tree.flatten(nt_vars)

            pos = len(t_vars_leaves) + len(nt_vars_leaves)

            args_treedef = graph.in_treedefs[2]
            kwargs_treedef = graph.in_treedefs[3]

            num_args_leaves = args_treedef.num_leaves
            args_avals = all_in_avals[pos : pos + num_args_leaves]
            kwargs_avals = all_in_avals[pos + num_args_leaves :]

            args_aval_tree = jax.tree.unflatten(args_treedef, args_avals)
            kwargs_aval_tree = jax.tree.unflatten(kwargs_treedef, kwargs_avals)

            dummy_args = jax.tree.map(
                lambda x: np.zeros(x.shape, x.dtype), args_aval_tree
            )
            dummy_kwargs = jax.tree.map(
                lambda x: np.zeros(x.shape, x.dtype), kwargs_aval_tree
            )

            def fn_to_trace(trainable_params, *fn_args, **fn_kwargs):
                """A function that executes the original jaxpr computation."""
                all_leaves = (
                    jax.tree.leaves(trainable_params)
                    + jax.tree.leaves(nt_vars)
                    + jax.tree.leaves(fn_args)
                    + jax.tree.leaves(fn_kwargs)
                )
                return jax.core.eval_jaxpr(
                    graph.jaxpr.jaxpr, graph.jaxpr.consts, *all_leaves
                )

            (param_assignments, *_) = shard_model(
                fn_to_trace, t_vars, dummy_args, **dummy_kwargs
            )

            param_vars_flat, _ = jax.tree.flatten(t_vars)
            param_layouts_flat, _ = jax.tree.flatten(param_assignments)

            parameter_layout_dict = {
                var.path: tuple(layout) if layout else None
                for var, layout in zip(param_vars_flat, param_layouts_flat)
            }
            return parameter_layout_dict

    class _JaxShardApplier:
        """Applies a sharding plan to a model's variables."""

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
