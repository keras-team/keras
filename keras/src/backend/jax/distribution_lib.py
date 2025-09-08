"""Utilities for distribution strategy with JAX backend."""

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