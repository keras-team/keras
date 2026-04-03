import collections

from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.config import backend
from keras.src.backend.config import is_nnx_enabled
from keras.src.ops.operation import Operation


@keras_export("keras.Function")
class Function(Operation):
    """Class that encapsulates a computation graph of Keras operations.

    You can use a `Function` to capture the computation graph linking
    some input tensors to some output tensors, and reapply the same
    computation on new inputs.

    A `Function` is similar to a Functional Model, with the difference
    that it is stateless (it does not track state variables)
    and does not implement the `Layer` API.

    Example:

    ```python
    input_1 = keras.KerasTensor(shape=(None, 2, 3))
    input_2 = keras.KerasTensor(shape=(None, 2, 3))
    x = input_1 + input_2
    output = keras.ops.sigmoid(x)
    fn = keras.Function(inputs=[input_1, input_2], outputs=output)

    input_1_val = np.random.random((4, 2, 3))
    input_2_val = np.random.random((4, 2, 3))
    output_val = fn([input_1_val, input_2_val])
    ```

    Args:
        inputs: `KerasTensor` instance or nested structured of
            `KerasTensor` instances.
        outputs: `KerasTensor` instance or nested structured of
            `KerasTensor` instances. They should be computable
            given only the values of `inputs`.
        name: String. The name of the function.
    """

    def __init__(self, inputs, outputs, name=None):
        super().__init__(name=name)

        if backend() == "tensorflow":
            # Temporary work around for
            # https://github.com/keras-team/keras/issues/931
            # This stop tensorflow from wrapping tf.function output in a
            # _DictWrapper object.
            _self_setattr_tracking = getattr(
                self, "_self_setattr_tracking", True
            )
            self._self_setattr_tracking = False
        self._inputs_struct = tree.map_structure(lambda x: x, inputs)
        self._outputs_struct = tree.map_structure(lambda x: x, outputs)
        self._inputs = tree.flatten(inputs)
        self._outputs = tree.flatten(outputs)
        if not self._inputs:
            raise ValueError(
                "`inputs` argument cannot be empty. Received:\n"
                f"inputs={inputs}\n"
                f"outputs={outputs}"
            )
        if not self._outputs:
            raise ValueError(
                "`outputs` argument cannot be empty. Received:\n"
                f"inputs={inputs}\n"
                f"outputs={outputs}"
            )

        if backend() == "tensorflow":
            self._self_setattr_tracking = _self_setattr_tracking

        (nodes, nodes_by_depth, operations, operations_by_depth) = map_graph(
            self._inputs, self._outputs
        )
        self._nodes = nodes
        self._nodes_by_depth = nodes_by_depth
        self._operations = operations
        self._operations_by_depth = operations_by_depth

        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)
        compiled_nodes = []
        for depth in depth_keys:
            for node in nodes_by_depth[depth]:
                if not node.operation or node.is_input:
                    continue
                compiled_nodes.append(node)
        self._compiled_nodes = compiled_nodes

        # Pre-compute per-node metadata for the hot loop in
        # _run_through_graph.  Each tuple contains:
        #   (node, input_ids, context_args, single_output_id, op)
        # - input_ids: tuple of id(x) for input tensors (avoid id()
        #   calls in hot loop)
        # - context_args: frozenset of accepted context arg names, or
        #   None if empty
        # - single_output_id: id(outputs[0]) when len(outputs)==1,
        #   else None
        self._compiled_node_data = []
        for node in compiled_nodes:
            input_ids = tuple(id(x) for x in node.input_tensors)
            ctx_args = getattr(node.operation, "_call_context_args", None)
            ctx_args = frozenset(ctx_args) if ctx_args else None
            single_out_id = (
                id(node.outputs[0]) if len(node.outputs) == 1 else None
            )
            self._compiled_node_data.append(
                (node, input_ids, ctx_args, single_out_id, node.operation)
            )

        # Run through graph to check all outputs are connected to the inputs.
        def empty_op_outputs(op, *args, **kwargs):
            return [None] * len(tree.flatten(op.output))

        self._run_through_graph(
            [None] * len(self._inputs), call_fn=empty_op_outputs
        )

        # Special handling for NNX to ensure consistent operation instance usage
        if is_nnx_enabled():
            self._setup_nnx_op_mapping()

        # Pre-compile an index-based execution plan for fast inference.
        # Uses integer slot indices instead of id()-based dict lookups.
        self._compile_forward_plan()

    @property
    def operations(self):
        return self._operations[:]

    @property
    def inputs(self):
        """Flat list of the symbolic inputs of the Function."""
        return self._inputs

    @property
    def outputs(self):
        """Flat list of the symbolic outputs of the Function."""
        return self._outputs

    def _setup_nnx_op_mapping(self):
        """Setup operation mapping for NNX"""
        # Create a mapping from operation id to operation instance
        self._nnx_op_mapping = {}

        # Assign the list of operations to a single attribute for NNX traversal
        self.nnx_operations = self._operations[:]
        for operation in self._operations:
            # Map the operation id to this operation instance
            self._nnx_op_mapping[id(operation)] = operation

    def _compile_forward_plan(self):
        """Pre-compile an index-based execution plan for fast inference.

        Instead of using id()-based dictionary lookups and fill_in() per
        node, this builds a flat list of (callable, input_slot_indices,
        output_slot_index, static_kwargs) tuples.  The execution engine
        uses a simple Python list as the slot array, with integer indexing
        replacing all dict and id() overhead.

        For layers with ``_fast_call=True``, the plan stores
        ``layer.call`` directly, bypassing ``Layer.__call__`` overhead.
        """
        idx_map = {}
        for i, inp in enumerate(self._inputs):
            idx_map[id(inp)] = i

        next_idx = len(self._inputs)
        plan = []

        for node in self._compiled_nodes:
            input_ids = [id(x) for x in node.input_tensors]
            if any(xid not in idx_map for xid in input_ids):
                continue

            op = node.operation
            sa = node.arguments

            # Use layer.call directly when safe (bypasses __call__).
            call_fn = op
            if (
                hasattr(op, "_fast_call")
                and op._fast_call
                and op.built
                and hasattr(op, "call")
            ):
                call_fn = op.call

            if sa._single_positional_tensor is not None:
                in_slots = (idx_map[id(sa._single_positional_tensor)],)
                pattern = 1
                static_kw = None
            elif sa._dual_positional_tensors is not None:
                t0, t1 = sa._dual_positional_tensors
                in_slots = (idx_map[id(t0)], idx_map[id(t1)])
                pattern = 2
                static_kw = None
            elif sa._dual_tensors_static_kwargs is not None:
                t0, t1, kw = sa._dual_tensors_static_kwargs
                in_slots = (idx_map[id(t0)], idx_map[id(t1)])
                pattern = 3
                static_kw = kw
            else:
                in_slots = tuple(idx_map[xid] for xid in input_ids)
                pattern = 0
                static_kw = node
                call_fn = op  # General case: go through __call__

            if len(node.outputs) == 1:
                out_slot = next_idx
                idx_map[id(node.outputs[0])] = next_idx
                next_idx += 1
                multi_out = None
            else:
                out_slot = next_idx
                multi_out = []
                for out_t in node.outputs:
                    idx_map[id(out_t)] = next_idx
                    multi_out.append(next_idx)
                    next_idx += 1
                multi_out = tuple(multi_out)

            plan.append(
                (call_fn, in_slots, out_slot, pattern, static_kw, multi_out)
            )

        output_slots = tuple(idx_map.get(id(x)) for x in self._outputs)
        self._forward_plan = tuple(plan)
        self._forward_plan_slots = next_idx
        self._forward_output_slots = output_slots
        self._forward_plan_single_output = len(output_slots) == 1
        # Store idx_map for the general fallback path.
        self._forward_idx_map = idx_map

    def _execute_forward_plan(self, inputs):
        """Execute the pre-compiled forward plan.

        Uses a flat list with integer indexing for intermediate results,
        eliminating dict lookups, id() calls, and fill_in() overhead.
        """
        plan = self._forward_plan
        slots = [None] * self._forward_plan_slots

        for i, inp in enumerate(inputs):
            slots[i] = inp

        for fn, in_slots, out_slot, pattern, static_kw, multi_out in plan:
            if pattern == 1:
                result = fn(slots[in_slots[0]])
            elif pattern == 2:
                result = fn(slots[in_slots[0]], slots[in_slots[1]])
            elif pattern == 3:
                result = fn(
                    slots[in_slots[0]], slots[in_slots[1]], **static_kw
                )
            else:
                # General: use fill_in with id-based tensor_dict.
                node = static_kw
                tensor_dict = {}
                for j, ref_inp in enumerate(self._inputs):
                    tensor_dict[id(ref_inp)] = slots[j]
                # Reconstruct tensor_dict for all computed outputs.
                for prev_node in self._compiled_nodes:
                    for out_t in prev_node.outputs:
                        oid = id(out_t)
                        if oid in self._forward_idx_map:
                            si = self._forward_idx_map[oid]
                            if slots[si] is not None:
                                tensor_dict[oid] = slots[si]
                args, kwargs = node.arguments.fill_in(tensor_dict)
                result = fn(*args, **kwargs)

            if multi_out is None:
                slots[out_slot] = result
            else:
                flat = tree.flatten(result)
                for slot_idx, val in zip(multi_out, flat):
                    slots[slot_idx] = val

        if self._forward_plan_single_output:
            return slots[self._forward_output_slots[0]]
        return [slots[i] for i in self._forward_output_slots]

    def _get_operation_for_node(self, node):
        """Get the operation for a node, using NNX mapping if enabled."""
        operation = node.operation
        if hasattr(self, "_nnx_op_mapping") and id(operation) in getattr(
            self, "_nnx_op_mapping", {}
        ):
            return self._nnx_op_mapping[id(operation)]
        return operation

    def compute_output_spec(self, inputs):
        self._assert_input_compatibility(inputs)
        # Check if input shapes are identical to ref input shapes,
        # if so take a shortcut.
        shortcut = True
        for x, x_ref in zip(tree.flatten(inputs), self._inputs):
            if x.shape != x_ref.shape:
                shortcut = False
                break
        if shortcut:
            return tree.map_structure(
                lambda x: KerasTensor(shape=x.shape, dtype=x.dtype),
                self._outputs_struct,
            )
        # No luck; take the long road through the graph.
        # Original Keras used a cache to avoid recomputing all this
        # when known input shapes where seen again. Perhaps a good
        # idea to bring that back.
        return self._run_through_graph(
            inputs, operation_fn=lambda op: op.compute_output_spec
        )

    def compute_output_shape(self, input_shape):
        # Wrap `input_shape` into the structure of KerasTensor to utilize
        # `compute_output_spec`.
        input_shape_struct = tree.map_shape_structure(
            lambda x: KerasTensor(shape=x), input_shape
        )
        # Ensure that dtype and sparse settings are the same as self._inputs,
        # because we only care about the shape in this function.
        for x, x_ref in zip(tree.flatten(input_shape_struct), self._inputs):
            x._dtype = x_ref.dtype
            x._sparse = x_ref.sparse
        output_spec = self.compute_output_spec(input_shape_struct)
        return tree.map_structure(lambda x: x.shape, output_spec)

    def call(self, inputs):
        """Computes output tensors for new inputs."""
        self._assert_input_compatibility(inputs)
        return self._run_through_graph(inputs)

    def _run_through_graph(
        self, inputs, operation_fn=None, call_fn=None, call_context_dict=None
    ):
        """Execute the graph.

        At each node we compute outputs via
        `operation_fn(node.operation)(*args, **kwargs)`.
        """
        inputs = tree.flatten(inputs)

        # Dictionary mapping reference tensors to computed tensors.
        tensor_dict = {}
        for x, y in zip(self.inputs, inputs):
            tensor_dict[id(x)] = y

        # Cache NNX mapping lookup outside the loop.
        nnx_map = getattr(self, "_nnx_op_mapping", None)
        td_contains = tensor_dict.__contains__
        td_set = tensor_dict.__setitem__

        # Hot loop over compiled nodes.
        # The common inference path has operation_fn=None, call_fn=None,
        # so we hoist those checks out of the loop.
        if call_fn is None and operation_fn is None:
            # Use pre-computed node data to avoid per-iteration getattr,
            # id() calls, and len() checks.
            node_data = getattr(self, "_compiled_node_data", None)
            if node_data is not None and nnx_map is None:
                # Fastest path: pre-computed data, no NNX mapping.
                if call_context_dict:
                    for (
                        node,
                        input_ids,
                        ctx_args,
                        single_out_id,
                        op,
                    ) in node_data:
                        if any(not td_contains(t) for t in input_ids):
                            continue
                        args, kwargs = node.arguments.fill_in(tensor_dict)
                        if ctx_args:
                            kwargs = dict(kwargs)
                            for name, value in call_context_dict.items():
                                if name in ctx_args and value is not None:
                                    kwargs[name] = value
                        outputs = op(*args, **kwargs)
                        if single_out_id is not None:
                            td_set(single_out_id, outputs)
                        else:
                            for x, y in zip(
                                node.outputs, tree.flatten(outputs)
                            ):
                                td_set(id(x), y)
                else:
                    for (
                        node,
                        input_ids,
                        ctx_args,
                        single_out_id,
                        op,
                    ) in node_data:
                        if any(not td_contains(t) for t in input_ids):
                            continue
                        args, kwargs = node.arguments.fill_in(tensor_dict)
                        outputs = op(*args, **kwargs)
                        if single_out_id is not None:
                            td_set(single_out_id, outputs)
                        else:
                            for x, y in zip(
                                node.outputs, tree.flatten(outputs)
                            ):
                                td_set(id(x), y)
            else:
                # Fallback: original loop (NNX mapping or no pre-computed data)
                for node in self._compiled_nodes:
                    if any(not td_contains(id(x)) for x in node.input_tensors):
                        continue

                    args, kwargs = node.arguments.fill_in(tensor_dict)

                    if call_context_dict:
                        valid_context_args = getattr(
                            node.operation, "_call_context_args", {}
                        )
                        if valid_context_args:
                            kwargs = dict(kwargs)
                            for name, value in call_context_dict.items():
                                if (
                                    name in valid_context_args
                                    and value is not None
                                ):
                                    kwargs[name] = value

                    # Resolve operation: prefer NNX mapping if present.
                    op = node.operation
                    if nnx_map is not None:
                        op = nnx_map.get(id(op), op)

                    outputs = op(*args, **kwargs)

                    # Fast path: single-output nodes (most common) skip
                    # tree.flatten overhead.
                    if len(node.outputs) == 1:
                        td_set(id(node.outputs[0]), outputs)
                    else:
                        for x, y in zip(node.outputs, tree.flatten(outputs)):
                            td_set(id(x), y)
        else:
            for node in self._compiled_nodes:
                if any(not td_contains(id(x)) for x in node.input_tensors):
                    continue

                args, kwargs = node.arguments.fill_in(tensor_dict)

                if call_context_dict:
                    valid_context_args = getattr(
                        node.operation, "_call_context_args", {}
                    )
                    if valid_context_args:
                        kwargs = dict(kwargs)
                        for name, value in call_context_dict.items():
                            if name in valid_context_args and value is not None:
                                kwargs[name] = value

                if call_fn is not None:
                    op = (
                        operation_fn(node.operation)
                        if operation_fn is not None
                        else node.operation
                    )
                    outputs = call_fn(op, *args, **kwargs)
                else:
                    op = node.operation
                    if nnx_map is not None:
                        op = nnx_map.get(id(op), op)
                    op = operation_fn(op) if operation_fn is not None else op
                    outputs = op(*args, **kwargs)

                if len(node.outputs) == 1:
                    td_set(id(node.outputs[0]), outputs)
                else:
                    for x, y in zip(node.outputs, tree.flatten(outputs)):
                        td_set(id(x), y)

        output_tensors = []
        for i, x in enumerate(self.outputs):
            if id(x) not in tensor_dict:
                path = tree.flatten_with_path(self._outputs_struct)[i][0]
                path = ".".join(str(p) for p in path)
                raise ValueError(
                    f"Output with path `{path}` is not connected to `inputs`"
                )
            output_tensors.append(tensor_dict[id(x)])

        return tree.pack_sequence_as(self._outputs_struct, output_tensors)

    def _assert_input_compatibility(self, inputs):
        try:
            tree.assert_same_structure(inputs, self._inputs_struct)
        except ValueError:
            raise ValueError(
                "Function was called with an invalid input structure. "
                f"Expected input structure: {self._inputs_struct}\n"
                f"Received input structure: {inputs}"
            )
        for x, x_ref in zip(tree.flatten(inputs), self._inputs):
            if len(x.shape) != len(x_ref.shape):
                raise ValueError(
                    f"{self.__class__.__name__} was passed "
                    f"incompatible inputs. For input '{x_ref.name}', "
                    f"expected shape {x_ref.shape}, but received "
                    f"instead a tensor with shape {x.shape}."
                )
            for dim, ref_dim in zip(x.shape, x_ref.shape):
                if ref_dim is not None and dim is not None:
                    if dim != ref_dim:
                        raise ValueError(
                            f"{self.__class__.__name__} was passed "
                            f"incompatible inputs. For input '{x_ref.name}', "
                            f"expected shape {x_ref.shape}, but received "
                            f"instead a tensor with shape {x.shape}."
                        )


def make_node_key(op, node_index):
    return f"{id(op)}_ib-{node_index}"


def map_graph(inputs, outputs):
    """Validates a graph's topology and gather its operations and nodes.

    Args:
        inputs: List of input tensors.
        outputs: List of outputs tensors.

    Returns:
        A tuple `(nodes, nodes_by_depth, operations, operations_by_depth)`.
        - nodes: set of Node instances
        - nodes_by_depth: dict mapping ints (depth) to lists of node instances.
        - operations: list of Operation instances.
        - operations_by_depth: dict mapping ints (depth) to lists of Operation
            instances.
    """
    # "depth" is number of operations between output Node and the Node.
    # Nodes are ordered from inputs -> outputs.
    nodes_in_decreasing_depth, operation_indices = _build_map(inputs, outputs)
    nodes_in_graph = set(nodes_in_decreasing_depth)
    network_nodes = {
        make_node_key(node.operation, node.operation._inbound_nodes.index(node))
        for node in nodes_in_decreasing_depth
    }

    nodes_depths = {}  # dict {node: depth value}
    operations_depths = {}  # dict {operation: depth value}

    for node in reversed(nodes_in_decreasing_depth):
        # If the depth is not set, the node has no outbound nodes (depth 0).
        depth = nodes_depths.setdefault(node, 0)

        # Update the depth of the corresponding operation
        previous_depth = operations_depths.get(node.operation, 0)
        # If we've seen this operation before at a higher depth,
        # we should use that depth instead of the node depth.
        # This is necessary for shared operations that have inputs at different
        # depth levels in the graph.
        depth = max(depth, previous_depth)
        operations_depths[node.operation] = depth
        nodes_depths[node] = depth

        # Update the depth of inbound nodes.
        # The "depth" of a node is the max of the depths
        # of all nodes it is connected to + 1.
        # Only update nodes that are actually part of the graph.
        for node_dep in node.parent_nodes:
            if node_dep not in nodes_in_graph:
                continue
            previous_depth = nodes_depths.get(node_dep, 0)
            nodes_depths[node_dep] = max(depth + 1, previous_depth)

    # Handle inputs that are not connected to outputs.
    # We do not error out here because the inputs may be used to compute losses
    # and metrics.
    for input_t in inputs:
        input_operation = input_t._keras_history[0]
        if input_operation and input_operation not in operations_depths:
            node_index = input_t._keras_history.node_index
            node = input_operation._inbound_nodes[node_index]
            # Add InputLayer operations (unused inputs) unconditionally.
            # Skip non-InputLayer operations, as they produce intermediate
            # tensors used as Function inputs and are outside the graph.
            if node.is_input:
                operations_depths[input_operation] = 0
                operation_indices[input_operation] = -1
                nodes_depths[node] = 0
                network_nodes.add(make_node_key(input_operation, node_index))

    # Build a dict {depth: list of nodes with this depth}
    nodes_by_depth = collections.defaultdict(list)
    for node, depth in nodes_depths.items():
        nodes_by_depth[depth].append(node)

    # Build a dict {depth: list of operations with this depth}
    operations_by_depth = collections.defaultdict(list)
    for operation, depth in operations_depths.items():
        operations_by_depth[depth].append(operation)

    # Get sorted list of operation depths.
    depth_keys = list(operations_by_depth.keys())
    depth_keys.sort(reverse=True)

    # Set self.operations ordered by depth.
    operations = []
    for depth in depth_keys:
        operations_for_depth = operations_by_depth[depth]
        # Network.operations needs to have a deterministic order:
        # here we order them by traversal order.
        operations_for_depth.sort(key=lambda x: operation_indices[x])
        operations.extend(operations_for_depth)

    # Get sorted list of node depths.
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    # Check that all tensors required are computable.
    # computable_tensors: all tensors in the graph
    # that can be computed from the inputs provided.
    computable_tensors = set()
    for x in inputs:
        computable_tensors.add(x)

    operations_with_complete_input = []  # To provide a better error msg.
    for depth in depth_keys:
        for node in nodes_by_depth[depth]:
            for x in tree.flatten(node.input_tensors):
                if x not in computable_tensors:
                    operation = node.operation
                    raise ValueError(
                        "Graph disconnected: cannot find parent for "
                        f"tensor {x} at operation '{operation}'. "
                        "The following previous operations were accessed "
                        f"without issue: {operations_with_complete_input}"
                    )
                operations_with_complete_input.append(node.operation.name)

            for x in tree.flatten(node.outputs):
                computable_tensors.add(x)

    # Ensure name unicity, which will be crucial for serialization
    # (since serialized nodes refer to operations by their name).
    all_names = [operation.name for operation in operations]
    for name in all_names:
        if all_names.count(name) != 1:
            raise ValueError(
                f'The name "{name}" is used {all_names.count(name)} '
                "times in the model. All operation names should be unique."
            )
    return network_nodes, nodes_by_depth, operations, operations_by_depth


def _build_map(inputs, outputs):
    """Topologically sort nodes in order from inputs to outputs.

    It uses a depth-first search to topologically sort nodes that appear in the
    _keras_history connectivity metadata of `outputs`.

    Args:
        outputs: the output tensors whose _keras_history metadata should be
                walked. This may be an arbitrary nested structure.

    Returns:
        A tuple like (ordered_nodes, operation_to_first_traversal_index)
        ordered_nodes: list of nodes appearing in the keras history,
            topologically sorted from original inputs to the `outputs`.
            (If outputs have different sets of ancestors, the inputs to one
            output may appear after a different output).
        operation_to_first_traversal_index:
            A dict mapping operation to the traversal index in the DFS where it
            is seen. Note: if a operation is shared by several nodes, the dict
            will onlystore the index corresponding to the *first* time the
            operation seen.
    """
    finished_nodes = set()
    nodes_in_progress = set()
    nodes_in_decreasing_depth = []  # nodes from inputs -> outputs.
    operation_indices = {}  # operation -> in traversal order.
    for output in tree.flatten(outputs):
        _build_map_helper(
            inputs,
            output,
            finished_nodes,
            nodes_in_progress,
            nodes_in_decreasing_depth,
            operation_indices,
        )
    return nodes_in_decreasing_depth, operation_indices


def _build_map_helper(
    inputs,
    tensor,
    finished_nodes,
    nodes_in_progress,
    nodes_in_decreasing_depth,
    operation_indices,
):
    """Recursive helper for `_build_map`."""
    (
        operation,
        node_index,
        _,
    ) = tensor._keras_history
    if not operation:
        return

    node = operation._inbound_nodes[node_index]

    # Don't repeat work for shared subgraphs
    if node in finished_nodes:
        return

    # If this tensor is one of the declared inputs and its producing
    # operation is not an InputLayer, stop traversal here. The operation
    # that produced this tensor is outside the Function's graph.
    flat_inputs = tree.flatten(inputs)
    if not node.is_input and tensor in flat_inputs:
        finished_nodes.add(node)
        return

    # Prevent cycles.
    if node in nodes_in_progress:
        raise ValueError(
            f"Tensor {tensor} from operation '{operation.name}' is part of a "
            "cycle."
        )

    # Store the traversal order for operation sorting.
    if operation not in operation_indices:
        operation_indices[operation] = len(operation_indices)

    # Propagate to all previous tensors connected to this node.
    nodes_in_progress.add(node)
    if not node.is_input:
        for input_tensor in node.input_tensors:
            _build_map_helper(
                inputs,
                input_tensor,
                finished_nodes,
                nodes_in_progress,
                nodes_in_decreasing_depth,
                operation_indices,
            )

    finished_nodes.add(node)
    nodes_in_progress.remove(node)
    nodes_in_decreasing_depth.append(node)
