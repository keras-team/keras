import collections

from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.backend import KerasTensor
from keras.src.backend.config import backend
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

    @property
    def operations(self):
        return self._operations[:]

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

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

    def call(self, inputs):
        """Computes output tensors for new inputs."""
        self._assert_input_compatibility(inputs)
        return self._run_through_graph(inputs, operation_fn=lambda op: op)

    def _run_through_graph(self, inputs, operation_fn, call_fn=None):
        """Execute the graph.

        At each node we compute outputs via
        `operation_fn(node.operation)(*args, **kwargs)`.
        """
        inputs = tree.flatten(inputs)

        # Dictionary mapping reference tensors to computed tensors.
        tensor_dict = {}
        for x, y in zip(self.inputs, inputs):
            tensor_dict[id(x)] = y

        nodes_by_depth = self._nodes_by_depth
        depth_keys = list(nodes_by_depth.keys())
        depth_keys.sort(reverse=True)

        for depth in depth_keys:
            nodes = nodes_by_depth[depth]
            for node in nodes:
                if not node.operation or node.is_input:
                    continue  # Input tensors already exist.

                if any(id(x) not in tensor_dict for x in node.input_tensors):
                    continue  # Node is not computable, try skipping.

                args, kwargs = node.arguments.fill_in(tensor_dict)
                op = operation_fn(node.operation)
                if call_fn is not None:
                    outputs = call_fn(op, *args, **kwargs)
                else:
                    outputs = op(*args, **kwargs)

                # Update tensor_dict.
                for x, y in zip(node.outputs, tree.flatten(outputs)):
                    tensor_dict[id(x)] = y

        output_tensors = []
        for x in self.outputs:
            output_tensors.append(tensor_dict[id(x)])

        return tree.pack_sequence_as(self._outputs_struct, output_tensors)

    def _assert_input_compatibility(self, inputs):
        try:
            tree.assert_same_structure(
                inputs, self._inputs_struct, check_types=False
            )
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
    return str(id(op)) + "_ib-" + str(node_index)


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
        for node_dep in node.parent_nodes:
            previous_depth = nodes_depths.get(node_dep, 0)
            nodes_depths[node_dep] = max(depth + 1, previous_depth)

    # Handle inputs that are not connected to outputs.
    # We do not error out here because the inputs may be used to compute losses
    # and metrics.
    for input_t in inputs:
        input_operation = input_t._keras_history[0]
        if input_operation and input_operation not in operations_depths:
            operations_depths[input_operation] = 0
            operation_indices[input_operation] = -1
            nodes_depths[input_operation._inbound_nodes[0]] = 0
            network_nodes.add(make_node_key(input_operation, 0))

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
                operations_with_complete_input.append(operation.name)

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
    if not node.is_input and tensor not in tree.flatten(inputs):
        for tensor in node.input_tensors:
            _build_map_helper(
                inputs,
                tensor,
                finished_nodes,
                nodes_in_progress,
                nodes_in_decreasing_depth,
                operation_indices,
            )

    finished_nodes.add(node)
    nodes_in_progress.remove(node)
    nodes_in_decreasing_depth.append(node)
