import collections

from keras.src import tree
from keras.src.backend import KerasTensor
from keras.src.ops.symbolic_arguments import SymbolicArguments


class Node:
    """A `Node` describes an operation `__call__()` event.

    A Keras Function is a DAG with `Node` instances as nodes, and
    `KerasTensor` instances as edges. Nodes aren't `Operation` instances,
    because a single operation could be called multiple times, which would
    result in graph cycles.

    A `__call__()` event involves input tensors (and other input arguments),
    the operation that was called, and the resulting output tensors.
    A `Node` will include all this information.

    Since a single `Operation` could be called multiple times,
    the `Node` instances are stored on operations as a list.
    Each time an operation is called, a node is added to `op._inbound_nodes`.
    Each time the output of an operation is used by another operation,
    a node is added to `op._outbound_nodes`.

    Every `KerasTensor` instance has a `KerasHistory` object attached,
    which tracks the `Node` that records the `__call__()` event that created
    the tensor. By recursively walking through `Node` instances
    via the `KerasHistory` metadata of `KerasTensor` instances, once can
    retrieve the entire DAG of a Keras Function.

    Args:
        operation: The Operation that was called in the `op.__call__()`
            event that this node represents.
        call_args: The positional arguments the operation was called with.
        call_kwargs: The keyword arguments the operation was called with.
        outputs: The output tensors of the `op.__call__()` call.
    """

    def __init__(
        self, operation, call_args=None, call_kwargs=None, outputs=None
    ):
        self.operation = operation
        self.arguments = SymbolicArguments(*call_args, **call_kwargs)
        self.outputs = [] if outputs is None else tree.flatten(outputs)
        for x in self.outputs:
            if not isinstance(x, KerasTensor):
                raise ValueError(
                    "All operation outputs must be tensors. "
                    f"Operation {operation} returned a non-tensor. "
                    f"Non-tensor received: {x}"
                )

        zero_history = any(
            not x.record_history for x in self.arguments.keras_tensors
        )

        # If inputs don't have metadata yet, add it.
        if not zero_history:
            for tensor in self.arguments.keras_tensors:
                if not hasattr(tensor, "_keras_history"):
                    tensor._keras_history = KerasHistory(
                        operation=None, node_index=0, tensor_index=0
                    )

        # Wire up Node to Operations.
        self.operation._inbound_nodes.append(self)
        for kt in self.arguments.keras_tensors:
            inbound_op = kt._keras_history.operation
            if inbound_op is not None:  # It's a graph entry point.
                inbound_op._outbound_nodes.append(self)

        # Set metadata on outputs.
        if not zero_history:
            node_index = len(self.operation._inbound_nodes) - 1
            for i, tensor in enumerate(self.outputs):
                tensor._keras_history = KerasHistory(
                    operation=operation, node_index=node_index, tensor_index=i
                )

        # Whether this is a root node.
        self.is_input = not self.arguments.keras_tensors

    def __repr__(self):
        return f"<Node operation={self.operation.name}, id={id(self)}>"

    @property
    def input_tensors(self):
        return self.arguments.keras_tensors

    @property
    def output_tensors(self):
        return self.outputs

    @property
    def parent_nodes(self):
        """The parent `Node`s.

        Returns:
            all the `Node`s whose output this node immediately depends on.
        """
        node_deps = []
        for kt in self.arguments.keras_tensors:
            op = kt._keras_history.operation
            node_index = kt._keras_history.node_index
            if op is not None:  # `None` for `Input` tensors.
                node_deps.append(op._inbound_nodes[node_index])
        return node_deps


class KerasHistory(
    collections.namedtuple(
        "KerasHistory", ["operation", "node_index", "tensor_index"]
    )
):
    """Tracks the Operation call that created a Tensor.

    During construction of Keras Functions, this metadata is added to
    each Tensor produced as the output of an Operation.
    This allows Keras to track how each Tensor was produced, and
    this information is later retraced by the `Function` class to
    reconstruct the Operations graph.

    Attributes:
      operation: The Operation instance that produced the Tensor.
      node_index: The specific call to the Operation that produced this Tensor.
        Operations can be called multiple times in order to share weights. A new
        node is created every time an Operation is called. The corresponding
        node that represents the call event that produced the Tensor can be
        found at `op._inbound_nodes[node_index]`.
      tensor_index: The output index for this Tensor.
        Always zero if the Operation that produced this Tensor
        only has one output. Nested structures of
        Tensors are deterministically assigned an index via `nest.flatten`.
    """

    # Added to maintain memory and performance characteristics of `namedtuple`
    # while subclassing.
    __slots__ = ()


def is_keras_tensor(obj):
    return hasattr(obj, "_keras_history")
