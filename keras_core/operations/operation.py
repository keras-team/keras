from keras_core.backend.keras_tensor import any_symbolic_tensors
from keras_core.operations.node import Node
from keras_core.utils.naming import auto_name
from keras_core import backend


class Operation:
    def __init__(self, name=None):
        self.name = name or auto_name(self.__class__.__name__)
        self._inbound_nodes = []
        self._outbound_nodes = []

    def __call__(self, *args, **kwargs):
        if any_symbolic_tensors(args, kwargs):
            return self.symbolic_call(*args, **kwargs)
        return self.call(*args, **kwargs)

    def symbolic_call(self, *args, **kwargs):
        # Perform shape/dtype inference.
        outputs = self.compute_output_spec(*args, **kwargs)
        # Record a new node in the operations graph.
        # The Node wires itself to inbound and outbound ops.  The
        # Node constructor updates this op's self._inbound_nodes,
        # sets _keras_history on the outputs, and adds itself to the
        # `_outbound_nodes` of the ops that produced the inputs to this
        # call.
        Node(
            operation=self, call_args=args, call_kwargs=kwargs, outputs=outputs
        )
        return outputs

    def call(self, *args, **kwargs):
        raise NotImplementedError

    def compute_output_spec(self, *args, **kwargs):
        try:
            return backend.compute_output_spec(self.call, *args, **kwargs)
        except Exception as e:
            raise RuntimeError(
                "Could not automatically infer the output shape / dtype of this operation. "
                "Please implement the `compute_output_spec` method "
                f"on your object ({self.__class__.__name__}). "
                f"Error encountered: {e}"
            )

    def get_config(self):
        return {"name": self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __repr__(self):
        return f"<Operation name={self.name}>"

    @property
    def input(self):
        """Retrieves the input tensor(s) of a symbolic operation.

        Only returns the tensor(s) corresponding to the *first time*
        the operation was called.

        Returns:
            Input tensor or list of input tensors.
        """
        return self._get_node_attribute_at_index(0, "input_tensors", "input")

    @property
    def output(self):
        """Retrieves the output tensor(s) of a layer.

        Only returns the tensor(s) corresponding to the *first time*
        the operation was called.

        Returns:
            Output tensor or list of output tensors.
        """
        return self._get_node_attribute_at_index(0, "output_tensors", "output")

    def _get_node_attribute_at_index(self, node_index, attr, attr_name):
        """Private utility to retrieves an attribute (e.g. inputs) from a node.

        This is used to implement the properties:
        - output
        - input

        Args:
            node_index: Integer index of the node from which
                to retrieve the attribute.
            attr: Exact node attribute name.
            attr_name: Human-readable attribute name, for error messages.

        Returns:
            The operation's attribute `attr` at the node of index `node_index`.
        """
        if not self._inbound_nodes:
            raise RuntimeError(
                f"The layer {self.name} has never been called "
                f"and thus has no defined {attr_name}."
            )
        if not len(self._inbound_nodes) > node_index:
            raise ValueError(
                f"Asked to get {attr_name} at node "
                f"{node_index}, but the operation has only "
                f"{len(self._inbound_nodes)} inbound nodes."
            )
        values = getattr(self._inbound_nodes[node_index], attr)
        if isinstance(values, list) and len(values) == 1:
            return values[0]
        else:
            return values
