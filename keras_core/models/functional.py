import copy
import inspect
import warnings

from tensorflow import nest

from keras_core import backend
from keras_core import operations as ops
from keras_core.layers.layer import Layer
from keras_core.models.model import Model
from keras_core.operations.function import Function
from keras_core.operations.function import make_node_key
from keras_core.saving import serialization_lib
from keras_core.utils import tracking


class Functional(Function, Model):
    """A `Functional` model is a `Model` defined as a directed graph of layers.

    Three types of `Model` exist: subclassed `Model`, `Functional` model,
    and `Sequential` (a special case of `Functional`).

    A `Functional` model can be instantiated by passing two arguments to
    `__init__()`. The first argument is the `keras_core.Input` objects
    that represent the inputs to the model.
    The second argument specifies the output tensors that represent
    the outputs of this model. Both arguments can be a nested structure
    of tensors.

    Example:

    ```
    inputs = {'x1': keras_core.Input(shape=(10,)),
              'x2': keras_core.Input(shape=(1,))}
    t = keras_core.layers.Dense(1, activation='relu')(inputs['x1'])
    outputs = keras_core.layers.Add()([t, inputs['x2'])
    model = keras_core.Model(inputs, outputs)
    ```

    A `Functional` model constructed using the Functional API can also
    include raw Keras Core ops.

    Example:

    ```python
    inputs = keras_core.Input(shape=(10,))
    x = keras_core.layers.Dense(1)(inputs)
    outputs = ops.nn.relu(x)
    model = keras_core.Model(inputs, outputs)
    ```

    A new `Functional` model can also be created by using the
    intermediate tensors. This enables you to quickly extract sub-components
    of the model.

    Example:

    ```python
    inputs = keras_core.Input(shape=(None, None, 3))
    processed = keras_core.layers.RandomCrop(width=32, height=32)(inputs)
    conv = keras_core.layers.Conv2D(filters=2, kernel_size=3)(processed)
    pooling = keras_core.layers.GlobalAveragePooling2D()(conv)
    feature = keras_core.layers.Dense(10)(pooling)

    full_model = keras_core.Model(inputs, feature)
    backbone = keras_core.Model(processed, conv)
    activations = keras_core.Model(conv, feature)
    ```

    Note that the `backbone` and `activations` models are not
    created with `keras_core.Input` objects, but with the tensors
    that are originated from `keras_core.Input` objects.
    Under the hood, the layers and weights will
    be shared across these models, so that user can train the `full_model`, and
    use `backbone` or `activations` to do feature extraction.
    The inputs and outputs of the model can be nested structures of tensors as
    well, and the created models are standard `Functional` model that support
    all the existing API.

    Args:
        inputs: List of input tensors (must be created via `keras_core.Input()`
            or originated from `keras_core.Input()`).
        outputs: List of output tensors.
        name: String, optional. Name of the model.
        trainable: Boolean, optional. If the model's variables should be
            trainable.
    """

    @tracking.no_automatic_dependency_tracking
    def __init__(self, inputs, outputs, name=None, **kwargs):
        if isinstance(inputs, dict):
            for k, v in inputs.items():
                if not isinstance(v, backend.KerasTensor):
                    raise ValueError(
                        "When providing `inputs` as a dict, all values in the "
                        f"dict must be KerasTensors. Received: inputs={inputs} "
                        f"including invalid value {v} of type {type(v)}"
                    )
                if k != v.name:
                    # TODO: maybe make this a warning
                    raise ValueError(
                        "When providing `inputs` as a dict, all keys in the "
                        "dict must match the names of the corresponding "
                        f"tensors. Received key '{k}' mapping to value {v} "
                        f"which has name '{v.name}'. Change the tensor name to "
                        f"'{k}' (via `Input(..., name='{k}')`)"
                    )
        elif isinstance(inputs, (list, tuple)):
            for x in inputs:
                if not isinstance(x, backend.KerasTensor):
                    raise ValueError(
                        "When providing `inputs` as a list/tuple, all values "
                        f"in the list/tuple must be KerasTensors. Received: "
                        f"inputs={inputs} including invalid value {x} of type "
                        f"{type(x)}"
                    )
        elif not isinstance(inputs, backend.KerasTensor):
            raise ValueError(
                f"Unrecognized type for `inputs`: {inputs} "
                f"(of type {type(inputs)})"
            )
        if isinstance(outputs, dict):
            for k, v in outputs.items():
                if not isinstance(v, backend.KerasTensor):
                    raise ValueError(
                        "When providing `outputs` as a dict, all values in the "
                        f"dict must be KerasTensors. Received: "
                        f"outputs={outputs} including invalid value {v} of "
                        f"type {type(v)}"
                    )
        elif isinstance(outputs, (list, tuple)):
            for x in outputs:
                if not isinstance(x, backend.KerasTensor):
                    raise ValueError(
                        "When providing `outputs` as a list/tuple, all values "
                        f"in the list/tuple must be KerasTensors. Received: "
                        f"outputs={outputs} including invalid value {x} of "
                        f"type {type(x)}"
                    )
        elif not isinstance(outputs, backend.KerasTensor):
            raise ValueError(
                f"Unrecognized type for `outputs`: {outputs} "
                f"(of type {type(outputs)})"
            )

        Function.__init__(self, inputs, outputs, name=name, **kwargs)
        self._layers = self.layers
        self.built = True

    @property
    def layers(self):
        layers = []
        for operation in self._operations:
            if isinstance(operation, Layer):
                layers.append(operation)
        return layers

    def call(self, inputs, training=None, mask=None):
        # Add support for traning, masking
        inputs = self._standardize_inputs(inputs)
        if mask is None:
            masks = [None] * len(inputs)
        else:
            masks = self._flatten_to_reference_inputs(mask)
            for x, mask in zip(inputs, masks):
                x._keras_mask = mask
        outputs = self._run_through_graph(
            inputs, operation_fn=lambda op: operation_fn(op, training=training)
        )
        return unpack_singleton(outputs)

    def compute_output_spec(self, inputs, training=None, mask=None):
        # From Function
        return super().compute_output_spec(inputs)

    def build(self, input_shape):
        self.built = True

    @property
    def input_shape(self):
        input_shapes = nest.map_structure(lambda x: x.shape, self.inputs)
        if isinstance(input_shapes, list) and len(input_shapes) == 1:
            return input_shapes[0]
        return input_shapes

    @property
    def output_shape(self):
        output_shapes = nest.map_structure(lambda x: x.shape, self.outputs)
        if isinstance(output_shapes, list) and len(output_shapes) == 1:
            return output_shapes[0]
        return output_shapes

    def _assert_input_compatibility(self, *args):
        return super(Model, self)._assert_input_compatibility(*args)

    def _flatten_to_reference_inputs(self, inputs, allow_extra_keys=True):
        if isinstance(inputs, dict):
            ref_inputs = self._inputs_struct
            if not nest.is_nested(ref_inputs):
                ref_inputs = [self._nested_inputs]
            if isinstance(ref_inputs, dict):
                # In the case that the graph is constructed with dict input
                # tensors, We will use the original dict key to map with the
                # keys in the input data. Note that the model.inputs is using
                # nest.flatten to process the input tensors, which means the
                # dict input tensors are ordered by their keys.
                ref_input_names = sorted(ref_inputs.keys())
            else:
                ref_input_names = [
                    inp._keras_history.operation.name for inp in ref_inputs
                ]
            # Raise an warning if there are more input data comparing to input
            # tensor
            if allow_extra_keys and len(inputs) > len(ref_input_names):
                warnings.warn(
                    "Input dict contained keys {} which did not match any "
                    "model input. They will be ignored by the model.".format(
                        [n for n in inputs.keys() if n not in ref_input_names]
                    ),
                    stacklevel=2,
                )
            # Flatten in the order `Input`s were passed during Model
            # construction.
            return [inputs[n] for n in ref_input_names]
        # Otherwise both ref inputs and inputs will already be in same order.
        return nest.flatten(inputs)

    def _adjust_input_rank(self, flat_inputs):
        flat_ref_shapes = [x.shape for x in self._inputs]
        adjusted = []
        for x, ref_shape in zip(flat_inputs, flat_ref_shapes):
            x_rank = len(x.shape)
            ref_rank = len(ref_shape)
            if x_rank == ref_rank:
                adjusted.append(x)
                continue
            if x_rank == ref_rank + 1:
                if x.shape[-1] == 1:
                    adjusted.append(ops.squeeze(x, axis=-1))
                    continue
            if x_rank == ref_rank - 1:
                if ref_shape[-1] == 1:
                    adjusted.append(ops.expand_dims(x, axis=-1))
                    continue
            raise ValueError(
                f"Invalid input shape for input {x}. Expected shape "
                f"{ref_shape}, but input has incompatible shape {x.shape}"
            )
        # Add back metadata.
        for i in range(len(flat_inputs)):
            if hasattr(flat_inputs[i], "_keras_history"):
                adjusted[i]._keras_history = flat_inputs[i]._keras_history
            if hasattr(flat_inputs[i], "_keras_mask"):
                adjusted[i]._keras_mask = flat_inputs[i]._keras_mask
        return adjusted

    def _standardize_inputs(self, inputs):
        flat_inputs = self._flatten_to_reference_inputs(inputs)
        return self._adjust_input_rank(flat_inputs)

    def add_loss(self, loss):
        # Symbolic only. TODO
        raise NotImplementedError

    def get_config(self):
        if not functional_like_constructor(self.__class__):
            # Subclassed networks are not serializable
            # (unless serialization is implemented by
            # the author of the subclassed network).
            return Model.get_config(self)

        config = {
            "name": self.name,
            "trainable": self.trainable,
        }
        # Build a map from a layer unique name (make_node_key)
        # to the index of the nodes that are saved in the config.
        # Only nodes in network_nodes are saved.
        node_reindexing_map = {}
        for operation in self.operations:
            if issubclass(operation.__class__, Functional):
                # Functional models start with a pre-existing node
                # linking their input to output.
                kept_nodes = 1
            else:
                kept_nodes = 0
            for original_node_index, node in enumerate(
                operation._inbound_nodes
            ):
                node_key = make_node_key(operation, original_node_index)
                if node_key in self._nodes:
                    # i.e. we mark it to be saved
                    node_reindexing_map[node_key] = kept_nodes
                    kept_nodes += 1

        # serialize and save the layers in layer_configs
        layer_configs = []
        for operation in self.operations:  # From the earliest layers on.
            filtered_inbound_nodes = []
            for original_node_index, node in enumerate(
                operation._inbound_nodes
            ):
                node_key = make_node_key(operation, original_node_index)
                if node_key in self._nodes:
                    # The node is relevant to the model:
                    # add to filtered_inbound_nodes.
                    node_data = serialize_node(node, node_reindexing_map)
                    if node_data is not None:
                        filtered_inbound_nodes.append(node_data)

            layer_config = serialization_lib.serialize_keras_object(operation)
            layer_config["name"] = operation.name
            layer_config["inbound_nodes"] = filtered_inbound_nodes
            layer_configs.append(layer_config)
        config["layers"] = layer_configs

        # Gather info about inputs and outputs.
        model_inputs = []
        for tensor in self._inputs:
            operation = tensor._keras_history[0]
            node_index = tensor._keras_history[1]
            tensor_index = tensor._keras_history[2]
            node_key = make_node_key(operation, node_index)
            if node_key not in self._nodes:
                continue
            new_node_index = node_reindexing_map[node_key]
            model_inputs.append([operation.name, new_node_index, tensor_index])
        config["input_layers"] = model_inputs
        model_outputs = []
        for tensor in self._outputs:
            operation = tensor._keras_history[0]
            node_index = tensor._keras_history[1]
            tensor_index = tensor._keras_history[2]
            node_key = make_node_key(operation, node_index)
            if node_key not in self._nodes:
                continue
            new_node_index = node_reindexing_map[node_key]
            model_outputs.append([operation.name, new_node_index, tensor_index])
        config["output_layers"] = model_outputs
        return copy.deepcopy(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        functional_config_keys = [
            "name",
            "layers",
            "input_layers",
            "output_layers",
        ]
        is_functional_config = all(
            key in config for key in functional_config_keys
        )
        argspec = inspect.getfullargspec(cls.__init__)
        functional_init_args = inspect.getfullargspec(Functional.__init__).args[
            1:
        ]
        revivable_as_functional = (
            cls in {Functional, Model}
            or argspec.args[1:] == functional_init_args
            or (argspec.varargs == "args" and argspec.varkw == "kwargs")
        )
        if is_functional_config and revivable_as_functional:
            # Revive Functional model
            # (but not Functional subclasses with a custom __init__)
            return cls._from_config(config, custom_objects=custom_objects)

        # Either the model has a custom __init__, or the config
        # does not contain all the information necessary to
        # revive a Functional model. This happens when the user creates
        # subclassed models where `get_config()` is returning
        # insufficient information to be considered a Functional model.
        # In this case, we fall back to provide all config into the
        # constructor of the class.
        try:
            return cls(**config)
        except TypeError as e:
            raise TypeError(
                "Unable to revive model from config. When overriding "
                "the `get_config()` method, make sure that the "
                "returned config contains all items used as arguments "
                f"in the  constructor to {cls}, "
                "which is the default behavior. "
                "You can override this default behavior by defining a "
                "`from_config(cls, config)` class method to specify "
                "how to create an "
                f"instance of {cls.__name__} from its config.\n\n"
                f"Received config={config}\n\n"
                f"Error encountered during deserialization: {e}"
            )

    @classmethod
    def _from_config(cls, config, custom_objects=None):
        """Instantiates a Model from its config (output of `get_config()`)."""
        # Layer instances created during
        # the graph reconstruction process
        created_layers = {}

        # Dictionary mapping layer instances to
        # node data that specifies a layer call.
        # It acts as a queue that maintains any unprocessed
        # layer call until it becomes possible to process it
        # (i.e. until the input tensors to the call all exist).
        unprocessed_nodes = {}

        def add_unprocessed_node(layer, node_data):
            """Add node to layer list

            Arg:
                layer: layer object
                node_data: Node data specifying layer call
            """
            if layer not in unprocessed_nodes:
                unprocessed_nodes[layer] = [node_data]
            else:
                unprocessed_nodes[layer].append(node_data)

        def process_node(layer, node_data):
            """Reconstruct node by linking to inbound layers

            Args:
                layer: Layer to process
                node_data: List of layer configs
            """
            args, kwargs = deserialize_node(node_data, created_layers)
            # Call layer on its inputs, thus creating the node
            # and building the layer if needed.
            layer(*args, **kwargs)

        def process_layer(layer_data):
            """Deserializes a layer, then call it on appropriate inputs.

            Args:
                layer_data: layer config dict.
            """
            layer_name = layer_data["name"]

            # Instantiate layer.
            layer = serialization_lib.deserialize_keras_object(
                layer_data, custom_objects=custom_objects
            )
            created_layers[layer_name] = layer

            # Gather layer inputs.
            inbound_nodes_data = layer_data["inbound_nodes"]
            for node_data in inbound_nodes_data:
                # We don't process nodes (i.e. make layer calls)
                # on the fly because the inbound node may not yet exist,
                # in case of layer shared at different topological depths
                # (e.g. a model such as A(B(A(B(x)))))
                add_unprocessed_node(layer, node_data)

        # First, we create all layers and enqueue nodes to be processed
        for layer_data in config["layers"]:
            process_layer(layer_data)

        # Then we process nodes in order of layer depth.
        # Nodes that cannot yet be processed (if the inbound node
        # does not yet exist) are re-enqueued, and the process
        # is repeated until all nodes are processed.
        while unprocessed_nodes:
            for layer_data in config["layers"]:
                layer = created_layers[layer_data["name"]]

                # Process all nodes in layer, if not yet processed
                if layer in unprocessed_nodes:
                    node_data_list = unprocessed_nodes[layer]

                    # Process nodes in order
                    node_index = 0
                    while node_index < len(node_data_list):
                        node_data = node_data_list[node_index]
                        try:
                            process_node(layer, node_data)

                        # If the node does not have all inbound layers
                        # available, stop processing and continue later
                        except IndexError:
                            break

                        node_index += 1

                    # If not all nodes processed then store unprocessed nodes
                    if node_index < len(node_data_list):
                        unprocessed_nodes[layer] = node_data_list[node_index:]
                    # If all nodes processed remove the layer
                    else:
                        del unprocessed_nodes[layer]

        # Create lits of input and output tensors and return new class
        name = config.get("name")
        input_tensors = []
        output_tensors = []
        for layer_data in config["input_layers"]:
            layer_name, node_index, tensor_index = layer_data
            assert layer_name in created_layers
            layer = created_layers[layer_name]
            layer_output_tensors = layer._inbound_nodes[
                node_index
            ].output_tensors
            input_tensors.append(layer_output_tensors[tensor_index])
        for layer_data in config["output_layers"]:
            layer_name, node_index, tensor_index = layer_data
            assert layer_name in created_layers
            layer = created_layers[layer_name]
            layer_output_tensors = layer._inbound_nodes[
                node_index
            ].output_tensors
            output_tensors.append(layer_output_tensors[tensor_index])
        return cls(inputs=input_tensors, outputs=output_tensors, name=name)


def operation_fn(operation, training):
    def call(*args, **kwargs):
        if (
            hasattr(operation, "_call_has_training_arg")
            and operation._call_has_training_arg()
        ):
            kwargs["training"] = training
        return operation(*args, **kwargs)

    return call


def functional_like_constructor(cls):
    init_args = inspect.getfullargspec(cls.__init__).args[1:]
    functional_init_args = inspect.getfullargspec(Functional.__init__).args[1:]
    if init_args == functional_init_args:
        return True
    return False


def unpack_singleton(x):
    if isinstance(x, (list, tuple)) and len(x) == 1:
        return x[0]
    return x


def serialize_node(node, node_reindexing_map):
    if not node.input_tensors:
        # Does not need to be serialized.
        return

    args = node.arguments.args
    kwargs = node.arguments.kwargs
    return {
        "args": serialization_lib.serialize_keras_object(args),
        "kwargs": serialization_lib.serialize_keras_object(kwargs),
    }


def deserialize_node(node_data, created_layers):
    """Return (args, kwargs) for calling the node layer."""
    if not node_data:
        return [], {}

    if isinstance(node_data, list):
        # Legacy case.
        input_tensors = []
        for input_data in node_data:
            inbound_layer_name = input_data[0]
            inbound_node_index = input_data[1]
            inbound_tensor_index = input_data[2]
            if len(input_data) == 3:
                kwargs = {}
            elif len(input_data) == 4:
                kwargs = input_data[3]
            else:
                raise ValueError(
                    "Cannot deserialize the model (invalid config data?)"
                )
            inbound_layer = created_layers[inbound_layer_name]

            # Raise an error if the corresponding layer node
            # has not yet been created
            if len(inbound_layer._inbound_nodes) <= inbound_node_index:
                raise IndexError(
                    "Layer node index out of bounds.\n"
                    f"inbound_layer = {inbound_layer}\n"
                    "inbound_layer._inbound_nodes = "
                    f"{inbound_layer._inbound_nodes}\n"
                    f"inbound_node_index = {inbound_node_index}"
                )
            inbound_node = inbound_layer._inbound_nodes[inbound_node_index]
            input_tensors.append(
                inbound_node.output_tensors[inbound_tensor_index]
            )
        return [unpack_singleton(input_tensors)], kwargs

    args = serialization_lib.deserialize_keras_object(node_data["args"])
    kwargs = serialization_lib.deserialize_keras_object(node_data["kwargs"])

    def convert_revived_tensor(x):
        if isinstance(x, backend.KerasTensor):
            history = x._pre_serialization_keras_history
            if history is None:
                return x
            layer = created_layers.get(history[0], None)
            if layer is None:
                raise ValueError(f"Unknown layer: {history[0]}")
            inbound_node_index = history[1]
            inbound_tensor_index = history[2]
            if len(layer._inbound_nodes) <= inbound_node_index:
                raise ValueError(
                    "Layer node index out of bounds.\n"
                    f"inbound_layer = {layer}\n"
                    f"inbound_layer._inbound_nodes = {layer._inbound_nodes}\n"
                    f"inbound_node_index = {inbound_node_index}"
                )
            inbound_node = layer._inbound_nodes[inbound_node_index]
            return inbound_node.output_tensors[inbound_tensor_index]
        return x

    args = nest.map_structure(convert_revived_tensor, args)
    kwargs = nest.map_structure(convert_revived_tensor, kwargs)
    return args, kwargs
