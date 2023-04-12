from tensorflow import nest
from keras_core import backend
from keras_core.utils import io_utils
from keras_core.utils import dtype_utils
from keras_core.utils import text_rendering
import math
import re


def count_params(weights):
    shapes = [v.shape for v in weights]
    return int(sum(math.prod(p) for p in shapes))


def weight_memory_size(weights):
    """Compute the memory footprint for weights based on their dtypes.

    Args:
        weights: An iterable contains the weights to compute weight size.

    Returns:
        The total memory size (in Bytes) of the weights.
    """
    unique_weights = set(weights)
    total_memory_size = 0
    for w in unique_weights:
        weight_shape = math.prod(w.shape)
        dtype = backend.standardize_dtype(w.dtype)
        per_param_size = dtype_utils.float_dtype_size(dtype)
        total_memory_size += weight_shape * per_param_size
    return total_memory_size


def readable_memory_size(weight_memory_size):
    """Convert the weight memory size (Bytes) to a readable string."""
    units = ["Byte", "KB", "MB", "GB", "TB", "PB"]
    scale = 1024
    for unit in units:
        if weight_memory_size / scale < 1:
            return "{:.2f} {}".format(weight_memory_size, unit)
        else:
            weight_memory_size /= scale
    return "{:.2f} {}".format(weight_memory_size, units[-1])


def print_summary(
    model,
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    show_trainable=False,
    layer_range=None,
):
    """Prints a summary of a model.

    Args:
        model: Keras model instance.
        line_length: Total length of printed lines
            (e.g. set this to adapt the display to different
            terminal window sizes).
        positions: Relative or absolute positions of log elements in each line.
            If not provided, defaults to `[0.3, 0.6, 0.70, 1.]`.
        print_fn: Print function to use.
            It will be called on each line of the summary.
            You can set it to a custom function
            in order to capture the string summary.
            It defaults to `print` (prints to stdout).
        expand_nested: Whether to expand the nested models.
            If not provided, defaults to `False`.
        show_trainable: Whether to show if a layer is trainable.
            If not provided, defaults to `False`.
        layer_range: List or tuple containing two strings,
            the starting layer name and ending layer name (both inclusive),
            indicating the range of layers to be printed in the summary. The
            strings could also be regexes instead of an exact name. In this
             case, the starting layer will be the first layer that matches
            `layer_range[0]` and the ending layer will be the last element that
            matches `layer_range[1]`. By default (`None`) all
            layers in the model are included in the summary.
    """
    from keras_core.models import Sequential, Functional

    if print_fn is None:
        print_fn = io_utils.print_msg

    if isinstance(model, Sequential):
        sequential_like = True
    elif not isinstance(model, Functional):
        # We treat subclassed models as a simple sequence of layers, for logging
        # purposes.
        sequential_like = True
    else:
        sequential_like = True
        nodes_by_depth = model._nodes_by_depth.values()
        nodes = []
        for v in nodes_by_depth:
            if (len(v) > 1) or (
                len(v) == 1 and len(nest.flatten(v[0].keras_inputs)) > 1
            ):
                # if the model has multiple nodes
                # or if the nodes have multiple inbound_layers
                # the model is no longer sequential
                sequential_like = False
                break
            nodes += v
        if sequential_like:
            # search for shared layers
            for layer in model.layers:
                flag = False
                for node in layer._inbound_nodes:
                    if node in nodes:
                        if flag:
                            sequential_like = False
                            break
                        else:
                            flag = True
                if not sequential_like:
                    break

    if sequential_like:
        line_length = line_length or 65
        positions = positions or [0.45, 0.85, 1.0]
        # header names for the different log elements
        header = ["Layer (type)", "Output Shape", "Param #"]
    else:
        line_length = line_length or 98
        positions = positions or [0.3, 0.6, 0.70, 1.0]
        # header names for the different log elements
        header = ["Layer (type)", "Output Shape", "Param #", "Connected to"]
        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v

    if show_trainable:
        line_length += 11
        positions.append(line_length)
        header.append("Trainable")

    layer_range = get_layer_index_bound_by_layer_name(model, layer_range)

    print_fn(f'Model: "{model.name}"')
    rows = []

    def print_layer_summary(layer, prefix=" "):
        """Prints a summary for a single layer.

        Args:
            layer: target layer.
            nested_level: level of nesting of the layer inside its parent layer
              (e.g. 0 for a top-level layer, 1 for a nested layer).
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = "multiple"
        except RuntimeError:  # output_shape unknown in Eager mode.
            output_shape = "?"
        name = prefix + layer.name
        cls_name = layer.__class__.__name__
        if not layer.built:
            # If a subclassed model has a layer that is not called in
            # Model.call, the layer will not be built and we cannot call
            # layer.count_params().
            params = "0 (unused)"
        else:
            params = layer.count_params()
        fields = [name + " (" + cls_name + ")", output_shape, params]

        if show_trainable:
            fields.append("Y" if layer.trainable else "N")
        rows.append(fields)

    def print_layer_summary_with_connections(layer, prefix=""):
        """Prints a summary for a single layer (including its connections).

        Args:
            layer: target layer.
            nested_level: level of nesting of the layer inside its parent layer
              (e.g. 0 for a top-level layer, 1 for a nested layer).
        """
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = "multiple"
        connections = []
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue
            for kt in node.keras_inputs:
                keras_history = kt._keras_history
                inbound_layer = keras_history.layer
                node_index = keras_history.node_index
                tensor_index = keras_history.tensor_index
                connections.append(
                    f"{inbound_layer.name}[{node_index}][{tensor_index}]"
                )
        name = prefix + layer.name
        cls_name = layer.__class__.__name__
        fields = [
            name + " (" + cls_name + ")",
            output_shape,
            layer.count_params(),
            connections,
        ]
        if show_trainable:
            fields.append("Y" if layer.trainable else "N")
        rows.append(fields)

    def print_layer(layer, nested_level=0):
        if sequential_like:
            print_layer_summary(layer, prefix=">>>" * nested_level + " ")
        else:
            print_layer_summary_with_connections(
                layer, prefix=">>>" * nested_level + " "
            )

        if expand_nested and hasattr(layer, "layers") and layer.layers:
            nested_layers = layer.layers
            nested_level += 1
            for i in range(len(nested_layers)):
                print_layer(nested_layers[i], nested_level=nested_level)

    for layer in model.layers[layer_range[0] : layer_range[1]]:
        print_layer(layer)

    # Render summary as a table.
    table = text_rendering.Table(
        header=header,
        rows=rows,
        positions=positions,
        # Left align layer name, center-align everything else
        alignments=["left"] + ["center" for _ in range(len(header) - 1)],
    )
    print_fn(table.make())

    # After the table, append information about parameter count and  size.
    if hasattr(model, "_collected_trainable_weights"):
        trainable_count = count_params(model._collected_trainable_weights)
        trainable_memory_size = weight_memory_size(
            model._collected_trainable_weights
        )
    else:
        trainable_count = count_params(model.trainable_weights)
        trainable_memory_size = weight_memory_size(model.trainable_weights)

    non_trainable_count = count_params(model.non_trainable_weights)
    non_trainable_memory_size = weight_memory_size(model.non_trainable_weights)

    total_memory_size = trainable_memory_size + non_trainable_memory_size

    print_fn(
        f"Total params: {trainable_count + non_trainable_count} "
        f"({readable_memory_size(total_memory_size)})"
    )
    print_fn(
        f"Trainable params: {trainable_count} "
        f"({readable_memory_size(trainable_memory_size)})"
    )
    print_fn(
        f"Non-trainable params: {non_trainable_count} "
        f"({readable_memory_size(non_trainable_memory_size)})"
    )


def get_layer_index_bound_by_layer_name(model, layer_range=None):
    """Get the layer indexes from the model based on layer names.

    The layer indexes can be used to slice the model into sub models for
    display.

    Args:
        model: `Model` instance.
        layer_names: a list or tuple of 2 strings, the starting layer name and
            ending layer name (both inclusive) for the result. All layers will
            be included when `None` is provided.

    Returns:
        The index value of layer based on its unique name (layer_names).
        Output will be [first_layer_index, last_layer_index + 1].
    """
    if layer_range is not None:
        if len(layer_range) != 2:
            raise ValueError(
                "layer_range must be a list or tuple of length 2. Received: "
                f"layer_range = {layer_range} of length {len(layer_range)}"
            )
        if not isinstance(layer_range[0], str) or not isinstance(
            layer_range[1], str
        ):
            raise ValueError(
                "layer_range should contain string type only. "
                f"Received: {layer_range}"
            )
    else:
        return [0, len(model.layers)]

    lower_index = [
        idx
        for idx, layer in enumerate(model.layers)
        if re.match(layer_range[0], layer.name)
    ]
    upper_index = [
        idx
        for idx, layer in enumerate(model.layers)
        if re.match(layer_range[1], layer.name)
    ]

    if not lower_index or not upper_index:
        raise ValueError(
            "Passed layer_names do not match the layer names in the model. "
            f"Received: {layer_range}"
        )

    if min(lower_index) > max(upper_index):
        return [min(upper_index), max(lower_index) + 1]
    return [min(lower_index), max(upper_index) + 1]
