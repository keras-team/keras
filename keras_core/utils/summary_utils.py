import math
import re
import shutil

import rich
from tensorflow import nest

from keras_core import backend
from keras_core.utils import dtype_utils
from keras_core.utils import io_utils


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
    unique_weights_ids = set(id(w) for w in weights)
    unique_weights = [w for w in weights if id(w) in unique_weights_ids]
    total_memory_size = 0
    for w in unique_weights:
        weight_shape = math.prod(w.shape)
        dtype = backend.standardize_dtype(w.dtype)
        per_param_size = dtype_utils.dtype_size(dtype)
        total_memory_size += weight_shape * per_param_size
    return total_memory_size


def readable_memory_size(weight_memory_size):
    """Convert the weight memory size (Bytes) to a readable string."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    scale = 1024
    for unit in units:
        if weight_memory_size / scale < 1:
            return "{:.2f} {}".format(weight_memory_size, unit)
        else:
            weight_memory_size /= scale
    return "{:.2f} {}".format(weight_memory_size, units[-1])


def highlight_number(x):
    """Themes numbers in a summary using rich markup.

    We use a separate color for `None`s, e.g. in a layer shape.
    """
    if x is None:
        return f"[color(45)]{x}[/]"
    else:
        return f"[color(34)]{x}[/]"


def highlight_symbol(x):
    """Themes keras symbols in a summary using rich markup."""
    return f"[color(33)]{x}[/]"


def bold_text(x):
    """Bolds text using rich markup."""
    return f"[bold]{x}[/]"


def format_layer_shape(layer):
    if not layer._inbound_nodes:
        return "?"

    def format_shape(shape):
        highlighted = [highlight_number(x) for x in shape]
        return "(" + ", ".join(highlighted) + ")"

    for i in range(len(layer._inbound_nodes)):
        outputs = layer._inbound_nodes[i].output_tensors
        output_shapes = nest.map_structure(
            lambda x: format_shape(x.shape), outputs
        )
    if len(output_shapes) == 1:
        output_shapes = output_shapes[0]
    return str(output_shapes)


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
    from keras_core.models import Functional
    from keras_core.models import Sequential

    if not print_fn and not io_utils.is_interactive_logging_enabled():
        print_fn = io_utils.print_msg

    if isinstance(model, Sequential):
        sequential_like = True
        layers = model.layers
    elif not isinstance(model, Functional):
        # We treat subclassed models as a simple sequence of layers, for logging
        # purposes.
        sequential_like = True
        layers = model.layers
    else:
        layers = model._operations
        sequential_like = True
        nodes_by_depth = model._nodes_by_depth.values()
        nodes = []
        for v in nodes_by_depth:
            if (len(v) > 1) or (
                len(v) == 1 and len(nest.flatten(v[0].input_tensors)) > 1
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
        line_length = line_length or 84
        positions = positions or [0.45, 0.84, 1.0]
        # header names for the different log elements
        header = ["Layer (type)", "Output Shape", "Param #"]
    else:
        line_length = line_length or 108
        positions = positions or [0.3, 0.56, 0.70, 1.0]
        # header names for the different log elements
        header = ["Layer (type)", "Output Shape", "Param #", "Connected to"]
        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v

    if show_trainable:
        line_length += 8
        positions = [p * 0.86 for p in positions] + [1.0]
        header.append("Trainable")

    # Compute columns widths
    line_length = min(line_length, shutil.get_terminal_size().columns - 4)
    column_widths = []
    current = 0
    for pos in positions:
        width = int(pos * line_length) - current
        if width < 4:
            raise ValueError("Insufficient console width to print summary.")
        column_widths.append(width)
        current += width

    # Render summary as a rich table.
    columns = []
    # Right align parameter counts.
    alignment = ["left", "left", "right", "left", "left"]
    for i, name in enumerate(header):
        column = rich.table.Column(
            name,
            justify=alignment[i],
            width=column_widths[i],
        )
        columns.append(column)
    table = rich.table.Table(*columns, width=line_length, show_lines=True)

    def get_layer_fields(layer, prefix=""):
        output_shape = format_layer_shape(layer)
        name = prefix + layer.name
        cls_name = layer.__class__.__name__
        name = rich.markup.escape(name)
        name += f" ({highlight_symbol(rich.markup.escape(cls_name))})"

        if not hasattr(layer, "built"):
            params = highlight_number(0)
        elif not layer.built:
            params = highlight_number(0) + " (unbuilt)"
        else:
            params = highlight_number(f"{layer.count_params():,}")

        fields = [name, output_shape, params]
        if show_trainable:
            fields.append("Y" if layer.trainable else "N")
        return fields

    def get_connections(layer):
        connections = ""
        for node in layer._inbound_nodes:
            if relevant_nodes and node not in relevant_nodes:
                # node is not part of the current network
                continue
            for kt in node.input_tensors:
                keras_history = kt._keras_history
                inbound_layer = keras_history.operation
                node_index = highlight_number(keras_history.node_index)
                tensor_index = highlight_number(keras_history.tensor_index)
                if connections:
                    connections += ", "
                connections += (
                    f"{inbound_layer.name}[{node_index}][{tensor_index}]"
                )
        if not connections:
            connections = "-"
        return connections

    def print_layer(layer, nested_level=0):
        if nested_level:
            prefix = "   " * nested_level + "└" + " "
        else:
            prefix = ""

        fields = get_layer_fields(layer, prefix=prefix)
        if not sequential_like:
            fields.append(get_connections(layer))
        if show_trainable:
            fields.append("Y" if layer.trainable else "N")

        rows = [fields]
        if expand_nested and hasattr(layer, "layers") and layer.layers:
            nested_layers = layer.layers
            nested_level += 1
            for i in range(len(nested_layers)):
                rows.extend(
                    print_layer(nested_layers[i], nested_level=nested_level)
                )
        return rows

    # Render all layers to the rich table.
    layer_range = get_layer_index_bound_by_layer_name(layers, layer_range)
    for layer in layers[layer_range[0] : layer_range[1]]:
        for row in print_layer(layer):
            table.add_row(*row)

    # After the table, append information about parameter count and size.
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

    total_count = trainable_count + non_trainable_count
    total_memory_size = trainable_memory_size + non_trainable_memory_size

    # Create a rich console for printing. Capture for non-interactive logging.
    if print_fn:
        console = rich.console.Console(
            highlight=False, force_terminal=False, color_system=None
        )
        console.begin_capture()
    else:
        console = rich.console.Console(highlight=False)

    # Print the to the console.
    console.print(bold_text(f'Model: "{rich.markup.escape(model.name)}"'))
    console.print(table)
    console.print(
        bold_text(" Total params: ")
        + highlight_number(f"{total_count:,}")
        + f" ({readable_memory_size(total_memory_size)})"
    )
    console.print(
        bold_text(" Trainable params: ")
        + highlight_number(f"{trainable_count:,}")
        + f" ({readable_memory_size(trainable_memory_size)})"
    )
    console.print(
        bold_text(" Non-trainable params: ")
        + highlight_number(f"{non_trainable_count:,}")
        + f" ({readable_memory_size(non_trainable_memory_size)})"
    )

    # Output captured summary for non-interactive logging.
    if print_fn:
        print_fn(console.end_capture(), line_break=False)


def get_layer_index_bound_by_layer_name(layers, layer_range=None):
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
        return [0, len(layers)]

    lower_index = [
        idx
        for idx, layer in enumerate(layers)
        if re.match(layer_range[0], layer.name)
    ]
    upper_index = [
        idx
        for idx, layer in enumerate(layers)
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
