


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
    if print_fn is None:
        print_fn = io_utils.print_msg

    if model.__class__.__name__ == "Sequential":
        sequential_like = True
    elif not model._is_graph_network:
        # We treat subclassed models as a simple sequence of layers, for logging
        # purposes.
        sequential_like = True
    else:
        sequential_like = True
        nodes_by_depth = model._nodes_by_depth.values()
        nodes = []
        for v in nodes_by_depth:
            if (len(v) > 1) or (
                len(v) == 1 and len(tf.nest.flatten(v[0].keras_inputs)) > 1
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
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ["Layer (type)", "Output Shape", "Param #"]
    else:
        line_length = line_length or 98
        positions = positions or [0.3, 0.6, 0.70, 1.0]
        if positions[-1] <= 1:
            positions = [int(line_length * p) for p in positions]
        # header names for the different log elements
        to_display = ["Layer (type)", "Output Shape", "Param #", "Connected to"]
        relevant_nodes = []
        for v in model._nodes_by_depth.values():
            relevant_nodes += v

    if show_trainable:
        line_length += 11
        positions.append(line_length)
        to_display.append("Trainable")

    layer_range = get_layer_index_bound_by_layer_name(model, layer_range)

    def print_row(fields, positions, nested_level=0):
        left_to_print = [str(x) for x in fields]
        while any(left_to_print):
            line = ""
            for col in range(len(left_to_print)):
                if col > 0:
                    start_pos = positions[col - 1]
                else:
                    start_pos = 0
                end_pos = positions[col]
                # Leave room for 2 spaces to delineate columns
                # we don't need any if we are printing the last column
                space = 2 if col != len(positions) - 1 else 0
                cutoff = end_pos - start_pos - space
                # Except for last col, offset by one to align the start of col
                if col != len(positions) - 1:
                    cutoff -= 1
                if col == 0:
                    cutoff -= nested_level
                fit_into_line = left_to_print[col][:cutoff]
                # For nicer formatting we line-break on seeing end of
                # tuple/dict etc.
                line_break_conditions = ("),", "},", "],", "',")
                candidate_cutoffs = [
                    fit_into_line.find(x) + len(x)
                    for x in line_break_conditions
                    if fit_into_line.find(x) >= 0
                ]
                if candidate_cutoffs:
                    cutoff = min(candidate_cutoffs)
                    fit_into_line = fit_into_line[:cutoff]

                if col == 0:
                    line += "|" * nested_level + " "
                line += fit_into_line
                line += " " * space if space else ""
                left_to_print[col] = left_to_print[col][cutoff:]

                # Pad out to the next position
                # Make space for nested_level for last column
                if nested_level and col == len(positions) - 1:
                    line += " " * (positions[col] - len(line) - nested_level)
                else:
                    line += " " * (positions[col] - len(line))
            line += "|" * nested_level
            print_fn(line)

    print_fn(f'Model: "{model.name}"')
    print_fn("_" * line_length)
    print_row(to_display, positions)
    print_fn("=" * line_length)

    def print_layer_summary(layer, nested_level=0):
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
        name = layer.name
        cls_name = layer.__class__.__name__
        if not layer.built and not getattr(layer, "_is_graph_network", False):
            # If a subclassed model has a layer that is not called in
            # Model.call, the layer will not be built and we cannot call
            # layer.count_params().
            params = "0 (unused)"
        else:
            params = layer.count_params()
        fields = [name + " (" + cls_name + ")", output_shape, params]

        if show_trainable:
            fields.append("Y" if layer.trainable else "N")

        print_row(fields, positions, nested_level)

    def print_layer_summary_with_connections(layer, nested_level=0):
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

            for (
                inbound_layer,
                node_index,
                tensor_index,
                _,
            ) in node.iterate_inbound():
                connections.append(
                    f"{inbound_layer.name}[{node_index}][{tensor_index}]"
                )

        name = layer.name
        cls_name = layer.__class__.__name__
        fields = [
            name + " (" + cls_name + ")",
            output_shape,
            layer.count_params(),
            connections,
        ]

        if show_trainable:
            fields.append("Y" if layer.trainable else "N")

        print_row(fields, positions, nested_level)

    def print_layer(layer, nested_level=0, is_nested_last=False):
        if sequential_like:
            print_layer_summary(layer, nested_level)
        else:
            print_layer_summary_with_connections(layer, nested_level)

        if expand_nested and hasattr(layer, "layers") and layer.layers:
            print_fn(
                "|" * (nested_level + 1)
                + "¯" * (line_length - 2 * nested_level - 2)
                + "|" * (nested_level + 1)
            )

            nested_layer = layer.layers
            is_nested_last = False
            for i in range(len(nested_layer)):
                if i == len(nested_layer) - 1:
                    is_nested_last = True
                print_layer(nested_layer[i], nested_level + 1, is_nested_last)

            print_fn(
                "|" * nested_level
                + "¯" * (line_length - 2 * nested_level)
                + "|" * nested_level
            )

        if not is_nested_last:
            print_fn(
                "|" * nested_level
                + " " * (line_length - 2 * nested_level)
                + "|" * nested_level
            )

    for layer in model.layers[layer_range[0] : layer_range[1]]:
        print_layer(layer)
    print_fn("=" * line_length)

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
    print_fn("_" * line_length)

    print_dtensor_variable_summary(model, print_fn, line_length)
