"""Utilities related to model visualization."""

import os
import sys

from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.utils import io_utils

try:
    import pydot
except ImportError:
    # pydot_ng and pydotplus are older forks of pydot
    # which may still be used by some users
    try:
        import pydot_ng as pydot
    except ImportError:
        try:
            import pydotplus as pydot
        except ImportError:
            pydot = None


def check_pydot():
    """Returns True if PyDot is available."""
    return pydot is not None


def check_graphviz():
    """Returns True if both PyDot and Graphviz are available."""
    if not check_pydot():
        return False
    try:
        # Attempt to create an image of a blank graph
        # to check the pydot/graphviz installation.
        pydot.Dot.create(pydot.Dot())
        return True
    except (OSError, pydot.PydotException):
        return False


def add_edge(dot, src, dst):
    src_id = str(id(src))
    dst_id = str(id(dst))
    if not dot.get_edge(src_id, dst_id):
        edge = pydot.Edge(src_id, dst_id)
        edge.set("penwidth", "2")
        dot.add_edge(edge)


def get_layer_activation_name(layer):
    if hasattr(layer.activation, "name"):
        activation_name = layer.activation.name
    elif hasattr(layer.activation, "__name__"):
        activation_name = layer.activation.__name__
    else:
        activation_name = str(layer.activation)
    return activation_name


def make_layer_label(layer, **kwargs):
    class_name = layer.__class__.__name__

    show_layer_names = kwargs.pop("show_layer_names")
    show_layer_activations = kwargs.pop("show_layer_activations")
    show_dtype = kwargs.pop("show_dtype")
    show_shapes = kwargs.pop("show_shapes")
    show_trainable = kwargs.pop("show_trainable")
    if kwargs:
        raise ValueError(f"Invalid kwargs: {kwargs}")

    table = (
        '<<table border="0" cellborder="1" bgcolor="black" cellpadding="10">'
    )

    colspan_max = sum(int(x) for x in (show_dtype, show_trainable))
    if show_shapes:
        colspan_max += 2
    colspan = max(1, colspan_max)

    if show_layer_names:
        table += (
            f'<tr><td colspan="{colspan}" bgcolor="black">'
            '<font point-size="16" color="white">'
            f"<b>{layer.name}</b> ({class_name})"
            "</font></td></tr>"
        )
    else:
        table += (
            f'<tr><td colspan="{colspan}" bgcolor="black">'
            '<font point-size="16" color="white">'
            f"<b>{class_name}</b>"
            "</font></td></tr>"
        )
    if (
        show_layer_activations
        and hasattr(layer, "activation")
        and layer.activation is not None
    ):
        table += (
            f'<tr><td bgcolor="white" colspan="{colspan}">'
            '<font point-size="14">'
            f"Activation: <b>{get_layer_activation_name(layer)}</b>"
            "</font></td></tr>"
        )

    cols = []
    if show_shapes:
        input_shape = None
        output_shape = None
        try:
            input_shape = tree.map_structure(lambda x: x.shape, layer.input)
            output_shape = tree.map_structure(lambda x: x.shape, layer.output)
        except (ValueError, AttributeError):
            pass

        def format_shape(shape):
            if shape is not None:
                if isinstance(shape, dict):
                    shape_str = ", ".join(
                        [f"{k}: {v}" for k, v in shape.items()]
                    )
                else:
                    shape_str = f"{shape}"
                shape_str = shape_str.replace("}", "").replace("{", "")
            else:
                shape_str = "?"
            return shape_str

        if class_name != "InputLayer":
            cols.append(
                (
                    '<td bgcolor="white"><font point-size="14">'
                    f"Input shape: <b>{format_shape(input_shape)}</b>"
                    "</font></td>"
                )
            )
        cols.append(
            (
                '<td bgcolor="white"><font point-size="14">'
                f"Output shape: <b>{format_shape(output_shape)}</b>"
                "</font></td>"
            )
        )
    if show_dtype:
        dtype = None
        try:
            dtype = tree.map_structure(lambda x: x.dtype, layer.output)
        except (ValueError, AttributeError):
            pass
        cols.append(
            (
                '<td bgcolor="white"><font point-size="14">'
                f"Output dtype: <b>{dtype or '?'}</b>"
                "</font></td>"
            )
        )
    if show_trainable and hasattr(layer, "trainable") and layer.weights:
        if layer.trainable:
            cols.append(
                (
                    '<td bgcolor="forestgreen">'
                    '<font point-size="14" color="white">'
                    "<b>Trainable</b></font></td>"
                )
            )
        else:
            cols.append(
                (
                    '<td bgcolor="firebrick">'
                    '<font point-size="14" color="white">'
                    "<b>Non-trainable</b></font></td>"
                )
            )
    if cols:
        colspan = len(cols)
    else:
        colspan = 1

    if cols:
        table += f"<tr>{''.join(cols)}</tr>"
    table += "</table>>"
    return table


def make_node(layer, **kwargs):
    node = pydot.Node(str(id(layer)), label=make_layer_label(layer, **kwargs))
    node.set("fontname", "Helvetica")
    node.set("border", "0")
    node.set("margin", "0")
    return node


@keras_export("keras.utils.model_to_dot")
def model_to_dot(
    model,
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    subgraph=False,
    show_layer_activations=False,
    show_trainable=False,
    **kwargs,
):
    """Convert a Keras model to dot format.

    Args:
        model: A Keras model instance.
        show_shapes: whether to display shape information.
        show_dtype: whether to display layer dtypes.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot: `"TB"`
            creates a vertical plot; `"LR"` creates a horizontal plot.
        expand_nested: whether to expand nested Functional models
            into clusters.
        dpi: Image resolution in dots per inch.
        subgraph: whether to return a `pydot.Cluster` instance.
        show_layer_activations: Display layer activations (only for layers that
            have an `activation` property).
        show_trainable: whether to display if a layer is trainable.

    Returns:
        A `pydot.Dot` instance representing the Keras model or
        a `pydot.Cluster` instance representing nested model if
        `subgraph=True`.
    """
    from keras.src.ops.function import make_node_key

    if not model.built:
        raise ValueError(
            "This model has not yet been built. "
            "Build the model first by calling `build()` or by calling "
            "the model on a batch of data."
        )

    from keras.src.models import functional
    from keras.src.models import sequential

    # from keras.src.layers import Wrapper

    if not check_pydot():
        raise ImportError(
            "You must install pydot (`pip install pydot`) for "
            "model_to_dot to work."
        )

    if subgraph:
        dot = pydot.Cluster(style="dashed", graph_name=model.name)
        dot.set("label", model.name)
        dot.set("labeljust", "l")
    else:
        dot = pydot.Dot()
        dot.set("rankdir", rankdir)
        dot.set("concentrate", True)
        dot.set("dpi", dpi)
        dot.set("splines", "ortho")
        dot.set_node_defaults(shape="record")

    if kwargs.pop("layer_range", None) is not None:
        raise ValueError("Argument `layer_range` is no longer supported.")
    if kwargs:
        raise ValueError(f"Unrecognized keyword arguments: {kwargs}")

    kwargs = {
        "show_layer_names": show_layer_names,
        "show_layer_activations": show_layer_activations,
        "show_dtype": show_dtype,
        "show_shapes": show_shapes,
        "show_trainable": show_trainable,
    }

    if isinstance(model, sequential.Sequential):
        layers = model.layers
    elif not isinstance(model, functional.Functional):
        # We treat subclassed models as a single node.
        node = make_node(model, **kwargs)
        dot.add_node(node)
        return dot
    else:
        layers = model._operations

    # Create graph nodes.
    for i, layer in enumerate(layers):
        # Process nested functional and sequential models.
        if expand_nested and isinstance(
            layer, (functional.Functional, sequential.Sequential)
        ):
            submodel = model_to_dot(
                layer,
                show_shapes,
                show_dtype,
                show_layer_names,
                rankdir,
                expand_nested,
                subgraph=True,
                show_layer_activations=show_layer_activations,
                show_trainable=show_trainable,
            )
            dot.add_subgraph(submodel)

        else:
            node = make_node(layer, **kwargs)
            dot.add_node(node)

    # Connect nodes with edges.
    if isinstance(model, sequential.Sequential):
        if not expand_nested:
            # Single Sequential case.
            for i in range(len(layers) - 1):
                add_edge(dot, layers[i], layers[i + 1])
            return dot
        else:
            # The first layer is connected to the `InputLayer`, which is not
            # represented for Sequential models, so we skip it. What will draw
            # the incoming edge from outside of the sequential model is the
            # edge connecting the Sequential model itself.
            layers = model.layers[1:]

    # Functional and nested Sequential case.
    for layer in layers:
        # Go from current layer to input `Node`s.
        for inbound_index, inbound_node in enumerate(layer._inbound_nodes):
            # `inbound_node` is a `Node`.
            if (
                isinstance(model, functional.Functional)
                and make_node_key(layer, inbound_index) not in model._nodes
            ):
                continue

            # Go from input `Node` to `KerasTensor` representing that input.
            for input_index, input_tensor in enumerate(
                inbound_node.input_tensors
            ):
                # `input_tensor` is a `KerasTensor`.
                # `input_history` is a `KerasHistory`.
                input_history = input_tensor._keras_history
                if input_history.operation is None:
                    # Operation is `None` for `Input` tensors.
                    continue

                # Go from input `KerasTensor` to the `Operation` that produced
                # it as an output.
                input_node = input_history.operation._inbound_nodes[
                    input_history.node_index
                ]
                output_index = input_history.tensor_index

                # Tentative source and destination of the edge.
                source = input_node.operation
                destination = layer

                if not expand_nested:
                    # No nesting, connect directly.
                    add_edge(dot, source, layer)
                    continue

                # ==== Potentially nested models case ====

                # ---- Resolve the source of the edge ----
                while isinstance(
                    source,
                    (functional.Functional, sequential.Sequential),
                ):
                    # When `source` is a `Functional` or `Sequential` model, we
                    # need to connect to the correct box within that model.
                    # Functional and sequential models do not have explicit
                    # "output" boxes, so we need to find the correct layer that
                    # produces the output we're connecting to, which can be
                    # nested several levels deep in sub-models. Hence the while
                    # loop to continue going into nested models until we
                    # encounter a real layer that's not a `Functional` or
                    # `Sequential`.
                    source, _, output_index = source.outputs[
                        output_index
                    ]._keras_history

                # ---- Resolve the destination of the edge ----
                while isinstance(
                    destination,
                    (functional.Functional, sequential.Sequential),
                ):
                    if isinstance(destination, functional.Functional):
                        # When `destination` is a `Functional`, we point to the
                        # specific `InputLayer` in the model.
                        destination = destination.inputs[
                            input_index
                        ]._keras_history.operation
                    else:
                        # When `destination` is a `Sequential`, there is no
                        # explicit "input" box, so we want to point to the first
                        # box in the model, but it may itself be another model.
                        # Hence the while loop to continue going into nested
                        # models until we encounter a real layer that's not a
                        # `Functional` or `Sequential`.
                        destination = destination.layers[0]

                add_edge(dot, source, destination)
    return dot


@keras_export("keras.utils.plot_model")
def plot_model(
    model,
    to_file="model.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=False,
    rankdir="TB",
    expand_nested=False,
    dpi=200,
    show_layer_activations=False,
    show_trainable=False,
    **kwargs,
):
    """Converts a Keras model to dot format and save to a file.

    Example:

    ```python
    inputs = ...
    outputs = ...
    model = keras.Model(inputs=inputs, outputs=outputs)

    dot_img_file = '/tmp/model_1.png'
    keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
    ```

    Args:
        model: A Keras model instance
        to_file: File name of the plot image.
        show_shapes: whether to display shape information.
        show_dtype: whether to display layer dtypes.
        show_layer_names: whether to display layer names.
        rankdir: `rankdir` argument passed to PyDot,
            a string specifying the format of the plot: `"TB"`
            creates a vertical plot; `"LR"` creates a horizontal plot.
        expand_nested: whether to expand nested Functional models
            into clusters.
        dpi: Image resolution in dots per inch.
        show_layer_activations: Display layer activations (only for layers that
            have an `activation` property).
        show_trainable: whether to display if a layer is trainable.

    Returns:
        A Jupyter notebook Image object if Jupyter is installed.
        This enables in-line display of the model plots in notebooks.
    """

    if not model.built:
        raise ValueError(
            "This model has not yet been built. "
            "Build the model first by calling `build()` or by calling "
            "the model on a batch of data."
        )
    if not check_pydot():
        message = (
            "You must install pydot (`pip install pydot`) "
            "for `plot_model` to work."
        )
        if "IPython.core.magics.namespace" in sys.modules:
            # We don't raise an exception here in order to avoid crashing
            # notebook tests where graphviz is not available.
            io_utils.print_msg(message)
            return
        else:
            raise ImportError(message)
    if not check_graphviz():
        message = (
            "You must install graphviz "
            "(see instructions at https://graphviz.gitlab.io/download/) "
            "for `plot_model` to work."
        )
        if "IPython.core.magics.namespace" in sys.modules:
            # We don't raise an exception here in order to avoid crashing
            # notebook tests where graphviz is not available.
            io_utils.print_msg(message)
            return
        else:
            raise ImportError(message)

    if kwargs.pop("layer_range", None) is not None:
        raise ValueError("Argument `layer_range` is no longer supported.")
    if kwargs:
        raise ValueError(f"Unrecognized keyword arguments: {kwargs}")

    dot = model_to_dot(
        model,
        show_shapes=show_shapes,
        show_dtype=show_dtype,
        show_layer_names=show_layer_names,
        rankdir=rankdir,
        expand_nested=expand_nested,
        dpi=dpi,
        show_layer_activations=show_layer_activations,
        show_trainable=show_trainable,
    )
    to_file = str(to_file)
    if dot is None:
        return
    _, extension = os.path.splitext(to_file)
    if not extension:
        extension = "png"
    else:
        extension = extension[1:]
    # Save image to disk.
    dot.write(to_file, format=extension)
    # Return the image as a Jupyter Image object, to be displayed in-line.
    # Note that we cannot easily detect whether the code is running in a
    # notebook, and thus we always return the Image if Jupyter is available.
    if extension != "pdf":
        try:
            from IPython import display

            return display.Image(filename=to_file)
        except ImportError:
            pass
