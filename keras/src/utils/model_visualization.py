"""Utilities related to model visualization."""

import os
import sys

from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.utils import io_utils

try:
    # pydot-ng is a fork of pydot that is better maintained.
    import pydot_ng as pydot
except ImportError:
    # pydotplus is an improved version of pydot
    try:
        import pydotplus as pydot
    except ImportError:
        # Fall back on pydot if necessary.
        try:
            import pydot
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
    except (OSError, pydot.InvocationException):
        return False


def add_edge(dot, src, dst):
    if not dot.get_edge(src, dst):
        edge = pydot.Edge(src, dst)
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
                f'Output dtype: <b>{dtype or "?"}</b>'
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
        table += "<tr>" + "".join(cols) + "</tr>"
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
    sub_n_first_node = {}
    sub_n_last_node = {}
    for i, layer in enumerate(layers):
        # Process nested functional models.
        if expand_nested and isinstance(layer, functional.Functional):
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
            # sub_n : submodel
            sub_n_nodes = submodel.get_nodes()
            sub_n_first_node[layer.name] = sub_n_nodes[0]
            sub_n_last_node[layer.name] = sub_n_nodes[-1]
            dot.add_subgraph(submodel)

        else:
            node = make_node(layer, **kwargs)
            dot.add_node(node)

    # Connect nodes with edges.
    # Sequential case.
    if isinstance(model, sequential.Sequential):
        for i in range(len(layers) - 1):
            inbound_layer_id = str(id(layers[i]))
            layer_id = str(id(layers[i + 1]))
            add_edge(dot, inbound_layer_id, layer_id)
        return dot

    # Functional case.
    for i, layer in enumerate(layers):
        layer_id = str(id(layer))
        for i, node in enumerate(layer._inbound_nodes):
            node_key = make_node_key(layer, i)
            if node_key in model._nodes:
                for parent_node in node.parent_nodes:
                    inbound_layer = parent_node.operation
                    inbound_layer_id = str(id(inbound_layer))
                    if not expand_nested:
                        assert dot.get_node(inbound_layer_id)
                        assert dot.get_node(layer_id)
                        add_edge(dot, inbound_layer_id, layer_id)
                    else:
                        # if inbound_layer is not Functional
                        if not isinstance(inbound_layer, functional.Functional):
                            # if current layer is not Functional
                            if not isinstance(layer, functional.Functional):
                                assert dot.get_node(inbound_layer_id)
                                assert dot.get_node(layer_id)
                                add_edge(dot, inbound_layer_id, layer_id)
                            # if current layer is Functional
                            elif isinstance(layer, functional.Functional):
                                add_edge(
                                    dot,
                                    inbound_layer_id,
                                    sub_n_first_node[layer.name].get_name(),
                                )
                        # if inbound_layer is Functional
                        elif isinstance(inbound_layer, functional.Functional):
                            name = sub_n_last_node[
                                inbound_layer.name
                            ].get_name()
                            if isinstance(layer, functional.Functional):
                                output_name = sub_n_first_node[
                                    layer.name
                                ].get_name()
                                add_edge(dot, name, output_name)
                            else:
                                add_edge(dot, name, layer_id)
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
