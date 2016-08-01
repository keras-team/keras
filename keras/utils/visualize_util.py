try:
    # pydot-ng is a fork of pydot that is better maintained
    import pydot_ng as pydot
except ImportError:
    # fall back on pydot if necessary
    import pydot
if not pydot.find_graphviz():
    raise RuntimeError('Failed to import pydot. You must install pydot'
                       ' and graphviz for `pydotprint` to work.')


def model_to_dot(model, show_shapes=False, show_layer_names=True,
                 verbose=False):
    """
    # Parameters
    verbose: boolean, optional (default False)
        If true, then more layer parameters will be reported in the
        generated plot.
    ...
    """
    dot = pydot.Dot()
    dot.set('rankdir', 'TB')
    dot.set('concentrate', True)
    dot.set_node_defaults(shape='record')

    if model.__class__.__name__ == 'Sequential':
        if not model.built:
            model.build()
        model = model.model
    layers = model.layers

    # first, populate the nodes of the graph
    for layer in layers:
        layer_id = str(id(layer))
        if show_layer_names:
            label = str(layer.name) + ' (' + layer.__class__.__name__ + ')'
        else:
            label = layer.__class__.__name__

        if show_shapes:
            # Build the label that will actually contain a table with the
            # input/output
            try:
                outputlabels = str(layer.output_shape)
            except:
                outputlabels = 'multiple'
            if hasattr(layer, 'input_shape'):
                inputlabels = str(layer.input_shape)
            elif hasattr(layer, 'input_shapes'):
                inputlabels = ', '.join(
                    [str(ishape) for ishape in layer.input_shapes])
            else:
                inputlabels = 'multiple'

            def _get_val(stuff):
                if hasattr(stuff, "__name__"):
                    return stuff.__name__
                else:
                    return str(stuff)

            # Show more more layer parameters ?
            additional_params = ""
            additional_vals = ""
            if verbose:
                # List of things which could be interesting to the user. This
                # is likeliy to change.
                param_names = ["activation",
                               "strides",
                               "pool_size",
                               "nb_filter",
                               "subsample",
                               "nb_row",
                               "nb_col",
                               "kernel_dim1",
                               "kernel_dim2",
                               "kernel_dim3",
                               "inner_activation"]
                if hasattr(layer, "activation"):
                    if _get_val(layer.activation) == "linear":
                        param_names.remove("activation")
                additional_params = "".join(["|%s:" % param
                                             for param in param_names
                                             if hasattr(layer, param)])
                additional_vals = "".join(["|{%s}" % _get_val(getattr(layer,
                                                                      param))
                                           for param in param_names
                                           if hasattr(layer, param)])
            label = '%s\n|{input:|output:%s}|{{%s}|{%s}%s}' % (
                label, additional_params, inputlabels, outputlabels,
                additional_vals)

        node = pydot.Node(layer_id, label=label)
        dot.add_node(node)

    # second, add the edges
    for layer in layers:
        layer_id = str(id(layer))
        for i, node in enumerate(layer.inbound_nodes):
            node_key = layer.name + '_ib-' + str(i)
            if node_key in model.container_nodes:
                # add edges
                for inbound_layer in node.inbound_layers:
                    inbound_layer_id = str(id(inbound_layer))
                    layer_id = str(id(layer))
                    dot.add_edge(pydot.Edge(inbound_layer_id, layer_id))
    return dot


def plot(model, to_file='model.png', show_shapes=False, show_layer_names=True,
         verbose=False):
    """
    # Parameters
    verbose: boolean, optional (default False)
        If true, then more layer parameters will be reported in the
        generated plot.
    ...
    """
    dot = model_to_dot(model, show_shapes=show_shapes,
                       show_layer_names=show_layer_names, verbose=verbose)
    dot.write_png(to_file)
