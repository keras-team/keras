import itertools
from keras.layers.containers import Graph, Sequential
from keras.layers.core import Merge

try:
    # pydot-ng is a fork of pydot that is better maintained
    import pydot_ng as pydot
except ImportError:
    # fall back on pydot if necessary
    import pydot
if not pydot.find_graphviz():
    raise RuntimeError("Failed to import pydot. You must install pydot"
                       " and graphviz for `pydotprint` to work.")


def layer_typename(layer):
    return type(layer).__module__ + "." + type(layer).__name__


def get_layer_to_name(model):
    """Returns a dict mapping layer to their name in the model"""
    if not isinstance(model, Graph):
        return {}
    else:
        node_to_name = itertools.chain(
            model.nodes.items(), model.inputs.items(), model.outputs.items()
        )
        return {v: k for k, v in node_to_name}


class ModelToDot(object):
    """
    This is a helper class which visits a keras model (Sequential or Graph) and
    returns a pydot.Graph representation.

    This is implemented as a class because we need to maintain various states.

    Use it as ```ModelToDot()(model)```

    Keras models can have an arbitrary number of inputs and outputs. A given
    layer can have multiple inputs but has a single output. We therefore
    explore the model by starting at its output and crawling "up" the tree.
    """
    def _pydot_node_for_layer(self, layer, label):
        """
        Returns the pydot.Node corresponding to the given layer.
        `label` specify the name of the layer (only used if the layer isn't yet
            associated with a pydot.Node)
        """
        # Check if this already exists (will be the case for nodes that
        # serve as input to more than one layer)
        if layer in self.layer_to_pydotnode:
            node = self.layer_to_pydotnode[layer]
        else:
            layer_id = 'layer%d' % self.idgen
            self.idgen += 1

            label = label + " (" + layer_typename(layer) + ")"

            if self.show_shape:
                # Build the label that will actually contain a table with the
                # input/output
                outputlabels = str(layer.output_shape)
                if hasattr(layer, 'input_shape'):
                    inputlabels = str(layer.input_shape)
                elif hasattr(layer, 'input_shapes'):
                    inputlabels = ', '.join(
                        [str(ishape) for ishape in layer.input_shapes])
                else:
                    inputlabels = ''
                label = "%s\n|{input:|output:}|{{%s}|{%s}}" % (
                        label, inputlabels, outputlabels)

            node = pydot.Node(layer_id, label=label)
            self.g.add_node(node)
            self.layer_to_pydotnode[layer] = node
        return node

    def _process_layer(self, layer, layer_to_name=None, connect_to=None):
        """
        Process a layer, adding its node to the graph and creating edges to its
        outputs.

        `connect_to` specify where the output of the current layer will be
            connected
        `layer_to_name` is a dict mapping layer to their name in the Graph
            model. Should be {} when processing a Sequential model
        """
        # The layer can be a container layer, in which case we can recurse
        is_graph = isinstance(layer, Graph)
        is_seq = isinstance(layer, Sequential)
        if self.recursive and (is_graph or is_seq):
            # We got a container layer, recursively transform it
            if is_graph:
                child_layers = layer.outputs.values()
            else:
                child_layers = [layer.layers[-1]]
            for l in child_layers:
                self._process_layer(l, layer_to_name=get_layer_to_name(layer),
                                    connect_to=connect_to)
        else:
            # This is a simple layer.
            label = layer_to_name.get(layer, '')
            layer_node = self._pydot_node_for_layer(layer, label=label)

            if connect_to is not None:
                self.g.add_edge(pydot.Edge(layer_node, connect_to))

            # Proceed upwards to the parent(s). Only Merge layers have more
            # than one parent
            if isinstance(layer, Merge):  # Merge layer
                for l in layer.layers:
                    self._process_layer(l, layer_to_name,
                                        connect_to=layer_node)
            elif hasattr(layer, 'previous') and layer.previous is not None:
                self._process_layer(layer.previous, layer_to_name,
                                    connect_to=layer_node)

    def __call__(self, model, recursive=True, show_shape=False,
                 connect_to=None):
        self.idgen = 0
        # Maps keras layer to the pydot.Node representing them
        self.layer_to_pydotnode = {}
        self.recursive = recursive
        self.show_shape = show_shape

        self.g = pydot.Dot()
        self.g.set('rankdir', 'TB')
        self.g.set('concentrate', True)
        self.g.set_node_defaults(shape='record')

        if hasattr(model, 'outputs'):
            # Graph
            for name, l in model.outputs.items():
                self._process_layer(l, get_layer_to_name(model),
                                    connect_to=connect_to)
        else:
            # Sequential container
            self._process_layer(model.layers[-1], {}, connect_to=connect_to)

        return self.g


def to_graph(model, **kwargs):
    """
    `recursive` controls whether we recursively explore container layers
    `show_shape` controls whether the shape is shown in the graph
    """
    return ModelToDot()(model, **kwargs)


def plot(model, to_file='model.png'):
    graph = to_graph(model)
    graph.write_png(to_file)
