from __future__ import print_function
import numpy as np
import theano

def print_graph_layer_shapes(graph, input_shapes):
    """
    Utility function to print the shape of the output at each layer of a Graph

    Arguments:
        graph: An instance of models.Graph
        input_shapes: A dict that gives a shape for each input to the Graph
    """
    input_vars = [graph.inputs[name].input
                  for name in graph.input_order]
    output_vars = [graph.outputs[name].get_output()
                   for name in graph.output_order]
    input_dummy = [np.zeros(input_shapes[name], dtype=np.float32)
                   for name in graph.input_order]

    print("input shapes : ", input_shapes)
    for name, l in graph.nodes.items():
        shape_f = theano.function(input_vars,
                                  l.get_output(train=False).shape,
                                  on_unused_input='ignore')
        out_shape = shape_f(*input_dummy)
        print('shape after', l.get_config()['name'], "(", name, ") :", out_shape)

def print_model_layer_shapes(model, input_shapes):
    """
    Utility function that prints the shape of the output at each layer.

    Arguments:
        model: An instance of models.Model
        input_shape: The shape of the input you will provide to the model.
                     Either a tuple (for a single input) or a list of tuple
    """
    # This is to handle the case where a model has been connected to a previous
    # layer (and therefore get_input would recurse into previous layer's
    # output).
    if hasattr(model.layers[0], 'previous'):
        # TODO: If the model is used as a part of another model, get_input will
        # return the input of the whole model and this won't work. So this is
        # not handled yet
        raise Exception("This function doesn't work on model used as subparts "
                        " for other models")

    # We allow the shortcut input_shapes=(1, 1, 28) instead of
    # input_shapes=[(1, 1, 28)].
    if not isinstance(input_shapes[0], tuple):
        input_shapes = [input_shapes]
    input_vars = model.get_input(train=False)
    # theano.function excepts a list of variables
    if not isinstance(input_vars, list):
        input_vars = [input_vars]
    input_dummy = [np.zeros(shape, dtype=np.float32)
                   for shape in input_shapes]

    print("input shapes : ", input_shapes)
    for l in model.layers:
        shape_f = theano.function(input_vars,
                                  l.get_output(train=False).shape)
        out_shape = shape_f(*input_dummy)
        print('shape after', l.get_config()['name'], ":", out_shape)
