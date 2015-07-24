from __future__ import print_function
import numpy as np
import theano


def print_layer_shapes(model, input_shapes):
    """
    Utility function to print the shape of the output at each layer of a Model

    Arguments:
        model: instance of Model / Merge
        input_shapes: dict (Graph), list of tuples (Merge) or tuple (Sequential)
    """
    if model.__class__.__name__ in ['Sequential', 'Merge']:
        # in this case input_shapes is a tuple, or a list [shape1, shape2]
        if not isinstance(input_shapes[0], tuple):
            input_shapes = [input_shapes]

        inputs = model.get_input(train=False)
        if not isinstance(inputs, list):
            inputs = [inputs]
        input_dummy = [np.zeros(shape, dtype=np.float32)
                       for shape in input_shapes]
        layers = model.layers

    elif model.__class__.__name__ == 'Graph':
        # in this case input_shapes is a dictionary
        inputs = [model.inputs[name].input
                  for name in model.input_order]
        input_dummy = [np.zeros(input_shapes[name], dtype=np.float32)
                       for name in model.input_order]
        layers = [model.nodes[c['name']] for c in model.node_config]

    print("input shapes : ", input_shapes)
    for l in layers:
        shape_f = theano.function(inputs, l.get_output(train=False).shape,
                                  on_unused_input='ignore')
        out_shape = tuple(shape_f(*input_dummy))
        config = l.get_config()
        print('shape after %s: %s' % (config['name'], out_shape))
