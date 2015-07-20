from __future__ import print_function
import numpy as np
import theano


def print_layer_shapes(model, input_shape):
    """
    Utility function that prints the shape of the output at each layer.

    Arguments:
        model: An instance of models.Model
        input_shape: The shape of the input you will provide to the model.
    """
    input_var = model.get_input(train=False)
    input_tmp = np.zeros(input_shape, dtype=np.float32)
    print("input shape : ", input_shape)
    for l in model.layers:
        shape_f = theano.function([input_var], l.get_output(train=False).shape)
        out_shape = shape_f(input_tmp)
        print('shape after', l.get_config()['name'], ":", out_shape)
