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
    # This is to handle the case where a model has been connected to a previous
    # layer (and therefore get_input would recurse into previous layer's
    # output).
    if hasattr(model.layers[0], 'previous'):
        # TODO: If the model is used as a part of another model, get_input will
        # return the input of the whole model and this won't work. So this is
        # not handled yet
        raise Exception("This function doesn't work on model used as subparts "
                        " for other models")

    input_var = model.get_input(train=False)
    input_tmp = np.zeros(input_shape, dtype=np.float32)
    print("input shape : ", input_shape)
    for l in model.layers:
        shape_f = theano.function([input_var], l.get_output(train=False).shape)
        out_shape = shape_f(input_tmp)
        print('shape after', l.get_config()['name'], ":", out_shape)
