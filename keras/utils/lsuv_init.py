from __future__ import print_function
import numpy as np
from keras.models import Model
from keras.layers import Dense, Convolution2D


def svd_orthonormal(shape):
    # Orthonorm init code is taked from Lasagne
    # https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
    if len(shape) < 2:
        raise RuntimeError("Only shapes of length 2 or more are supported.")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.standard_normal(flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return q


def get_activations(model, layer, X_batch):
    intermediate_layer_model = Model(
        inputs=model.get_input_at(0),
        outputs=layer.get_output_at(0)
    )
    activations = intermediate_layer_model.predict(X_batch)
    return activations


def LSUVinit(model, batch, verbose=True, margin=0.1, max_iter=10):
    # only these layer classes considered for LSUV initialization; add more if needed
    classes_to_consider = (Dense, Convolution2D)

    needed_variance = 1.0

    layers_inintialized = 0
    for layer in model.layers:
        if verbose:
            print(layer.name)
        if not isinstance(layer, classes_to_consider):
            continue
        # avoid small layers where activation variance close to zero, esp. for small batches
        if np.prod(layer.get_output_shape_at(0)[1:]) < 32:
            if verbose:
                print(layer.name, 'too small')
            continue
        if verbose:
            print('LSUV initializing', layer.name)

        layers_inintialized += 1
        weights, biases = layer.get_weights()
        weights = svd_orthonormal(weights.shape)
        layer.set_weights([weights, biases])
        activations = get_activations(model, layer, batch)
        variance = np.var(activations)
        iteration = 0
        if verbose:
            print(variance)
        while abs(needed_variance - variance) > margin:
            if np.abs(np.sqrt(variance)) < 1e-7:
                # avoid zero division
                break

            weights, biases = layer.get_weights()
            weights /= np.sqrt(variance) / np.sqrt(needed_variance)
            layer.set_weights([weights, biases])
            activations = get_activations(model, layer, batch)
            variance = np.var(activations)

            iteration += 1
            if verbose:
                print(variance)
            if iteration >= max_iter:
                break
    if verbose:
        print('LSUV: total layers initialized', layers_inintialized)
    return model
