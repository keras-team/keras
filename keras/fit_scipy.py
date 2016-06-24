"""
Fit a Keras model with any method from scipy.optimize.minimize.

fit_scipy(..) is the only function which needs to be called externally.
"""

from __future__ import division
import numpy as np
import scipy as sp
import scipy.optimize
import keras
from keras import backend as K


def pack_theta(trainable_weights):
    """ Flattens a set of shared variables (trainable_weights)"""
    x = np.empty(0)
    for t in trainable_weights:
        x = np.concatenate((x, K.get_value(t).reshape(-1)))
    return x


def unpack_theta(model, theta):
    """ Converts flattened theta back to tensor shapes of Keras model params """
    weights = []
    idx = 0
    for layer in model.layers:
        layer_weights = []
        for param in layer.get_weights():
            plen = np.prod(param.shape)
            layer_weights.append(
                np.asarray(
                    theta[
                        idx:(
                            idx +
                            plen)].reshape(
                        param.shape),
                    dtype=np.float32))
            idx += plen
        weights.append(layer_weights)
    return weights


def set_model_params(model, theta):
    """ Sets the Keras model params from a flattened numpy array of theta """
    trainable_params = unpack_theta(model, theta)
    for trainable_param, layer in zip(trainable_params, model.layers):
        layer.set_weights(trainable_param)


def get_cost_grads(model):
    """ Returns the cost and flattened gradients for the model """
    trainable_params = get_trainable_params(model)

    cost = model.model.total_loss
    grads = K.gradients(cost, trainable_params)

    return cost, grads


def flatten_grads(grads):
    """ Flattens a set tensor variables (gradients) """
    x = np.empty(0)
    for g in grads:
        x = np.concatenate((x, g.reshape(-1)))
    return x


def get_trainable_params(model):
    trainable_weights = []
    for layer in model.layers:
        trainable_weights += keras.engine.training.collect_trainable_weights(
            layer)
    return trainable_weights


def get_training_function(model, x, y):
    cost, grads = get_cost_grads(model)
    outs = [cost]
    if type(grads) in {list, tuple}:
        outs += grads
    else:
        outs.append(grads)

    fn = K.function(
        inputs=[],
        outputs=outs,
        givens={
            model.model.inputs[0]: x,
            model.model.targets[0]: y,
            model.model.sample_weights[0]: np.ones(
                (x.shape[0],
                 ),
                dtype=np.float32),
            K.learning_phase(): np.uint8(1)})

    def train_fn(theta):
        set_model_params(model, theta)
        cost_grads = fn([])
        cost = np.asarray(cost_grads[0], dtype=np.float64)
        grads = np.asarray(flatten_grads(cost_grads[1:]), dtype=np.float64)

        return cost, grads

    return train_fn


def fit_scipy(model, x, y, nb_epoch=300, method='L-BFGS-B', **kwargs):
    trainable_params = get_trainable_params(model)
    theta0 = pack_theta(trainable_params)

    train_fn = get_training_function(model, x, y)

    weights = sp.optimize.minimize(
        train_fn, theta0, method=method, jac=True, options={
            'maxiter': nb_epoch, 'disp': False}, **kwargs)

    theta_final = weights.x
    set_model_params(model, theta_final)
