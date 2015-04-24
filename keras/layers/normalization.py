from ..layers.core import Layer
from ..utils.theano_utils import shared_zeros
from .. import initializations

import theano, numpy

class BatchNormalization(Layer):
    '''
        Reference: 
            Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
                http://arxiv.org/pdf/1502.03167v3.pdf
    '''
    def __init__(self, input_shape, epsilon=1e-6, weights=None):
        self.init = initializations.get("uniform")
        self.input_shape = input_shape
        self.epsilon = epsilon

        self.gamma = self.init((self.input_shape))
        self.beta = shared_zeros(self.input_shape)

        self.data_mean = shared_zeros(self.input_shape)
        self.data_std = shared_zeros(self.input_shape)

        self.params = [self.gamma, self.beta]
        if weights is not None:
            self.set_weights(weights)

    def output(self, train):
        X = self.get_input(train)
        if train:
            X_normed = (X - X.mean(keepdims=True)) / (X.std(keepdims=True) + self.epsilon)
        else:
            X_normed = (X - self.data_mean) / (self.data_std + self.epsilon)
        out = self.gamma * X_normed + self.beta
        return out

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_shape":self.input_shape,
            "epsilon":self.epsilon}

def set_activation_stats(model, X, batch_size, shuffle=False, verbose=False):
    from keras.models import make_batches
    # Find all BatchNormalization layers in the model
    bn_layers = [layer for layer in model.layers if (layer.__class__.__name__ == BatchNormalization.__name__)]
    for layer in bn_layers:
        if verbose:
            print('Setting activation statistics for layer {}'.format(layer))
        activations = layer.get_input(train=False)
        activation_stats = theano.function([model.layers[0].input], outputs=[activations.mean(axis=0), activations.std(axis=0)])
        
        # Prepare training data and compute activation statistics
        index_array = numpy.arange(len(X))
        if shuffle:
            numpy.random.shuffle(index_array)
        batches = make_batches(len(X), batch_size)
        X_shape = list(layer.input_shape)
        X_shape.insert(0, len(batches))
        batch_means = numpy.empty(X_shape)
        batch_stds = numpy.empty(X_shape)

        for batch_index, (batch_start, batch_end) in enumerate(batches):
            if shuffle:
                batch_ids = index_array[batch_start:batch_end]
            else:
                batch_ids = slice(batch_start, batch_end)
            X_batch = X[batch_ids]
            batch_means[batch_index,...], batch_stds[batch_index,...] = activation_stats(X_batch)
        layer.data_mean.set_value(batch_means.mean(axis=0))
        layer.data_std.set_value((batch_size/(batch_size-1))*batch_stds.mean(axis=0))
    return

