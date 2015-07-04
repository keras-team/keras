from __future__ import absolute_import
import theano
import theano.tensor as T

from .. import activations, initializations
from ..layers.core import Layer, MaskedLayer
from ..utils.theano_utils import sharedX

from ..constraints import unitnorm


class Embedding(Layer):
    '''
        Turn positive integers (indexes) into denses vectors of fixed size. 
        eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

        @input_dim: size of vocabulary (highest input integer + 1)
        @out_dim: size of dense representation
    '''
    def __init__(self, input_dim, output_dim, init='uniform',
        W_regularizer=None, activity_regularizer=None, W_constraint=None,
        mask_zero=False, weights=None):

        super(Embedding,self).__init__()
        self.init = initializations.get(init)
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.input = T.imatrix()
        self.W = self.init((self.input_dim, self.output_dim))
        self.mask_zero = mask_zero

        self.params = [self.W]
        self.constraints = [W_constraint]

        self.regularizers = []
        if W_regularizer:
            W_regularizer.set_param(self.W)
            self.regularizers.append(W_regularizer)
        if activity_regularizer:
            activity_regularizer.set_layer(self)
            self.regularizers.append(activity_regularizer)

        if weights is not None:
            self.set_weights(weights)

    def get_output_mask(self, train=None):
        X = self.get_input(train)
        if not self.mask_zero:
            return None
        else:
            return T.ones_like(X) * (1 - T.eq(X,0))

    def get_output(self, train=False):
        X = self.get_input(train)
        out = self.W[X]
        return out

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "output_dim":self.output_dim,
            "init":self.init.__name__}


class WordContextProduct(Layer):
    '''
        This layer turns a pair of words (a pivot word + a context word, 
        ie. a word from the same context, or a random, out-of-context word),
        indentified by their index in a vocabulary, into two dense reprensentations
        (word representation and context representation).

        Then it returns activation(dot(pivot_embedding, context_embedding)),
        which can be trained to encode the probability 
        of finding the context word in the context of the pivot word
        (or reciprocally depending on your training procedure).

        The layer ingests integer tensors of shape:
        (nb_samples, 2)
        and outputs a float tensor of shape
        (nb_samples, 1)

        The 2nd dimension encodes (pivot, context).
        input_dim is the size of the vocabulary.

        For more context, see Mikolov et al.:
            Efficient Estimation of Word reprensentations in Vector Space
            http://arxiv.org/pdf/1301.3781v3.pdf
    '''
    def __init__(self, input_dim, proj_dim=128, 
        init='uniform', activation='sigmoid', weights=None):
        super(WordContextProduct,self).__init__()
        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.input = T.imatrix()
        # two different embeddings for pivot word and its context
        # because p(w|c) != p(c|w)
        self.W_w = self.init((input_dim, proj_dim))
        self.W_c = self.init((input_dim, proj_dim))

        self.params = [self.W_w, self.W_c]

        if weights is not None:
            self.set_weights(weights)


    def get_output(self, train=False):
        X = self.get_input(train)
        w = self.W_w[X[:, 0]] # nb_samples, proj_dim
        c = self.W_c[X[:, 1]] # nb_samples, proj_dim

        dot = T.sum(w * c, axis=1)
        dot = theano.tensor.reshape(dot, (X.shape[0], 1))
        return self.activation(dot)

    def get_config(self):
        return {"name":self.__class__.__name__,
            "input_dim":self.input_dim,
            "proj_dim":self.proj_dim,
            "init":self.init.__name__,
            "activation":self.activation.__name__}

