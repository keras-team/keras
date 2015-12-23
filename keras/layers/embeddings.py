from __future__ import absolute_import
from .. import backend as K

from .. import activations, initializations, regularizers, constraints
from ..layers.core import Layer, MaskedLayer

from ..constraints import unitnorm


class Embedding(Layer):
    '''Turn positive integers (indexes) into dense vectors of fixed size.
    eg. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]

    This layer can only be used as the first layer in a model.

    # Input shape
        2D tensor with shape: `(nb_samples, sequence_length)`.

    # Output shape
        3D tensor with shape: `(nb_samples, sequence_length, output_dim)`.

    # Arguments
      input_dim: int >= 0. Size of the vocabulary, ie.
          1 + maximum integer index occurring in the input data.
      output_dim: int >= 0. Dimension of the dense embedding.
      init: name of initialization function for the weights
          of the layer (see: [initializations](../initializations.md)),
          or alternatively, Theano function to use for weights initialization.
          This parameter is only relevant if you don't pass a `weights` argument.
      weights: list of numpy arrays to set as initial weights.
          The list should have 1 element, of shape `(input_dim, output_dim)`.
      W_regularizer: instance of the [regularizers](../regularizers.md) module
        (eg. L1 or L2 regularization), applied to the embedding matrix.
      W_constraint: instance of the [constraints](../constraints.md) module
          (eg. maxnorm, nonneg), applied to the embedding matrix.
      mask_zero: Whether or not the input value 0 is a special "padding"
          value that should be masked out.
          This is useful for [recurrent layers](recurrent.md) which may take
          variable length input. If this is `True` then all subsequent layers
          in the model need to support masking or an exception will be raised.
      input_length: Length of input sequences, when it is constant.
          This argument is required if you are going to connect
          `Flatten` then `Dense` layers upstream
          (without it, the shape of the dense outputs cannot be computed).
    '''
    input_ndim = 2

    def __init__(self, input_dim, output_dim,
                 init='uniform', input_length=None,
                 W_regularizer=None, activity_regularizer=None,
                 W_constraint=None,
                 mask_zero=False,
                 weights=None, **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = initializations.get(init)
        self.input_length = input_length
        self.mask_zero = mask_zero

        self.W_constraint = constraints.get(W_constraint)
        self.constraints = [self.W_constraint]

        self.W_regularizer = regularizers.get(W_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.initial_weights = weights
        kwargs['input_shape'] = (self.input_dim,)
        super(Embedding, self).__init__(**kwargs)

    def build(self):
        self.input = K.placeholder(shape=(self.input_shape[0], self.input_length),
                                   dtype='int32')
        self.W = self.init((self.input_dim, self.output_dim))
        self.params = [self.W]
        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)

    def get_output_mask(self, train=None):
        X = self.get_input(train)
        if not self.mask_zero:
            return None
        else:
            if K._BACKEND == "tensorflow":
                raise Exception("Masking is Theano-only for the time being.")
            return K.ones_like(X) * (1 - K.equal(X, 0))

    @property
    def output_shape(self):
        return (self.input_shape[0], self.input_length, self.output_dim)

    def get_output(self, train=False):
        X = self.get_input(train)
        out = K.gather(self.W, X)
        return out

    def get_config(self):
        config = {"name": self.__class__.__name__,
                  "input_dim": self.input_dim,
                  "output_dim": self.output_dim,
                  "init": self.init.__name__,
                  "input_length": self.input_length,
                  "mask_zero": self.mask_zero,
                  "activity_regularizer": self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  "W_regularizer": self.W_regularizer.get_config() if self.W_regularizer else None,
                  "W_constraint": self.W_constraint.get_config() if self.W_constraint else None}
        base_config = super(Embedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
