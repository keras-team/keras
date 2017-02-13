# -*- coding: utf-8 -*-
from __future__ import absolute_import

from .. import backend as K
from .. import initializations, regularizers, constraints
from ..engine import Layer, InputSpec


def path_energy(y, x, U, b_start=None, b_end=None, mask=None):
    '''Calculates the energy of a tag path y for a given input x (with mask),
    transition energies U and boundary energies b_start, b_end.'''
    x = add_boundary_energy(x, b_start, b_end, mask)
    return path_energy0(y, x, U, mask)


def path_energy0(y, x, U, mask=None):
    '''Path energy without boundary potential handling.'''
    n_classes = K.shape(x)[2]
    y_one_hot = K.one_hot(y, n_classes)

    # Tag path energy
    energy = K.sum(x * y_one_hot, 2)
    energy = K.sum(energy, 1)

    # Transition energy
    y_t = y[:, :-1]
    y_tp1 = y[:, 1:]
    U_flat = K.reshape(U, [-1])
    # Convert 2-dim indices (y_t, y_tp1) of U to 1-dim indices of U_flat:
    flat_indices = y_t * n_classes + y_tp1
    U_y_t_tp1 = K.gather(U_flat, flat_indices)

    if mask is not None:
        mask = K.cast(mask, K.floatx())
        y_t_mask = mask[:, :-1]
        y_tp1_mask = mask[:, 1:]
        U_y_t_tp1 *= y_t_mask * y_tp1_mask

    energy += K.sum(U_y_t_tp1, axis=1)

    return energy


def sparse_chain_crf_loss(y, x, U, b_start=None, b_end=None, mask=None):
    '''Given the true sparsely encoded tag sequence y, input x (with mask),
    transition energies U, boundary energies b_start and b_end, it computes
    the loss function of a Linear Chain Conditional Random Field:

    loss(y, x) = NNL(P(y|x)), where P(y|x) = exp(E(y, x)) / Z.
    So, loss(y, x) = - E(y, x) + log(Z)

    Here, E(y, x) is the tag path energy, and Z is the normalization constant.
    The values log(Z) is also called free energy.
    '''
    x = add_boundary_energy(x, b_start, b_end, mask)
    energy = path_energy0(y, x, U, mask)
    energy -= free_energy0(x, U, mask)
    return K.expand_dims(-energy, -1)


def chain_crf_loss(y, x, U, b_start=None, b_end=None, mask=None):
    '''Variant of sparse_chain_crf_loss but with one-hot encoded tags y.'''
    y_sparse = K.argmax(y, -1)
    y_sparse = K.cast(y_sparse, 'int32')
    return sparse_chain_crf_loss(y_sparse, x, U, b_start, b_end, mask)


def add_boundary_energy(x, b_start=None, b_end=None, mask=None):
    '''Given the observations x, it adds the start boundary energy b_start (resp.
    end boundary energy b_end on the start (resp. end) elements and multiplies
    the mask.'''
    if mask is None:
        if b_start is not None:
            x = K.concatenate([x[:, :1, :] + b_start, x[:, 1:, :]], axis=1)
        if b_end is not None:
            x = K.concatenate([x[:, :-1, :], x[:, -1:, :] + b_end], axis=1)
    else:
        mask = K.cast(mask, K.floatx())
        mask = K.expand_dims(mask, 2)
        x *= mask
        if b_start is not None:
            mask_r = K.concatenate([K.zeros_like(mask[:, :1]), mask[:, :-1]], axis=1)
            start_mask = K.cast(K.greater(mask, mask_r), K.floatx())
            x = x + start_mask * b_start
        if b_end is not None:
            mask_l = K.concatenate([mask[:, 1:], K.zeros_like(mask[:, -1:])], axis=1)
            end_mask = K.cast(K.greater(mask, mask_l), K.floatx())
            x = x + end_mask * b_end
    return x


def viterbi_decode(x, U, b_start=None, b_end=None, mask=None):
    '''Computes the best tag sequence y for a given input x, i.e. the one that
    maximizes the value of path_energy.'''
    x = add_boundary_energy(x, b_start, b_end, mask)

    alpha_0 = x[:, 0, :]
    gamma_0 = K.zeros_like(alpha_0)
    initial_states = [gamma_0, alpha_0]
    _, gamma = _forward(x,
                        lambda B: [K.cast(K.argmax(B, axis=1), K.floatx()), K.max(B, axis=1)],
                        initial_states,
                        U,
                        mask)
    y = _backward(gamma, mask)
    return y


def free_energy(x, U, b_start=None, b_end=None, mask=None):
    '''Computes efficiently the sum of all path energies for input x, when
    runs over all possible tag sequences.'''
    x = add_boundary_energy(x, b_start, b_end, mask)
    return free_energy0(x, U, mask)


def free_energy0(x, U, mask=None):
    '''Free energy without boundary potential handling.'''
    initial_states = [x[:, 0, :]]
    last_alpha, _ = _forward(x,
                             lambda B: [K.logsumexp(B, axis=1)],
                             initial_states,
                             U,
                             mask)
    return last_alpha[:, 0]


def _forward(x, reduce_step, initial_states, U, mask=None):
    '''Forward recurrence of the linear chain crf.'''

    def _forward_step(energy_matrix_t, states):
        alpha_tm1 = states[-1]
        new_states = reduce_step(K.expand_dims(alpha_tm1, 2) + energy_matrix_t)
        return new_states[0], new_states

    U_shared = K.expand_dims(K.expand_dims(U, 0), 0)

    if mask is not None:
        mask = K.cast(mask, K.floatx())
        mask_U = K.expand_dims(K.expand_dims(mask[:, :-1] * mask[:, 1:], 2), 3)
        U_shared = U_shared * mask_U

    inputs = K.expand_dims(x[:, 1:, :], 2) + U_shared
    inputs = K.concatenate([inputs, K.zeros_like(inputs[:, -1:, :, :])], axis=1)

    last, values, _ = K.rnn(_forward_step, inputs, initial_states)
    return last, values


def _backward(gamma, mask):
    '''Backward recurrence of the linear chain crf.'''
    gamma = K.cast(gamma, 'int32')

    def _backward_step(gamma_t, states):
        y_tm1 = K.squeeze(states[0], 0)
        y_t = K.batch_gather(gamma_t, y_tm1)
        return y_t, [K.expand_dims(y_t, 0)]

    initial_states = [K.expand_dims(K.zeros_like(gamma[:, 0, 0]), 0)]
    _, y_rev, _ = K.rnn(_backward_step,
                        gamma,
                        initial_states,
                        go_backwards=True)
    y = K.reverse(y_rev, 1)

    if mask is not None:
        mask = K.cast(mask, dtype='int32')
        # mask output
        y *= mask
        # set masked values to -1
        y += -(1 - mask)
    return y


class ChainCRF(Layer):
    '''A Linear Chain Conditional Random Field output layer.

    It carries the loss function and its weights for computing
    the global tag sequence scores. While training it acts as
    the identity function that passes the inputs to the subsequently
    used loss function. While testing it applies Viterbi decoding
    and returns the best scoring tag sequence as one-hot encoded vectors.

    # Arguments
        init: weight initialization function for chain energies U.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the transition weight matrix.
        b_start_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the start bias b.
        b_end_regularizer: instance of [WeightRegularizer](../regularizers.md)
            module, applied to the end bias b.
        b_start_constraint: instance of the [constraints](../constraints.md)
            module, applied to the start bias b.
        b_end_regularizer: instance of the [constraints](../constraints.md)
            module, applied to the end bias b.
        weights: list of Numpy arrays for initializing [U, b_start, b_end].
            Thus it should be a list of 3 elements of shape
            [(n_classes, n_classes), (n_classes, ), (n_classes, )]

    # Input shape
        3D tensor with shape `(nb_samples, timesteps, nb_classes)`, where
        ´timesteps >= 2`and `nb_classes >= 2`.

    # Output shape
        Same shape as input.

    # Masking
        This layer supports masking for input sequences of variable length.

    # Example

    ```python
    # As the last layer of sequential layer with
    # model.output_shape == (None, timesteps, nb_classes)
    crf = ChainCRF()
    model.add(crf)
    # now: model.output_shape == (None, timesteps, nb_classes)

    # Compile model with chain crf loss (and one-hot encoded labels) and accuracy
    model.compile(loss=crf.loss, optimizer='sgd', metrics=['accuracy'])

    # Alternatively, compile model with sparsely encoded labels and sparse accuracy:
    model.compile(loss=crf.sparse_loss, optimizer='sgd', metrics=['sparse_categorical_accuracy'])
    ```

    # Gotchas

    ## Model loading

    When you want to load a saved model that has a crf output, then loading
    the model with 'keras.models.load_model' won't work properly because
    the reference of the loss function to the transition parameters is lost. To
    fix this, you need to use the parameter 'custom_objects' as follows:

    ```python
    from keras.layer.crf import create_custom_objects:
    model = keras.models.load_model(filename, custom_objects=create_custom_objects())
    ```

    ## Temporal sample weights

    Given a ChainCRF instance crf both loss functions, crf.loss and crf.sparse_loss
    return a tensor of shape (batch_size, 1) and not (batch_size, maxlen).
    that sample weighting in temporal mode.

    '''
    def __init__(self, init='glorot_uniform',
                 U_regularizer=None, b_start_regularizer=None, b_end_regularizer=None,
                 U_constraint=None, b_start_constraint=None, b_end_constraint=None,
                 weights=None,
                 **kwargs):
        self.supports_masking = True
        self.uses_learning_phase = True
        self.input_spec = [InputSpec(ndim=3)]
        self.init = initializations.get(init)

        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_start_regularizer = regularizers.get(b_start_regularizer)
        self.b_end_regularizer = regularizers.get(b_end_regularizer)
        self.U_constraint = constraints.get(U_constraint)
        self.b_start_constraint = constraints.get(b_start_constraint)
        self.b_end_constraint = constraints.get(b_end_constraint)

        self.initial_weights = weights

        super(ChainCRF, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 3
        return (input_shape[0], input_shape[1], input_shape[2])

    def compute_mask(self, input, mask=None):
        if mask is not None:
            return K.any(mask, axis=1)
        return mask

    def _fetch_mask(self):
        mask = None
        if self.inbound_nodes:
            mask = self.inbound_nodes[0].input_masks[0]
        return mask

    def build(self, input_shape):
        assert len(input_shape) == 3
        n_classes = input_shape[2]
        n_steps = input_shape[1]
        assert n_classes >= 2
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, n_steps, n_classes))]

        self.U = self.add_weight((n_classes, n_classes),
                                 initializer=self.init,
                                 name='{}_U'.format(self.name),
                                 regularizer=self.U_regularizer,
                                 constraint=self.U_constraint)

        self.b_start = self.add_weight((n_classes, ),
                                       initializer='zero',
                                       name='{}_b_start'.format(self.name),
                                       regularizer=self.b_start_regularizer,
                                       constraint=self.b_start_constraint)

        self.b_end = self.add_weight((n_classes, ),
                                     initializer='zero',
                                     name='{}_b_end'.format(self.name),
                                     regularizer=self.b_end_regularizer,
                                     constraint=self.b_end_constraint)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, x, mask=None):
        y_pred = viterbi_decode(x, self.U, self.b_start, self.b_end, mask)
        nb_classes = self.input_spec[0].shape[2]
        y_pred_one_hot = K.one_hot(y_pred, nb_classes)
        return K.in_train_phase(x, y_pred_one_hot)

    def loss(self, y_true, y_pred):
        '''Linear Chain Conditional Random Field loss function.
        '''
        mask = self._fetch_mask()
        return chain_crf_loss(y_true, y_pred, self.U, self.b_start, self.b_end, mask)

    def sparse_loss(self, y_true, y_pred):
        '''Linear Chain Conditional Random Field loss function with sparse
        tag sequences.
        '''
        y_true = K.cast(y_true, 'int32')
        y_true = K.squeeze(y_true, 2)
        mask = self._fetch_mask()
        return sparse_chain_crf_loss(y_true, y_pred, self.U, self.b_start, self.b_end, mask)

    def get_config(self):
        config = {'init': self.init.__name__,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_start_regularizer': self.b_start_regularizer.get_config() if self.b_start_regularizer else None,
                  'b_end_regularizer': self.b_end_regularizer.get_config() if self.b_end_regularizer else None,
                  'U_constraint': self.U_constraint.get_config() if self.U_constraint else None,
                  'b_start_constraint': self.b_start_constraint.get_config() if self.b_start_constraint else None,
                  'b_end_constraint': self.b_end_constraint.get_config() if self.b_end_constraint else None,
                  }
        base_config = super(ChainCRF, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def create_custom_objects():
    '''Returns the custom objects, needed for loading a persisted model.'''
    instanceHolder = {'instance': None}

    class ClassWrapper(ChainCRF):
        def __init__(self, *args, **kwargs):
            instanceHolder['instance'] = self
            super(ClassWrapper, self).__init__(*args, **kwargs)

    def loss(*args):
        method = getattr(instanceHolder['instance'], 'loss')
        return method(*args)

    def sparse_loss(*args):
        method = getattr(instanceHolder['instance'], 'sparse_loss')
        return method(*args)

    return {'ChainCRF': ClassWrapper, 'loss': loss, 'sparse_loss': sparse_loss}
