# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
np.set_printoptions(threshold=np.inf)


from .. import backend as K
from .. import activations, initializations, regularizers
from ..engine import Layer, InputSpec

# Access to attention layers from recurrent.py
from .recurrent import AttLSTM as AttLSTM
from .recurrent import AttLSTMCond as AttLSTMCond
from .recurrent import AttGRUCond as AttGRUCond


class Attention(Layer):
    ''' Attention layer that does not depend on temporal information. The output information
        provided are the attention vectors 'alpha' over the input data.

    # Arguments
        nb_attention: number of attention mechanisms applied over the input vectors
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
        forget_bias_init: initialization function for the bias of the forget gate.
            [Jozefowicz et al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
            recommend initializing with ones.
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
        inner_activation: activation function for the inner cells.
        w_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        W_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_a_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_a_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_w_a: float between 0 and 1.
        dropout_W_a: float between 0 and 1.
        dropout_U_a: float between 0 and 1.

    # Formulation

        The resulting attention vector 'phi' at time 't' is formed by applying a weighted sum over
        the set of inputs 'x_i' contained in 'X':

            phi(X, t) = ∑_i alpha_i(t) * x_i,

        where each 'alpha_i' at time 't' is a weighting vector over all the input dimension that
        accomplishes the following condition:

            ∑_i alpha_i = 1

        and is dynamically adapted at each timestep w.r.t. the following formula:

            alpha_i(t) = exp{e_i(t)} /  ∑_j exp{e_j(t)}

        where each 'e_i' at time 't' is calculated as:

            e_i(t) = wa' * tanh( W * x_i   +  b ),

        where the following are learnable with the respectively named sizes:
                w                W               b
            [input_dim] [input_dim, input_dim] [input_dim]

    '''
    def __init__(self, nb_attention,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 dropout_w=0., dropout_W=0., dropout_Wa=0.,
                 w_regularizer=None, W_regularizer=None, b_regularizer=None, Wa_regularizer=None, ba_regularizer=None,
                 **kwargs):
        self.nb_attention = nb_attention
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        # attention model learnable params
        self.w_regularizer = regularizers.get(w_regularizer)
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.Wa_regularizer = regularizers.get(Wa_regularizer)
        self.ba_regularizer = regularizers.get(ba_regularizer)
        self.dropout_w, self.dropout_W, self.dropout_Wa = dropout_w, dropout_W, dropout_Wa

        if self.dropout_w or self.dropout_W or self.dropout_Wa:
            self.uses_learning_phase = True
        super(Attention, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]


    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape, ndim=3)]
        self.input_dim = input_shape[-1]


        # Initialize Att model params (following the same format for any option of self.consume_less)
        self.w = self.init((self.input_dim,),
                               name='{}_w'.format(self.name))

        self.W = self.init((self.nb_attention, self.input_dim, self.input_dim),
                               name='{}_W'.format(self.name))

        self.b = K.variable((np.zeros(self.input_dim)),
                                name='{}_b'.format(self.name))

        self.Wa = self.init((self.nb_attention, self.nb_attention),
                           name='{}_Wa'.format(self.name))

        self.ba = K.variable((np.zeros(self.nb_attention)),
                            name='{}_ba'.format(self.name))

        self.trainable_weights = [self.w, self.W, self.b, self.Wa, self.ba]

        self.regularizers = []
        # Att regularizers
        if self.w_regularizer:
            self.w_regularizer.set_param(self.w)
            self.regularizers.append(self.w_regularizer)
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)
        if self.Wa_regularizer:
            self.Wa_regularizer.set_param(self.Wa)
            self.regularizers.append(self.Wa_regularizer)
        if self.ba_regularizer:
            self.ba_regularizer.set_param(self.ba)
            self.regularizers.append(self.ba_regularizer)

        #if self.initial_weights is not None:
        #    self.set_weights(self.initial_weights)
        #    del self.initial_weights

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        # input shape must be:
        #   (nb_samples, temporal_or_spatial_dimensions, input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        assert len(input_shape) == 3, 'Input shape must be: (nb_samples, temporal_or_spatial_dimensions, input_dim)'

        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of temporal_or_spatial_dimensions of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis. '
                                'Found input shape at layer ' + self.name +
                                ': ' + str(input_shape))

        constants = self.get_constants(x)
        preprocessed_input = self.preprocess_input(x)

        attention = self.attention_step(preprocessed_input, constants)

        return attention


    def attention_step(self, x, constants):

        # Att model dropouts
        B_w = constants[0]
        B_W = constants[1]
        B_Wa = constants[2]

        # AttModel (see Formulation in class header)
        e = K.dot(K.tanh(K.dot(x * B_W, self.W) + self.b) * B_w, self.w)

        # Attention spatial weights 'alpha'
        #e = e.dimshuffle((0, 2, 1))
        e = K.permute_dimensions(e, (0,2,1))
        alpha = K.softmax_3d(e)
        alpha = K.permute_dimensions(alpha, (0,2,1))
        #alpha = alpha.dimshuffle((0,2,1))

        # Attention class weights 'beta'
        beta = K.sigmoid(K.dot(alpha * B_Wa, self.Wa) + self.ba)

        # Sum over the in_timesteps dimension resulting in [batch_size, input_dim]
        #x_att = (x * alpha[:,:,None]).sum(axis=1)

        # TODO: complete formulas in class description

        return beta


    def get_constants(self, x):
        constants = []

        # AttModel
        if 0 < self.dropout_w < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, :, 0, 0], (-1, input_shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, 2)
            B_w = K.in_train_phase(K.dropout(ones, self.dropout_w), ones)
            constants.append(B_w)
        else:
            constants.append(K.cast_to_floatx(1.))

        if 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, :, 0, 0], (-1, input_shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, 2)
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append([K.cast_to_floatx(1.)])

        if 0 < self.dropout_Wa < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, :, 0, 0], (-1, input_shape[1], 1)))
            ones = K.concatenate([ones] * self.nb_attention, 2)
            B_Wa = K.in_train_phase(K.dropout(ones, self.dropout_Wa), ones)
            constants.append(B_Wa)
        else:
            constants.append([K.cast_to_floatx(1.)])

        return constants


    def get_output_shape_for(self, input_shape):
        return tuple(list(input_shape[:2])+[self.nb_attention])


    def get_config(self):
        config = {'nb_attention': self.nb_attention,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'forget_bias_init': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'w_regularizer': self.w_regularizer.get_config() if self.w_regularizer else None,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'Wa_regularizer': self.Wa_regularizer.get_config() if self.Wa_regularizer else None,
                  'ba_regularizer': self.ba_regularizer.get_config() if self.ba_regularizer else None,
                  'dropout_w': self.dropout_w,
                  'dropout_W': self.dropout_W,
                  'dropout_Wa': self.dropout_Wa}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

