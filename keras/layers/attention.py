# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np

np.set_printoptions(threshold=np.inf)

from .. import backend as K
from .. import activations, initializations, regularizers, constraints
from ..engine import Layer, InputSpec

# Access to attention layers from recurrent.py


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

    '''

    def __init__(self, nb_attention,
                 init='glorot_uniform', inner_init='orthogonal',
                 forget_bias_init='one', activation='tanh',
                 inner_activation='hard_sigmoid',
                 dropout_Wa=0.,
                 Wa_regularizer=None, ba_regularizer=None,
                 **kwargs):
        self.nb_attention = nb_attention
        self.init = initializations.get(init)
        self.inner_init = initializations.get(inner_init)
        self.forget_bias_init = initializations.get(forget_bias_init)
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)

        # attention model learnable params
        self.Wa_regularizer = regularizers.get(Wa_regularizer)
        self.ba_regularizer = regularizers.get(ba_regularizer)
        self.dropout_Wa = dropout_Wa

        if self.dropout_Wa:
            self.uses_learning_phase = True
        super(Attention, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape, ndim=3)]
        self.input_dim = input_shape[-1]

        # Initialize Att model params (following the same format for any option of self.consume_less)
        self.Wa = self.init((self.input_dim, self.nb_attention),
                            name='{}_Wa'.format(self.name))

        self.ba = K.variable((np.zeros(self.nb_attention)),
                             name='{}_ba'.format(self.name))

        self.trainable_weights = [self.Wa, self.ba]

        self.regularizers = []
        # Att regularizers
        if self.Wa_regularizer:
            self.Wa_regularizer.set_param(self.Wa)
            self.regularizers.append(self.Wa_regularizer)
        if self.ba_regularizer:
            self.ba_regularizer.set_param(self.ba)
            self.regularizers.append(self.ba_regularizer)

            # if self.initial_weights is not None:
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
        B_Wa = constants[0]

        # AttModel (see Formulation in class header)
        # e = K.dot(K.tanh(K.dot(x * B_W, self.W) + self.b) * B_w, self.w)

        # Attention spatial weights 'alpha'
        # e = K.permute_dimensions(e, (0,2,1))
        # alpha = K.softmax_3d(e)
        # alpha = K.permute_dimensions(alpha, (0,2,1))

        # Attention class weights 'beta'
        # beta = K.sigmoid(K.dot(alpha * B_Wa, self.Wa) + self.ba)
        beta = K.sigmoid(K.dot(x * B_Wa, self.Wa) + self.ba)

        # TODO: complete formulas in class description

        return beta

    def get_constants(self, x):
        constants = []

        # AttModel

        if 0 < self.dropout_Wa < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, :, 0, 0], (-1, input_shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, 2)
            B_Wa = K.in_train_phase(K.dropout(ones, self.dropout_Wa), ones)
            constants.append(B_Wa)
        else:
            constants.append([K.cast_to_floatx(1.)])

        return constants

    def get_output_shape_for(self, input_shape):
        return tuple(list(input_shape[:2]) + [self.nb_attention])

    def get_config(self):
        config = {'nb_attention': self.nb_attention,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'forget_bias_init': self.forget_bias_init.__name__,
                  'activation': self.activation.__name__,
                  'inner_activation': self.inner_activation.__name__,
                  'Wa_regularizer': self.Wa_regularizer.get_config() if self.Wa_regularizer else None,
                  'ba_regularizer': self.ba_regularizer.get_config() if self.ba_regularizer else None,
                  'dropout_Wa': self.dropout_Wa}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SoftAttention(Layer):
    ''' Simple soft Attention layer
    The output information provided are the attended input an the attention weights 'alpha' over the input data.

    # Arguments
        att_dim: Soft alignment MLP dimension
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        activation: activation function.
            Can be the name of an existing function (str),
            or a Theano function (see: [activations](../activations.md)).
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

            e_i(t) = wa' * tanh( Wa * x_i  +  Ua * h(t-1)  +  ba ),

        where the following are learnable with the respectively named sizes:
                wa                Wa                     Ua                 ba
            [input_dim] [input_dim, input_dim] [output_dim, input_dim] [input_dim]

    '''

    def __init__(self, att_dim,
                 init='glorot_uniform', activation='tanh',
                 dropout_Wa=0., dropout_Ua=0.,
                 wa_regularizer=None, Wa_regularizer=None, Ua_regularizer=None, ba_regularizer=None, ca_regularizer=None,
                 **kwargs):
        self.att_dim = att_dim
        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        self.dropout_Wa, self.dropout_Ua = dropout_Wa, dropout_Ua

        # attention model learnable params
        self.wa_regularizer = regularizers.get(wa_regularizer)
        self.Wa_regularizer = regularizers.get(Wa_regularizer)
        self.Ua_regularizer = regularizers.get(Ua_regularizer)
        self.ba_regularizer = regularizers.get(ba_regularizer)
        self.ca_regularizer = regularizers.get(ca_regularizer)


        if self.dropout_Wa or self.dropout_Ua :
            self.uses_learning_phase = True
        super(SoftAttention, self).__init__(**kwargs)
        #self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        assert len(input_shape) == 2, 'You should pass two inputs to SoftAttention '
        self.input_spec = [InputSpec(shape=input_shape[0]), InputSpec(shape=input_shape[1])]
        self.input_steps = input_shape[0][1]
        self.input_dim = input_shape[0][2]
        self.context_dim = input_shape[1][1]

        # Initialize Att model params (following the same format for any option of self.consume_less)
        self.wa = self.add_weight((self.att_dim, ),
                                   initializer=self.init,
                                   name='{}_wa'.format(self.name),
                                   regularizer=self.wa_regularizer)

        self.Wa = self.add_weight((self.input_dim, self.att_dim),
                                   initializer=self.init,
                                   name='{}_Wa'.format(self.name),
                                   regularizer=self.Wa_regularizer)

        self.Ua = self.add_weight((self.context_dim, self.att_dim),
                                   initializer=self.init,
                                   name='{}_Ua'.format(self.name),
                                   regularizer=self.Ua_regularizer)

        self.ba = self.add_weight(self.att_dim,
                                   initializer='zero',
                                   name='{}_ba'.format(self.name),
                                  regularizer=self.ba_regularizer)

        self.ca = self.add_weight(self.input_steps,
                                  initializer='zero',
                                   name='{}_ca'.format(self.name),
                                  regularizer=self.ca_regularizer)

        self.trainable_weights = [self.wa, self.Wa, self.Ua, self.ba, self.ca]  # AttModel parameters

        self.built = True

    def preprocess_input(self, x):
        return x

    def call(self, x, mask=None):
        # input shape must be:
        #   (nb_samples, temporal_or_spatial_dimensions, input_dim)
        # note that the .build() method of subclasses MUST define
        # self.input_spec with a complete input shape.
        input_shape = self.input_spec[0].shape
        state_below = x[0]
        self.context = x[1]
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

        constants = self.get_constants(state_below, mask[1])
        preprocessed_input = self.preprocess_input(state_below)

        [attended_representation, alphas] = self.attention_step(preprocessed_input, constants)

        return  [attended_representation, alphas]

    def attention_step(self, x, constants):
        # Att model dropouts
        B_Wa = constants[0]                                  # Dropout Wa

        pctx_ = constants[1]                               # Original context

        # Attention model (see Formulation in class header)
        p_state_ = K.dot(x * B_Wa[0], self.Wa)
        pctx_ = self.activation(pctx_[:, None, :] + p_state_)
        e = K.dot(pctx_, self.wa) + self.ca
        alphas_shape = e.shape
        alphas = K.softmax(e.reshape([alphas_shape[0], alphas_shape[1]]))

        # sum over the in_timesteps dimension resulting in [batch_size, input_dim]
        ctx_ = (x * alphas[:, :, None]).sum(axis=1)
        return [ctx_, alphas]

    def get_constants(self, x, mask_context):
        constants = []

        # constants[0]
        if 0 < self.dropout_Wa < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.concatenate([ones] * input_dim, 1)
            B_Wa = [K.in_train_phase(K.dropout(ones, self.dropout_Wa), ones)]
            constants.append(B_Wa)
        else:
            constants.append([K.cast_to_floatx(1.)])

        # constants[1]
        if 0 < self.dropout_Ua < 1:
            input_dim = self.context_dim
            ones = K.ones_like(K.reshape(self.context[:, :, 0], (-1, self.context.shape[1], 1)))
            ones = K.concatenate([ones] * input_dim, axis=2)
            B_Ua = [K.in_train_phase(K.dropout(ones, self.dropout_Ua), ones)]
            pctx = K.dot(self.context * B_Ua[0], self.Ua) + self.ba
        else:
            pctx = K.dot(self.context, self.Ua) + self.ba
        constants.append(pctx)

        return constants

    def get_output_shape_for(self, input_shape):
        dim_x_att = (input_shape[0][0], input_shape[0][2])
        dim_alpha_att = (input_shape[0][0], input_shape[0][1])
        main_out = [dim_x_att, dim_alpha_att]
        return main_out


    def compute_mask(self, input, input_mask=None):
        return [None, None]

    def get_config(self):
        config = {'att_dim': self.att_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'wa_regularizer': self.wa_regularizer.get_config() if self.wa_regularizer else None,
                  'Wa_regularizer': self.Wa_regularizer.get_config() if self.Wa_regularizer else None,
                  'Ua_regularizer': self.Ua_regularizer.get_config() if self.Ua_regularizer else None,
                  'ba_regularizer': self.ba_regularizer.get_config() if self.ba_regularizer else None,
                  'ca_regularizer': self.ca_regularizer.get_config() if self.ca_regularizer else None,
                  'dropout_Wa': self.dropout_Wa,
                  'dropout_Ua': self.dropout_Ua,
                  }
        base_config = super(SoftAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionComplex(Layer):
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
        super(AttentionComplex, self).__init__(**kwargs)
        self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape, ndim=3)]
        self.input_dim = input_shape[-1]

        # Initialize Att model params (following the same format for any option of self.consume_less)
        # self.w = self.add_weight((self.input_dim,),
        self.w = self.add_weight((self.input_dim, self.nb_attention),
                                 initializer=self.init,
                                 name='{}_w'.format(self.name),
                                 regularizer=self.w_regularizer)

        # self.W = self.add_weight((self.input_dim, self.nb_attention, self.input_dim),
        self.W = self.add_weight((self.input_dim, self.input_dim),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer)

        self.b = self.add_weight(self.input_dim,
                                 initializer='zero',
                                 regularizer=self.b_regularizer)

        """
        self.Wa = self.add_weight((self.nb_attention, self.nb_attention),
                                 initializer=self.init,
                                 name='{}_Wa'.format(self.name),
                                 regularizer=self.Wa_regularizer)

        self.ba = self.add_weight(self.input_dim,
                                  initializer= 'zero',
                                  regularizer=self.ba_regularizer)

        self.trainable_weights = [self.w, self.W, self.b, self.Wa, self.ba] # AttModel parameters
        """
        self.trainable_weights = [self.w, self.W, self.b]
        # if self.initial_weights is not None:
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
        return e

        # Attention spatial weights 'alpha'
        ##e = e.dimshuffle((0, 2, 1))
        e = K.permute_dimensions(e, (0, 2, 1))
        # alpha = K.softmax(e)
        # return alpha
        alpha = K.softmax_3d(e)
        alpha = K.permute_dimensions(alpha, (0, 2, 1))

        return alpha

        ##alpha = alpha.dimshuffle((0,2,1))

        # Attention class weights 'beta'
        beta = K.sigmoid(K.dot(alpha * B_Wa, self.Wa) + self.ba)
        ##beta = K.softmax_3d(K.dot(alpha * B_Wa, self.Wa) + self.ba)

        # Sum over the in_timesteps dimension resulting in [batch_size, input_dim]
        ##x_att = (x * alpha[:,:,None]).sum(axis=1)

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
            constants.append(K.cast_to_floatx(1.))

        if 0 < self.dropout_Wa < 1:
            input_shape = self.input_spec[0].shape
            ones = K.ones_like(K.reshape(x[:, :, 0, 0], (-1, input_shape[1], 1)))
            ones = K.concatenate([ones] * self.nb_attention, 2)
            B_Wa = K.in_train_phase(K.dropout(ones, self.dropout_Wa), ones)
            constants.append(B_Wa)
        else:
            constants.append(K.cast_to_floatx(1.))

        return constants

    def get_output_shape_for(self, input_shape):
        return tuple(list(input_shape[:2]) + [self.nb_attention])

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
        base_config = super(AttentionComplex, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConvAtt(Layer):
    '''Convolution operator for filtering windows of two-dimensional inputs with Attention mechanism.
    The first input corresponds to the image and the second input to the weighting vector (which contains a set of steps).
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(3, 128, 128)` for 128x128 RGB pictures. An additional input for modulating the attention is required.

    # Examples

    ```python
        # apply a 3x3 convolution with 64 output filters on a 256x256 image:
        model = Sequential()
        model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(3, 256, 256)))
        # now model.output_shape == (None, 64, 256, 256)

        # add a 3x3 convolution on top, with 32 output filters:
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        # now model.output_shape == (None, 32, 256, 256)
    ```
    # Arguments
            nb_filter: Number of convolution filters to use.
            init: name of initialization function for the weights of the layer
                (see [initializations](../initializations.md)), or alternatively,
                Theano function to use for weights initialization.
                This parameter is only relevant if you don't pass
                a `weights` argument.
            activation: name of activation function to use
                (see [activations](../activations.md)),
                or alternatively, elementwise Theano function.
                If you don't specify anything, no activation is applied
                (ie. "linear" activation: a(x) = x).
            weights: list of numpy arrays to set as initial weights.
            border_mode: 'valid', 'same' or 'full'. ('full' requires the Theano backend.)
            subsample: tuple of length 2. Factor by which to subsample output.
                Also called strides elsewhere.
            W_regularizer: instance of [WeightRegularizer](../regularizers.md)
                (eg. L1 or L2 regularization), applied to the main weights matrix.
            b_regularizer: instance of [WeightRegularizer](../regularizers.md),
                applied to the bias.
            activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
                applied to the network output.
            W_constraint: instance of the [constraints](../constraints.md) module
                (eg. maxnorm, nonneg), applied to the main weights matrix.
            b_constraint: instance of the [constraints](../constraints.md) module,
                applied to the bias.
            dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
                (the depth) is at index 1, in 'tf' mode is it at index 3.
                It defaults to the `image_dim_ordering` value found in your
                Keras config file at `~/.keras/keras.json`.
                If you never set it, then it will be "tf".
            bias: whether to include a bias
                (i.e. make the layer affine rather than linear).

        # Input shape
            4D tensor with shape:
            `(samples, channels, rows, cols)` if dim_ordering='th'
            or 4D tensor with shape:
            `(samples, rows, cols, channels)` if dim_ordering='tf'.
            and 4D tensor with shape:
            `(samples, steps, features)`

        # Output shape
            4D tensor with shape:
            `(samples, nb_filter, rows, cols)` if dim_ordering='th'
            or 4D tensor with shape:
            `(samples, rows, cols, nb_filter)` if dim_ordering='tf'.
            `rows` and `cols` values might have changed due to padding.
        '''

    def __init__(self, nb_embedding, nb_glimpses=1,
                 init='glorot_uniform', activation=None, weights=None,
                 border_mode='valid', dim_ordering='default',
                 W_regularizer=None, U_regularizer=None, V_regularizer=None, b_regularizer=None,
                 activity_regularizer=None,
                 W_constraint=None, U_constraint=None, V_constraint=None, b_constraint=None,
                 W_learning_rate_multiplier=None, b_learning_rate_multiplier=None,
                 bias=True, **kwargs):
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        if border_mode not in {'valid', 'same', 'full'}:
            raise ValueError('Invalid border mode for Convolution2D:', border_mode)
        self.nb_embedding = nb_embedding
        self.nb_glimpses = nb_glimpses
        self.nb_row = 1
        self.nb_col = 1
        self.init = initializations.get(init, dim_ordering=dim_ordering)
        self.activation = activations.get(activation)
        self.border_mode = border_mode
        self.subsample = tuple((1, 1))
        if dim_ordering not in {'tf', 'th'}:
            raise ValueError('dim_ordering must be in {tf, th}.')
        self.dim_ordering = dim_ordering
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.U_constraint = constraints.get(U_constraint)
        self.V_constraint = constraints.get(V_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.b_learning_rate_multiplier = b_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier, self.b_learning_rate_multiplier]

        self.bias = bias
        self.input_spec = [InputSpec(ndim=4)]
        self.initial_weights = weights
        self.supports_masking = True
        super(ConvAtt, self).__init__(**kwargs)

    def build(self, input_shape):
        self.num_words = input_shape[1][1]
        if self.dim_ordering == 'th':
            img_size = input_shape[0][1]
            qst_size = input_shape[1][2]
            self.U_shape = (self.nb_glimpses, self.nb_embedding, self.nb_row, self.nb_col)
            self.V_shape = (qst_size, self.nb_embedding)
            self.W_shape = (self.nb_embedding, img_size, self.nb_row, self.nb_col)
        elif self.dim_ordering == 'tf':
            img_size = input_shape[0][3]
            qst_size = input_shape[1][2]
            self.U_shape = (self.nb_row, self.nb_col, self.nb_embedding, self.nb_glimpses)
            self.V_shape = (qst_size, self.nb_embedding)
            self.W_shape = (self.nb_row, self.nb_col, img_size, self.nb_embedding)
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)
        self.U = self.add_weight(self.U_shape,
                                 initializer=self.init,
                                 name='{}_U'.format(self.name),
                                 regularizer=self.U_regularizer,
                                 constraint=self.U_constraint)
        self.V = self.add_weight(self.V_shape,
                                 initializer=self.init,
                                 name='{}_V'.format(self.name),
                                 regularizer=self.V_regularizer,
                                 constraint=self.V_constraint)
        self.W = self.add_weight(self.W_shape,
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((self.nb_embedding,),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def preprocess_input(self, x):
        return K.dot(x, self.V)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            rows = input_shape[0][2]
            cols = input_shape[0][3]
        elif self.dim_ordering == 'tf':
            rows = input_shape[0][1]
            cols = input_shape[0][2]
        else:
            raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        '''
        rows = conv_output_length(rows, self.nb_row,
                                  self.border_mode, self.subsample[0])
        cols = conv_output_length(cols, self.nb_col,
                                  self.border_mode, self.subsample[1])
        '''

        if self.dim_ordering == 'th':
            return (input_shape[0][0], self.nb_glimpses * self.num_words, rows, cols)
        elif self.dim_ordering == 'tf':
            return (input_shape[0][0], rows, cols, self.nb_glimpses * self.num_words)

    def call(self, x, mask=None):

        preprocessed_img = K.conv2d(x[0], self.W, strides=self.subsample,
                                    border_mode=self.border_mode,
                                    dim_ordering=self.dim_ordering,
                                    filter_shape=self.W_shape)

        preprocessed_input = self.preprocess_input(x[1])  # TODO: Dropout?

        if self.bias:
            if self.dim_ordering == 'th':
                preprocessed_img += K.reshape(self.b, (1, self.nb_embedding, 1, 1))
            elif self.dim_ordering == 'tf':
                preprocessed_img += K.reshape(self.b, (1, 1, 1, self.nb_embedding))
            else:
                raise ValueError('Invalid dim_ordering:', self.dim_ordering)

        last_output, outputs, states = K.rnn(self.step, preprocessed_input,
                                             [],
                                             go_backwards=False,
                                             mask=None,
                                             # mask[1], #TODO: What does this mask mean? How should it be applied?
                                             constants=[preprocessed_img],
                                             unroll=False,
                                             input_length=self.num_words)

        # Join temporal and glimpses dimensions
        outputs = K.permute_dimensions(outputs, (0,3,4,2,1))
        shp = outputs.shape
        outputs = K.reshape(outputs, (shp[0], shp[1], shp[2], -1))
        outputs = K.permute_dimensions(outputs, (0, 3, 1, 2))

        return outputs

    def step(self, x, states):
        context = states[0]

        a_t = K.conv2d(K.tanh(context + x[:, :, None, None]),
                       self.U,
                       strides=(1, 1),
                       border_mode='valid',
                       dim_ordering=self.dim_ordering,
                       filter_shape=self.U_shape)

        return a_t, []

    def compute_mask(self, input, mask):
        out_mask = K.repeat(mask[1], self.nb_glimpses)
        out_mask = K.flatten(out_mask)
        return out_mask

    def get_config(self):
        config = {'nb_embedding': self.nb_embedding,
                  'nb_glimpses': self.nb_glimpses,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'border_mode': self.border_mode,
                  'dim_ordering': self.dim_ordering,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'V_regularizer': self.V_regularizer.get_config() if self.V_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'U_constraint': self.U_constraint.get_config() if self.U_constraint else None,
                  'V_constraint': self.V_constraint.get_config() if self.V_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'W_learning_rate_multiplier': self.W_learning_rate_multiplier,
                  'b_learning_rate_multiplier': self.b_learning_rate_multiplier,
                  'bias': self.bias}
        base_config = super(ConvAtt, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def set_lr_multipliers(self, W_learning_rate_multiplier, b_learning_rate_multiplier):
        self.W_learning_rate_multiplier = W_learning_rate_multiplier
        self.b_learning_rate_multiplier = b_learning_rate_multiplier
        self.learning_rate_multipliers = [self.W_learning_rate_multiplier,
                                          self.b_learning_rate_multiplier]
