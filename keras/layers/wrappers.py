from ..engine import Layer, InputSpec
from .. import backend as K


class Wrapper(Layer):

    def __init__(self, layer, **kwargs):
        self.layer = layer
        self.uses_learning_phase = layer.uses_learning_phase
        super(Wrapper, self).__init__(**kwargs)

    def build(self, input_shape=None):
        '''Assumes that self.layer is already set.
        Should be called at the end of .build() in the
        children classes.
        '''
        self.trainable_weights = getattr(self.layer, 'trainable_weights', [])
        self.non_trainable_weights = getattr(self.layer, 'non_trainable_weights', [])
        self.updates = getattr(self.layer, 'updates', [])
        self.regularizers = getattr(self.layer, 'regularizers', [])
        self.constraints = getattr(self.layer, 'constraints', {})

    def get_weights(self):
        weights = self.layer.get_weights()
        return weights

    def set_weights(self, weights):
        self.layer.set_weights(weights)

    def get_config(self):
        config = {'layer': {'class_name': self.layer.__class__.__name__,
                            'config': self.layer.get_config()}}
        base_config = super(Wrapper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        from keras.utils.layer_utils import layer_from_config
        layer = layer_from_config(config.pop('layer'))
        return cls(layer, **config)


class TimeDistributed(Wrapper):
    """This wrapper allows to apply a layer to every
    temporal slice of an input.

    The input should be at least 3D,
    and the dimension of index one will be considered to be
    the temporal dimension.

    Consider a batch of 32 samples, where each sample is a sequence of 10
    vectors of 16 dimensions. The batch input shape of the layer is then `(32, 10, 16)`
    (and the `input_shape`, not including the samples dimension, is `(10, 16)`).

    You can then use `TimeDistributed` to apply a `Dense` layer to each of the 10 timesteps, independently:
    ```python
        # as the first layer in a model
        model = Sequential()
        model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
        # now model.output_shape == (None, 10, 8)

        # subsequent layers: no need for input_shape
        model.add(TimeDistributed(Dense(32)))
        # now model.output_shape == (None, 10, 32)
    ```

    The output will then have shape `(32, 10, 8)`.

    Note this is strictly equivalent to using `layers.core.TimeDistributedDense`.
    However what is different about `TimeDistributed`
    is that it can be used with arbitrary layers, not just `Dense`,
    for instance with a `Convolution2D` layer:

    ```python
        model = Sequential()
        model.add(TimeDistributed(Convolution2D(64, 3, 3), input_shape=(10, 3, 299, 299)))
    ```

    # Arguments
        layer: a layer instance.
    """
    def __init__(self, layer, **kwargs):
        self.supports_masking = True
        super(TimeDistributed, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        assert len(input_shape) >= 3
        self.input_spec = [InputSpec(shape=input_shape)]
        if K._BACKEND == 'tensorflow':
            if not input_shape[1]:
                raise Exception('When using TensorFlow, you should define '
                                'explicitly the number of timesteps of '
                                'your sequences.\n'
                                'If your first layer is an Embedding, '
                                'make sure to pass it an "input_length" '
                                'argument. Otherwise, make sure '
                                'the first layer has '
                                'an "input_shape" or "batch_input_shape" '
                                'argument, including the time axis.')
        child_input_shape = (input_shape[0],) + input_shape[2:]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(TimeDistributed, self).build()

    def get_output_shape_for(self, input_shape):
        child_input_shape = (input_shape[0],) + input_shape[2:]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        timesteps = input_shape[1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    def call(self, X, mask=None):
        input_shape = self.input_spec[0].shape
        if input_shape[0]:
            # batch size matters, use rnn-based implementation
            def step(x, states):
                output = self.layer.call(x)
                return output, []

            last_output, outputs, states = K.rnn(step, X,
                                                 initial_states=[])
            y = outputs
        else:
            # no batch size specified, therefore the layer will be able
            # to process batches of any size
            # we can go with reshape-based implementation for performance
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(X)[1]
            X = K.reshape(X, (-1, ) + input_shape[2:])  # (nb_samples * timesteps, ...)
            y = self.layer.call(X)  # (nb_samples * timesteps, ...)
            # (nb_samples, timesteps, ...)
            output_shape = self.get_output_shape_for(input_shape)
            y = K.reshape(y, (-1, input_length) + output_shape[2:])
        return y


class Bidirectional(Wrapper):
    ''' Bidirectional wrapper for RNNs

    # Arguments:
        layer: `Recurrent` instance.
        merge_mode: Mode by which outputs of the forward and reverse RNNs will be combined. One of {sum, mul, concat, ave}

    # Examples:
    ```python
    model = Sequential()
    model.add(Bidirectional(LSTM(10), input_shape=(5, 10)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    ```
    '''
    def __init__(self, layer, merge_mode='concat', weights=None, **kwargs):
        self.forward = layer
        self.reverse = layer.__class__.from_config(layer.get_config())
        self.forward.name = 'forward_' + self.forward.name
        self.reverse.name = 'reverse_' + self.reverse.name
        self.merge_mode = merge_mode
        if weights:
            nw = len(weights)
            self.forward.initial_weights = weights[:nw//2]
            self.reverse.initial_weights = weights[nw//2:]
        self.stateful = layer.stateful
        self.return_sequences = layer.return_sequences
        self.supports_masking = True
        super(Bidirectional, self).__init__(layer, **kwargs)

    def get_weights(self):
        return self.forward.get_weights() + self.reverse.get_weights()

    def set_weights(self, weights):
        nw = len(weights)
        self.forward.set_weights(weights[:nw//2])
        self.reverse.set_weights(weights[nw//2:])

    def get_output_shape_for(self, input_shape):
        if self.merge_mode in ['sum', 'ave', 'mul']:
            return self.forward.get_output_shape_for(input_shape)
        elif self.merge_mode == 'concat':
            shape = list(self.forward.get_output_shape_for(input_shape))
            shape[-1] *= 2
            return tuple(shape)

    def call(self, X, mask=None):

        def reverse(x):
            rev = K.permute_dimensions(x, (1, 0, 2))[::-1]                
            return K.permute_dimensions(rev, (1, 0, 2))

        Y = self.forward.call(X, mask)
        X_rev = reverse(X)
        mask_rev = reverse(mask) if mask else None
        Y_rev = self.reverse.call(X_rev, mask_rev)

        if self.return_sequences:
            Y_rev = reverse(Y_rev)

        if self.merge_mode == 'concat':
            return K.concatenate([Y, Y_rev])
        elif self.merge_mode == 'sum':
            return Y + Y_rev
        elif self.merge_mode == 'ave':
            return (Y + Y_rev) / 2
        elif self.merge_mode == 'mul':
            return Y * Y_rev

    def reset_states(self):
        self.forward.reset_states()
        self.reverse.reset_states()

    def build(self, input_shape):
        self.forward.build(input_shape)
        self.reverse.build(input_shape)
        params = ['trainable_weights', 'non_trainable_weights', 'updates', 'regularizers']
        for p in params:
            setattr(self, p, getattr(self.forward, p, []) + getattr(self.reverse, p, []))
        self.constraints = {}
        if hasattr(self.forward, 'constraints'):
            self.constraints.update(self.forward.constraints)
            self.constraints.update(self.reverse.constraints)

    def compute_mask(self, input, mask):
        if self.return_sequences:
            return mask
        else:
            return None

    def get_config(self):
        config = {"merge_mode": self.merge_mode}
        base_config = super(Bidirectional, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
