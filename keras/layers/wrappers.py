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
        if type(input_shape) != list:
            input_shape = [input_shape]
        for shape in input_shape:
            assert len(shape) >= 3
        self.input_spec = [InputSpec(shape=shape) for shape in input_shape]
        if K._BACKEND == 'tensorflow':
            for shape in input_shape:
                if not shape[1]:
                    raise Exception('When using TensorFlow, you should define '
                                    'explicitly the number of timesteps of '
                                    'your sequences.\n'
                                    'If your first layer is an Embedding, '
                                    'make sure to pass it an "input_length" '
                                    'argument. Otherwise, make sure '
                                    'the first layer has '
                                    'an "input_shape" or "batch_input_shape" '
                                    'argument, including the time axis.')
        child_input_shape = [((shape[0],) + shape[2:]) for shape in input_shape]
        if len(input_shape) == 1:
            child_input_shape = child_input_shape[0]
        if not self.layer.built:
            self.layer.build(child_input_shape)
            self.layer.built = True
        super(TimeDistributed, self).build()

    def get_output_shape_for(self, input_shape):
        if type(input_shape) == list:
            child_input_shape = [((shape[0],) + shape[2:]) for shape in input_shape]
            timesteps = input_shape[0][1]
           
        else:
            child_input_shape = (input_shape[0],) + input_shape[2:]
            timesteps = input_shape[1]
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    def compute_mask(self, input, input_mask):
        if type(input_mask) != list:
            return input_mask
        elif not any(input_mask):
            return None
        else:
            return K.prod(input_mask, axis=0)

    def call(self, X, mask=None):
        input_shapes = [input_spec.shape for input_spec in self.input_spec]
        batch_size = False
        for shape in input_shapes:
            if shape[0] is not None:
                batch_size = True
                break
        if batch_size:
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
            if type(X) != list:
                X = [X]
            input_length = input_shapes[0][1]
            if not input_length:
                input_length = K.shape(X[0])[1]
            X = [K.reshape(X[i], (-1, ) + input_shapes[i][2:]) for i in range(len(X))] # (nb_samples * timesteps, ...)
            if len(X) == 1:
                X = X[0]
            y = self.layer.call(X)  # (nb_samples * timesteps, ...)
            # (nb_samples, timesteps, ...)
            output_shape = self.get_output_shape_for(input_shapes)
            y = K.reshape(y, (-1, input_length) + output_shape[2:])
        return y
