from .core import MaskedLayer
from .. import backend as K


class TimeDistributed(MaskedLayer):
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
        model = Sequential()
        model.add(TimeDistributed(Dense(8), input_shape=(10, 16)))
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
        self.layer = layer
        super(TimeDistributed, self).__init__(**kwargs)

    def build(self):
        input_shape = self.input_shape
        assert len(input_shape) >= 3
        child_input_shape = (input_shape[0],) + input_shape[2:]
        self.layer.set_input_shape(child_input_shape)
        self.layer.build()

        trainable_weights, regularizers, constraints, updates = self.layer.get_params()
        self.trainable_weights = trainable_weights
        self.non_trainable_weights = self.layer.non_trainable_weights
        self.regularizers = regularizers
        self.constraints = constraints
        self.updates = updates

    @property
    def output_shape(self):
        child_output_shape = self.layer.output_shape
        timesteps = self.input_shape[1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    def get_output(self, train=False):
        X = self.get_input(train)
        mask = self.get_input_mask(train)

        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
                raise Exception('When using TensorFlow, you should define ' +
                                'explicitly the number of timesteps of ' +
                                'your sequences.\n' +
                                'If your first layer is an Embedding, ' +
                                'make sure to pass it an "input_length" ' +
                                'argument. Otherwise, make sure ' +
                                'the first layer has ' +
                                'an "input_shape" or "batch_input_shape" ' +
                                'argument, including the time axis.')

        if self.input_shape[0]:
            # batch size matters, use rnn-based implementation
            def step(x, states):
                output = self.layer(x, train=train)
                return output, []

            last_output, outputs, states = K.rnn(step, X,
                                                 initial_states=[],
                                                 mask=mask)
            y = outputs
        else:
            # no batch size specified, therefore the layer will be able
            # to process batches of any size
            # we can go with reshape-based implementation for performance
            input_shape = self.input_shape
            x = K.reshape(X, (-1, ) + input_shape[2:])  # (nb_samples * timesteps, ...)
            y = self.layer(x, train=False)  # (nb_samples * timesteps, ...)
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(X)[1]
            # (nb_samples, timesteps, ...)
            y = K.reshape(y, (-1, input_length) + self.layer.output_shape[1:])
        return y

    def get_weights(self):
        weights = self.layer.get_weights()
        return weights

    def set_weights(self, weights):
        self.layer.set_weights(weights)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'layer': self.layer.get_config()}
        base_config = super(TimeDistributed, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
