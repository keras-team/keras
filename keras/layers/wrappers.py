from ..engine import Layer
from .. import backend as K


class Wrapper(Layer):

    def __init__(self, layer, **kwargs):
        self.layer = layer
        super(Wrapper, self).__init__(**kwargs)

    @property
    def trainable_weights(self):
        return self.layer.trainable_weights

    @property
    def non_trainable_weights(self):
        return self.layer.non_trainable_weights

    @property
    def regularizers(self):
        return self.layer.regularizers

    @property
    def updates(self):
        return self.layer.updates

    def get_weights(self):
        weights = self.layer.get_weights()
        return weights

    def set_weights(self, weights):
        self.layer.set_weights(weights)

    def get_config(self):
        config = {'name': self.__class__.__name__,
                  'layer': {'class_name': self.layer.__class__.__name__,
                            'config': self.layer.get_config()}}
        base_config = super(TimeDistributed, self).get_config()
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

    def build(self, input_shape):
        assert len(input_shape) >= 3
        if K._BACKEND == 'tensorflow':
            if not self.input_shape[1]:
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
        self.layer.build(child_input_shape)

    def get_output_shape_for(self, input_shape):
        child_input_shape = (input_shape[0], input_shape[2:])
        child_output_shape = self.layer.get_output_shape_for(child_input_shape)
        timesteps = input_shape[1]
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    def call(self, X, mask=None):
        if self.input_shape[0]:
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
            input_shape = self.input_shape
            x = K.reshape(X, (-1, ) + input_shape[2:])  # (nb_samples * timesteps, ...)
            y = self.layer(x, train=False)  # (nb_samples * timesteps, ...)
            input_length = input_shape[1]
            if not input_length:
                input_length = K.shape(X)[1]
            # (nb_samples, timesteps, ...)
            y = K.reshape(y, (-1, input_length) + self.layer.output_shape[1:])
        return y
