from .core import MaskedLayer
from .. import backend as K


class TimeDistributed(MaskedLayer):

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

        def step(x, states):
            output = self.layer(x, train=train)
            return output, []

        last_output, outputs, states = K.rnn(step, X,
                                             initial_states=[],
                                             mask=mask)
        return outputs

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
