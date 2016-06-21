from ..engine import Layer, InputSpec
from .. import initializations
from .. import backend as K


class BatchNormalization(Layer):
    '''Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    # Arguments
        epsilon: small float > 0. Fuzz parameter.
        mode: integer, 0, 1 or 2.
            - 0: feature-wise normalization.
                Each feature map in the input will
                be normalized separately. The axis on which
                to normalize is specified by the `axis` argument.
                Note that if the input is a 4D image tensor
                using Theano conventions (samples, channels, rows, cols)
                then you should set `axis` to `1` to normalize along
                the channels axis.
                During training we use per-batch statistics to normalize
                the data, and during testing we use running averages
                computed during the training phase.
            - 1: sample-wise normalization. This mode assumes a 2D input.
            - 2: feature-wise normalization, like mode 0, but
                using per-batch statistics to normalize the data during both
                testing and training.
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializations](../initializations.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](http://jmlr.org/proceedings/papers/v37/ioffe15.html)
    '''
    def __init__(self, epsilon=1e-6, mode=0, axis=-1, momentum=0.9,
                 weights=None, beta_init='zero', gamma_init='one', **kwargs):
        self.beta_init = initializations.get(beta_init)
        self.gamma_init = initializations.get(gamma_init)
        self.epsilon = epsilon
        self.mode = mode
        self.axis = axis
        self.momentum = momentum
        self.initial_weights = weights
        if self.mode == 0:
            self.uses_learning_phase = True
        super(BatchNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)

        self.gamma = self.gamma_init(shape, name='{}_gamma'.format(self.name))
        self.beta = self.beta_init(shape, name='{}_beta'.format(self.name))
        self.trainable_weights = [self.gamma, self.beta]

        self.running_mean = K.zeros(shape,
                                    name='{}_running_mean'.format(self.name))
        self.running_std = K.ones(shape,
                                  name='{}_running_std'.format(self.name))
        self.non_trainable_weights = [self.running_mean, self.running_std]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True
        self.called_with = None

    def call(self, x, mask=None):
        if self.mode == 0 or self.mode == 2:
            assert self.built, 'Layer must be built before being called'
            input_shape = self.input_spec[0].shape

            reduction_axes = list(range(len(input_shape)))
            del reduction_axes[self.axis]
            broadcast_shape = [1] * len(input_shape)
            broadcast_shape[self.axis] = input_shape[self.axis]

            # case: train mode (uses stats of the current batch)
            mean = K.mean(x, axis=reduction_axes)
            brodcast_mean = K.reshape(mean, broadcast_shape)
            std = K.mean(K.square(x - brodcast_mean) + self.epsilon, axis=reduction_axes)
            std = K.sqrt(std)
            brodcast_std = K.reshape(std, broadcast_shape)
            mean_update = self.momentum * self.running_mean + (1 - self.momentum) * mean
            std_update = self.momentum * self.running_std + (1 - self.momentum) * std

            if self.mode == 2:
                x_normed = (x - brodcast_mean) / (brodcast_std + self.epsilon)
                out = K.reshape(self.gamma, broadcast_shape) * x_normed + K.reshape(self.beta, broadcast_shape)
            else:
                # mode 0
                if self.called_with not in {None, x}:
                    raise Exception('You are attempting to share a '
                                    'same `BatchNormalization` layer across '
                                    'different data flows. '
                                    'This is not possible. '
                                    'You should use `mode=2` in '
                                    '`BatchNormalization`, which has '
                                    'a similar behavior but is shareable '
                                    '(see docs for a description of '
                                    'the behavior).')
                self.called_with = x
                self.updates = [(self.running_mean, mean_update),
                                (self.running_std, std_update)]
                x_normed = (x - brodcast_mean) / (brodcast_std + self.epsilon)

                # case: test mode (uses running averages)
                brodcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
                brodcast_running_std = K.reshape(self.running_std, broadcast_shape)
                x_normed_running = ((x - brodcast_running_mean) / (brodcast_running_std + self.epsilon))

                # pick the normalized form of x corresponding to the training phase
                x_normed = K.in_train_phase(x_normed, x_normed_running)
                out = K.reshape(self.gamma, broadcast_shape) * x_normed + K.reshape(self.beta, broadcast_shape)

        elif self.mode == 1:
            # sample-wise normalization
            m = K.mean(x, axis=-1, keepdims=True)
            std = K.sqrt(K.var(x, axis=-1, keepdims=True) + self.epsilon)
            x_normed = (x - m) / (std + self.epsilon)
            out = self.gamma * x_normed + self.beta
        return out

    def get_config(self):
        config = {"epsilon": self.epsilon,
                  "mode": self.mode,
                  "axis": self.axis,
                  "momentum": self.momentum}
        base_config = super(BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
