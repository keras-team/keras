# -*- coding: utf-8 -*-
from __future__ import absolute_import

import copy
from ..engine import Layer
from ..engine import InputSpec
from .. import backend as K


class Wrapper(Layer):
    """Abstract wrapper base class.

    Wrappers take another layer and augment it in various ways.
    Do not use this class as a layer, it is only an abstract base class.
    Two usable wrappers are the `TimeDistributed` and `Bidirectional` wrappers.

    # Arguments
        layer: The layer to be wrapped.
    """

    def __init__(self, layer, **kwargs):
        self.layer = layer
        super(Wrapper, self).__init__(**kwargs)

    def build(self, input_shape=None):
        # Assumes that self.layer is already set.
        # Should be called at the end of .build() in the children classes.
        self.trainable_weights = getattr(self.layer, 'trainable_weights', [])
        self.non_trainable_weights = getattr(self.layer, 'non_trainable_weights', [])
        self.updates = getattr(self.layer, 'updates', [])
        self.losses = getattr(self.layer, 'losses', [])
        self.constraints = getattr(self.layer, 'constraints', {})
        self.built = True

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
        from . import deserialize as deserialize_layer
        layer = deserialize_layer(config.pop('layer'))
        return cls(layer, **config)


class TimeDistributed(Wrapper):
    """This wrapper allows to apply a layer to every temporal slice of an input.

    The input should be at least 3D, and the dimension of index one
    will be considered to be the temporal dimension.

    Consider a batch of 32 samples,
    where each sample is a sequence of 10 vectors of 16 dimensions.
    The batch input shape of the layer is then `(32, 10, 16)`,
    and the `input_shape`, not including the samples dimension, is `(10, 16)`.

    You can then use `TimeDistributed` to apply a `Dense` layer
    to each of the 10 timesteps, independently:

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

    `TimeDistributed` can be used with arbitrary layers, not just `Dense`,
    for instance with a `Conv2D` layer:

    ```python
        model = Sequential()
        model.add(TimeDistributed(Conv2D(64, (3, 3)),
                                  input_shape=(10, 299, 299, 3)))
    ```

    # Arguments
        layer: a layer instance.
    """

    def __init__(self, layer, **kwargs):
        super(TimeDistributed, self).__init__(layer, **kwargs)
        self.supports_masking = True

    def build(self, input_shape):
        if isinstance(input_shape, tuple):
            input_shape = [input_shape]
        assert all(len(shape) >= 3 for shape in input_shape), "Need 3 dims to TimeDistribute"
        all_timesteps = [i[1] for i in input_shape]
        assert len(set(all_timesteps)) == 1, "Tensors must have same number of timesteps"
        self.input_spec = [InputSpec(shape=shape) for shape in input_shape]
        if not self.layer.built:
            child_input_shape = [(shape[0],) + shape[2:] for shape in input_shape]
            if len(input_shape) == 1:
                child_input_shape = child_input_shape[0]
            self.layer.build(child_input_shape)
            self.layer.built = True
        self.built = True
        super(TimeDistributed, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        child_input_shape = [(shape[0],) + shape[2:] for shape in input_shape]
        timesteps = input_shape[0][1]
        if len(input_shape) == 1:
            child_input_shape = child_input_shape[0]
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    @staticmethod
    def reshape_inputs_and_masks(inputs, masks):
        reshaped_xs = []
        reshaped_masks = []
        for input_i, mask_i in zip(inputs, masks):
            input_shape = K.int_shape(input_i)
            reshaped_x = K.reshape(input_i, (-1,) + input_shape[2:])  # (batch_size * timesteps, ...)
            if mask_i is not None:
                mask_ndim = K.ndim(mask_i)
                input_ndim = K.ndim(input_i)
                if mask_ndim == input_ndim:
                    mask_shape = input_shape
                elif mask_ndim == input_ndim - 1:
                    mask_shape = input_shape[:-1]
                else:
                    raise Exception("Mask is of an unexpected shape. Mask's ndim: %s, input's ndim %s" %
                                    (mask_ndim, input_ndim))
                mask_i = K.reshape(mask_i, (-1,) + mask_shape[2:])  # (batch_size * timesteps, ...)
            reshaped_xs.append(reshaped_x)
            reshaped_masks.append(mask_i)
        if len(inputs) == 1:
            reshaped_xs = reshaped_xs[0]
            reshaped_masks = reshaped_masks[0]
        return reshaped_xs, reshaped_masks

    def call(self, inputs, mask=None):
        if not isinstance(inputs, list):
            inputs = [inputs]
            mask = [mask]
        timesteps = K.int_shape(inputs[0])[1]
        input_shape = [K.int_shape(input_i) for input_i in inputs]
        if len(inputs) == 1:
            input_shape = input_shape[0]
        first_input_shape = self.input_spec[0].shape
        if len(inputs) == 1 and first_input_shape[0]:
            # The batch size is passed when defining the layer in some cases (for
            # example if it is stateful).  We respect the input shape in that
            # case and don't reshape the input. This is slower.  K.rnn also
            # expects only a single tensor, so we can't do this if we have
            # multiple inputs.
            def step(input_i, states):
                output = self.layer.call(input_i)
                return output, []
            _, outputs, _ = K.rnn(step, inputs, mask=mask, input_states=[])
        else:
            reshaped_xs, reshaped_masks = self.reshape_inputs_and_masks(inputs, mask)
            outputs = self.layer.call(reshaped_xs, mask=reshaped_masks)
            output_shape = self.compute_output_shape(input_shape)
            outputs = K.reshape(outputs, (-1, timesteps) + output_shape[2:])
        return outputs

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            if not any(input_mask):
                return None
            else:
                raise RuntimeError("This version of TimeDistributed doesn't "
                                   "handle multiple masked inputs!")
        return input_mask


class Bidirectional(Wrapper):
    """Bidirectional wrapper for RNNs.

    # Arguments
        layer: `Recurrent` instance.
        merge_mode: Mode by which outputs of the
            forward and backward RNNs will be combined.
            One of {'sum', 'mul', 'concat', 'ave', None}.
            If None, the outputs will not be combined,
            they will be returned as a list.

    # Examples

    ```python
        model = Sequential()
        model.add(Bidirectional(LSTM(10, return_sequences=True), input_shape=(5, 10)))
        model.add(Bidirectional(LSTM(10)))
        model.add(Dense(5))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    ```
    """

    def __init__(self, layer, merge_mode='concat', weights=None, **kwargs):
        super(Bidirectional, self).__init__(layer, **kwargs)
        if merge_mode not in ['sum', 'mul', 'ave', 'concat', None]:
            raise ValueError('Invalid merge mode. '
                             'Merge mode should be one of '
                             '{"sum", "mul", "ave", "concat", None}')
        self.forward_layer = copy.copy(layer)
        config = layer.get_config()
        config['go_backwards'] = not config['go_backwards']
        self.backward_layer = layer.__class__.from_config(config)
        self.forward_layer.name = 'forward_' + self.forward_layer.name
        self.backward_layer.name = 'backward_' + self.backward_layer.name
        self.merge_mode = merge_mode
        if weights:
            nw = len(weights)
            self.forward_layer.initial_weights = weights[:nw // 2]
            self.backward_layer.initial_weights = weights[nw // 2:]
        self.stateful = layer.stateful
        self.return_sequences = layer.return_sequences
        self.supports_masking = True

    def get_weights(self):
        return self.forward_layer.get_weights() + self.backward_layer.get_weights()

    def set_weights(self, weights):
        nw = len(weights)
        self.forward_layer.set_weights(weights[:nw // 2])
        self.backward_layer.set_weights(weights[nw // 2:])

    def compute_output_shape(self, input_shape):
        if self.merge_mode in ['sum', 'ave', 'mul']:
            return self.forward_layer.compute_output_shape(input_shape)
        elif self.merge_mode == 'concat':
            shape = list(self.forward_layer.compute_output_shape(input_shape))
            shape[-1] *= 2
            return tuple(shape)
        elif self.merge_mode is None:
            return [self.forward_layer.compute_output_shape(input_shape)] * 2

    def call(self, inputs, mask=None):
        y = self.forward_layer.call(inputs, mask)
        y_rev = self.backward_layer.call(inputs, mask)
        if self.return_sequences:
            y_rev = K.reverse(y_rev, 1)
        if self.merge_mode == 'concat':
            return K.concatenate([y, y_rev])
        elif self.merge_mode == 'sum':
            return y + y_rev
        elif self.merge_mode == 'ave':
            return (y + y_rev) / 2
        elif self.merge_mode == 'mul':
            return y * y_rev
        elif self.merge_mode is None:
            return [y, y_rev]

    def reset_states(self):
        self.forward_layer.reset_states()
        self.backward_layer.reset_states()

    def build(self, input_shape):
        self.forward_layer.build(input_shape)
        self.backward_layer.build(input_shape)
        self.built = True

    def compute_mask(self, inputs, mask):
        if self.return_sequences:
            if not self.merge_mode:
                return [mask, mask]
            else:
                return mask
        else:
            return None

    @property
    def trainable_weights(self):
        if hasattr(self.forward_layer, 'trainable_weights'):
            return (self.forward_layer.trainable_weights +
                    self.backward_layer.trainable_weights)
        return []

    @property
    def non_trainable_weights(self):
        if hasattr(self.forward_layer, 'non_trainable_weights'):
            return (self.forward_layer.non_trainable_weights +
                    self.backward_layer.non_trainable_weights)
        return []

    @property
    def updates(self):
        if hasattr(self.forward_layer, 'updates'):
            return self.forward_layer.updates + self.backward_layer.updates
        return []

    @property
    def losses(self):
        if hasattr(self.forward_layer, 'losses'):
            return self.forward_layer.losses + self.backward_layer.losses
        return []

    @property
    def constraints(self):
        constraints = {}
        if hasattr(self.forward_layer, 'constraints'):
            constraints.update(self.forward_layer.constraints)
            constraints.update(self.backward_layer.constraints)
        return constraints

    def get_config(self):
        config = {'merge_mode': self.merge_mode}
        base_config = super(Bidirectional, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
