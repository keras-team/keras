"""Sequential model class.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import copy

from . import network
from .training import Model
from .base_layer import Layer
from .input_layer import Input
from .input_layer import InputLayer
from .. import backend as K
from .. import layers as layer_module

try:
    import h5py
except ImportError:
    h5py = None


class Sequential(Model):
    """Linear stack of layers.

    # Arguments
        layers: list of layers to add to the model.
        name: Name given to the model

    # Example

    ```python
    # Optionally, the first layer can receive an `input_shape` argument:
    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))

    # Afterwards, we do automatic shape inference:
    model.add(Dense(32))

    # This is identical to the following:
    model = Sequential()
    model.add(Dense(32, input_dim=500))

    # And to the following:
    model = Sequential()
    model.add(Dense(32, batch_input_shape=(None, 500)))

    # Note that you can also omit the `input_shape` argument:
    # In that case the model gets built the first time you call `fit` (or other
    # training and evaluation methods).
    model = Sequential()
    model.add(Dense(32))
    model.add(Dense(32))
    model.compile(optimizer=optimizer, loss=loss)

    # This builds the model for the first time:
    model.fit(x, y, batch_size=32, epochs=10)

    # Note that when using this delayed-build pattern
    # (no input shape specified),
    # the model doesn't have any weights until the first call
    # to a training/evaluation method (since it isn't yet built):
    model = Sequential()
    model.add(Dense(32))
    model.add(Dense(32))
    model.weights  # returns []

    # Whereas if you specify the input shape, the model gets built continuously
    # as you are adding layers:
    model = Sequential()
    model.add(Dense(32, input_shape=(500,)))
    model.add(Dense(32))
    model.weights  # returns list of length 4

    # When using the delayed-build pattern (no input shape specified), you can
    # choose to manually build your model by calling
    # `build(batch_input_shape)`:
    model = Sequential()
    model.add(Dense(32))
    model.add(Dense(32))
    model.build((None, 500))
    model.weights  # returns list of length 4
    ```
    """

    def __init__(self, layers=None, name=None):
        super(Sequential, self).__init__(name=name)
        self._build_input_shape = None

        # Add to the model any layers passed to the constructor.
        if layers:
            for layer in layers:
                self.add(layer)

    @property
    def layers(self):
        # Historically, `sequential.layers` only returns layers that were added
        # via `add`, and omits the auto-generated `InputLayer`
        # that comes at the bottom of the stack.
        if self._layers and isinstance(self._layers[0], InputLayer):
            return self._layers[1:]
        return self._layers

    @property
    def model(self):
        # Historically, `Sequential` was once
        # implemented as a wrapper for `Model` which maintained
        # its underlying `Model` as the `model` property.
        # We keep it for compatibility reasons.
        warnings.warn('`Sequential.model` is deprecated. '
                      '`Sequential` is a subclass of `Model`, you can '
                      'just use your `Sequential` instance directly.')
        return self

    def add(self, layer):
        """Adds a layer instance on top of the layer stack.

        # Arguments
            layer: layer instance.

        # Raises
            TypeError: If `layer` is not a layer instance.
            ValueError: In case the `layer` argument does not
                know its input shape.
            ValueError: In case the `layer` argument has
                multiple output tensors, or is already connected
                somewhere else (forbidden in `Sequential` models).
        """
        if not isinstance(layer, Layer):
            raise TypeError('The added layer must be '
                            'an instance of class Layer. '
                            'Found: ' + str(layer))
        self.built = False
        if not self._layers:
            set_inputs = False
            # First layer in model: check that it is an input layer.
            if not isinstance(layer, InputLayer):
                # Create an input tensor and call `layer` on the input tensor.
                # First, we need to infer the expected input shape and dtype.
                first_layer = layer
                if isinstance(layer, (Model, Sequential)):
                    # We were passed a model as first layer.
                    # This requires a specific way to figure out the
                    # input shape and dtype.
                    if not layer.layers:
                        raise ValueError('Cannot add an empty model '
                                         'to a `Sequential` model.')
                    # In case of nested models: recover the first layer
                    # of the deepest model to infer input shape and dtype.
                    first_layer = layer.layers[0]
                    while isinstance(first_layer, (Model, Sequential)):
                        first_layer = first_layer.layers[0]

                if hasattr(first_layer, 'batch_input_shape'):
                    batch_shape = first_layer.batch_input_shape
                    dtype = first_layer.dtype
                    # Instantiate the input layer.
                    x = Input(
                        batch_shape=batch_shape,
                        dtype=dtype,
                        name=layer.name + '_input')
                    # This will build the current layer
                    # and create the node connecting the current layer
                    # to the input layer we just created.
                    layer(x)
                    set_inputs = True
            else:
                # Corner case where the user passes an InputLayer via `add`.
                assert len(layer._inbound_nodes[-1].output_tensors) == 1
                set_inputs = True

            if set_inputs:
                if len(layer._inbound_nodes[-1].output_tensors) != 1:
                    raise ValueError('All layers in a Sequential model '
                                     'should have a single output tensor. '
                                     'For multi-output layers, '
                                     'use the functional API.')
                self.outputs = [layer._inbound_nodes[-1].output_tensors[0]]
                self.inputs = network.get_source_inputs(self.outputs[0])
        elif self.outputs:
            output_tensor = layer(self.outputs[0])
            if isinstance(output_tensor, list):
                raise TypeError('All layers in a Sequential model '
                                'should have a single output tensor. '
                                'For multi-output layers, '
                                'use the functional API.')
            self.outputs = [output_tensor]
        if self.inputs:
            self.build()
        else:
            self._layers.append(layer)

    def pop(self):
        """Removes the last layer in the model.

        # Raises
            TypeError: if there are no layers in the model.
        """
        if not self.layers:
            raise TypeError('There are no layers in the model.')

        self._layers.pop()
        self.built = False
        if not self.layers:
            self.outputs = None
            self.inputs = None
        elif self.outputs:
            self.layers[-1]._outbound_nodes = []
            self.outputs = [self.layers[-1].output]
            self.build()

    def build(self, input_shape=None):
        if input_shape and not self.inputs:
            batch_shape = tuple(input_shape)
            dtype = K.floatx()
            x = Input(batch_shape=batch_shape,
                      dtype=dtype,
                      name=self.name + '_input')
            self.inputs = [x]
            for layer in self._layers:
                x = layer(x)
            self.outputs = [x]
            self._build_input_shape = input_shape

        if self.inputs:
            self._init_graph_network(self.inputs,
                                     self.outputs,
                                     name=self.name)
            self.built = True

    def predict_proba(self, x, batch_size=32, verbose=0):
        """Generates class probability predictions for the input samples.

        The input samples are processed batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A Numpy array of probability predictions.
        """
        preds = self.predict(x, batch_size, verbose)
        if preds.min() < 0. or preds.max() > 1.:
            warnings.warn('Network returning invalid probability values. '
                          'The last layer might not normalize predictions '
                          'into probabilities '
                          '(like softmax or sigmoid would).')
        return preds

    def predict_classes(self, x, batch_size=32, verbose=0):
        """Generate class predictions for the input samples.

        The input samples are processed batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A numpy array of class predictions.
        """
        proba = self.predict(x, batch_size=batch_size, verbose=verbose)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')

    def get_config(self):
        layer_configs = []
        for layer in self.layers:
            layer_configs.append({
                'class_name': layer.__class__.__name__,
                'config': layer.get_config()
            })
        config = {
            'name': self.name,
            'layers': copy.deepcopy(layer_configs)
        }
        if self._build_input_shape:
            config['build_input_shape'] = self._build_input_shape
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if 'name' in config:
            name = config['name']
            build_input_shape = config.get('build_input_shape')
            layer_configs = config['layers']
        else:  # legacy config file
            name = build_input_shape = None
            layer_configs = config
        model = cls(name=name)
        for conf in layer_configs:
            layer = layer_module.deserialize(conf,
                                             custom_objects=custom_objects)
            model.add(layer)
        if not model.inputs and build_input_shape:
            model.build(build_input_shape)
        return model
