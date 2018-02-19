# -*- coding: utf-8 -*-
"""Sequential model class and model-related utilities.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import copy
import json
import os
import yaml
import numpy as np

from . import backend as K
from . import optimizers
from . import layers as layer_module
from .utils.io_utils import ask_to_proceed_with_overwrite
from .utils.generic_utils import has_arg
from .engine.training import Model
from .engine import topology
from .engine.topology import Layer
from .engine.topology import Input
from .engine.topology import InputLayer
from .legacy import layers as legacy_layers
from .legacy import models as legacy_models
from .legacy import interfaces

try:
    import h5py
except ImportError:
    h5py = None


def save_model(model, filepath, overwrite=True, include_optimizer=True):
    """Save a model to a HDF5 file.

    The saved model contains:
        - the model's configuration (topology)
        - the model's weights
        - the model's optimizer's state (if any)

    Thus the saved model can be reinstantiated in
    the exact same state, without any of the code
    used for model definition or training.

    # Arguments
        model: Keras model instance to be saved.
        filepath: String, path where to save the model.
        overwrite: Whether we should overwrite any existing
            model at the target location, or instead
            ask the user with a manual prompt.
        include_optimizer: If True, save optimizer's state together.

    # Raises
        ImportError: if h5py is not available.
    """

    if h5py is None:
        raise ImportError('`save_model` requires h5py.')

    def get_json_type(obj):
        """Serialize any object to a JSON-serializable structure.

        # Arguments
            obj: the object to serialize

        # Returns
            JSON-serializable structure representing `obj`.

        # Raises
            TypeError: if `obj` cannot be serialized.
        """
        # if obj is a serializable Keras class instance
        # e.g. optimizer, layer
        if hasattr(obj, 'get_config'):
            return {'class_name': obj.__class__.__name__,
                    'config': obj.get_config()}

        # if obj is any numpy type
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return {'type': type(obj),
                        'value': obj.tolist()}
            else:
                return obj.item()

        # misc functions (e.g. loss function)
        if callable(obj):
            return obj.__name__

        # if obj is a python 'type'
        if type(obj).__name__ == type.__name__:
            return obj.__name__

        raise TypeError('Not JSON Serializable:', obj)

    from . import __version__ as keras_version

    # If file exists and should not be overwritten.
    if not overwrite and os.path.isfile(filepath):
        proceed = ask_to_proceed_with_overwrite(filepath)
        if not proceed:
            return

    with h5py.File(filepath, mode='w') as f:
        f.attrs['keras_version'] = str(keras_version).encode('utf8')
        f.attrs['backend'] = K.backend().encode('utf8')
        f.attrs['model_config'] = json.dumps({
            'class_name': model.__class__.__name__,
            'config': model.get_config()
        }, default=get_json_type).encode('utf8')

        model_weights_group = f.create_group('model_weights')
        if legacy_models.needs_legacy_support(model):
            model_layers = legacy_models.legacy_sequential_layers(model)
        else:
            model_layers = model.layers
        topology.save_weights_to_hdf5_group(model_weights_group, model_layers)

        if include_optimizer and hasattr(model, 'optimizer'):
            if isinstance(model.optimizer, optimizers.TFOptimizer):
                warnings.warn(
                    'TensorFlow optimizers do not '
                    'make it possible to access '
                    'optimizer attributes or optimizer state '
                    'after instantiation. '
                    'As a result, we cannot save the optimizer '
                    'as part of the model save file.'
                    'You will have to compile your model again '
                    'after loading it. '
                    'Prefer using a Keras optimizer instead '
                    '(see keras.io/optimizers).')
            else:
                f.attrs['training_config'] = json.dumps({
                    'optimizer_config': {
                        'class_name': model.optimizer.__class__.__name__,
                        'config': model.optimizer.get_config()
                    },
                    'loss': model.loss,
                    'metrics': model.metrics,
                    'sample_weight_mode': model.sample_weight_mode,
                    'loss_weights': model.loss_weights,
                }, default=get_json_type).encode('utf8')

                # Save optimizer weights.
                symbolic_weights = getattr(model.optimizer, 'weights')
                if symbolic_weights:
                    optimizer_weights_group = f.create_group('optimizer_weights')
                    weight_values = K.batch_get_value(symbolic_weights)
                    weight_names = []
                    for i, (w, val) in enumerate(zip(symbolic_weights,
                                                     weight_values)):
                        # Default values of symbolic_weights is /variable
                        # for Theano and CNTK
                        if K.backend() == 'theano' or K.backend() == 'cntk':
                            if hasattr(w, 'name'):
                                if w.name.split('/')[-1] == 'variable':
                                    name = str(w.name) + '_' + str(i)
                                else:
                                    name = str(w.name)
                            else:
                                name = 'param_' + str(i)
                        else:
                            if hasattr(w, 'name') and w.name:
                                name = str(w.name)
                            else:
                                name = 'param_' + str(i)
                        weight_names.append(name.encode('utf8'))
                    optimizer_weights_group.attrs['weight_names'] = weight_names
                    for name, val in zip(weight_names, weight_values):
                        param_dset = optimizer_weights_group.create_dataset(
                            name,
                            val.shape,
                            dtype=val.dtype)
                        if not val.shape:
                            # scalar
                            param_dset[()] = val
                        else:
                            param_dset[:] = val
        f.flush()


def load_model(filepath, custom_objects=None, compile=True):
    """Loads a model saved via `save_model`.

    # Arguments
        filepath: String, path to the saved model.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model
            after loading.

    # Returns
        A Keras model instance. If an optimizer was found
        as part of the saved model, the model is already
        compiled. Otherwise, the model is uncompiled and
        a warning will be displayed. When `compile` is set
        to False, the compilation is omitted without any
        warning.

    # Raises
        ImportError: if h5py is not available.
        ValueError: In case of an invalid savefile.
    """
    if h5py is None:
        raise ImportError('`load_model` requires h5py.')

    if not custom_objects:
        custom_objects = {}

    def convert_custom_objects(obj):
        """Handles custom object lookup.

        # Arguments
            obj: object, dict, or list.

        # Returns
            The same structure, where occurrences
                of a custom object name have been replaced
                with the custom object.
        """
        if isinstance(obj, list):
            deserialized = []
            for value in obj:
                deserialized.append(convert_custom_objects(value))
            return deserialized
        if isinstance(obj, dict):
            deserialized = {}
            for key, value in obj.items():
                deserialized[key] = convert_custom_objects(value)
            return deserialized
        if obj in custom_objects:
            return custom_objects[obj]
        return obj
    with h5py.File(filepath, mode='r') as f:
        # instantiate model
        model_config = f.attrs.get('model_config')
        if model_config is None:
            raise ValueError('No model found in config file.')
        model_config = json.loads(model_config.decode('utf-8'))
        model = model_from_config(model_config, custom_objects=custom_objects)

        # set weights
        topology.load_weights_from_hdf5_group(f['model_weights'], model.layers)

        # Early return if compilation is not required.
        if not compile:
            return model

        # instantiate optimizer
        training_config = f.attrs.get('training_config')
        if training_config is None:
            warnings.warn('No training configuration found in save file: '
                          'the model was *not* compiled. Compile it manually.')
            return model
        training_config = json.loads(training_config.decode('utf-8'))
        optimizer_config = training_config['optimizer_config']
        optimizer = optimizers.deserialize(optimizer_config,
                                           custom_objects=custom_objects)

        # Recover loss functions and metrics.
        loss = convert_custom_objects(training_config['loss'])
        metrics = convert_custom_objects(training_config['metrics'])
        sample_weight_mode = training_config['sample_weight_mode']
        loss_weights = training_config['loss_weights']

        # Compile model.
        model.compile(optimizer=optimizer,
                      loss=loss,
                      metrics=metrics,
                      loss_weights=loss_weights,
                      sample_weight_mode=sample_weight_mode)

        # Set optimizer weights.
        if 'optimizer_weights' in f:
            # Build train function (to get weight updates).
            if isinstance(model, Sequential):
                model.model._make_train_function()
            else:
                model._make_train_function()
            optimizer_weights_group = f['optimizer_weights']
            optimizer_weight_names = [n.decode('utf8') for n in
                                      optimizer_weights_group.attrs['weight_names']]
            optimizer_weight_values = [optimizer_weights_group[n] for n in
                                       optimizer_weight_names]
            try:
                model.optimizer.set_weights(optimizer_weight_values)
            except ValueError:
                warnings.warn('Error in loading the saved optimizer '
                              'state. As a result, your model is '
                              'starting with a freshly initialized '
                              'optimizer.')
    return model


def model_from_config(config, custom_objects=None):
    """Instantiates a Keras model from its config.

    # Arguments
        config: Configuration dictionary.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    # Returns
        A Keras model instance (uncompiled).

    # Raises
        TypeError: if `config` is not a dictionary.
    """
    if isinstance(config, list):
        raise TypeError('`model_from_config` expects a dictionary, not a list. '
                        'Maybe you meant to use '
                        '`Sequential.from_config(config)`?')
    return layer_module.deserialize(config, custom_objects=custom_objects)


def model_from_yaml(yaml_string, custom_objects=None):
    """Parses a yaml model configuration file and returns a model instance.

    # Arguments
        yaml_string: YAML string encoding a model configuration.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    # Returns
        A Keras model instance (uncompiled).
    """
    config = yaml.load(yaml_string)
    return layer_module.deserialize(config, custom_objects=custom_objects)


def model_from_json(json_string, custom_objects=None):
    """Parses a JSON model configuration file and returns a model instance.

    # Arguments
        json_string: JSON string encoding a model configuration.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    # Returns
        A Keras model instance (uncompiled).
    """
    config = json.loads(json_string)
    return layer_module.deserialize(config, custom_objects=custom_objects)


class Sequential(Model):
    """Linear stack of layers.

    # Arguments
        layers: list of layers to add to the model.

    # Note
        The first layer passed to a Sequential model
        should have a defined input shape. What that
        means is that it should have received an `input_shape`
        or `batch_input_shape` argument,
        or for some type of layers (recurrent, Dense...)
        an `input_dim` argument.

    # Example

        ```python
            model = Sequential()
            # first layer must have a defined input shape
            model.add(Dense(32, input_dim=500))
            # afterwards, Keras does automatic shape inference
            model.add(Dense(32))

            # also possible (equivalent to the above):
            model = Sequential()
            model.add(Dense(32, input_shape=(500,)))
            model.add(Dense(32))

            # also possible (equivalent to the above):
            model = Sequential()
            # here the batch dimension is None,
            # which means any batch size will be accepted by the model.
            model.add(Dense(32, batch_input_shape=(None, 500)))
            model.add(Dense(32))
        ```
    """

    def __init__(self, layers=None, name=None):
        self.layers = []  # Stack of layers.
        self.model = None  # Internal Model instance.
        self.inputs = []  # List of input tensors
        self.outputs = []  # List of length 1: the output tensor (unique).
        self._trainable = True
        self._initial_weights = None

        # Model attributes.
        self._inbound_nodes = []
        self._outbound_nodes = []
        self.built = False

        # Set model name.
        if not name:
            prefix = 'sequential_'
            name = prefix + str(K.get_uid(prefix))
        self.name = name

        # Add to the model any layers passed to the constructor.
        if layers:
            for layer in layers:
                self.add(layer)

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
        if not self.outputs:
            # First layer in model: check that it is an input layer.
            if not isinstance(layer, (InputLayer, legacy_layers.Merge)):
                # Create an input layer.
                # First, we need to infer its expected input shape and dtype.
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
                    batch_shape = first_layer.batch_input_shape
                    dtype = first_layer.dtype
                else:
                    # We were passed a regular layer, and it should
                    # know about its input shape. Otherwise, that's an error.
                    if not hasattr(layer, 'batch_input_shape'):
                        raise ValueError('The first layer in a '
                                         'Sequential model must '
                                         'get an `input_shape` or '
                                         '`batch_input_shape` argument.')
                    batch_shape = layer.batch_input_shape
                    dtype = layer.dtype
                # Instantiate the input layer.
                x = Input(batch_shape=batch_shape,
                          dtype=dtype,
                          name=layer.name + '_input')
                # This will build the current layer
                # and create the node connecting the current layer
                # to the input layer we just created.
                layer(x)

            if len(layer._inbound_nodes[-1].output_tensors) != 1:
                raise ValueError('All layers in a Sequential model '
                                 'should have a single output tensor. '
                                 'For multi-output layers, '
                                 'use the functional API.')

            self.outputs = [layer._inbound_nodes[-1].output_tensors[0]]
            self.inputs = topology.get_source_inputs(self.outputs[0])

            # We create an input node, which we will keep updated
            # as we add more layers
            topology.Node(outbound_layer=self,
                          inbound_layers=[],
                          node_indices=[],
                          tensor_indices=[],
                          input_tensors=self.inputs,
                          output_tensors=self.outputs,
                          # no model-level masking for now
                          input_masks=[None for _ in self.inputs],
                          output_masks=[None],
                          input_shapes=[x._keras_shape for x in self.inputs],
                          output_shapes=[self.outputs[0]._keras_shape])
        else:
            output_tensor = layer(self.outputs[0])
            if isinstance(output_tensor, list):
                raise TypeError('All layers in a Sequential model '
                                'should have a single output tensor. '
                                'For multi-output layers, '
                                'use the functional API.')
            self.outputs = [output_tensor]
            # update self._inbound_nodes
            self._inbound_nodes[0].output_tensors = self.outputs
            self._inbound_nodes[0].output_shapes = [self.outputs[0]._keras_shape]

        self.layers.append(layer)
        self.built = False

    def pop(self):
        """Removes the last layer in the model.

        # Raises
            TypeError: if there are no layers in the model.
        """
        if not self.layers:
            raise TypeError('There are no layers in the model.')

        self.layers.pop()
        if not self.layers:
            self.outputs = []
            self._inbound_nodes = []
            self._outbound_nodes = []
        else:
            self.layers[-1]._outbound_nodes = []
            self.outputs = [self.layers[-1].output]
            # update self._inbound_nodes
            self._inbound_nodes[0].output_tensors = self.outputs
            self._inbound_nodes[0].output_shapes = [self.outputs[0]._keras_shape]
        self.built = False

    def get_layer(self, name=None, index=None):
        """Retrieve a layer that is part of the model.

        Returns a layer based on either its name (unique)
        or its index in the graph. Indices are based on
        order of horizontal graph traversal (bottom-up).

        # Arguments
            name: string, name of layer.
            index: integer, index of layer.

        # Returns
            A layer instance.
        """
        if not self.built:
            self.build()
        return self.model.get_layer(name, index)

    def call(self, inputs, mask=None):
        if not self.built:
            self.build()
        return self.model.call(inputs, mask)

    def build(self, input_shape=None):
        if not self.inputs or not self.outputs:
            raise TypeError('Sequential model cannot be built: model is empty.'
                            ' Add some layers first.')
        # actually create the model
        self.model = Model(self.inputs, self.outputs[0],
                           name=self.name + '_model')
        self.model.trainable = self.trainable

        # mirror model attributes
        self.supports_masking = self.model.supports_masking
        self._output_mask_cache = self.model._output_mask_cache
        self._output_tensor_cache = self.model._output_tensor_cache
        self._output_shape_cache = self.model._output_shape_cache
        self.input_layers = self.model.input_layers
        self.input_layers_node_indices = self.model.input_layers_node_indices
        self.input_layers_tensor_indices = self.model.input_layers_tensor_indices
        self.output_layers = self.model.output_layers
        self.output_layers_node_indices = self.model.output_layers_node_indices
        self.output_layers_tensor_indices = self.model.output_layers_tensor_indices
        self._nodes_by_depth = self.model._nodes_by_depth
        self.output_names = self.model.output_names
        self.input_names = self.model.input_names
        self._feed_input_names = self.model._feed_input_names
        self._feed_inputs = self.model._feed_inputs
        self._container_nodes = self.model._container_nodes

        # Make sure child model callbacks
        # will call the parent Sequential model.
        self.model.callback_model = self
        self.built = True

    @property
    def uses_learning_phase(self):
        if not self.built:
            self.build()
        return self.model.uses_learning_phase

    @property
    def _flattened_layers(self):
        layers = []
        if self.layers:
            # Support for legacy models
            if isinstance(self.layers[0], legacy_layers.Merge):
                merge = self.layers[0]
                for layer in merge.layers:
                    if hasattr(layer, '_flattened_layers'):
                        for sublayer in layer._flattened_layers:
                            if sublayer not in layers:
                                layers.append(sublayer)
                    elif hasattr(layer, 'layers'):
                        for sublayer in layer.layers:
                            if sublayer not in layers:
                                layers.append(sublayer)
                    else:
                        if layer not in layers:
                            layers.append(layer)
            else:
                if self.layers[0] not in layers:
                    layers.append(self.layers[0])
            for layer in self.layers[1:]:
                if layer not in layers:
                    layers.append(layer)
        return layers

    def _gather_list_attr(self, attr):
        all_attrs = []
        for layer in self._flattened_layers:
            all_attrs += getattr(layer, attr, [])
        return all_attrs

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        if self.built:
            self.model.trainable = value
        self._trainable = value

    @property
    def trainable_weights(self):
        if not self.trainable:
            return []
        # Support for legacy behavior
        return self._gather_list_attr('trainable_weights')

    @property
    def non_trainable_weights(self):
        # Support for legacy behavior
        weights = self._gather_list_attr('non_trainable_weights')
        if not self.trainable:
            trainable_weights = self._gather_list_attr('trainable_weights')
            return trainable_weights + weights
        return weights

    @property
    def updates(self):
        if not self.built:
            self.build()
        return self.model.updates

    @property
    def state_updates(self):
        if not self.built:
            self.build()
        return self.model.state_updates

    def get_updates_for(self, inputs):
        if not self.built:
            self.build()
        return self.model.get_updates_for(inputs)

    @property
    def losses(self):
        if not self.built:
            self.build()
        return self.model.losses

    def get_losses_for(self, inputs):
        if not self.built:
            self.build()
        return self.model.get_losses_for(inputs)

    @property
    def regularizers(self):
        if not self.built:
            self.build()
        return self.model.regularizers

    def get_weights(self):
        """Retrieves the weights of the model.

        # Returns
            A flat list of Numpy arrays
            (one array per model weight).
        """
        # Legacy support
        if legacy_models.needs_legacy_support(self):
            layers = legacy_models.legacy_sequential_layers(self)
            weights = []
            for layer in layers:
                weights.append(layer.get_weights())
            return weights

        if not self.built:
            self.build()
        return self.model.get_weights()

    def set_weights(self, weights):
        """Sets the weights of the model.

        # Arguments
            weights: Should be a list
                of Numpy arrays with shapes and types matching
                the output of `model.get_weights()`.
        """
        # Legacy support
        if legacy_models.needs_legacy_support(self):
            layers = legacy_models.legacy_sequential_layers(self)
            for layer in layers:
                nb_param = len(layer.weights)
                layer.set_weights(weights[:nb_param])
                weights = weights[nb_param:]

        if not self.built:
            self.build()
        self.model.set_weights(weights)

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, reshape=False):
        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        with h5py.File(filepath, mode='r') as f:
            if 'layer_names' not in f.attrs and 'model_weights' in f:
                f = f['model_weights']

            # Legacy support
            if legacy_models.needs_legacy_support(self):
                layers = legacy_models.legacy_sequential_layers(self)
            else:
                layers = self.layers
            if by_name:
                topology.load_weights_from_hdf5_group_by_name(f, layers,
                                                              skip_mismatch=skip_mismatch,
                                                              reshape=reshape)
            else:
                topology.load_weights_from_hdf5_group(f, layers, reshape=reshape)

    def save_weights(self, filepath, overwrite=True):
        if h5py is None:
            raise ImportError('`save_weights` requires h5py.')
        # If file exists and should not be overwritten:
        if not overwrite and os.path.isfile(filepath):
            proceed = ask_to_proceed_with_overwrite(filepath)
            if not proceed:
                return
        # Legacy support
        if legacy_models.needs_legacy_support(self):
            layers = legacy_models.legacy_sequential_layers(self)
        else:
            layers = self.layers

        with h5py.File(filepath, 'w') as f:
            topology.save_weights_to_hdf5_group(f, layers)
            f.flush()

    def compile(self, optimizer, loss,
                metrics=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                target_tensors=None,
                **kwargs):
        """Configures the model for training.

        # Arguments
            optimizer: String (name of optimizer) or optimizer object.
                See [optimizers](/optimizers).
            loss: String (name of objective function) or objective function.
                See [losses](/losses).
                If the model has multiple outputs, you can use a different loss
                on each output by passing a dictionary or a list of losses.
                The loss value that will be minimized by the model
                will then be the sum of all individual losses.
            metrics: List of metrics to be evaluated by the model
                during training and testing.
                Typically you will use `metrics=['accuracy']`.
                To specify different metrics for different outputs of a
                multi-output model, you could also pass a dictionary,
                such as `metrics={'output_a': 'accuracy'}`.
            sample_weight_mode: If you need to do timestep-wise
                sample weighting (2D weights), set this to `"temporal"`.
                `None` defaults to sample-wise weights (1D).
                If the model has multiple outputs, you can use a different
                `sample_weight_mode` on each output by passing a
                dictionary or a list of modes.
            weighted_metrics: List of metrics to be evaluated and weighted
                by sample_weight or class_weight during training and testing.
            target_tensors: By default, Keras will create a placeholder for the
                model's target, which will be fed with the target data during
                training. If instead you would like to use your own
                target tensor (in turn, Keras will not expect external
                Numpy data for these targets at training time), you
                can specify them via the `target_tensors` argument.
                It should be a single tensor
                (for a single-output `Sequential` model).
            **kwargs: When using the Theano/CNTK backends, these arguments
                are passed into `K.function`.
                When using the TensorFlow backend,
                these arguments are passed into `tf.Session.run`.

        # Raises
            ValueError: In case of invalid arguments for
                `optimizer`, `loss`, `metrics` or `sample_weight_mode`.

        # Example
            ```python
                model = Sequential()
                model.add(Dense(32, input_shape=(500,)))
                model.add(Dense(10, activation='softmax'))
                model.compile(optimizer='rmsprop',
                              loss='categorical_crossentropy',
                              metrics=['accuracy'])
            ```
        """
        # create the underlying model
        self.build()
        # call compile method of Model class
        self.model.compile(optimizer, loss,
                           metrics=metrics,
                           sample_weight_mode=sample_weight_mode,
                           weighted_metrics=weighted_metrics,
                           target_tensors=target_tensors,
                           **kwargs)
        self.optimizer = self.model.optimizer
        self.loss = self.model.loss
        self.metrics = self.model.metrics
        self.loss_weights = self.model.loss_weights
        self.sample_weight_mode = self.model.sample_weight_mode
        self.weighted_metrics = self.model.weighted_metrics
        self.targets = self.model.targets
        self.metrics_tensors = self.model.metrics_tensors
        self.metrics_names = self.model.metrics_names
        self.sample_weights = self.model.sample_weights
        self.total_loss = self.model.total_loss

    def fit(self,
            x=None,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            **kwargs):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        # Arguments
            x: Numpy array of training data.
                If the input layer in the model is named, you can also pass a
                dictionary mapping the input name to a Numpy array.
                `x` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            y: Numpy array of target (label) data.
                If the output layer in the model is named, you can also pass a
                dictionary mapping the output name to a Numpy array.
                `y` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, it will default to 32.
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided.
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See [callbacks](/callbacks).
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling.
            validation_data: tuple `(x_val, y_val)` or tuple
                `(x_val, y_val, val_sample_weights)` on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                This will override `validation_split`.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch').
                'batch' is a special option for dealing with the
                limitations of HDF5 data; it shuffles in batch-sized chunks.
                Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify
                `sample_weight_mode="temporal"` in `compile()`.
            initial_epoch: Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined.
            validation_steps: Only relevant if `steps_per_epoch`
                is specified. Total number of steps (batches of samples)
                to validate before stopping.

        # Returns
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        # Raises
            RuntimeError: If the model was never compiled.
            ValueError: In case of mismatch between the provided input data
                and what the model expects.
        """
        # Legacy support
        if 'nb_epoch' in kwargs:
            warnings.warn('The `nb_epoch` argument in `fit` '
                          'has been renamed `epochs`.')
            epochs = kwargs.pop('nb_epoch')
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        if not self.built:
            raise RuntimeError('The model needs to be compiled '
                               'before being used.')
        return self.model.fit(x, y,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              callbacks=callbacks,
                              validation_split=validation_split,
                              validation_data=validation_data,
                              shuffle=shuffle,
                              class_weight=class_weight,
                              sample_weight=sample_weight,
                              initial_epoch=initial_epoch,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps)

    def evaluate(self, x=None, y=None,
                 batch_size=None,
                 verbose=1,
                 sample_weight=None,
                 steps=None):
        """Computes the loss on some input data, batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
                `x` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            y: labels, as a Numpy array.
                `y` can be `None` (default) if feeding from
                framework-native tensors (e.g. TensorFlow data tensors).
            batch_size: Integer. If unspecified, it will default to 32.
            verbose: verbosity mode, 0 or 1.
            sample_weight: sample weights, as a Numpy array.
            steps: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring the evaluation round finished.
                Ignored with the default value of `None`.

        # Returns
            Scalar test loss (if the model has no metrics)
            or list of scalars (if the model computes other metrics).
            The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.

        # Raises
            RuntimeError: if the model was never compiled.
        """
        if not self.built:
            raise RuntimeError('The model needs to be compiled '
                               'before being used.')
        return self.model.evaluate(x, y,
                                   batch_size=batch_size,
                                   verbose=verbose,
                                   sample_weight=sample_weight,
                                   steps=steps)

    def predict(self, x, batch_size=None, verbose=0, steps=None):
        """Generates output predictions for the input samples.

        The input samples are processed batch by batch.

        # Arguments
            x: the input data, as a Numpy array.
            batch_size: Integer. If unspecified, it will default to 32.
            verbose: verbosity mode, 0 or 1.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.

        # Returns
            A Numpy array of predictions.
        """
        if not self.built:
            self.build()
        return self.model.predict(x, batch_size=batch_size, verbose=verbose,
                                  steps=steps)

    def predict_on_batch(self, x):
        """Returns predictions for a single batch of samples.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).

        # Returns
            A Numpy array of predictions.
        """
        if not self.built:
            self.build()
        return self.model.predict_on_batch(x)

    def train_on_batch(self, x, y, class_weight=None,
                       sample_weight=None):
        """Single gradient update over one batch of samples.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            y: labels, as a Numpy array.
            class_weight: dictionary mapping classes to a weight value,
                used for scaling the loss function (during training only).
            sample_weight: sample weights, as a Numpy array.

        # Returns
            Scalar training loss (if the model has no metrics)
            or list of scalars (if the model computes other metrics).
            The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.

        # Raises
            RuntimeError: if the model was never compiled.
        """
        if not self.built:
            raise RuntimeError('The model needs to be compiled '
                               'before being used.')
        return self.model.train_on_batch(x, y,
                                         sample_weight=sample_weight,
                                         class_weight=class_weight)

    def test_on_batch(self, x, y,
                      sample_weight=None):
        """Evaluates the model over a single batch of samples.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            y: labels, as a Numpy array.
            sample_weight: sample weights, as a Numpy array.

        # Returns
            Scalar test loss (if the model has no metrics)
            or list of scalars (if the model computes other metrics).
            The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.

        # Raises
            RuntimeError: if the model was never compiled.
        """
        if not self.built:
            raise RuntimeError('The model needs to be compiled '
                               'before being used.')
        return self.model.test_on_batch(x, y,
                                        sample_weight=sample_weight)

    def predict_proba(self, x, batch_size=None, verbose=0, steps=None):
        """Generates class probability predictions for the input samples.

        The input samples are processed batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            verbose: verbosity mode, 0 or 1.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.


        # Returns
            A Numpy array of probability predictions.
        """
        preds = self.predict(x, batch_size, verbose, steps=steps)
        if preds.min() < 0. or preds.max() > 1.:
            warnings.warn('Network returning invalid probability values. '
                          'The last layer might not normalize predictions '
                          'into probabilities '
                          '(like softmax or sigmoid would).')
        return preds

    def predict_classes(self, x, batch_size=None, verbose=0, steps=None):
        """Generate class predictions for the input samples.

        The input samples are processed batch by batch.

        # Arguments
            x: input data, as a Numpy array or list of Numpy arrays
                (if the model has multiple inputs).
            batch_size: Integer. If unspecified, it will default to 32.
            verbose: verbosity mode, 0 or 1.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.

        # Returns
            A numpy array of class predictions.
        """
        proba = self.predict(x, batch_size=batch_size, verbose=verbose,
                             steps=steps)
        if proba.shape[-1] > 1:
            return proba.argmax(axis=-1)
        else:
            return (proba > 0.5).astype('int32')

    @interfaces.legacy_generator_methods_support
    def fit_generator(self, generator,
                      steps_per_epoch=None,
                      epochs=1,
                      verbose=1,
                      callbacks=None,
                      validation_data=None,
                      validation_steps=None,
                      class_weight=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      shuffle=True,
                      initial_epoch=0):
        """Fits the model on data generated batch-by-batch by a Python generator.

        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.

        # Arguments
            generator: A generator.
                The output of the generator must be either
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
                All arrays should contain the same number of samples.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `steps_per_epoch`
                batches have been seen by the model.
            steps_per_epoch: Total number of steps (batches of samples)
                to yield from `generator` before declaring one epoch
                finished and starting the next epoch. It should typically
                be equal to the number of samples of your dataset
                divided by the batch size.
                Optional for `Sequence`: if unspecified, will use
                the `len(generator)` as a number of steps.
            epochs: Integer, total number of iterations on the data.
                Note that in conjunction with initial_epoch, the parameter
                epochs is to be understood as "final epoch". The model is
                not trained for n steps given by epochs, but until the
                epoch epochs is reached.
            verbose: Verbosity mode, 0, 1, or 2.
            callbacks: List of callbacks to be called during training.
            validation_data: This can be either
                - A generator for the validation data
                - A tuple (inputs, targets)
                - A tuple (inputs, targets, sample_weights).
            validation_steps: Only relevant if `validation_data`
                is a generator.
                Number of steps to yield from validation generator
                at the end of every epoch. It should typically
                be equal to the number of samples of your
                validation dataset divided by the batch size.
                Optional for `Sequence`: if unspecified, will use
                the `len(validation_data)` as a number of steps.
            class_weight: Dictionary mapping class indices to a weight
                for the class.
            max_queue_size: Maximum size for the generator queue
            workers: Maximum number of processes to spin up
            use_multiprocessing: if True, use process based threading.
                Note that because
                this implementation relies on multiprocessing,
                you should not pass
                non picklable arguments to the generator
                as they can't be passed
                easily to children processes.
            shuffle: Whether to shuffle the order of the batches at
                the beginning of each epoch. Only used with instances
                of `Sequence` (keras.utils.Sequence).
            initial_epoch: Epoch at which to start training
                (useful for resuming a previous training run).

        # Returns
            A `History` object.

        # Raises
            RuntimeError: if the model was never compiled.

        # Example

        ```python
            def generate_arrays_from_file(path):
                while 1:
                    with open(path) as f:
                        for line in f:
                            # create Numpy arrays of input data
                            # and labels, from each line in the file
                            x, y = process_line(line)
                            yield (x, y)

            model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                                steps_per_epoch=1000, epochs=10)
        ```
        """
        if not self.built:
            raise RuntimeError('The model needs to be compiled '
                               'before being used.')
        return self.model.fit_generator(generator,
                                        steps_per_epoch,
                                        epochs,
                                        verbose=verbose,
                                        callbacks=callbacks,
                                        validation_data=validation_data,
                                        validation_steps=validation_steps,
                                        class_weight=class_weight,
                                        max_queue_size=max_queue_size,
                                        workers=workers,
                                        use_multiprocessing=use_multiprocessing,
                                        shuffle=shuffle,
                                        initial_epoch=initial_epoch)

    @interfaces.legacy_generator_methods_support
    def evaluate_generator(self, generator, steps=None,
                           max_queue_size=10, workers=1,
                           use_multiprocessing=False):
        """Evaluates the model on a data generator.

        The generator should return the same kind of data
        as accepted by `test_on_batch`.

        # Arguments
            generator: Generator yielding tuples (inputs, targets)
                or (inputs, targets, sample_weights)
            steps: Total number of steps (batches of samples)
                to yield from `generator` before stopping.
                Optional for `Sequence`: if unspecified, will use
                the `len(generator)` as a number of steps.
            max_queue_size: maximum size for the generator queue
            workers: maximum number of processes to spin up
            use_multiprocessing: if True, use process based threading.
                Note that because this implementation
                relies on multiprocessing, you should not pass
                non picklable arguments to the generator
                as they can't be passed easily to children processes.

        # Returns
            Scalar test loss (if the model has no metrics)
            or list of scalars (if the model computes other metrics).
            The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.

        # Raises
            RuntimeError: if the model was never compiled.
        """
        if not self.built:
            raise RuntimeError('The model needs to be compiled '
                               'before being used.')
        return self.model.evaluate_generator(generator,
                                             steps,
                                             max_queue_size=max_queue_size,
                                             workers=workers,
                                             use_multiprocessing=use_multiprocessing)

    @interfaces.legacy_generator_methods_support
    def predict_generator(self, generator, steps=None,
                          max_queue_size=10, workers=1,
                          use_multiprocessing=False, verbose=0):
        """Generates predictions for the input samples from a data generator.

        The generator should return the same kind of data as accepted by
        `predict_on_batch`.

        # Arguments
            generator: generator yielding batches of input samples.
            steps: Total number of steps (batches of samples)
                to yield from `generator` before stopping.
                Optional for `Sequence`: if unspecified, will use
                the `len(generator)` as a number of steps.
            max_queue_size: maximum size for the generator queue
            workers: maximum number of processes to spin up
            use_multiprocessing: if True, use process based threading.
                Note that because this implementation
                relies on multiprocessing, you should not pass
                non picklable arguments to the generator
                as they can't be passed easily to children processes.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A Numpy array of predictions.
        """
        if not self.built:
            self.build()
        return self.model.predict_generator(generator, steps,
                                            max_queue_size=max_queue_size,
                                            workers=workers,
                                            use_multiprocessing=use_multiprocessing,
                                            verbose=verbose)

    def get_config(self):
        if isinstance(self.layers[0], legacy_layers.Merge):
            return self.legacy_get_config()

        config = []
        for layer in self.layers:
            config.append({'class_name': layer.__class__.__name__,
                           'config': layer.get_config()})
        return copy.deepcopy(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if 'class_name' not in config[0] or config[0]['class_name'] == 'Merge':
            return cls.legacy_from_config(config)

        model = cls()
        for conf in config:
            layer = layer_module.deserialize(conf, custom_objects=custom_objects)
            model.add(layer)
        return model

    def legacy_get_config(self):
        """Retrieves the model configuration as a Python list.

        # Returns
            A list of dicts (each dict is a layer config).
        """
        config = []
        if isinstance(self.layers[0], legacy_layers.Merge):
            assert hasattr(self.layers[0], 'layers')
            layers = []
            for layer in self.layers[0].layers:
                layer_config = {'class_name': layer.__class__.__name__,
                                'config': layer.get_config()}
                layers.append(layer_config)
            merge_config = self.layers[0].get_config()
            merge_config['layers'] = layers
            config.append({'class_name': 'Merge', 'config': merge_config})
        else:
            config.append({'class_name': self.layers[0].__class__.__name__,
                           'config': self.layers[0].get_config()})
        for layer in self.layers[1:]:
            config.append({'class_name': layer.__class__.__name__,
                           'config': layer.get_config()})
        return copy.deepcopy(config)

    @classmethod
    def legacy_from_config(cls, config, layer_cache=None):
        """Load a model from a legacy configuration.

        # Arguments
            config: dictionary with configuration.
            layer_cache: cache to draw pre-existing layer.

        # Returns
            The loaded Model.
        """
        if not layer_cache:
            layer_cache = {}

        def normalize_legacy_config(conf):
            if 'class_name' not in conf:
                class_name = conf['name']
                name = conf.get('custom_name')
                conf['name'] = name
                return {'class_name': class_name,
                        'config': conf}
            return conf

        # the model we will return
        model = cls()

        def get_or_create_layer(layer_data):
            name = layer_data['config'].get('name')
            if name in layer_cache:
                return layer_cache[name]
            layer = layer_module.deserialize(layer_data)
            layer_cache[name] = layer
            return layer

        first_layer = config[0]
        first_layer = normalize_legacy_config(first_layer)
        if first_layer['class_name'] == 'Merge':
            merge_inputs = []
            first_layer_config = first_layer['config']
            for merge_input_config in first_layer_config.pop('layers'):
                merge_input = layer_module.deserialize(merge_input_config)
                merge_inputs.append(merge_input)
            first_layer_config['layers'] = merge_inputs
            merge = legacy_layers.Merge.from_config(first_layer_config)
            model.add(merge)
        else:
            layer = get_or_create_layer(first_layer)
            model.add(layer)

        for conf in config[1:]:
            conf = normalize_legacy_config(conf)
            layer = get_or_create_layer(conf)
            model.add(layer)
        return model


def _clone_functional_model(model, input_tensors=None):
    """Clone a functional `Model` instance.

    Model cloning is similar to calling a model on new inputs,
    except that it creates new layers (and thus new weights) instead
    of sharing the weights of the existing layers.

    # Arguments
        model: Instance of `Model`.
        input_tensors: optional list of input tensors
            to build the model upon. If not provided,
            placeholders will be created.

    # Returns
        An instance of `Model` reproducing the behavior
        of the original model, on top of new inputs tensors,
        using newly instantiated weights.

    # Raises
        ValueError: in case of invalid `model` argument value.
    """
    if not isinstance(model, Model):
        raise ValueError('Expected `model` argument '
                         'to be a `Model` instance, got ', model)
    if isinstance(model, Sequential):
        raise ValueError('Expected `model` argument '
                         'to be a functional `Model` instance, '
                         'got a `Sequential` instance instead:', model)

    layer_map = {}  # Cache for created layers.
    tensor_map = {}  # Map {reference_tensor: (corresponding_tensor, mask)}
    if input_tensors is None:
        # Create placeholders to build the model on top of.
        input_layers = []
        input_tensors = []
        for layer in model.input_layers:
            input_tensor = Input(batch_shape=layer.batch_input_shape,
                                 dtype=layer.dtype,
                                 sparse=layer.sparse,
                                 name=layer.name)
            input_tensors.append(input_tensor)
            # Cache newly created input layer.
            newly_created_input_layer = input_tensor._keras_history[0]
            layer_map[layer] = newly_created_input_layer
        for original_input_layer, cloned_input_layer in zip(model.input_layers, input_layers):
            layer_map[original_input_layer] = cloned_input_layer
    else:
        # Make sure that all input tensors come from a Keras layer.
        # If tensor comes from an input layer: cache the input layer.
        input_tensors = topology._to_list(input_tensors)
        _input_tensors = []
        for i, x in enumerate(input_tensors):
            if not K.is_keras_tensor(x):
                name = model.input_layers[i].name
                input_tensor = Input(tensor=x,
                                     name='input_wrapper_for_' + name)
                _input_tensors.append(input_tensor)
                # Cache newly created input layer.
                original_input_layer = x._keras_history[0]
                newly_created_input_layer = input_tensor._keras_history[0]
                layer_map[original_input_layer] = newly_created_input_layer
            else:
                _input_tensors.append(x)
        input_tensors = _input_tensors

    for x, y in zip(model.inputs, input_tensors):
        tensor_map[x] = (y, None)  # tensor, mask

    # Iterated over every node in the reference model, in depth order.
    depth_keys = list(model._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
        nodes = model._nodes_by_depth[depth]
        for node in nodes:
            # Recover the corresponding layer.
            layer = node.outbound_layer

            # Get or create layer.
            if layer not in layer_map:
                # Clone layer.
                new_layer = layer.__class__.from_config(layer.get_config())
                layer_map[layer] = new_layer
                layer = new_layer
            else:
                # Reuse previously cloned layer.
                layer = layer_map[layer]
                # Don't call InputLayer multiple times.
                if isinstance(layer, topology.InputLayer):
                    continue

            # Gather inputs to call the new layer.
            reference_input_tensors = node.input_tensors
            reference_output_tensors = node.output_tensors

            # If all previous input tensors are available in tensor_map,
            # then call node.inbound_layer on them.
            computed_data = []  # List of tuples (input, mask).
            for x in reference_input_tensors:
                if x in tensor_map:
                    computed_data.append(tensor_map[x])

            if len(computed_data) == len(reference_input_tensors):
                # Call layer.
                if node.arguments:
                    kwargs = node.arguments
                else:
                    kwargs = {}
                if len(computed_data) == 1:
                    computed_tensor, computed_mask = computed_data[0]
                    if has_arg(layer.call, 'mask'):
                        if 'mask' not in kwargs:
                            kwargs['mask'] = computed_mask
                    output_tensors = topology._to_list(
                        layer(computed_tensor, **kwargs))
                    output_masks = topology._to_list(
                        layer.compute_mask(computed_tensor,
                                           computed_mask))
                    computed_tensors = [computed_tensor]
                    computed_masks = [computed_mask]
                else:
                    computed_tensors = [x[0] for x in computed_data]
                    computed_masks = [x[1] for x in computed_data]
                    if has_arg(layer.call, 'mask'):
                        if 'mask' not in kwargs:
                            kwargs['mask'] = computed_masks
                    output_tensors = topology._to_list(
                        layer(computed_tensors, **kwargs))
                    output_masks = topology._to_list(
                        layer.compute_mask(computed_tensors,
                                           computed_masks))
                # Update tensor_map.
                for x, y, mask in zip(reference_output_tensors,
                                      output_tensors,
                                      output_masks):
                    tensor_map[x] = (y, mask)

    # Check that we did compute the model outputs,
    # then instantiate a new model from inputs and outputs.
    output_tensors = []
    for x in model.outputs:
        assert x in tensor_map, 'Could not compute output ' + str(x)
        tensor, _ = tensor_map[x]
        output_tensors.append(tensor)
    return Model(input_tensors, output_tensors, name=model.name)


def _clone_sequential_model(model, input_tensors=None):
    """Clone a `Sequential` model instance.

    Model cloning is similar to calling a model on new inputs,
    except that it creates new layers (and thus new weights) instead
    of sharing the weights of the existing layers.

    # Arguments
        model: Instance of `Sequential`.
        input_tensors: optional list of input tensors
            to build the model upon. If not provided,
            placeholders will be created.

    # Returns
        An instance of `Sequential` reproducing the behavior
        of the original model, on top of new inputs tensors,
        using newly instantiated weights.

    # Raises
        ValueError: in case of invalid `model` argument value.
    """
    if not isinstance(model, Sequential):
        raise ValueError('Expected `model` argument '
                         'to be a `Sequential` model instance, '
                         'but got:', model)

    def clone(layer):
        return layer.__class__.from_config(layer.get_config())

    layers = [clone(layer) for layer in model.layers]
    if input_tensors is None:
        return Sequential(layers=layers, name=model.name)
    else:
        if len(topology._to_list(input_tensors)) != 1:
            raise ValueError('To clone a `Sequential` model, we expect '
                             ' at most one tensor '
                             'as part of `input_tensors`.')
        x = topology._to_list(input_tensors)[0]
        if K.is_keras_tensor(x):
            origin_layer = x._keras_history[0]
            if isinstance(origin_layer, topology.InputLayer):
                return Sequential(layers=[origin_layer] + layers,
                                  name=model.name)
            else:
                raise ValueError('Cannot clone a `Sequential` model on top '
                                 'of a tensor that comes from a Keras layer '
                                 'other than an `InputLayer`. '
                                 'Use the functional API instead.')
        input_tensor = Input(tensor=x,
                             name='input_wrapper_for_' + str(x.name))
        input_layer = input_tensor._keras_history[0]
        return Sequential(layers=[input_layer] + layers, name=model.name)


def clone_model(model, input_tensors=None):
    """Clone any `Model` instance.

    Model cloning is similar to calling a model on new inputs,
    except that it creates new layers (and thus new weights) instead
    of sharing the weights of the existing layers.

    # Arguments
        model: Instance of `Model`
            (could be a functional model or a Sequential model).
        input_tensors: optional list of input tensors
            to build the model upon. If not provided,
            placeholders will be created.

    # Returns
        An instance of `Model` reproducing the behavior
        of the original model, on top of new inputs tensors,
        using newly instantiated weights.

    # Raises
        ValueError: in case of invalid `model` argument value.
    """
    if isinstance(model, Sequential):
        return _clone_sequential_model(model, input_tensors=input_tensors)
    else:
        return _clone_functional_model(model, input_tensors=input_tensors)
