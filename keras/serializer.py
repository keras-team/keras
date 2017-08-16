import numpy as np
import warnings
try:
    import h5py
except ImportError:
    h5py = None


def _slash_get(dictionary, key):
    """Gets the value from the dictionary associated with the key as if the key represented a path.
    For example,

    >>> _slash_get({'a': {'b': 'c'}}, 'a/b')
    'c'

    When the key starts with `/`, the retrieved item is without `/`.

    # Arguments
        dictionary: a dictionary
        key: a string

    # Returns
        an item of the dictionary
    """
    if key.startswith('/'):
        key = key[1:]

    sub_keys = key.split('/')
    if len(sub_keys) == 1:
        dictionary = dictionary[key]
    else:
        for sub_layer in sub_keys:
            dictionary = dictionary[sub_layer]
    return dictionary


def _slash_set(dictionary, key, value):
    """Sets the key of the dictionary to the value as if the key represented a path.
    For example,

    >>> model_dict = {}
    >>> _slash_set(model_dict, 'a/b', 'c')
    model_dict == {'a': {'b': 'c'}}

    When the key starts with `/`, the retrieved item is without `/`.

    # Arguments
        dictionary: a dictionary
        key: a string

    # Returns
        an item of the dictionary
    """
    if key.startswith('/'):
        key = key[1:]

    sub_layers = key.split('/')
    if len(sub_layers) == 1:
        dictionary[key] = value
    else:
        base = dictionary
        for sub_layer in sub_layers[:-1]:
            if sub_layer not in base:
                base[sub_layer] = {}
            base = base[sub_layer]
        base[sub_layers[-1]] = value


def save_dict_to_hdf5_group(f, dictionary):
    """Saves a dictionary to an H5 group `f`.

    # Arguments
        f: an H5 group
        dictionary: a dictionary

    # Returns
        None

    # Raises
        TypeError: if the dictionary contains a non-serializable type
    """
    for key, value in dictionary.items():
        if isinstance(value, list):
            f.attrs[key] = [x.encode('utf-8') for x in value]
        elif isinstance(value, str):
            f.attrs[key] = value.encode('utf-8')
        elif isinstance(value, (np.ndarray, np.generic)):
            param_dset = f.create_dataset(key, value.shape, dtype=value.dtype)
            if not value.shape:
                # scalar
                param_dset[()] = value
            else:
                param_dset[:] = value
        elif isinstance(value, dict):
            save_dict_to_hdf5_group(f.create_group(key), value)
        else:
            raise TypeError('Cannot save type "%s" to HDF5' % type(value))


def load_dict_from_hdf5_group(f):
    """Load a dictionary from an H5 group `f`.

    # Arguments
        f: an H5 group

    # Returns
        a dictionary

    # Raises
        TypeError: if the group contains a non-serializable object
    """
    if isinstance(f, h5py.Dataset):
        return f.value
    # else, it is a group, so we recursively get data
    layers_dict = {}
    for key, value in f.attrs.items():
        if isinstance(value, bytes):
            layers_dict[key] = value.decode('utf-8')
        elif isinstance(value, (np.ndarray, np.generic)):
            if value.dtype.kind == 'S':  # an array of strings
                layers_dict[key] = [x.decode('utf-8') for x in value.tolist()]
            else:
                layers_dict[key] = value
        else:
            raise TypeError('Can\'t decode object "%s"' % type(value))
    for key, value in f.items():
        layers_dict[key] = load_dict_from_hdf5_group(value)
    return layers_dict


def serialize_weights(layers):
    """Convert a list of layers into a dictionary

    # Arguments
        layers: list of layers (e.g. model.layers)

    # Returns
        a dictionary
    """
    from keras import backend as K
    from . import __version__ as keras_version
    result = {'layer_names': [layer.name for layer in layers],
              'backend': K.backend(),
              'keras_version': str(keras_version)}

    for layer in layers:
        result[layer.name] = {}
        symbolic_weights = layer.weights
        weight_values = K.batch_get_value(symbolic_weights)
        weight_names = []
        for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
            if hasattr(w, 'name') and w.name:
                name = str(w.name)
            else:
                name = 'param_' + str(i)
            weight_names.append(name)

        result[layer.name]['weight_names'] = weight_names
        for name, val in zip(weight_names, weight_values):
            _slash_set(result[layer.name], name, val)
    return result


def deserialize_weights(layers_dict, layers):
    """Populates a list of `layer`s with the data from `layers_dict`.

    # Arguments
        layers: list of layers (e.g. model.layers)

    # Returns
        a dictionary

    # Raises
        ValueError: when the number of layers in the dictionary differs from the number of `layers`.
        ValueError: when the number of weights of a layer in the dictionary differs from the number of weights in the corresponding layer.
    """
    from keras import backend as K
    from keras.engine import topology
    filtered_layers = []

    for layer in layers:
        weights = layer.weights
        if weights:
            filtered_layers.append(layer)

    layer_names = [n for n in layers_dict['layer_names']]
    filtered_layer_names = []
    for layer_name in layer_names:
        root = _slash_get(layers_dict, layer_name)
        weight_names = [n for n in root['weight_names']]
        if weight_names:
            filtered_layer_names.append(layer_name)
    layer_names = filtered_layer_names
    if len(layer_names) != len(filtered_layers):
        raise ValueError('You are trying to load a weight dict '
                         'containing ' + str(len(layer_names)) +
                         ' layers into a model with ' +
                         str(len(filtered_layers)) + ' layers.')

    # We batch weight value assignments in a single backend call
    # which provides a speedup in TensorFlow.
    weight_value_tuples = []
    for k, layer_name in enumerate(layer_names):
        root = _slash_get(layers_dict, layer_name)
        weight_names = [n for n in root['weight_names']]
        weight_values = [_slash_get(root, weight_name) for weight_name in weight_names]
        layer = filtered_layers[k]
        symbolic_weights = layer.weights
        weight_values = topology.preprocess_weights_for_loading(layer, weight_values)
        if len(weight_values) != len(symbolic_weights):
            raise ValueError('Layer #' + str(k) +
                             ' (named "' + layer.name +
                             '" in the current model) was found to '
                             'correspond to layer ' + layer_name +
                             ' in the save file. '
                             'However the new layer ' + layer.name +
                             ' expects ' + str(len(symbolic_weights)) +
                             ' weights, but the saved weights have ' +
                             str(len(weight_values)) +
                             ' elements.')
        weight_value_tuples += zip(symbolic_weights, weight_values)
    K.batch_set_value(weight_value_tuples)


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
    from . import layers as layer_module
    if isinstance(config, list):
        raise TypeError('`model_from_config` expects a dictionary, not a list. '
                        'Maybe you meant to use '
                        '`Sequential.from_config(config)`?')
    return layer_module.deserialize(config, custom_objects=custom_objects)


def model_to_dict(model, include_optimizer=True):
    """Converts a model into a dictionary.

    # Arguments
        model: the model
        include_optimizer: whether to include the optimizer (default: True)

    # Returns
        a dictionary
    """
    from . import optimizers
    from .legacy import models as legacy_models
    from keras import backend as K
    from . import __version__ as keras_version

    def convert_custom_objects(obj):
        """Handles custom object lookup.

        # Arguments
            obj: object, dict, or list.

        # Returns
            The same structure, where occurrences of a custom
            object name have been replaced with a string identifying it
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
        if callable(obj):
            return obj.__name__

        # if obj is a python 'type'
        if type(obj).__name__ == type.__name__:
            return obj.__name__

        return obj

    model_dict = {
        'keras_version': str(keras_version),
        'backend': K.backend(),
        'model_config': {
            'class_name': model.__class__.__name__,
            'config': model.get_config()
        }
    }

    if legacy_models.needs_legacy_support(model):
        model_layers = legacy_models.legacy_sequential_layers(model)
    else:
        model_layers = model.layers
    model_dict['model_weights'] = serialize_weights(model_layers)

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
            model_dict['training_config'] = {
                'optimizer_config': {
                    'class_name': model.optimizer.__class__.__name__,
                    'config': convert_custom_objects(model.optimizer.get_config())
                },
                'loss': convert_custom_objects(model.loss),
                'metrics': convert_custom_objects(model.metrics),
                'sample_weight_mode': convert_custom_objects(model.sample_weight_mode),
                'loss_weights': model.loss_weights,
            }
            symbolic_weights = getattr(model.optimizer, 'weights')
            if symbolic_weights:
                weight_values = K.batch_get_value(symbolic_weights)
                weight_names = []
                model_dict['optimizer_weights'] = {}
                for i, (w, val) in enumerate(zip(symbolic_weights, weight_values)):
                    if K.backend() == 'theano' or K.backend() == 'cntk':
                        if hasattr(w, 'name') and w.name != "/variable":
                            name = str(w.name)
                        else:
                            name = 'param_' + str(i)
                    else:
                        if hasattr(w, 'name') and w.name:
                            name = str(w.name)
                        else:
                            name = 'param_' + str(i)
                    weight_names.append(name)
                model_dict['optimizer_weights']['weight_names'] = weight_names
                for name, val in zip(weight_names, weight_values):
                    _slash_set(model_dict['optimizer_weights'], name, val)

    return model_dict


def model_from_dict(model_dict, custom_objects=None, compile=True):
    """Converts dictionary to a model.

    # Arguments
        model_dict: the dictionary (constructed from `model_to_dict`)
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.
        compile: Boolean, whether to compile the model
            after loading.

    # Raises
        ValueError: In case of an invalid dictionary.
    """
    from . import optimizers
    if not custom_objects:
        custom_objects = {}

    if 'model_config' not in model_dict:
        raise ValueError('No model found in config file.')

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

    model = model_from_config(model_dict['model_config'], custom_objects=custom_objects)

    # set weights
    deserialize_weights(model_dict['model_weights'], model.layers)

    # Early return if compilation is not required.
    if not compile:
        return model

    # instantiate optimizer
    if 'training_config' not in model_dict:
        warnings.warn('No training configuration found in save file: '
                      'the model was *not* compiled. Compile it manually.')
        return model
    optimizer_config = model_dict['training_config']['optimizer_config']
    optimizer = optimizers.deserialize(optimizer_config,
                                       custom_objects=custom_objects)

    # Recover loss functions and metrics.
    loss = convert_custom_objects(model_dict['training_config']['loss'])
    metrics = convert_custom_objects(model_dict['training_config']['metrics'])
    sample_weight_mode = model_dict['training_config']['sample_weight_mode']
    loss_weights = model_dict['training_config']['loss_weights']

    # Compile model.
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics,
                  loss_weights=loss_weights,
                  sample_weight_mode=sample_weight_mode)

    # Set optimizer weights.
    if 'optimizer_weights' in model_dict:
        # Build train function (to get weight updates).
        model._make_train_function()
        optimizer_weight_values = [_slash_get(model_dict['optimizer_weights'], n) for n in
                                   model_dict['optimizer_weights']['weight_names']]
        try:
            model.optimizer.set_weights(optimizer_weight_values)
        except ValueError:
            warnings.warn('Error in loading the saved optimizer '
                          'state. As a result, your model is '
                          'starting with a freshly initialized '
                          'optimizer.')
    return model
