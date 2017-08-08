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
