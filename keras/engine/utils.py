# -*- coding: utf-8 -*-
"""Utils for the Keras engine.
"""
import numpy as np


def _standardize_input_data(data, names, shapes=None,
                            check_batch_axis=True,
                            exception_prefix=''):
    """Normalizes inputs and targets provided by users.

    Users may pass data as a list of arrays, dictionary of arrays,
    or as a single array. We normalize this to an ordered list of
    arrays (same order as `names`), while checking that the provided
    arrays have shapes that match the network's expectations.

    # Arguments
        data: User-provided input data (polymorphic).
        names: List of expected array names.
        shapes: Optional list of expected array shapes.
        check_batch_axis: Boolean; whether to check that
            the batch axis of the arrays matches the expected
            value found in `shapes`.
        exception_prefix: String prefix used for exception formatting.

    # Returns
        List of standardized input arrays (one array per model input).

    # Raises
        ValueError: in case of improperly formatted user-provided data.
    """
    if not names:
        if data is not None and hasattr(data, '__len__') and len(data):
            raise ValueError('Error when checking model ' +
                             exception_prefix + ': '
                             'expected no data, but got:', data)
        return []
    if data is None:
        return [None for _ in range(len(names))]

    if isinstance(data, dict):
        try:
            data = [data[x].values if data[x].__class__.__name__ == 'DataFrame' else data[x] for x in names]
        except KeyError as e:
            raise ValueError(
                'No data provided for "' + e.args[0] + '". Need data '
                'for each key in: ' + str(names))
    elif isinstance(data, list):
        if len(names) == 1 and data and isinstance(data[0], (float, int)):
            data = [np.asarray(data)]
        else:
            data = [x.values if x.__class__.__name__ == 'DataFrame' else x for x in data]
    else:
        data = data.values if data.__class__.__name__ == 'DataFrame' else data
        data = [data]
    data = [np.expand_dims(x, 1) if x is not None and x.ndim == 1 else x for x in data]

    if len(data) != len(names):
        if data and hasattr(data[0], 'shape'):
            raise ValueError(
                'Error when checking model ' + exception_prefix +
                ': the list of Numpy arrays that you are passing to '
                'your model is not the size the model expected. '
                'Expected to see ' + str(len(names)) + ' array(s), '
                'but instead got the following list of ' +
                str(len(data)) + ' arrays: ' + str(data)[:200] + '...')
        elif len(names) > 1:
            raise ValueError(
                'Error when checking model ' + exception_prefix +
                ': you are passing a list as input to your model, '
                'but the model expects a list of ' + str(len(names)) +
                ' Numpy arrays instead. The list you passed was: ' +
                str(data)[:200])
        elif len(data) == 1 and not hasattr(data[0], 'shape'):
            raise TypeError(
                'Error when checking model ' + exception_prefix +
                ': data should be a Numpy array, or list/dict of '
                'Numpy arrays. Found: ' + str(data)[:200] + '...')
        elif len(names) == 1:
            data = [np.asarray(data)]

    # Check shapes compatibility.
    if shapes:
        for i in range(len(names)):
            if shapes[i] is not None:
                data_shape = data[i].shape
                shape = shapes[i]
                if data[i].ndim != len(shape):
                    raise ValueError(
                        'Error when checking ' + exception_prefix +
                        ': expected ' + names[i] + ' to have ' +
                        str(len(shape)) + ' dimensions, but got array '
                        'with shape ' + str(data_shape))
                if not check_batch_axis:
                    data_shape = data_shape[1:]
                    shape = shape[1:]
                for dim, ref_dim in zip(data_shape, shape):
                    if ref_dim != dim and ref_dim:
                        raise ValueError(
                            'Error when checking ' + exception_prefix +
                            ': expected ' + names[i] + ' to have shape ' +
                            str(shape) + ' but got array with shape ' +
                            str(data_shape))
    return data


def _standardize_sample_or_class_weights(x_weight, output_names, weight_type):
    """Maps `sample_weight` or `class_weight` to model outputs.

    # Arguments
        x_weight: User-provided `sample_weight` or `class_weight` argument.
        output_names: List of output names (strings) in the model.
        weight_type: A string used purely for exception printing.

    # Returns
        A list of `sample_weight` or `class_weight` where there are exactly
            one element per model output.

    # Raises
        ValueError: In case of invalid user-provided argument.
    """
    if x_weight is None or len(x_weight) == 0:
        return [None for _ in output_names]
    if len(output_names) == 1:
        if isinstance(x_weight, list) and len(x_weight) == 1:
            return x_weight
        if isinstance(x_weight, dict) and output_names[0] in x_weight:
            return [x_weight[output_names[0]]]
        else:
            return [x_weight]
    if isinstance(x_weight, list):
        if len(x_weight) != len(output_names):
            raise ValueError('Provided `' + weight_type + '` was a list of ' +
                             str(len(x_weight)) +
                             ' elements, but the model has ' +
                             str(len(output_names)) + ' outputs. '
                             'You should provide one `' + weight_type + '`'
                             'array per model output.')
        return x_weight
    if isinstance(x_weight, dict):
        x_weights = []
        for name in output_names:
            x_weights.append(x_weight.get(name))
        return x_weights
    else:
        raise TypeError('The model has multiple outputs, so `' +
                        weight_type + '` '
                        'should be either a list or a dict. '
                        'Provided `' + weight_type +
                        '` type not understood: ' +
                        str(x_weight))


def _standardize_class_weights(class_weight, output_names):
    return _standardize_sample_or_class_weights(class_weight,
                                                output_names,
                                                'class_weight')


def _standardize_sample_weights(sample_weight, output_names):
    return _standardize_sample_or_class_weights(sample_weight,
                                                output_names,
                                                'sample_weight')
