"""Training-related utilities.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect
import collections
import copy
import numpy as np
import six
import warnings
from collections import OrderedDict

from .. import backend as K
from .. import losses
from .. import metrics as metrics_module
from ..utils import Sequence
from ..utils import generic_utils
from ..utils import losses_utils


def standardize_single_array(x):
    if x is None:
        return None
    elif K.is_tensor(x):
        shape = K.int_shape(x)
        if shape is None or shape[0] is None:
            raise ValueError(
                'When feeding symbolic tensors to a model, we expect the '
                'tensors to have a static batch size. '
                'Got tensor with shape: %s' % str(shape))
        return x
    elif x.ndim == 1:
        x = np.expand_dims(x, 1)
    return x


def standardize_input_data(data,
                           names,
                           shapes=None,
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
            data = [
                data[x].values
                if data[x].__class__.__name__ == 'DataFrame' else data[x]
                for x in names
            ]
        except KeyError as e:
            raise ValueError('No data provided for "' + e.args[0] +
                             '". Need data '
                             'for each key in: ' + str(names))
    elif isinstance(data, list):
        if isinstance(data[0], list):
            data = [np.asarray(d) for d in data]
        elif len(names) == 1 and isinstance(data[0], (float, int)):
            data = [np.asarray(data)]
        else:
            data = [
                x.values if x.__class__.__name__ == 'DataFrame'
                else x for x in data
            ]
    else:
        data = data.values if data.__class__.__name__ == 'DataFrame' else data
        data = [data]
    data = [standardize_single_array(x) for x in data]

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
                ' Numpy arrays instead. '
                'The list you passed was: ' + str(data)[:200])
        elif len(data) == 1 and not hasattr(data[0], 'shape'):
            raise TypeError('Error when checking model ' + exception_prefix +
                            ': data should be a Numpy array, or list/dict of '
                            'Numpy arrays. Found: ' + str(data)[:200] + '...')
        elif len(names) == 1:
            data = [np.asarray(data)]

    # Check shapes compatibility.
    if shapes:
        for i in range(len(names)):
            if shapes[i] is not None and not K.is_tensor(data[i]):
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


def standardize_sample_or_class_weights(x_weight,
                                        output_names,
                                        weight_type):
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


def standardize_class_weights(class_weight, output_names):
    return standardize_sample_or_class_weights(class_weight,
                                               output_names,
                                               'class_weight')


def standardize_sample_weights(sample_weight, output_names):
    return standardize_sample_or_class_weights(sample_weight,
                                               output_names,
                                               'sample_weight')


def check_array_length_consistency(inputs, targets, weights=None):
    """Checks if batch axes are the same for Numpy arrays.

    # Arguments
        inputs: list of Numpy arrays of inputs.
        targets: list of Numpy arrays of targets.
        weights: list of Numpy arrays of sample weights.

    # Raises
        ValueError: in case of incorrectly formatted data.
    """
    def set_of_lengths(x):
        # return a set with the variation between
        # different shapes, with None => 0
        if x is None:
            return {0}
        else:
            return set([0 if y is None else int(y.shape[0]) for y in x])

    set_x = set_of_lengths(inputs)
    set_y = set_of_lengths(targets)
    set_w = set_of_lengths(weights)
    if len(set_x) > 1:
        raise ValueError('All input arrays (x) should have '
                         'the same number of samples. Got array shapes: ' +
                         str([x.shape for x in inputs]))
    if len(set_y) > 1:
        raise ValueError('All target arrays (y) should have '
                         'the same number of samples. Got array shapes: ' +
                         str([y.shape for y in targets]))
    if set_x and set_y and list(set_x)[0] != list(set_y)[0]:
        raise ValueError('Input arrays should have '
                         'the same number of samples as target arrays. '
                         'Found ' + str(list(set_x)[0]) + ' input samples '
                         'and ' + str(list(set_y)[0]) + ' target samples.')
    if len(set_w) > 1:
        raise ValueError('All sample_weight arrays should have '
                         'the same number of samples. Got array shapes: ' +
                         str([w.shape for w in weights]))
    if set_y and set_w and list(set_y)[0] != list(set_w)[0]:
        raise ValueError('Sample_weight arrays should have '
                         'the same number of samples as target arrays. Got ' +
                         str(list(set_y)[0]) + ' input samples and ' +
                         str(list(set_w)[0]) + ' target samples.')


def check_loss_and_target_compatibility(targets, loss_fns, output_shapes):
    """Does validation on the compatibility of targets and loss functions.

    This helps prevent users from using loss functions incorrectly. This check
    is purely for UX purposes.

    # Arguments
        targets: list of Numpy arrays of targets.
        loss_fns: list of loss functions.
        output_shapes: list of shapes of model outputs.

    # Raises
        ValueError: if a loss function or target array
            is incompatible with an output.
    """
    key_loss_fns = {
        losses.mean_squared_error, losses.binary_crossentropy,
        losses.categorical_crossentropy
    }
    key_loss_classes = (losses.MeanSquaredError, losses.BinaryCrossentropy,
                        losses.CategoricalCrossentropy)
    for y, loss, shape in zip(targets, loss_fns, output_shapes):
        if y is None or loss is None:
            continue
        if losses.is_categorical_crossentropy(loss):
            if y.shape[-1] == 1:
                raise ValueError(
                    'You are passing a target array of shape ' + str(y.shape) +
                    ' while using as loss `categorical_crossentropy`. '
                    '`categorical_crossentropy` expects '
                    'targets to be binary matrices (1s and 0s) '
                    'of shape (samples, classes). '
                    'If your targets are integer classes, '
                    'you can convert them to the expected format via:\n'
                    '```\n'
                    'from keras.utils import to_categorical\n'
                    'y_binary = to_categorical(y_int)\n'
                    '```\n'
                    '\n'
                    'Alternatively, you can use the loss function '
                    '`sparse_categorical_crossentropy` instead, '
                    'which does expect integer targets.')
        is_loss_wrapper = isinstance(loss, losses.LossFunctionWrapper)
        if (isinstance(loss, key_loss_classes) or (is_loss_wrapper and
                                                   (loss.fn in key_loss_fns))):
            for target_dim, out_dim in zip(y.shape[1:], shape[1:]):
                if out_dim is not None and target_dim != out_dim:
                    loss_name = loss.name
                    if loss_name is None:
                        loss_type = loss.fn if is_loss_wrapper else type(loss)
                        loss_name = loss_type.__name__
                    raise ValueError(
                        'A target array with shape ' + str(y.shape) +
                        ' was passed for an output of shape ' + str(shape) +
                        ' while using as loss `' + loss_name + '`. '
                        'This loss expects targets to have the same shape '
                        'as the output.')


def check_generator_arguments(y=None, sample_weight=None,
                              validation_split=None):
    """Validates arguments passed when using a generator."""
    if y is not None:
        raise ValueError('`y` argument is not supported when data is'
                         'a generator or Sequence instance. Instead pass targets'
                         ' as the second element of the generator.')
    if sample_weight is not None:
        raise ValueError('`sample_weight` argument is not supported when data is'
                         'a generator or Sequence instance. Instead pass sample'
                         ' weights as the third element of the generator.')
    if validation_split:
        raise ValueError('If your data is in the form of a Python generator, '
                         'you cannot use `validation_split`.')


def batch_shuffle(index_array, batch_size):
    """Shuffles an array in a batch-wise fashion.

    Useful for shuffling HDF5 arrays
    (where one cannot access arbitrary indices).

    # Arguments
        index_array: array of indices to be shuffled.
        batch_size: integer.

    # Returns
        The `index_array` array, shuffled in a batch-wise fashion.
    """
    batch_count = int(len(index_array) / batch_size)
    # to reshape we need to be cleanly divisible by batch size
    # we stash extra items and reappend them after shuffling
    last_batch = index_array[batch_count * batch_size:]
    index_array = index_array[:batch_count * batch_size]
    index_array = index_array.reshape((batch_count, batch_size))
    np.random.shuffle(index_array)
    index_array = index_array.flatten()
    return np.append(index_array, last_batch)


def make_batches(size, batch_size):
    """Returns a list of batch indices (tuples of indices).

    # Arguments
        size: Integer, total size of the data to slice into batches.
        batch_size: Integer, batch size.

    # Returns
        A list of tuples of array indices.
    """
    num_batches = (size + batch_size - 1) // batch_size  # round up
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(num_batches)]


def weighted_masked_objective(fn):
    """Adds support for masking and sample-weighting to an objective function.

    It transforms an objective function `fn(y_true, y_pred)`
    into a sample-weighted, cost-masked objective function
    `fn(y_true, y_pred, weights, mask)`.

    # Arguments
        fn: The objective function to wrap,
            with signature `fn(y_true, y_pred)`.

    # Returns
        A function with signature `fn(y_true, y_pred, weights, mask)`.
    """
    if fn is None:
        return None

    def weighted(y_true, y_pred, weights, mask=None):
        """Wrapper function.

        # Arguments
            y_true: `y_true` argument of `fn`.
            y_pred: `y_pred` argument of `fn`.
            weights: Weights tensor.
            mask: Mask tensor.

        # Returns
            Scalar tensor.
        """
        # score_array has ndim >= 2
        score_array = fn(y_true, y_pred)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in Theano
            mask = K.cast(mask, K.floatx())
            # mask should have the same shape as score_array
            score_array *= mask
            #  the loss per batch should be proportional
            #  to the number of unmasked samples.
            score_array /= K.mean(mask) + K.epsilon()

        # apply sample weighting
        if weights is not None:
            # reduce score_array to same ndim as weight array
            ndim = K.ndim(score_array)
            weight_ndim = K.ndim(weights)
            score_array = K.mean(score_array,
                                 axis=list(range(weight_ndim, ndim)))
            score_array *= weights
            score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
        return K.mean(score_array)
    return weighted


def standardize_weights(y,
                        sample_weight=None,
                        class_weight=None,
                        sample_weight_mode=None):
    """Performs sample weight validation and standardization.

    Everything gets normalized to a single sample-wise (or timestep-wise)
    weight array. If both `sample_weights` and `class_weights` are provided,
    the weights are multiplied together.

    # Arguments
        y: Numpy array of model targets to be weighted.
        sample_weight: User-provided `sample_weight` argument.
        class_weight: User-provided `class_weight` argument.
        sample_weight_mode: One of `None` or `"temporal"`.
            `"temporal"` indicated that we expect 2D weight data
            that will be applied to the last 2 dimensions of
            the targets (i.e. we are weighting timesteps, not samples).

    # Returns
        A Numpy array of target weights, one entry per sample to weight.

    # Raises
        ValueError: In case of invalid user-provided arguments.
    """
    if sample_weight_mode is not None:
        if sample_weight_mode != 'temporal':
            raise ValueError('"sample_weight_mode '
                             'should be None or "temporal". '
                             'Found: ' + str(sample_weight_mode))
        if len(y.shape) < 3:
            raise ValueError('Found a sample_weight array for '
                             'an input with shape ' +
                             str(y.shape) + '. '
                             'Timestep-wise sample weighting (use of '
                             'sample_weight_mode="temporal") is restricted to '
                             'outputs that are at least 3D, i.e. that have '
                             'a time dimension.')
        if sample_weight is not None and len(sample_weight.shape) != 2:
            raise ValueError('Found a sample_weight array with shape ' +
                             str(sample_weight.shape) + '. '
                             'In order to use timestep-wise sample weighting, '
                             'you should pass a 2D sample_weight array.')
    else:
        if sample_weight is not None and len(sample_weight.shape) != 1:
            raise ValueError('Found a sample_weight array with shape ' +
                             str(sample_weight.shape) + '. '
                             'In order to use timestep-wise sample weights, '
                             'you should specify '
                             'sample_weight_mode="temporal" '
                             'in compile(). If you just mean to use '
                             'sample-wise weights, make sure your '
                             'sample_weight array is 1D.')

    if sample_weight is not None:
        if len(sample_weight.shape) > len(y.shape):
            raise ValueError('Found a sample_weight with shape' +
                             str(sample_weight.shape) + '.'
                             'Expected sample_weight with rank '
                             'less than or equal to ' + str(len(y.shape)))

        if y.shape[:sample_weight.ndim] != sample_weight.shape:
            raise ValueError('Found a sample_weight array with shape ' +
                             str(sample_weight.shape) +
                             ' for an input with shape ' +
                             str(y.shape) + '. '
                             'sample_weight cannot be broadcast.')

    class_sample_weight = None
    if isinstance(class_weight, dict):
        if len(y.shape) > 2:
            raise ValueError('`class_weight` not supported for '
                             '3+ dimensional targets.')
        if len(y.shape) == 2:
            if y.shape[1] > 1:
                y_classes = np.argmax(y, axis=1)
            elif y.shape[1] == 1:
                y_classes = np.reshape(y, y.shape[0])
        else:
            y_classes = y

        class_sample_weight = np.asarray(
            [class_weight[cls] for cls in y_classes if cls in class_weight])

        if len(class_sample_weight) != len(y_classes):
            # subtract the sets to pick all missing classes
            existing_classes = set(y_classes)
            existing_class_weight = set(class_weight.keys())
            raise ValueError('`class_weight` must contain '
                             'all classes in the data.'
                             ' The classes %s exist in the data but not in '
                             '`class_weight`.'
                             % (existing_classes - existing_class_weight))

    if sample_weight is not None and class_sample_weight is not None:
        return sample_weight * class_sample_weight
    if sample_weight is not None:
        return sample_weight
    if class_sample_weight is not None:
        return class_sample_weight

    # Everything has weight 1 by default.
    if sample_weight_mode is None:
        return np.ones((y.shape[0],), dtype=K.floatx())
    else:
        return np.ones((y.shape[0], y.shape[1]), dtype=K.floatx())


def check_num_samples(ins,
                      batch_size=None,
                      steps=None,
                      steps_name='steps'):
    """Checks the number of samples provided for training and evaluation.

    The number of samples is not defined when running with `steps`,
    in which case the number of samples is set to `None`.

    # Arguments
        ins: List of tensors to be fed to the Keras function.
        batch_size: Integer batch size or `None` if not defined.
        steps: Total number of steps (batches of samples)
            before declaring `predict_loop` finished.
            Ignored with the default value of `None`.
        steps_name: The public API's parameter name for `steps`.

    # Raises
        ValueError: when `steps` is `None` and the attribute `ins.shape`
        does not exist. Also raises ValueError when `steps` is not `None`
        and `batch_size` is not `None` because they are mutually
        exclusive.

    # Returns
        When `steps` is `None`, returns the number of samples to be
        processed based on the size of the first dimension of the
        first input Numpy array. When `steps` is not `None` and
        `batch_size` is `None`, returns `None`.

    # Raises
        ValueError: In case of invalid arguments.
    """
    if steps is not None and batch_size is not None:
        raise ValueError(
            'If ' + steps_name + ' is set, the `batch_size` must be None.')

    if not ins or any(K.is_tensor(x) for x in ins):
        if steps is None:
            raise ValueError(
                'If your data is in the form of symbolic tensors, '
                'you should specify the `' + steps_name + '` argument '
                '(instead of the `batch_size` argument, '
                'because symbolic tensors are expected to produce '
                'batches of input data).')
        return None

    if hasattr(ins[0], 'shape'):
        return int(ins[0].shape[0])
    return None  # Edge case where ins == [static_learning_phase]


def iter_sequence_infinite(seq):
    """Iterate indefinitely over a Sequence.

    # Arguments
        seq: Sequence object

    # Returns
        Generator yielding batches.
    """
    while True:
        for item in seq:
            yield item


def is_sequence(seq):
    """Determine if an object follows the Sequence API.

    # Arguments
        seq: a possible Sequence object

    # Returns
        boolean, whether the object follows the Sequence API.
    """
    # TODO Dref360: Decide which pattern to follow. First needs a new TF Version.
    return (getattr(seq, 'use_sequence_api', False)
            or set(dir(Sequence())).issubset(set(dir(seq) + ['use_sequence_api'])))


def is_generator_or_sequence(x):
    """Check if `x` is a Keras generator type."""
    return inspect.isgenerator(x) or is_sequence(x)


def should_run_validation(validation_freq, epoch):
    """Checks if validation should be run this epoch.

    # Arguments
        validation_freq: Integer or list. If an integer, specifies how many training
          epochs to run before a new validation run is performed. If a list,
          specifies the epochs on which to run validation.
        epoch: Integer, the number of the training epoch just completed.

    # Returns
        Bool, True if validation should be run.

    # Raises
        ValueError: if `validation_freq` is an Integer and less than 1, or if
        it is neither an Integer nor a Sequence.
    """
    # `epoch` is 0-indexed internally but 1-indexed in the public API.
    one_indexed_epoch = epoch + 1

    if isinstance(validation_freq, int):
        if validation_freq < 1:
            raise ValueError('`validation_freq` can not be less than 1.')
        return one_indexed_epoch % validation_freq == 0

    if not isinstance(validation_freq, collections.Container):
        raise ValueError('`validation_freq` must be an Integer or '
                         '`collections.Container` (e.g. list, tuple, etc.)')
    return one_indexed_epoch in validation_freq


def get_static_batch_size(layer):
    """Gets the static batch size of a Layer.

    # Arguments
        layer: a `Layer` instance.

    # Returns
        The static batch size of a Layer.
    """
    batch_input_shape, _ = get_input_shape_and_dtype(layer)
    if batch_input_shape is not None:
        return batch_input_shape[0]
    return None


def get_input_shape_and_dtype(layer):
    """Retrieves input shape and input dtype of layer if applicable.

    # Arguments
        layer: Layer (or model) instance.

    # Returns
        Tuple (input_shape, input_dtype). Both could be None if the layer
        does not have a defined input shape.

    # Raises
      ValueError: in case an empty Sequential or Functional model is passed.
    """
    def _is_graph_model(layer):
        return ((hasattr(layer, '_is_graph_network') and layer._is_graph_network) or
                layer.__class__.__name__ == 'Sequential')

    # In case of nested models: recover the first layer
    # of the deepest model to infer input shape and dtype.
    # Subclassed Models may not have been built so can't be checked.
    while _is_graph_model(layer):
        if not layer.layers:
            raise ValueError('An empty Model cannot be used as a Layer.')
        layer = layer.layers[0]

    if hasattr(layer, '_batch_input_shape'):
        return layer._batch_input_shape, layer.dtype
    return None, None


def get_loss_function(loss):
    """Returns the loss corresponding to the loss input in `compile` API."""
    if loss is None or isinstance(loss, losses.Loss):
        return loss

    # Deserialize loss configuration, if needed.
    if isinstance(loss, collections.Mapping):
        loss = losses.get(loss)

    # Custom callable class.
    if callable(loss) and not hasattr(loss, '__name__'):
        return loss

    # Wrap loss function with signature `(y_true, y_pred, **kwargs)`
    # in `LossFunctionWrapper` class.
    loss_fn = losses.get(loss)

    # For losses which are given as strings/functions in the compile API,
    # we always set the loss reduction type to be `SUM_OVER_BATCH_SIZE`..
    return losses.LossFunctionWrapper(
        loss_fn,
        name=loss_fn.__name__,
        reduction=losses_utils.Reduction.SUM_OVER_BATCH_SIZE)


def get_output_sample_weight_and_mode(skip_target_weighing_indices,
                                      sample_weight_mode, output_name,
                                      output_index):
    """Returns the sample weight and weight mode for a single output."""
    if output_index in skip_target_weighing_indices:
        return None, None

    if sample_weight_mode == 'temporal':
        shape = [None, None]
        mode = 'temporal'
    else:
        shape = [None]
        mode = None
    weight = K.placeholder(
        shape=shape,
        name=output_name + '_sample_weights')
    return weight, mode


def prepare_sample_weights(output_names, sample_weight_mode,
                           skip_target_weighing_indices):
    """Prepares sample weights for the model.

    # Arguments
        output_names: List of model output names.
        sample_weight_mode: sample weight mode user input passed from compile API.
        skip_target_weighing_indices: Indices of output for which sample weights
            should be skipped.

    # Returns
        A pair of list of sample weights and sample weight modes
            (one for each output).

    # Raises
        ValueError: In case of invalid `sample_weight_mode` input.
    """
    sample_weights = []
    sample_weight_modes = []
    if isinstance(sample_weight_mode, dict):
        unknown_output = set(sample_weight_mode.keys()) - set(output_names)
        if unknown_output:
            raise ValueError(
                'Unknown entry in '
                'sample_weight_mode dictionary: "' + str(unknown_output) +
                '". Only expected the following keys: ' + str(output_names))
        for i, name in enumerate(output_names):
            if (i not in skip_target_weighing_indices and
                    name not in sample_weight_mode):
                raise ValueError(
                    'Output missing from sample_weight_modes dictionary')
            weight, mode = get_output_sample_weight_and_mode(
                skip_target_weighing_indices,
                sample_weight_mode.get(name),
                name,
                i)
            sample_weights.append(weight)
            sample_weight_modes.append(mode)
    elif isinstance(sample_weight_mode, list):
        if len(sample_weight_mode) != len(output_names):
            raise ValueError('When passing a list as sample_weight_mode, '
                             'it should have one entry per model output. '
                             'The model has ' + str(len(output_names)) +
                             ' outputs, but you passed ' +
                             str(len(sample_weight_mode)) + 'sample_weight_modes')
        for i, name in enumerate(output_names):
            weight, mode = get_output_sample_weight_and_mode(
                skip_target_weighing_indices, sample_weight_mode[i], name, i)
            sample_weights.append(weight)
            sample_weight_modes.append(mode)
    else:
        for i, name in enumerate(output_names):
            weight, mode = get_output_sample_weight_and_mode(
                skip_target_weighing_indices, sample_weight_mode, name, i)
            sample_weights.append(weight)
            sample_weight_modes.append(mode)
    return sample_weights, sample_weight_modes


def prepare_loss_functions(loss, output_names):
    """Converts loss to a list of loss functions.

    # Arguments
        loss: String (name of objective function), objective function or
            `Loss` instance. If the model has multiple outputs, you can use
            a different loss on each output by passing a dictionary or a
            list of losses. The loss value that will be minimized by the model
            will then be the sum of all individual losses.
        output_names: List of model output names.

    # Returns
        A list of loss objective functions.

    # Raises:
        ValueError: If loss is a dict with keys not in model output names,
            or if loss is a list with len not equal to model outputs.
    """
    if isinstance(loss, collections.Mapping):
        generic_utils.check_for_unexpected_keys('loss', loss, output_names)
        loss_functions = []
        for name in output_names:
            if name not in loss:
                warnings.warn(
                    'Output {0} missing from loss dictionary. We assume '
                    'this was done on purpose. The fit and evaluate APIs will not '
                    'be expecting any data to be passed to {0}.'.format(name))
            loss_functions.append(get_loss_function(loss.get(name, None)))
    elif isinstance(loss, six.string_types):
        loss_functions = [get_loss_function(loss) for _ in output_names]
    elif isinstance(loss, collections.Sequence):
        if len(loss) != len(output_names):
            raise ValueError('When passing a list as loss, it should have one entry '
                             'per model outputs. The model has {} outputs, but you '
                             'passed loss={}'.format(len(output_names), loss))
        loss_functions = [get_loss_function(l) for l in loss]
    else:
        loss_functions = [get_loss_function(loss) for _ in range(len(output_names))]

    return loss_functions


def prepare_loss_weights(output_names, loss_weights=None):
    """Converts loss weights to a list of loss weights.

    # Arguments
        output_names: List of model output names.
        loss_weights: Optional list or dictionary specifying scalar coefficients
            (Python floats) to weight the loss contributions of different model
            outputs. The loss value that will be minimized by the model will then be
            the *weighted sum* of all individual losses, weighted by the
            `loss_weights` coefficients. If a list, it is expected to have a 1:1
            mapping to the model's outputs. If a dict, it is expected to map
            output names (strings) to scalar coefficients.

    # Returns
        A list of loss weights of python floats.

    # Raises
        ValueError: If loss weight is a dict with key not in model output names,
            or if loss is a list with len not equal to model outputs.
    """
    if loss_weights is None:
        weights_list = [1.] * len(output_names)
    elif isinstance(loss_weights, collections.Mapping):
        generic_utils.check_for_unexpected_keys('loss_weights', loss_weights,
                                                output_names)
        weights_list = [loss_weights.get(name, 1.) for name in output_names]
    elif isinstance(loss_weights, list):
        if len(loss_weights) != len(output_names):
            raise ValueError('When passing a list as loss_weights, '
                             'it should have one entry per model output. '
                             'The model has ' + str(len(output_names)) +
                             ' outputs, but you passed loss_weights=' +
                             str(loss_weights))
        weights_list = loss_weights
    else:
        raise TypeError('Could not interpret loss_weights argument: ' +
                        str(loss_weights) + ' - expected a list of dicts.')

    return weights_list


def collect_per_output_metric_info(metrics,
                                   output_names,
                                   output_shapes,
                                   loss_fns,
                                   is_weighted=False):
    """Maps metric names and functions to model outputs.

    # Arguments
        metrics: a list or a list of lists or a dict of metric functions.
        output_names: a list of the names (strings) of model outputs.
        output_shapes: a list of the shapes (strings) of model outputs.
        loss_fns: a list of the loss functions corresponding to the model outputs.
        is_weighted: Boolean indicating whether the given metrics are weighted.

    # Returns
        A list (one entry per model output) of dicts.
        For instance, if the model has 2 outputs, and for the first output
        we want to compute "binary_accuracy" and "binary_crossentropy",
        and just "binary_accuracy" for the second output,
        the list would look like: `[{
            'acc': binary_accuracy(),
            'ce': binary_crossentropy(),
        }, {
            'acc': binary_accuracy(),
        }]`

    # Raises
        TypeError: if an incorrect type is passed for the `metrics` argument.
    """
    if not metrics:
        return [{} for _ in output_names]

    if isinstance(metrics, list):
        any_sub_list = any(isinstance(m, list) for m in metrics)
        if any_sub_list:
            if len(metrics) != len(output_names):
                raise ValueError('When passing a list of lists as `metrics`, '
                                 'it should have one entry per model output. '
                                 'The model has ' + str(len(output_names)) +
                                 ' outputs, but you passed metrics=' + str(metrics))
            # User has provided a list of len = len(outputs).
            nested_metrics = [generic_utils.to_list(m) for m in metrics]
        else:
            # If it is a single list we then apply all metrics to all outputs.
            if len(output_names) > 1:
                nested_metrics = []
                for _ in output_names:
                    nested_metrics.append(
                        [metrics_module.clone_metric(m) for m in metrics])
            else:
                nested_metrics = [metrics]
    elif isinstance(metrics, collections.Mapping):
        generic_utils.check_for_unexpected_keys('metrics', metrics, output_names)
        nested_metrics = []
        for name in output_names:
            output_metrics = generic_utils.to_list(metrics.get(name, []))
            nested_metrics.append(output_metrics)
    else:
        raise TypeError('Type of `metrics` argument not understood. '
                        'Expected a list or dictionary, found: ' + str(metrics))

    per_output_metrics = []
    for i, metrics in enumerate(nested_metrics):
        metrics_dict = OrderedDict()
        for metric in metrics:
            metric_name = get_metric_name(metric, is_weighted)
            metric_fn = get_metric_function(
                metric, output_shape=output_shapes[i], loss_fn=loss_fns[i])

            # If the metric function is not stateful, we create a stateful version.
            if not isinstance(metric_fn, metrics_module.Metric):
                metric_fn = metrics_module.MeanMetricWrapper(
                    metric_fn, name=metric_name)
            metrics_dict[metric_name] = metric_fn
        per_output_metrics.append(metrics_dict)

    return per_output_metrics


def get_metric_name(metric, weighted=False):
    """Returns the name corresponding to the given metric input.

    # Arguments
        metric: Metric function name or reference.
        weighted: Boolean indicating if the given metric is weighted.

    # Returns
        The metric name.
    """
    # We keep the string that the user has set in compile as the metric name.
    if isinstance(metric, six.string_types):
        return metric

    metric = metrics_module.get(metric)
    return metric.name if hasattr(metric, 'name') else metric.__name__


def get_metric_function(metric, output_shape=None, loss_fn=None):
    """Returns the metric function corresponding to the given metric input.

    # Arguments
        metric: Metric function name or reference.
        output_shape: The shape of the output that this metric will be calculated
            for.
        loss_fn: The loss function used.

    # Returns
        The metric function.
    """
    if metric not in ['accuracy', 'acc', 'crossentropy', 'ce']:
        return metrics_module.get(metric)

    is_sparse_categorical_crossentropy = (
        isinstance(loss_fn, losses.SparseCategoricalCrossentropy) or
        (isinstance(loss_fn, losses.LossFunctionWrapper) and
         loss_fn.fn == losses.sparse_categorical_crossentropy))

    is_binary_crossentropy = (
        isinstance(loss_fn, losses.BinaryCrossentropy) or
        (isinstance(loss_fn, losses.LossFunctionWrapper) and
         loss_fn.fn == losses.binary_crossentropy))

    if metric in ['accuracy', 'acc']:
        if output_shape[-1] == 1 or is_binary_crossentropy:
            return metrics_module.binary_accuracy
        elif is_sparse_categorical_crossentropy:
            return metrics_module.sparse_categorical_accuracy
        # If the output_shape[-1] is not 1, then we know output is `categorical`.
        # We assume it is sparse categorical only if loss is explicitly given
        # as sparse categorical crossentropy loss.
        return metrics_module.categorical_accuracy
    else:
        if output_shape[-1] == 1 or is_binary_crossentropy:
            return metrics_module.binary_crossentropy
        elif is_sparse_categorical_crossentropy:
            return metrics_module.sparse_categorical_crossentropy
        return metrics_module.categorical_crossentropy


def call_metric_function(metric_fn,
                         y_true,
                         y_pred=None,
                         weights=None,
                         mask=None):
    """Invokes metric function and returns the metric result tensor."""
    if mask is not None:
        mask = K.cast(mask, y_pred.dtype)
        if weights is None:
            # Use mask as sample weight.
            weights = mask
        else:
            # Update dimensions of weights to match with mask.
            mask, _, weights = losses_utils.squeeze_or_expand_dimensions(
                mask, sample_weight=weights)
            weights *= mask

    if y_pred is not None:
        update_ops = metric_fn.update_state(y_true, y_pred, sample_weight=weights)
        with K.control_dependencies(update_ops):  # For TF
            metric_fn.result()
    else:
        # `Mean` metric only takes a single value.
        update_ops = metric_fn.update_state(y_true, sample_weight=weights)
        with K.control_dependencies(update_ops):  # For TF
            metric_fn.result()
