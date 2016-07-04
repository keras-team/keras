from __future__ import print_function
from __future__ import absolute_import

import warnings
import copy
import time
import numpy as np
import multiprocessing
import threading
try:
    import queue
except ImportError:
    import Queue as queue

from .topology import Container
from .. import backend as K
from .. import optimizers
from .. import objectives
from .. import metrics as metrics_module
from ..utils.generic_utils import Progbar
from .. import callbacks as cbks


def standardize_input_data(data, names, shapes=None,
                           check_batch_dim=True,
                           exception_prefix=''):
    '''Users may pass data as a list of arrays, dictionary of arrays,
    or as a single array. We normalize this to an ordered list of
    arrays (same order as `names`), while checking that the provided
    arrays have shapes that match the network's expectations.
    '''
    if type(data) is dict:
        arrays = []
        for name in names:
            if name not in data:
                raise Exception('No data provided for "' +
                                name + '". Need data for each key in: ' +
                                str(data.keys()))
            arrays.append(data[name])
    elif type(data) is list:
        if len(data) != len(names):
            if len(data) > 0 and hasattr(data[0], 'shape'):
                raise Exception('Error when checking ' + exception_prefix +
                                ': the list of Numpy arrays '
                                'that you are passing to your model '
                                'is not the size the model expected. '
                                'Expected to see ' + str(len(names)) +
                                ' arrays but instead got '
                                'the following list of ' + str(len(data)) +
                                ' arrays: ' + str(data)[:200] +
                                '...')
            else:
                if len(names) == 1:
                    data = [np.asarray(data)]
                else:
                    raise Exception('Error when checking ' + exception_prefix +
                                    ': you are passing a list as '
                                    'input to your model, '
                                    'but the model expects '
                                    'a list of ' + str(len(names)) +
                                    ' Numpy arrays instead. '
                                    'The list you passed was: ' +
                                    str(data)[:200])
        arrays = data
    else:
        if not hasattr(data, 'shape'):
            raise Exception('Error when checking ' + exception_prefix +
                            ': data should be a Numpy array, '
                            'or list/dict of Numpy arrays. '
                            'Found: ' + str(data)[:200] + '...')
        if len(names) != 1:
            # case: model expects multiple inputs but only received
            # a single Numpy array
            raise Exception('The model expects ' + str(len(names)) +
                            ' input arrays, but only received one array. '
                            'Found: array with shape ' + str(data.shape))
        arrays = [data]

    # make arrays at least 2D
    for i in range(len(names)):
        array = arrays[i]
        if len(array.shape) == 1:
            array = np.expand_dims(array, 1)
            arrays[i] = array

    # check shapes compatibility
    if shapes:
        for i in range(len(names)):
            if shapes[i] is None:
                continue
            array = arrays[i]
            if len(array.shape) != len(shapes[i]):
                raise Exception('Error when checking ' + exception_prefix +
                                ': expected ' + names[i] +
                                ' to have ' + str(len(shapes[i])) +
                                ' dimensions, but got array with shape ' +
                                str(array.shape))
            for j, (dim, ref_dim) in enumerate(zip(array.shape, shapes[i])):
                if not j and not check_batch_dim:
                    # skip the first axis
                    continue
                if ref_dim:
                    if ref_dim != dim:
                        raise Exception('Error when checking ' + exception_prefix +
                                        ': expected ' + names[i] +
                                        ' to have shape ' + str(shapes[i]) +
                                        ' but got array with shape ' +
                                        str(array.shape))
    return arrays


def standardize_sample_or_class_weights(x_weight, output_names, weight_type):
    if x_weight is None or len(x_weight) == 0:
        return [None for _ in output_names]
    if len(output_names) == 1:
        if type(x_weight) is list and len(x_weight) == 1:
            return x_weight
        if type(x_weight) is dict and output_names[0] in x_weight:
            return [x_weight[output_names[0]]]
        else:
            return [x_weight]
    if type(x_weight) is list:
        if len(x_weight) != len(output_names):
            raise Exception('Provided `' + weight_type + '` was a list of ' +
                            str(len(x_weight)) +
                            ' elements, but the model has ' +
                            str(len(output_names)) + ' outputs. '
                            'You should provide one `' + weight_type + '`'
                            'array per model output.')
        return x_weight
    if type(x_weight) is dict:
        x_weights = []
        for name in output_names:
            x_weights.append(x_weight.get(name))
        return x_weights
    else:
        raise Exception('The model has multiple outputs, so `' +
                        weight_type + '` '
                        'should be either a list of a dict. '
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


def check_array_lengths(X, Y, W):
    x_lengths = [x.shape[0] for x in X]
    y_lengths = [y.shape[0] for y in Y]
    w_lengths = [w.shape[0] for w in W]
    set_x = set(x_lengths)
    if len(set_x) != 1:
        raise Exception('All input arrays (x) should have '
                        'the same number of samples.')
    set_y = set(y_lengths)
    if len(set_y) != 1:
        raise Exception('All target arrays (y) should have '
                        'the same number of samples.')
    set_w = set(w_lengths)
    if len(set_w) != 1:
        raise Exception('All sample_weight arrays should have '
                        'the same number of samples.')
    if list(set_x)[0] != list(set_y)[0]:
        raise Exception('Input arrays should have '
                        'the same number of samples as target arrays. Found ' +
                        str(list(set_x)[0]) + ' input samples and ' +
                        str(list(set_y)[0]) + ' target samples.')
    if list(set_x)[0] != list(set_w)[0]:
        raise Exception('Sample_weight arrays should have '
                        'the same number of samples as input arrays. Found ' +
                        str(list(set_x)[0]) + ' input samples and ' +
                        str(list(set_w)[0]) + ' target samples.')


def check_loss_and_target_compatibility(targets, losses, output_shapes):
    assert len(targets) == len(losses) == len(output_shapes)
    key_losses = {'mean_square_error',
                  'binary_crossentropy',
                  'categorical_crossentropy'}
    for y, loss, shape in zip(targets, losses, output_shapes):
        if loss.__name__ == 'categorical_crossentropy':
            if y.shape[1] == 1:
                raise Exception('You are passing a target array of shape ' + str(y.shape) +
                                ' while using as loss `categorical_crossentropy`. '
                                '`categorical_crossentropy` expects '
                                'targets to be binary matrices (1s and 0s) '
                                'of shape (samples, classes). '
                                'If your targets are integer classes, '
                                'you can convert them to the expected format via:\n'
                                '```\n'
                                'from keras.utils.np_utils import to_categorical\n'
                                'y_binary = to_categorical(y_int)\n'
                                '```\n'
                                '\n'
                                'Alternatively, you can use the loss function '
                                '`sparse_categorical_crossentropy` instead, '
                                'which does expect integer targets.')
        if loss.__name__ in key_losses and shape[1] is not None and y.shape[1] != shape[1]:
            raise Exception('A target array with shape ' + str(y.shape) +
                            ' was passed for an output of shape ' + str(shape) +
                            ' while using as loss `' + loss.__name__ + '`. '
                            'This loss expects '
                            'targets to have the same shape '
                            'as the output.')


def collect_metrics(metrics, output_names):
    if not metrics:
        return [[] for _ in output_names]
    if type(metrics) is list:
        # we then apply all metrics to all outputs.
        return [copy.copy(metrics) for _ in output_names]
    elif type(metrics) is dict:
        nested_metrics = []
        for name in output_names:
            output_metrics = metrics.get(name, [])
            if type(output_metrics) is not list:
                output_metrics = [output_metrics]
            nested_metrics.append(output_metrics)
        return nested_metrics
    else:
        raise Exception('Type of `metrics` argument not understood. '
                        'Expected a list or dictionary, found: ' +
                        str(metrics))


def collect_trainable_weights(layer):
    '''Collects all `trainable_weights` attributes,
    excluding any sublayers where `trainable` is set the `False`.
    '''
    trainable = getattr(layer, 'trainable', True)
    if not trainable:
        return []
    weights = []
    if layer.__class__.__name__ == 'Sequential':
        for sublayer in layer.flattened_layers:
            weights += collect_trainable_weights(sublayer)
    elif layer.__class__.__name__ == 'Model':
        for sublayer in layer.layers:
            weights += collect_trainable_weights(sublayer)
    elif layer.__class__.__name__ == 'Graph':
        for sublayer in layer._graph_nodes.values():
            weights += collect_trainable_weights(sublayer)
    else:
        weights += layer.trainable_weights
    # dedupe weights
    weights = list(set(weights))
    weights.sort(key=lambda x: x.name)
    return weights


def batch_shuffle(index_array, batch_size):
    '''This shuffles an array in a batch-wise fashion.
    Useful for shuffling HDF5 arrays
    (where one cannot access arbitrary indices).
    '''
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
    '''Returns a list of batch indices (tuples of indices).
    '''
    nb_batch = int(np.ceil(size / float(batch_size)))
    return [(i * batch_size, min(size, (i + 1) * batch_size))
            for i in range(0, nb_batch)]


def slice_X(X, start=None, stop=None):
    '''This takes an array-like, or a list of
    array-likes, and outputs:
        - X[start:stop] if X is an array-like
        - [x[start:stop] for x in X] if X in a list

    Can also work on list/array of indices: `slice_X(x, indices)`

    # Arguments:
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.
    '''
    if type(X) == list:
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in X]
        else:
            return [x[start:stop] for x in X]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return X[start]
        else:
            return X[start:stop]


def weighted_objective(fn):
    '''Transforms an objective function `fn(y_true, y_pred)`
    into a sample-weighted, cost-masked objective function
    `fn(y_true, y_pred, weights, mask)`.
    '''
    def weighted(y_true, y_pred, weights, mask=None):
        # score_array has ndim >= 2
        score_array = fn(y_true, y_pred)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            mask = K.cast(mask, K.floatx())
            # mask should have the same shape as score_array
            score_array *= mask
            #  the loss per batch should be proportional
            #  to the number of unmasked samples.
            score_array /= K.mean(mask)

        # reduce score_array to same ndim as weight array
        ndim = K.ndim(score_array)
        weight_ndim = K.ndim(weights)
        score_array = K.mean(score_array, axis=list(range(weight_ndim, ndim)))

        # apply sample weighting
        if weights is not None:
            score_array *= weights
            score_array /= K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
        return K.mean(score_array)
    return weighted


def standardize_weights(y, sample_weight=None, class_weight=None,
                        sample_weight_mode=None):
    '''Performs weight input validation and standardization
    to a single sample-wise (or timestep-wise) weight array.
    '''
    if sample_weight_mode is not None:
        if sample_weight_mode != 'temporal':
            raise Exception('"sample_weight_mode '
                            'should be None or "temporal". '
                            'Found: ' + str(sample_weight_mode))
        if len(y.shape) < 3:
            raise Exception('Found a sample_weight array for '
                            'an input with shape ' +
                            str(y.shape) + '. '
                            'Timestep-wise sample weighting (use of '
                            'sample_weight_mode="temporal") is restricted to '
                            'outputs that are at least 3D, i.e. that have '
                            'a time dimension.')
        if sample_weight is not None and len(sample_weight.shape) != 2:
            raise Exception('Found a sample_weight array with shape ' +
                            str(sample_weight.shape) + '. '
                            'In order to use timestep-wise sample weighting, '
                            'you should pass a 2D sample_weight array.')
    else:
        if sample_weight is not None and len(sample_weight.shape) != 1:
            raise Exception('Found a sample_weight array with shape ' +
                            str(sample_weight.shape) + '. '
                            'In order to use timestep-wise sample weights, '
                            'you should specify sample_weight_mode="temporal" '
                            'in compile(). If you just mean to use '
                            'sample-wise weights, make sure your '
                            'sample_weight array is 1D.')

    if sample_weight is not None:
        assert len(sample_weight.shape) <= len(y.shape)
        # TODO: proper error message
        assert y.shape[:sample_weight.ndim] == sample_weight.shape
        return sample_weight
    elif isinstance(class_weight, dict):
        if len(y.shape) > 2:
            raise Exception('class_weight not supported for '
                            '3+ dimensional targets.')
        if y.shape[1] > 1:
            y_classes = y.argmax(axis=1)
        elif y.shape[1] == 1:
            y_classes = np.reshape(y, y.shape[0])
        else:
            y_classes = y
        weights = np.asarray([class_weight[cls] for cls in y_classes])
        return weights
    else:
        if sample_weight_mode is None:
            return np.ones((y.shape[0],))
        else:
            return np.ones((y.shape[0], y.shape[1]))


def generator_queue(generator, max_q_size=10,
                    wait_time=0.05, nb_worker=1, pickle_safe=False):
    '''Builds a queue out of a data generator.
    If pickle_safe, use a multiprocessing approach. Else, use threading.
    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    '''

    generator_threads = []
    if pickle_safe:
        q = multiprocessing.Queue(maxsize=max_q_size)
        _stop = multiprocessing.Event()
    else:
        q = queue.Queue()
        _stop = threading.Event()

    try:

        def data_generator_task():
            while not _stop.is_set():
                try:
                    if q.qsize() < max_q_size:
                        try:
                            generator_output = next(generator)
                        except ValueError:
                            continue
                        q.put(generator_output)
                    else:
                        time.sleep(wait_time)
                except Exception:
                    _stop.set()
                    raise

        for i in range(nb_worker):
            if pickle_safe:
                # Reset random seed else all children processes share the same seed
                np.random.seed()
                thread = multiprocessing.Process(target=data_generator_task)
            else:
                thread = threading.Thread(target=data_generator_task)
            generator_threads.append(thread)
            thread.daemon = True
            thread.start()

    except:
        _stop.set()
        if pickle_safe:
            # Terminate all daemon processes
            for p in generator_threads:
                if p.is_alive():
                    p.terminate()
            q.close()
        raise

    return q, _stop


class Model(Container):

    def compile(self, optimizer, loss, metrics=[], loss_weights=None,
                sample_weight_mode=None, **kwargs):
        '''Configures the model for training.

        # Arguments
            optimizer: str (name of optimizer) or optimizer object.
                See [optimizers](/optimizers).
            loss: str (name of objective function) or objective function.
                See [objectives](/objectives).
                If the model has multiple outputs, you can use a different loss
                on each output by passing a dictionary or a list of objectives.
            metrics: list of metrics to be evaluated by the model
                during training and testing.
                Typically you will use `metrics=['accuracy']`.
                To specify different metrics for different outputs of a
                multi-output model, you could also pass a dictionary,
                such as `metrics={'output_a': 'accuracy'}`.
            sample_weight_mode: if you need to do timestep-wise
                sample weighting (2D weights), set this to "temporal".
                "None" defaults to sample-wise weights (1D).
                If the model has multiple outputs, you can use a different
                `sample_weight_mode` on each output by passing a
                dictionary or a list of modes.
            kwargs: when using the Theano backend, these arguments
                are passed into K.function. Ignored for Tensorflow backend.
        '''
        self.optimizer = optimizers.get(optimizer)
        self.sample_weight_mode = sample_weight_mode
        self.loss = loss
        self.loss_weights = loss_weights

        # prepare loss weights
        if loss_weights is None:
            loss_weights_list = [1. for _ in range(len(self.outputs))]
        elif type(loss_weights) is dict:
            for name in loss_weights:
                if name not in self.output_names:
                    raise Exception('Unknown entry in loss_weights '
                                    'dictionary: "' + name + '". '
                                    'Only expected the following keys: ' +
                                    str(self.output_names))
            loss_weights_list = []
            for name in self.output_names:
                loss_weights_list.append(loss_weights.get(name, 1.))
        elif type(loss_weights) is list:
            if len(loss_weights) != len(self.outputs):
                raise Exception('When passing a list as loss_weights, '
                                'it should have one entry per model outputs. '
                                'The model has ' + str(len(self.outputs)) +
                                ' outputs, but you passed loss_weights=' +
                                str(loss))
            loss_weights_list = loss_weights
        else:
            raise Exception('Could not interpret loss_weights argument: ' +
                            str(loss_weights))

        # prepare loss functions
        if type(loss) is dict:
            for name in loss:
                if name not in self.output_names:
                    raise Exception('Unknown entry in loss '
                                    'dictionary: "' + name + '". '
                                    'Only expected the following keys: ' +
                                    str(self.output_names))
            loss_functions = []
            for name in self.output_names:
                if name not in loss:
                    raise Exception('Output "' + name +
                                    '" missing from loss dictionary')
                loss_functions.append(objectives.get(loss[name]))
        elif type(loss) is list:
            if len(loss) != len(self.outputs):
                raise Exception('When passing a list as loss, '
                                'it should have one entry per model outputs. '
                                'The model has ' + str(len(self.outputs)) +
                                ' outputs, but you passed loss=' +
                                str(loss))
            loss_functions = [objectives.get(l) for l in loss]
        else:
            loss_function = objectives.get(loss)
            loss_functions = [loss_function for _ in range(len(self.outputs))]
        self.loss_functions = loss_functions
        weighted_losses = [weighted_objective(fn) for fn in loss_functions]

        # prepare output masks
        masks = self.compute_mask(self.inputs, mask=None)
        if masks is None:
            masks = [None for _ in self.outputs]
        if type(masks) is not list:
            masks = [masks]

        # prepare sample weights
        if type(sample_weight_mode) is dict:
            for name in sample_weight_mode:
                if name not in self.output_names:
                    raise Exception('Unknown entry in '
                                    'sample_weight_mode dictionary: "' +
                                    name + '". '
                                    'Only expected the following keys: ' +
                                    str(self.output_names))
            sample_weights = []
            sample_weight_modes = []
            for name in self.output_names:
                if name not in sample_weight_mode:
                    raise Exception('Output "' + name +
                                    '" missing from sample_weight_modes '
                                    'dictionary')
                if sample_weight_mode.get(name) == 'temporal':
                    weight = K.placeholder(ndim=2, name=name + '_sample_weights')
                    sample_weight_modes.append('temporal')
                else:
                    weight = K.placeholder(ndim=1, name=name + '_sample_weights')
                    sample_weight_modes.append(None)
                sample_weights.append(weight)
        elif type(sample_weight_mode) is list:
            if len(sample_weight_mode) != len(self.outputs):
                raise Exception('When passing a list as sample_weight_mode, ' +
                                'it should have one entry per model outputs. '
                                'The model has ' + str(len(self.outputs)) +
                                ' outputs, but you passed sample_weight_mode=' +
                                str(sample_weight_mode))
            sample_weights = []
            sample_weight_modes = []
            for mode, name in zip(sample_weight_mode, self.output_names):
                if mode == 'temporal':
                    weight = K.placeholder(ndim=2, name=name + '_sample_weights')
                    sample_weight_modes.append('temporal')
                else:
                    weight = K.placeholder(ndim=1, name=name + '_sample_weights')
                    sample_weight_modes.append(None)
                sample_weights.append(weight)
        else:
            if sample_weight_mode == 'temporal':
                sample_weights = [K.placeholder(ndim=2, name=name + '_sample_weights')
                                  for name in self.output_names]
                sample_weight_modes = ['temporal' for name in self.output_names]
            else:
                sample_weights = [K.placeholder(ndim=1, name=name + '_sample_weights')
                                  for name in self.output_names]
                sample_weight_modes = [None for name in self.output_names]
        self.sample_weight_modes = sample_weight_modes

        # prepare targets of model
        self.targets = []
        for i in range(len(self.outputs)):
            shape = self.internal_output_shapes[i]
            name = self.output_names[i]
            self.targets.append(K.placeholder(ndim=len(shape), name=name + '_target'))

        # prepare metrics
        self.metrics_names = ['loss']
        self.metrics = []

        # compute total loss
        total_loss = None
        for i in range(len(self.outputs)):
            y_true = self.targets[i]
            y_pred = self.outputs[i]
            weighted_loss = weighted_losses[i]
            sample_weight = sample_weights[i]
            mask = masks[i]
            loss_weight = loss_weights_list[i]
            output_loss = weighted_loss(y_true, y_pred,
                                        sample_weight, mask)
            if len(self.outputs) > 1:
                self.metrics.append(output_loss)
                self.metrics_names.append(self.output_names[i] + '_loss')
            if total_loss is None:
                total_loss = loss_weight * output_loss
            else:
                total_loss += loss_weight * output_loss

        # add regularization penalties to the loss
        for r in self.regularizers:
            total_loss = r(total_loss)

        # list of same size as output_names.
        # contains tuples (metrics for output, names of metrics)
        nested_metrics = collect_metrics(metrics, self.output_names)
        for i in range(len(self.outputs)):
            y_true = self.targets[i]
            y_pred = self.outputs[i]
            output_metrics = nested_metrics[i]

            for metric in output_metrics:
                if metric == 'accuracy' or metric == 'acc':
                    # custom handling of accuracy (because of class mode duality)
                    output_shape = self.internal_output_shapes[i]
                    if output_shape[-1] == 1 or self.loss_functions[i] == objectives.binary_crossentropy:
                        # case: binary accuracy
                        self.metrics.append(metrics_module.binary_accuracy(y_true, y_pred))
                    elif self.loss_functions[i] == objectives.sparse_categorical_crossentropy:
                        # case: categorical accuracy with sparse targets
                        self.metrics.append(
                            metrics_module.sparse_categorical_accuracy(y_true, y_pred))
                    else:
                        # case: categorical accuracy with dense targets
                        self.metrics.append(metrics_module.categorical_accuracy(y_true, y_pred))
                    if len(self.output_names) == 1:
                        self.metrics_names.append('acc')
                    else:
                        self.metrics_names.append(self.output_layers[i].name + '_acc')
                else:
                    metric_fn = metrics_module.get(metric)
                    self.metrics.append(metric_fn(y_true, y_pred))
                    if len(self.output_names) == 1:
                        self.metrics_names.append(metric_fn.__name__)
                    else:
                        self.metrics_names.append(self.output_layers[i].name + '_' + metric_fn.__name__)

        # prepare gradient updates and state updates
        self.optimizer = optimizers.get(optimizer)
        self.total_loss = total_loss
        self.sample_weights = sample_weights

        # functions for train, test and predict will
        # be compiled lazily when required.
        # This saves time when the user is not using all functions.
        self._function_kwargs = kwargs

        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def _make_train_function(self):
        if not hasattr(self, 'train_function'):
            raise Exception('You must compile your model before using it.')
        if self.train_function is None:
            if self.uses_learning_phase:
                inputs = self.inputs + self.targets + self.sample_weights + [K.learning_phase()]
            else:
                inputs = self.inputs + self.targets + self.sample_weights

            # get trainable weights
            trainable_weights = collect_trainable_weights(self)
            training_updates = self.optimizer.get_updates(trainable_weights, self.constraints, self.total_loss)
            updates = self.updates + training_updates

            # returns loss and metrics. Updates weights at each call.
            self.train_function = K.function(inputs,
                                             [self.total_loss] + self.metrics,
                                             updates=updates,
                                             **self._function_kwargs)

    def _make_test_function(self):
        if not hasattr(self, 'test_function'):
            raise Exception('You must compile your model before using it.')
        if self.test_function is None:
            if self.uses_learning_phase:
                inputs = self.inputs + self.targets + self.sample_weights + [K.learning_phase()]
            else:
                inputs = self.inputs + self.targets + self.sample_weights
            # return loss and metrics, no gradient updates.
            # Does update the network states.
            self.test_function = K.function(inputs,
                                            [self.total_loss] + self.metrics,
                                            updates=self.state_updates,
                                            **self._function_kwargs)

    def _make_predict_function(self):
        if not hasattr(self, 'predict_function'):
            self.predict_function = None
        if self.predict_function is None:
            if self.uses_learning_phase:
                inputs = self.inputs + [K.learning_phase()]
            else:
                inputs = self.inputs
            # returns network outputs. Does not update weights.
            # Does update the network states.
            kwargs = getattr(self, '_function_kwargs', {})
            self.predict_function = K.function(inputs,
                                               self.outputs,
                                               updates=self.state_updates,
                                               **kwargs)

    def _fit_loop(self, f, ins, out_labels=[], batch_size=32,
                  nb_epoch=100, verbose=1, callbacks=[],
                  val_f=None, val_ins=None, shuffle=True,
                  callback_metrics=[]):
        '''Abstract fit function for f(ins).
        Assume that f returns a list, labeled by out_labels.

        # Arguments
            f: Keras function returning a list of tensors
            ins: list of tensors to be fed to `f`
            out_labels: list of strings, display names of
                the outputs of `f`
            batch_size: integer batch size
            nb_epoch: number of times to iterate over the data
            verbose: verbosity mode, 0, 1 or 2
            callbacks: list of callbacks to be called during training
            val_f: Keras function to call for validation
            val_ins: list of tensors to be fed to `val_f`
            shuffle: whether to shuffle the data at the beginning of each epoch
            callback_metrics: list of strings, the display names of the metrics
                passed to the callbacks. They should be the
                concatenation of list the display names of the outputs of
                 `f` and the list of display names of the outputs of `f_val`.

        # Returns
            `History` object.
        '''
        do_validation = False
        if val_f and val_ins:
            do_validation = True
            if verbose:
                print('Train on %d samples, validate on %d samples' %
                      (len(ins[0]), len(val_ins[0])))

        nb_train_sample = len(ins[0])
        index_array = np.arange(nb_train_sample)

        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + callbacks + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)

        # it's possible to callback a different model than self
        # (used by Sequential models)
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self

        callbacks._set_model(callback_model)
        callbacks._set_params({
            'batch_size': batch_size,
            'nb_epoch': nb_epoch,
            'nb_sample': nb_train_sample,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()
        callback_model.stop_training = False
        self.validation_data = val_ins

        for epoch in range(nb_epoch):
            callbacks.on_epoch_begin(epoch)
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)
            epoch_logs = {}
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    if type(ins[-1]) is float:
                        # do not slice the training phase flag
                        ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
                    else:
                        ins_batch = slice_X(ins, batch_ids)
                except TypeError:
                    raise Exception('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = f(ins_batch)
                if type(outs) != list:
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                if batch_index == len(batches) - 1:  # last batch
                    # validation
                    if do_validation:
                        # replace with self._evaluate
                        val_outs = self._test_loop(val_f, val_ins,
                                                   batch_size=batch_size,
                                                   verbose=0)
                        if type(val_outs) != list:
                            val_outs = [val_outs]
                        # same labels assumed
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o
            callbacks.on_epoch_end(epoch, epoch_logs)
            if callback_model.stop_training:
                break
        callbacks.on_train_end()
        return self.history

    def _predict_loop(self, f, ins, batch_size=32, verbose=0):
        '''Abstract method to loop over some data in batches.

        # Arguments
            f: Keras function returning a list of tensors.
            ins: list of tensors to be fed to `f`.
            batch_size: integer batch size.
            verbose: verbosity mode.

        # Returns
            Array of predictions (if the model has a single output)
            or list of arrays of predictions
            (if the model has multiple outputs).
        '''
        nb_sample = len(ins[0])
        outs = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            if type(ins[-1]) is float:
                # do not slice the training phase flag
                ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = slice_X(ins, batch_ids)

            batch_outs = f(ins_batch)
            if type(batch_outs) != list:
                batch_outs = [batch_outs]
            if batch_index == 0:
                for batch_out in batch_outs:
                    shape = (nb_sample,) + batch_out.shape[1:]
                    outs.append(np.zeros(shape))

            for i, batch_out in enumerate(batch_outs):
                outs[i][batch_start:batch_end] = batch_out
            if verbose == 1:
                progbar.update(batch_end)
        if len(outs) == 1:
            return outs[0]
        return outs

    def _test_loop(self, f, ins, batch_size=32, verbose=0):
        '''Abstract method to loop over some data in batches.

        # Arguments
            f: Keras function returning a list of tensors.
            ins: list of tensors to be fed to `f`.
            batch_size: integer batch size.
            verbose: verbosity mode.

        # Returns
            Scalar loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        '''
        nb_sample = len(ins[0])
        outs = []
        if verbose == 1:
            progbar = Progbar(target=nb_sample)
        batches = make_batches(nb_sample, batch_size)
        index_array = np.arange(nb_sample)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            if type(ins[-1]) is float:
                # do not slice the training phase flag
                ins_batch = slice_X(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = slice_X(ins, batch_ids)

            batch_outs = f(ins_batch)
            if type(batch_outs) == list:
                if batch_index == 0:
                    for batch_out in enumerate(batch_outs):
                        outs.append(0.)
                for i, batch_out in enumerate(batch_outs):
                    outs[i] += batch_out * len(batch_ids)
            else:
                if batch_index == 0:
                    outs.append(0.)
                outs[0] += batch_outs * len(batch_ids)

            if verbose == 1:
                progbar.update(batch_end)
        for i, out in enumerate(outs):
            outs[i] /= nb_sample
        if len(outs) == 1:
            return outs[0]
        return outs

    def _standardize_user_data(self, x, y,
                               sample_weight=None, class_weight=None,
                               check_batch_dim=True, batch_size=None):
        if not hasattr(self, 'optimizer'):
            raise Exception('You must compile a model before training/testing.'
                            ' Use `model.compile(optimizer, loss)`.')

        output_shapes = []
        for output_shape, loss_fn in zip(self.internal_output_shapes, self.loss_functions):
            if loss_fn.__name__ == 'sparse_categorical_crossentropy':
                output_shapes.append(output_shape[:-1] + (1,))
            elif getattr(objectives, loss_fn.__name__, None) is None:
                output_shapes.append(None)
            else:
                output_shapes.append(output_shape)
        x = standardize_input_data(x, self.input_names,
                                   self.internal_input_shapes,
                                   check_batch_dim=False,
                                   exception_prefix='model input')
        y = standardize_input_data(y, self.output_names,
                                   output_shapes,
                                   check_batch_dim=False,
                                   exception_prefix='model target')
        sample_weights = standardize_sample_weights(sample_weight,
                                                    self.output_names)
        class_weights = standardize_class_weights(class_weight,
                                                  self.output_names)
        sample_weights = [standardize_weights(ref, sw, cw, mode)
                          for (ref, sw, cw, mode)
                          in zip(y, sample_weights, class_weights, self.sample_weight_modes)]
        check_array_lengths(x, y, sample_weights)
        check_loss_and_target_compatibility(y, self.loss_functions, self.internal_output_shapes)
        if self.stateful and batch_size:
            if x[0].shape[0] % batch_size != 0:
                raise Exception('In a stateful network, '
                                'you should only pass inputs with '
                                'a number of samples that can be '
                                'divided by the batch size. Found: ' +
                                str(x[0].shape[0]) + ' samples')
        return x, y, sample_weights

    def fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=[],
            validation_split=0., validation_data=None, shuffle=True,
            class_weight=None, sample_weight=None):
        '''Trains the model for a fixed number of epochs (iterations on a dataset).

        # Arguments
            x: Numpy array of training data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named, you can also pass a dictionary
                mapping input names to Numpy arrays.
            y: Numpy array of target data,
                or list of Numpy arrays if the model has multiple outputs.
                If all outputs in the model are named, you can also pass a dictionary
                mapping output names to Numpy arrays.
            batch_size: integer. Number of samples per gradient update.
            nb_epoch: integer, the number of times to iterate over the training data arrays.
            verbose: 0, 1, or 2. Verbosity mode. 0 = silent, 1 = verbose, 2 = one log line per epoch.
            callbacks: list of callbacks to be called during training.
                See [callbacks](/callbacks).
            validation_split: float between 0 and 1:
                fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate the loss and any model metrics
                on this data at the end of each epoch.
            validation_data: data on which to evaluate the loss and any model metrics
                at the end of each epoch. The model will not be trained on this data.
                This could be a tuple (x_val, y_val) or a tuple (val_x, val_y, val_sample_weights).
            shuffle: boolean, whether to shuffle the training data before each epoch.
            class_weight: optional dictionary mapping class indices (integers) to
                a weight (float) to apply to the model's loss for the samples
                from this class during training.
                This can be useful to tell the model to "pay more attention" to
                samples from an under-represented class.
            sample_weight: optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify sample_weight_mode="temporal" in compile().


        # Returns
            A `History` instance. Its `history` attribute contains
            all information collected during training.
        '''
        # validate user data
        x, y, sample_weights = self._standardize_user_data(x, y,
                                                           sample_weight=sample_weight,
                                                           class_weight=class_weight,
                                                           check_batch_dim=False,
                                                           batch_size=batch_size)
        # prepare validation data
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise
            val_x, val_y, val_sample_weights = self._standardize_user_data(val_x, val_y,
                                                                           sample_weight=val_sample_weight,
                                                                           check_batch_dim=False,
                                                                           batch_size=batch_size)
            self._make_test_function()
            val_f = self.test_function
            if self.uses_learning_phase:
                val_ins = val_x + val_y + val_sample_weights + [0.]
            else:
                val_ins = val_x + val_y + val_sample_weights

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (slice_X(x, 0, split_at), slice_X(x, split_at))
            y, val_y = (slice_X(y, 0, split_at), slice_X(y, split_at))
            sample_weights, val_sample_weights = (
                slice_X(sample_weights, 0, split_at), slice_X(sample_weights, split_at))
            self._make_test_function()
            val_f = self.test_function
            if self.uses_learning_phase:
                val_ins = val_x + val_y + val_sample_weights + [0.]
            else:
                val_ins = val_x + val_y + val_sample_weights
        else:
            do_validation = False
            val_f = None
            val_ins = None

        # prepare input arrays and training function
        if self.uses_learning_phase:
            ins = x + y + sample_weights + [1.]
        else:
            ins = x + y + sample_weights
        self._make_train_function()
        f = self.train_function

        # prepare display labels
        out_labels = self.metrics_names

        # rename duplicated metrics name
        # (can happen with an output layer shared among multiple dataflows)
        deduped_out_labels = []
        for i, label in enumerate(out_labels):
            new_label = label
            if out_labels.count(label) > 1:
                dup_idx = out_labels[:i].count(label)
                new_label += '_' + str(dup_idx + 1)
            deduped_out_labels.append(new_label)
        out_labels = deduped_out_labels

        if do_validation:
            callback_metrics = copy.copy(out_labels) + ['val_' + n for n in out_labels]
        else:
            callback_metrics = copy.copy(out_labels)

        # delegate logic to _fit_loop
        return self._fit_loop(f, ins, out_labels=out_labels,
                              batch_size=batch_size, nb_epoch=nb_epoch,
                              verbose=verbose, callbacks=callbacks,
                              val_f=val_f, val_ins=val_ins, shuffle=shuffle,
                              callback_metrics=callback_metrics)

    def evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None):
        '''Returns the loss value and metrics values for the model
        in test mode. Computation is done in batches.

        # Arguments
            x: Numpy array of test data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named, you can also pass a dictionary
                mapping input names to Numpy arrays.
            y: Numpy array of target data,
                or list of Numpy arrays if the model has multiple outputs.
                If all outputs in the model are named, you can also pass a dictionary
                mapping output names to Numpy arrays.
            batch_size: integer. Number of samples per gradient update.

        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        '''
        # validate user data
        x, y, sample_weights = self._standardize_user_data(x, y,
                                                           sample_weight=sample_weight,
                                                           check_batch_dim=False,
                                                           batch_size=batch_size)
        # prepare inputs, delegate logic to _test_loop
        if self.uses_learning_phase:
            ins = x + y + sample_weights + [0.]
        else:
            ins = x + y + sample_weights
        self._make_test_function()
        f = self.test_function
        return self._test_loop(f, ins,
                               batch_size=batch_size,
                               verbose=verbose)

    def predict(self, x, batch_size=32, verbose=0):
        '''Generates output predictions for the input samples,
        processing the samples in a batched way.

        # Arguments
            x: the input data, as a Numpy array
                (or list of Numpy arrays if the model has multiple outputs).
            batch_size: integer.
            verbose: verbosity mode, 0 or 1.

        # Returns
            A Numpy array of predictions.
        '''
        # validate user data
        x = standardize_input_data(x, self.input_names,
                                   self.internal_input_shapes,
                                   check_batch_dim=False)
        if self.stateful:
            if x[0].shape[0] > batch_size and x[0].shape[0] % batch_size != 0:
                raise Exception('In a stateful network, '
                                'you should only pass inputs with '
                                'a number of samples that can be '
                                'divided by the batch size. Found: ' +
                                str(x[0].shape[0]) + ' samples. '
                                'Batch size: ' + str(batch_size) + '.')

        # prepare inputs, delegate logic to _predict_loop
        if self.uses_learning_phase:
            ins = x + [0.]
        else:
            ins = x
        self._make_predict_function()
        f = self.predict_function
        return self._predict_loop(f, ins,
                                  batch_size=batch_size, verbose=verbose)

    def train_on_batch(self, x, y,
                       sample_weight=None, class_weight=None):
        '''Runs a single gradient update on a single batch of data.

        # Arguments
            x: Numpy array of training data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named, you can also pass a dictionary
                mapping input names to Numpy arrays.
            y: Numpy array of target data,
                or list of Numpy arrays if the model has multiple outputs.
                If all outputs in the model are named, you can also pass a dictionary
                mapping output names to Numpy arrays.
            sample_weight: optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify sample_weight_mode="temporal" in compile().
            class_weight: optional dictionary mapping class indices (integers) to
                a weight (float) to apply to the model's loss for the samples
                from this class during training.
                This can be useful to tell the model to "pay more attention" to
                samples from an under-represented class.

        # Returns
            Scalar training loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        '''
        x, y, sample_weights = self._standardize_user_data(x, y,
                                                           sample_weight=sample_weight,
                                                           class_weight=class_weight,
                                                           check_batch_dim=True)
        if self.uses_learning_phase:
            ins = x + y + sample_weights + [1.]
        else:
            ins = x + y + sample_weights
        self._make_train_function()
        outputs = self.train_function(ins)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def test_on_batch(self, x, y, sample_weight=None):
        '''Test the model on a single batch of samples.

        # Arguments
            x: Numpy array of test data,
                or list of Numpy arrays if the model has multiple inputs.
                If all inputs in the model are named, you can also pass a dictionary
                mapping input names to Numpy arrays.
            y: Numpy array of target data,
                or list of Numpy arrays if the model has multiple outputs.
                If all outputs in the model are named, you can also pass a dictionary
                mapping output names to Numpy arrays.
            sample_weight: optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape (samples, sequence_length),
                to apply a different weight to every timestep of every sample.
                In this case you should make sure to specify sample_weight_mode="temporal" in compile().

        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        '''
        x, y, sample_weights = self._standardize_user_data(x, y,
                                                           sample_weight=sample_weight,
                                                           check_batch_dim=True)
        if self.uses_learning_phase:
            ins = x + y + sample_weights + [0.]
        else:
            ins = x + y + sample_weights
        self._make_test_function()
        outputs = self.test_function(ins)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def predict_on_batch(self, x):
        '''Returns predictions for a single batch of samples.
        '''
        x = standardize_input_data(x, self.input_names,
                                   self.internal_input_shapes)
        if self.uses_learning_phase:
            ins = x + [0.]
        else:
            ins = x
        self._make_predict_function()
        outputs = self.predict_function(ins)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

    def fit_generator(self, generator, samples_per_epoch, nb_epoch,
                      verbose=1, callbacks=[],
                      validation_data=None, nb_val_samples=None,
                      class_weight={}, max_q_size=10, nb_worker=1, pickle_safe=False):
        '''Fits the model on data generated batch-by-batch by
        a Python generator.
        The generator is run in parallel to the model, for efficiency.
        For instance, this allows you to do real-time data augmentation
        on images on CPU in parallel to training your model on GPU.

        # Arguments
            generator: a generator.
                The output of the generator must be either
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
                All arrays should contain the same number of samples.
                The generator is expected to loop over its data
                indefinitely. An epoch finishes when `samples_per_epoch`
                samples have been seen by the model.
            samples_per_epoch: integer, number of samples to process before
                going to the next epoch.
            nb_epoch: integer, total number of iterations on the data.
            verbose: verbosity mode, 0, 1, or 2.
            callbacks: list of callbacks to be called during training.
            validation_data: this can be either
                - a generator for the validation data
                - a tuple (inputs, targets)
                - a tuple (inputs, targets, sample_weights).
            nb_val_samples: only relevant if `validation_data` is a generator.
                number of samples to use from validation generator
                at the end of every epoch.
            class_weight: dictionary mapping class indices to a weight
                for the class.
            max_q_size: maximum size for the generator queue
            nb_worker: maximum number of processes to spin up when using process based threading
            pickle_safe: if True, use process based threading. Note that because
                this implementation relies on multiprocessing, you should not pass non
                non picklable arguments to the generator as they can't be passed
                easily to children processes.

        # Returns
            A `History` object.

        # Example

        ```python
            def generate_arrays_from_file(path):
                while 1:
                    f = open(path)
                    for line in f:
                        # create numpy arrays of input data
                        # and labels, from each line in the file
                        x1, x2, y = process_line(line)
                        yield ({'input_1': x1, 'input_2': x2}, {'output': y})
                    f.close()

            model.fit_generator(generate_arrays_from_file('/my_file.txt'),
                                samples_per_epoch=10000, nb_epoch=10)
        ```
        '''
        wait_time = 0.01  # in seconds
        epoch = 0

        do_validation = bool(validation_data)
        self._make_train_function()
        if do_validation:
            self._make_test_function()

        # python 2 has 'next', 3 has '__next__'
        # avoid any explicit version checks
        val_gen = (hasattr(validation_data, 'next') or
                   hasattr(validation_data, '__next__'))
        if val_gen and not nb_val_samples:
            raise Exception('When using a generator for validation data, '
                            'you must specify a value for "nb_val_samples".')

        out_labels = self.metrics_names
        callback_metrics = out_labels + ['val_' + n for n in out_labels]

        # prepare callbacks
        self.history = cbks.History()
        callbacks = [cbks.BaseLogger()] + callbacks + [self.history]
        if verbose:
            callbacks += [cbks.ProgbarLogger()]
        callbacks = cbks.CallbackList(callbacks)

        # it's possible to callback a different model than self:
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self
        callbacks._set_model(callback_model)
        callbacks._set_params({
            'nb_epoch': nb_epoch,
            'nb_sample': samples_per_epoch,
            'verbose': verbose,
            'do_validation': do_validation,
            'metrics': callback_metrics,
        })
        callbacks.on_train_begin()

        if do_validation and not val_gen:
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise Exception('validation_data should be a tuple '
                                '(val_x, val_y, val_sample_weight) '
                                'or (val_x, val_y). Found: ' + str(validation_data))
            val_x, val_y, val_sample_weights = self._standardize_user_data(val_x, val_y, val_sample_weight)
            self.validation_data = val_x + [val_y, val_sample_weights]
        else:
            self.validation_data = None

        # start generator thread storing batches into a queue
        data_gen_queue, _stop = generator_queue(generator, max_q_size=max_q_size, nb_worker=nb_worker,
                                                pickle_safe=pickle_safe)

        callback_model.stop_training = False
        while epoch < nb_epoch:
            callbacks.on_epoch_begin(epoch)
            samples_seen = 0
            batch_index = 0
            while samples_seen < samples_per_epoch:
                generator_output = None
                while not _stop.is_set():
                    if not data_gen_queue.empty():
                        generator_output = data_gen_queue.get()
                        break
                    else:
                        time.sleep(wait_time)

                if not hasattr(generator_output, '__len__'):
                    _stop.set()
                    raise Exception('output of generator should be a tuple '
                                    '(x, y, sample_weight) '
                                    'or (x, y). Found: ' + str(generator_output))
                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    _stop.set()
                    raise Exception('output of generator should be a tuple '
                                    '(x, y, sample_weight) '
                                    'or (x, y). Found: ' + str(generator_output))
                # build batch logs
                batch_logs = {}
                if type(x) is list:
                    batch_size = len(x[0])
                elif type(x) is dict:
                    batch_size = len(list(x.values())[0])
                else:
                    batch_size = len(x)
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_batch_begin(batch_index, batch_logs)

                try:
                    outs = self.train_on_batch(x, y,
                                               sample_weight=sample_weight,
                                               class_weight=class_weight)
                except:
                    _stop.set()
                    raise

                if type(outs) != list:
                    outs = [outs]
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_batch_end(batch_index, batch_logs)

                # construct epoch logs
                epoch_logs = {}
                batch_index += 1
                samples_seen += batch_size

                # epoch finished
                if samples_seen > samples_per_epoch:
                    warnings.warn('Epoch comprised more than '
                                  '`samples_per_epoch` samples, '
                                  'which might affect learning results. '
                                  'Set `samples_per_epoch` correctly '
                                  'to avoid this warning.')
                if samples_seen >= samples_per_epoch and do_validation:
                    if val_gen:
                        val_outs = self.evaluate_generator(validation_data,
                                                           nb_val_samples,
                                                           max_q_size=max_q_size)
                    else:
                        # no need for try/except because
                        # data has already been validated
                        val_outs = self.evaluate(val_x, val_y,
                                                 sample_weight=val_sample_weights,
                                                 verbose=0)
                    if type(val_outs) is not list:
                        val_outs = [val_outs]
                    # same labels assumed
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            if callback_model.stop_training:
                break

        _stop.set()
        if pickle_safe:
            data_gen_queue.close()
        callbacks.on_train_end()
        return self.history

    def evaluate_generator(self, generator, val_samples, max_q_size=10, nb_worker=1, pickle_safe=False):
        '''Evaluates the model on a data generator. The generator should
        return the same kind of data as accepted by `test_on_batch`.

        Arguments:
            generator:
                generator yielding tuples (inputs, targets)
                or (inputs, targets, sample_weights)
            val_samples:
                total number of samples to generate from `generator`
                before returning.
            max_q_size: maximum size for the generator queue
            nb_worker: maximum number of processes to spin up when using process based threading
            pickle_safe: if True, use process based threading. Note that because
                this implementation relies on multiprocessing, you should not pass non
                non picklable arguments to the generator as they can't be passed
                easily to children processes.

        # Returns
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        '''
        self._make_test_function()

        processed_samples = 0
        wait_time = 0.01
        all_outs = []
        weights = []
        data_gen_queue, _stop = generator_queue(generator, max_q_size=max_q_size, nb_worker=nb_worker,
                                                pickle_safe=pickle_safe)

        while processed_samples < val_samples:
            generator_output = None
            while not _stop.is_set():
                if not data_gen_queue.empty():
                    generator_output = data_gen_queue.get()
                    break
                else:
                    time.sleep(wait_time)

            if not hasattr(generator_output, '__len__'):
                _stop.set()
                raise Exception('output of generator should be a tuple '
                                '(x, y, sample_weight) '
                                'or (x, y). Found: ' + str(generator_output))
            if len(generator_output) == 2:
                x, y = generator_output
                sample_weight = None
            elif len(generator_output) == 3:
                x, y, sample_weight = generator_output
            else:
                _stop.set()
                raise Exception('output of generator should be a tuple '
                                '(x, y, sample_weight) '
                                'or (x, y). Found: ' + str(generator_output))
            try:
                outs = self.test_on_batch(x, y, sample_weight=sample_weight)
            except:
                _stop.set()
                raise

            if type(x) is list:
                nb_samples = len(x[0])
            elif type(x) is dict:
                nb_samples = len(list(x.values())[0])
            else:
                nb_samples = len(x)
            all_outs.append(outs)

            processed_samples += nb_samples
            weights.append(nb_samples)

        _stop.set()
        if pickle_safe:
            data_gen_queue.close()
        if type(outs) is not list:
            return np.average(np.asarray(all_outs),
                              weights=weights)
        else:
            averages = []
            for i in range(len(outs)):
                averages.append(np.average([out[i] for out in all_outs],
                                           weights=weights))
            return averages

    def predict_generator(self, generator, val_samples, max_q_size=10, nb_worker=1, pickle_safe=False):
        '''Generates predictions for the input samples from a data generator.
        The generator should return the same kind of data as accepted by
        `predict_on_batch`.

        # Arguments
            generator: generator yielding batches of input samples.
            val_samples: total number of samples to generate from `generator`
                before returning.
            max_q_size: maximum size for the generator queue
            nb_worker: maximum number of processes to spin up when using process based threading
            pickle_safe: if True, use process based threading. Note that because
                this implementation relies on multiprocessing, you should not pass non
                non picklable arguments to the generator as they can't be passed
                easily to children processes.

        # Returns
            Numpy array(s) of predictions.
        '''
        self._make_predict_function()

        processed_samples = 0
        wait_time = 0.01
        all_outs = []
        data_gen_queue, _stop = generator_queue(generator, max_q_size=max_q_size, nb_worker=nb_worker,
                                                pickle_safe=pickle_safe)

        while processed_samples < val_samples:
            generator_output = None
            while not _stop.is_set():
                if not data_gen_queue.empty():
                    generator_output = data_gen_queue.get()
                    break
                else:
                    time.sleep(wait_time)

            if isinstance(generator_output, tuple):
                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    _stop.set()
                    raise Exception('output of generator should be a tuple '
                                    '(x, y, sample_weight) '
                                    'or (x, y). Found: ' + str(generator_output))
            else:
                x = generator_output

            try:
                outs = self.predict_on_batch(x)
            except:
                _stop.set()
                raise

            if type(x) is list:
                nb_samples = len(x[0])
            elif type(x) is dict:
                nb_samples = len(list(x.values())[0])
            else:
                nb_samples = len(x)

            if type(outs) != list:
                outs = [outs]

            if len(all_outs) == 0:
                for out in outs:
                    shape = (val_samples,) + out.shape[1:]
                    all_outs.append(np.zeros(shape))

            for i, out in enumerate(outs):
                all_outs[i][processed_samples:(processed_samples + nb_samples)] = out

            processed_samples += nb_samples

        _stop.set()
        if pickle_safe:
            data_gen_queue.close()
        if len(all_outs) == 1:
            return all_outs[0]
        return all_outs
