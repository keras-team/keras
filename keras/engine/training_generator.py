"""Part of the training engine related to Python generators of array data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np

from .training_utils import iter_sequence_infinite
from .. import backend as K
from ..utils.data_utils import Sequence
from ..utils.data_utils import GeneratorEnqueuer
from ..utils.data_utils import OrderedEnqueuer
from ..utils.generic_utils import Progbar
from ..utils.generic_utils import to_list
from ..utils.generic_utils import unpack_singleton
from .. import callbacks as cbks


def fit_generator(model,
                  generator,
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
    """See docstring for `Model.fit_generator`."""
    epoch = initial_epoch
    do_validation = bool(validation_data)

    enqueuer, generator, steps_per_epoch = init_generator(
        generator, steps_per_epoch,
        workers, max_queue_size,
        use_multiprocessing, shuffle, wait_time=0.01,
        steps_arg='steps_per_epoch')

    val_enqueuer = None
    if do_validation:
        # python 2 has 'next', 3 has '__next__'
        # avoid any explicit version checks
        val_gen = (hasattr(validation_data, 'next') or
                   hasattr(validation_data, '__next__') or
                   isinstance(validation_data, Sequence))

        if val_gen:
            # todo: necessary? (repeated in evaluate_generator (with different wait_time)?)
            val_enqueuer, val_generator, validation_steps = init_generator(
                validation_data, validation_steps,
                workers, max_queue_size,
                use_multiprocessing,
                steps_arg='validation_steps')
        else:
            # todo: share with training_array?
            # Prepare data for validation
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('`validation_data` should be a tuple '
                                 '`(val_x, val_y, val_sample_weight)` '
                                 'or `(val_x, val_y)`. Found: ' +
                                 str(validation_data))
            val_x, val_y, val_sample_weights = model._standardize_user_data(
                val_x, val_y, val_sample_weight)
            val_data = val_x + val_y + val_sample_weights
            if model._uses_dynamic_learning_phase():
                val_data += [0.]

    # prepare display labels.
    out_labels = model.metrics_names
    callback_metrics = out_labels
    if do_validation:
        callback_metrics += ['val_' + n for n in out_labels]

    # prepare callbacks
    model.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=model.stateful_metric_names)]
    if verbose:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_param='steps',
                stateful_metrics=model.stateful_metric_names))
    _callbacks += (callbacks or []) + [model.history]
    callbacks_arg = callbacks
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than self:
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model
    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'val_steps': validation_steps,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics or [],
    })
    callbacks.on_train_begin()
    callback_model.stop_training = False
    if do_validation and not val_gen:
        for cbk in callbacks:
            cbk.validation_data = val_data

    try:
        # Construct epoch logs.
        epoch_logs = {}
        while epoch < epochs:
            for m in model.stateful_metric_functions:
                m.reset_states()
            callbacks.on_epoch_begin(epoch)
            batch_index = 0
            while batch_index < steps_per_epoch:
                x, y, sample_weight = get_batch(generator)
                # build batch logs
                batch_size = get_batch_size(x)
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = batch_size
                callbacks.on_fit_batch_begin(batch_index, batch_logs)

                batch_ins = model.prepare_inputs(x, y,
                                                 sample_weight=sample_weight,
                                                 class_weight=class_weight)
                batch_outs = model.train_function(batch_ins)

                for l, o in zip(out_labels, batch_outs):
                    batch_logs[l] = o

                callbacks.on_fit_batch_end(batch_index, batch_logs)

                batch_index += 1

                # Epoch finished.
                if batch_index >= steps_per_epoch and do_validation:
                    if val_gen:
                        val_outs = model.evaluate_generator(
                            val_generator,
                            validation_steps,
                            workers=0,
                            callbacks=callbacks_arg)
                    else:
                        # No need for try/except because
                        # data has already been validated.
                        val_outs = model.evaluate(
                            val_x, val_y,
                            batch_size=batch_size,
                            sample_weight=val_sample_weights,
                            verbose=0,
                            callbacks=callbacks_arg)
                    val_outs = to_list(val_outs)
                    # Same labels assumed.
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o

                if callback_model.stop_training:
                    break

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            if callback_model.stop_training:
                break

    finally:
        try:
            if enqueuer is not None:
                enqueuer.stop()
        finally:
            if val_enqueuer is not None:
                val_enqueuer.stop()

    callbacks.on_train_end()
    return model.history


def evaluate_generator(model, generator,
                       steps=None,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=False,
                       verbose=0,
                       callbacks=None):
    """See docstring for `Model.evaluate_generator`."""
    if hasattr(model, 'metrics'):
        for m in model.stateful_metric_functions:
            m.reset_states()
        stateful_metric_indices = [
            i for i, name in enumerate(model.metrics_names)
            if str(name) in model.stateful_metric_names]
    else:
        stateful_metric_indices = []

    batch_index = 0
    outs_per_batch = []
    batch_sizes = []
    enqueuer, generator, steps = init_generator(
        generator, steps,
        workers, max_queue_size,
        use_multiprocessing, wait_time=0.01)

    out_labels = ['val_' + n for n in model.metrics_names]

    _callbacks = []
    if verbose == 1:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_param='val_steps',
                stateful_metrics=model.stateful_metric_names))

    _callbacks += callbacks or []
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than itself
    # (used by Sequential models)
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'val_steps': steps,
        'val_metrics': out_labels,
        'verbose': verbose
    })
    callback_model.stop_evaluating = False

    try:
        while batch_index < steps:
            x, y, sample_weight = get_batch(generator)

            # build batch logs
            batch_size = get_batch_size(x)
            batch_logs = {}
            batch_logs['batch'] = batch_index
            batch_logs['size'] = batch_size

            callbacks.on_evaluate_batch_begin(batch_index, batch_logs)

            batch_ins = model.prepare_inputs(x, y,
                                             sample_weight=sample_weight)
            batch_outs = model.test_function(batch_ins)

            for l, o in zip(out_labels, batch_outs):
                batch_logs[l] = o

            callbacks.on_evaluate_batch_end(batch_index, batch_logs)

            batch_index += 1
            outs_per_batch.append(batch_outs)
            batch_sizes.append(batch_size)
            if callback_model.stop_evaluating:
                break

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    averages = []
    for i in range(len(batch_outs)):
        if i not in stateful_metric_indices:
            averages.append(np.average([out[i] for out in outs_per_batch],
                                       weights=batch_sizes))
        else:
            averages.append(np.float64(outs_per_batch[-1][i]))
    return unpack_singleton(averages)


def predict_generator(model, generator,
                      steps=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      verbose=0,
                      callbacks=None):
    """See docstring for `Model.predict_generator`."""
    batch_index = 0
    # todo: wait_time?
    enqueuer, generator, steps = init_generator(
        generator, steps,
        workers, max_queue_size,
        use_multiprocessing, wait_time=0.01)

    _callbacks = []
    if verbose == 1:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_param='pred_steps',
                stateful_metrics=model.stateful_metric_names))

    _callbacks += callbacks or []
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than itself
    # (used by Sequential models)
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'pred_steps': steps,
        'verbose': verbose
    })
    callback_model.stop_predicting = False

    # Step-based predictions.
    # Since we do not know how many samples
    # we will see, we cannot pre-allocate
    # the returned Numpy arrays.
    # Instead, we store one array per batch seen
    # and concatenate them upon returning.
    unconcatenated_outs = []
    try:
        while batch_index < steps:
            x, _, _ = get_batch(generator, require_output=False)

            # build batch logs
            batch_size = get_batch_size(x)
            batch_logs = {}
            batch_logs['batch'] = batch_index
            batch_logs['size'] = batch_size

            callbacks.on_predict_batch_begin(batch_index, batch_logs)

            batch_ins = model.prepare_inputs(x)
            batch_outs = model.predict_function(batch_ins)

            callbacks.on_predict_batch_end(batch_index, batch_logs)

            if batch_index == 0:
                for _ in batch_outs:
                    unconcatenated_outs.append([])

            for i, out in enumerate(batch_outs):
                unconcatenated_outs[i].append(out)

            batch_index += 1
            if callback_model.stop_predicting:
                break

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    concatenated_outs = [np.concatenate(out) for out in unconcatenated_outs]
    return unpack_singleton(concatenated_outs)


def init_generator(generator, steps,
                   workers, max_queue_size,
                   use_multiprocessing, shuffle=False, wait_time=0.05,
                   steps_arg=None):
    # todo: defaults repeated from Enqueuers
    is_sequence = isinstance(generator, Sequence)
    enqueuer = None

    if not is_sequence and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the`keras.utils.Sequence'
                        ' class.'))
    if steps is None:
        if is_sequence:
            steps = len(generator)
        else:
            raise ValueError('`{0}=None` is only valid for a'
                             ' generator based on the '
                             '`keras.utils.Sequence`'
                             ' class. Please specify `{0}`'
                             ' or use the `keras.utils.Sequence`'
                             ' class.'.format(steps_arg or 'steps'))
    if workers > 0:
        if is_sequence:
            enqueuer = OrderedEnqueuer(
                generator,
                use_multiprocessing=use_multiprocessing,
                shuffle=shuffle)
        else:
            enqueuer = GeneratorEnqueuer(
                generator,
                use_multiprocessing=use_multiprocessing,
                wait_time=wait_time)
        enqueuer.start(workers=workers,
                       max_queue_size=max_queue_size)
        output_generator = enqueuer.get()
    else:
        if is_sequence:
            output_generator = iter_sequence_infinite(generator)
        else:
            output_generator = generator
    return enqueuer, output_generator, steps


def get_batch(generator, require_output=True):
    # todo: might break generators that (incorrectly) return list / np.array
    generator_output = next(generator)
    y = sample_weight = None
    if isinstance(generator_output, tuple):
        if len(generator_output) == 2:
            x, y = generator_output
        elif len(generator_output) == 3:
            x, y, sample_weight = generator_output
        else:
            raise generator_output_error(generator_output)
    else:
        if require_output:
            raise generator_output_error(generator_output)
        else:
            # Assumes a generator that only
            # yields inputs (not targets and sample weights).
            x = generator_output

    return x, y, sample_weight


def generator_output_error(generator_output):
    return ValueError('Output of generator should be '
                      'a tuple `(x, y, sample_weight)` '
                      'or `(x, y)`. Found: ' +
                      str(generator_output))


def get_batch_size(x):
    # todo: rename: step_size?
    if x is None or len(x) == 0:
        # Handle data tensors support when no input given
        # step-size = 1 for data tensors
        batch_size = 1
    elif isinstance(x, list):
        batch_size = x[0].shape[0]
    elif isinstance(x, dict):
        batch_size = list(x.values())[0].shape[0]
    else:
        batch_size = x.shape[0]
    if batch_size == 0:
        raise ValueError('Received an empty batch. '
                         'Batches should contain '
                         'at least one item.')
    return batch_size
