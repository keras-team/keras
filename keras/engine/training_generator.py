"""Part of the training engine related to Python generators of array data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
import numpy as np

from .training_utils import is_sequence
from .training_utils import iter_sequence_infinite
from .training_utils import should_run_validation
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
                  validation_freq=1,
                  class_weight=None,
                  max_queue_size=10,
                  workers=1,
                  use_multiprocessing=False,
                  shuffle=True,
                  initial_epoch=0):
    """See docstring for `Model.fit_generator`."""
    epoch = initial_epoch

    do_validation = bool(validation_data)
    model._make_train_function()
    if do_validation:
        model._make_test_function()

    use_sequence_api = is_sequence(generator)
    if not use_sequence_api and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the `keras.utils.Sequence'
                        ' class.'))
    if steps_per_epoch is None:
        if use_sequence_api:
            steps_per_epoch = len(generator)
        else:
            raise ValueError('`steps_per_epoch=None` is only valid for a'
                             ' generator based on the '
                             '`keras.utils.Sequence`'
                             ' class. Please specify `steps_per_epoch` '
                             'or use the `keras.utils.Sequence` class.')

    # python 2 has 'next', 3 has '__next__'
    # avoid any explicit version checks
    val_use_sequence_api = is_sequence(validation_data)
    val_gen = (hasattr(validation_data, 'next') or
               hasattr(validation_data, '__next__') or
               val_use_sequence_api)
    if (val_gen and not val_use_sequence_api and
            not validation_steps):
        raise ValueError('`validation_steps=None` is only valid for a'
                         ' generator based on the `keras.utils.Sequence`'
                         ' class. Please specify `validation_steps` or use'
                         ' the `keras.utils.Sequence` class.')

    # Prepare display labels.
    out_labels = model.metrics_names
    callback_metrics = out_labels + ['val_' + n for n in out_labels]

    # prepare callbacks
    model.history = cbks.History()
    _callbacks = [cbks.BaseLogger(
        stateful_metrics=model.stateful_metric_names)]
    if verbose:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_mode='steps',
                stateful_metrics=model.stateful_metric_names))
    _callbacks += (callbacks or []) + [model.history]
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than self:
    callback_model = model._get_callback_model()

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'epochs': epochs,
        'steps': steps_per_epoch,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics,
    })
    callbacks._call_begin_hook('train')

    enqueuer = None
    val_enqueuer = None

    try:
        if do_validation:
            if val_gen and workers > 0:
                # Create an Enqueuer that can be reused
                val_data = validation_data
                if is_sequence(val_data):
                    val_enqueuer = OrderedEnqueuer(
                        val_data,
                        use_multiprocessing=use_multiprocessing)
                    validation_steps = validation_steps or len(val_data)
                else:
                    val_enqueuer = GeneratorEnqueuer(
                        val_data,
                        use_multiprocessing=use_multiprocessing)
                val_enqueuer.start(workers=workers,
                                   max_queue_size=max_queue_size)
                val_enqueuer_gen = val_enqueuer.get()
            elif val_gen:
                val_data = validation_data
                if is_sequence(val_data):
                    val_enqueuer_gen = iter_sequence_infinite(val_data)
                    validation_steps = validation_steps or len(val_data)
                else:
                    val_enqueuer_gen = val_data
            else:
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
                if model.uses_learning_phase and not isinstance(K.learning_phase(),
                                                                int):
                    val_data += [0.]
                for cbk in callbacks:
                    cbk.validation_data = val_data

        if workers > 0:
            if use_sequence_api:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing,
                    shuffle=shuffle)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if use_sequence_api:
                output_generator = iter_sequence_infinite(generator)
            else:
                output_generator = generator

        callbacks.model.stop_training = False
        # Construct epoch logs.
        epoch_logs = {}
        while epoch < epochs:
            for m in model.stateful_metric_functions:
                m.reset_states()
            callbacks.on_epoch_begin(epoch)
            steps_done = 0
            batch_index = 0
            while steps_done < steps_per_epoch:
                generator_output = next(output_generator)

                if not hasattr(generator_output, '__len__'):
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))

                if len(generator_output) == 2:
                    x, y = generator_output
                    sample_weight = None
                elif len(generator_output) == 3:
                    x, y, sample_weight = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))
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
                # build batch logs
                batch_logs = {'batch': batch_index, 'size': batch_size}
                callbacks.on_batch_begin(batch_index, batch_logs)

                outs = model.train_on_batch(x, y,
                                            sample_weight=sample_weight,
                                            class_weight=class_weight)

                outs = to_list(outs)
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks._call_batch_hook('train', 'end', batch_index, batch_logs)

                batch_index += 1
                steps_done += 1

                # Epoch finished.
                if (steps_done >= steps_per_epoch and
                        do_validation and
                        should_run_validation(validation_freq, epoch)):
                    # Note that `callbacks` here is an instance of
                    # `keras.callbacks.CallbackList`
                    if val_gen:
                        val_outs = model.evaluate_generator(
                            val_enqueuer_gen,
                            validation_steps,
                            callbacks=callbacks,
                            workers=0)
                    else:
                        # No need for try/except because
                        # data has already been validated.
                        val_outs = model.evaluate(
                            val_x, val_y,
                            batch_size=batch_size,
                            sample_weight=val_sample_weights,
                            callbacks=callbacks,
                            verbose=0)
                    val_outs = to_list(val_outs)
                    # Same labels assumed.
                    for l, o in zip(out_labels, val_outs):
                        epoch_logs['val_' + l] = o

                if callbacks.model.stop_training:
                    break

            callbacks.on_epoch_end(epoch, epoch_logs)
            epoch += 1
            if callbacks.model.stop_training:
                break

    finally:
        try:
            if enqueuer is not None:
                enqueuer.stop()
        finally:
            if val_enqueuer is not None:
                val_enqueuer.stop()

    callbacks._call_end_hook('train')
    return model.history


def evaluate_generator(model, generator,
                       steps=None,
                       callbacks=None,
                       max_queue_size=10,
                       workers=1,
                       use_multiprocessing=False,
                       verbose=0):
    """See docstring for `Model.evaluate_generator`."""
    model._make_test_function()

    if hasattr(model, 'metrics'):
        for m in model.stateful_metric_functions:
            m.reset_states()
        stateful_metric_indices = [
            i for i, name in enumerate(model.metrics_names)
            if str(name) in model.stateful_metric_names]
    else:
        stateful_metric_indices = []

    steps_done = 0
    outs_per_batch = []
    batch_sizes = []
    use_sequence_api = is_sequence(generator)
    if not use_sequence_api and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the `keras.utils.Sequence'
                        ' class.'))
    if steps is None:
        if use_sequence_api:
            steps = len(generator)
        else:
            raise ValueError('`steps=None` is only valid for a generator'
                             ' based on the `keras.utils.Sequence` class.'
                             ' Please specify `steps` or use the'
                             ' `keras.utils.Sequence` class.')
    enqueuer = None

    # Check if callbacks have not been already configured
    if not isinstance(callbacks, cbks.CallbackList):
        callbacks = cbks.CallbackList(callbacks)
        callback_model = model._get_callback_model()
        callbacks.set_model(callback_model)
        callback_metrics = []
        if hasattr(model, 'metrics_names'):
            callback_metrics = list(model.metrics_names)
        callback_params = {
            'steps': steps,
            'verbose': verbose,
            'metrics': callback_metrics,
        }
        callbacks.set_params(callback_params)

    callbacks.model.stop_training = False
    callbacks._call_begin_hook('test')

    try:
        if workers > 0:
            if use_sequence_api:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if use_sequence_api:
                output_generator = iter_sequence_infinite(generator)
            else:
                output_generator = generator

        if verbose == 1:
            progbar = Progbar(target=steps)

        while steps_done < steps:
            generator_output = next(output_generator)
            if not hasattr(generator_output, '__len__'):
                raise ValueError('Output of generator should be a tuple '
                                 '(x, y, sample_weight) '
                                 'or (x, y). Found: ' +
                                 str(generator_output))
            if len(generator_output) == 2:
                x, y = generator_output
                sample_weight = None
            elif len(generator_output) == 3:
                x, y, sample_weight = generator_output
            else:
                raise ValueError('Output of generator should be a tuple '
                                 '(x, y, sample_weight) '
                                 'or (x, y). Found: ' +
                                 str(generator_output))

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

            batch_logs = {'batch': steps_done, 'size': batch_size}
            callbacks._call_batch_hook('test', 'begin', steps_done, batch_logs)
            outs = model.test_on_batch(x, y, sample_weight=sample_weight)
            outs = to_list(outs)
            outs_per_batch.append(outs)

            if hasattr(model, 'metrics_names'):
                for l, o in zip(model.metrics_names, outs):
                    batch_logs[l] = o
            callbacks._call_batch_hook('test', 'end', steps_done, batch_logs)

            steps_done += 1
            batch_sizes.append(batch_size)

            if verbose == 1:
                progbar.update(steps_done)
        callbacks._call_end_hook('test')

    finally:
        if enqueuer is not None:
            enqueuer.stop()

    averages = []
    for i in range(len(outs)):
        if i not in stateful_metric_indices:
            averages.append(np.average([out[i] for out in outs_per_batch],
                                       weights=batch_sizes))
        else:
            averages.append(np.float64(outs_per_batch[-1][i]))
    return unpack_singleton(averages)


def predict_generator(model, generator,
                      steps=None,
                      callbacks=None,
                      max_queue_size=10,
                      workers=1,
                      use_multiprocessing=False,
                      verbose=0):
    """See docstring for `Model.predict_generator`."""
    model._make_predict_function()

    steps_done = 0
    all_outs = []
    use_sequence_api = is_sequence(generator)
    if not use_sequence_api and use_multiprocessing and workers > 1:
        warnings.warn(
            UserWarning('Using a generator with `use_multiprocessing=True`'
                        ' and multiple workers may duplicate your data.'
                        ' Please consider using the `keras.utils.Sequence'
                        ' class.'))
    if steps is None:
        if use_sequence_api:
            steps = len(generator)
        else:
            raise ValueError('`steps=None` is only valid for a generator'
                             ' based on the `keras.utils.Sequence` class.'
                             ' Please specify `steps` or use the'
                             ' `keras.utils.Sequence` class.')
    enqueuer = None

    # Check if callbacks have not been already configured
    if not isinstance(callbacks, cbks.CallbackList):
        callbacks = cbks.CallbackList(callbacks)
        callback_model = model._get_callback_model()
        callbacks.set_model(callback_model)
        callback_params = {
            'steps': steps,
            'verbose': verbose,
        }
        callbacks.set_params(callback_params)

    callbacks.model.stop_training = False
    callbacks._call_begin_hook('predict')

    try:
        if workers > 0:
            if use_sequence_api:
                enqueuer = OrderedEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            else:
                enqueuer = GeneratorEnqueuer(
                    generator,
                    use_multiprocessing=use_multiprocessing)
            enqueuer.start(workers=workers, max_queue_size=max_queue_size)
            output_generator = enqueuer.get()
        else:
            if use_sequence_api:
                output_generator = iter_sequence_infinite(generator)
            else:
                output_generator = generator

        if verbose == 1:
            progbar = Progbar(target=steps)

        while steps_done < steps:
            generator_output = next(output_generator)
            if isinstance(generator_output, tuple):
                # Compatibility with the generators
                # used for training.
                if len(generator_output) == 2:
                    x, _ = generator_output
                elif len(generator_output) == 3:
                    x, _, _ = generator_output
                else:
                    raise ValueError('Output of generator should be '
                                     'a tuple `(x, y, sample_weight)` '
                                     'or `(x, y)`. Found: ' +
                                     str(generator_output))
            else:
                # Assumes a generator that only
                # yields inputs (not targets and sample weights).
                x = generator_output

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

            batch_logs = {'batch': steps_done, 'size': batch_size}
            callbacks._call_batch_hook('predict', 'begin', steps_done, batch_logs)

            outs = model.predict_on_batch(x)
            outs = to_list(outs)

            if not all_outs:
                for out in outs:
                    all_outs.append([])

            for i, out in enumerate(outs):
                all_outs[i].append(out)

            batch_logs['outputs'] = outs
            callbacks._call_batch_hook('predict', 'end', steps_done, batch_logs)

            steps_done += 1
            if verbose == 1:
                progbar.update(steps_done)
        callbacks._call_end_hook('predict')
    finally:
        if enqueuer is not None:
            enqueuer.stop()

    if len(all_outs) == 1:
        if steps_done == 1:
            return all_outs[0][0]
        else:
            return np.concatenate(all_outs[0])
    if steps_done == 1:
        return [out[0] for out in all_outs]
    else:
        return [np.concatenate(out) for out in all_outs]
