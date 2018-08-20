"""Part of the training engine related to plain array data (e.g. Numpy).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.sparse import issparse

from .training_utils import batch_shuffle
from .training_utils import make_batches
from .training_utils import check_num_samples
from .. import backend as K
from .. import callbacks as cbks
from ..utils.generic_utils import slice_arrays
from ..utils.generic_utils import to_list
from ..utils.generic_utils import unpack_singleton


def fit_loop(model, ins,
             batch_size=None,
             epochs=100,
             verbose=1,
             callbacks=None,
             val_ins=None,
             shuffle=True,
             initial_epoch=0,
             steps_per_epoch=None,
             validation_steps=None):
    """Abstract fit function to loop over data in batches and epochs.

    # Arguments
        model: Keras model instance.
        ins: List of tensors to be fed to the train function
        batch_size: Integer batch size or None if unknown.
        epochs: Number of times to iterate over the data
        verbose: Verbosity mode, 0, 1 or 2
        callbacks: List of callbacks to be called during training
        val_ins: List of tensors to be fed to the test function
        shuffle: Whether to shuffle the data at the beginning of each epoch
        initial_epoch: Epoch at which to start training
            (useful for resuming a previous training run)
        steps_per_epoch: Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. Ignored with the default value of `None`.
        validation_steps: Number of steps to run validation for
            (only if doing validation from data tensors).
            Ignored with the default value of `None`.

    # Returns
        `History` object.
    """
    do_validation = False
    if val_ins:
        do_validation = True
        if (verbose and ins and
           hasattr(ins[0], 'shape') and hasattr(val_ins[0], 'shape')):
            print('Train on %d samples, validate on %d samples' %
                  (ins[0].shape[0], val_ins[0].shape[0]))
    if validation_steps:
        do_validation = True
        if steps_per_epoch is None:
            raise ValueError('Can only use `validation_steps` '
                             'when doing step-wise '
                             'training, i.e. `steps_per_epoch` '
                             'must be set.')
    elif do_validation:
        if steps_per_epoch:
            raise ValueError('Must specify `validation_steps` '
                             'to perform validation '
                             'when doing step-wise training.')

    num_train_samples = check_num_samples(ins,
                                          batch_size=batch_size,
                                          steps=steps_per_epoch,
                                          steps_name='steps_per_epoch')
    if num_train_samples is not None:
        index_array = np.arange(num_train_samples)

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
        if steps_per_epoch is not None:
            count_mode = 'steps'
        else:
            count_mode = 'samples'
        _callbacks.append(
            cbks.ProgbarLogger(
                count_mode,
                stateful_metrics=model.stateful_metric_names))
    _callbacks += (callbacks or []) + [model.history]
    callbacks = cbks.CallbackList(_callbacks)

    # it's possible to callback a different model than itself
    # (used by Sequential models)
    if hasattr(model, 'callback_model') and model.callback_model:
        callback_model = model.callback_model
    else:
        callback_model = model

    callbacks.set_model(callback_model)
    callbacks.set_params({
        'batch_size': batch_size,
        'epochs': epochs,
        'steps': steps_per_epoch,
        'samples': num_train_samples,
        'val_steps': validation_steps,
        'verbose': verbose,
        'do_validation': do_validation,
        'metrics': callback_metrics or [],
    })
    callbacks.on_train_begin()
    callback_model.stop_training = False
    for cbk in callbacks:
        cbk.validation_data = val_ins

    # To prevent a slowdown,
    # we find beforehand the arrays that need conversion.
    feed = (model._feed_inputs +
            model._feed_targets +
            model._feed_sample_weights)
    indices_for_conversion_to_dense = []
    for i in range(len(feed)):
        if issparse(ins[i]) and not K.is_sparse(feed[i]):
            indices_for_conversion_to_dense.append(i)

    for epoch in range(initial_epoch, epochs):
        # Reset stateful metrics
        for m in model.stateful_metric_functions:
            m.reset_states()
        callbacks.on_epoch_begin(epoch)
        epoch_logs = {}
        if steps_per_epoch is not None:
            for step_index in range(steps_per_epoch):
                batch_logs = {}
                batch_logs['batch'] = step_index
                batch_logs['size'] = 1
                callbacks.on_fit_batch_begin(step_index, batch_logs)

                outs = model.train_function(ins)

                outs = to_list(outs)
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_fit_batch_end(step_index, batch_logs)
                if callback_model.stop_training:
                    break

            if do_validation:
                val_outs = evaluate_loop(model, val_ins,
                                         steps=validation_steps,
                                         verbose=0,
                                         callbacks=callbacks)
                val_outs = to_list(val_outs)
                # Same labels assumed.
                for l, o in zip(out_labels, val_outs):
                    epoch_logs['val_' + l] = o
        else:
            if shuffle == 'batch':
                index_array = batch_shuffle(index_array, batch_size)
            elif shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(num_train_samples, batch_size)
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    if isinstance(ins[-1], float):
                        # Do not slice the training phase flag.
                        ins_batch = slice_arrays(
                            ins[:-1], batch_ids) + [ins[-1]]
                    else:
                        ins_batch = slice_arrays(ins, batch_ids)
                except TypeError:
                    raise TypeError('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')
                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_fit_batch_begin(batch_index, batch_logs)
                for i in indices_for_conversion_to_dense:
                    ins_batch[i] = ins_batch[i].toarray()

                outs = model.train_function(ins_batch)
                outs = to_list(outs)
                for l, o in zip(out_labels, outs):
                    batch_logs[l] = o

                callbacks.on_fit_batch_end(batch_index, batch_logs)
                if callback_model.stop_training:
                    break

                if batch_index == len(batches) - 1:  # Last batch.
                    if do_validation:
                        val_outs = evaluate_loop(model, val_ins,
                                                 batch_size=batch_size,
                                                 verbose=0,
                                                 callbacks=callbacks)
                        val_outs = to_list(val_outs)
                        # Same labels assumed.
                        for l, o in zip(out_labels, val_outs):
                            epoch_logs['val_' + l] = o
        callbacks.on_epoch_end(epoch, epoch_logs)
        if callback_model.stop_training:
            break
    callbacks.on_train_end()
    return model.history


def predict_loop(model, ins, batch_size=32, verbose=0, callbacks=None, steps=None):
    """Abstract method to loop over some data in batches.

    # Arguments
        model: Keras model instance.
        ins: list of tensors to be fed to the predict function.
        batch_size: integer batch size.
        verbose: verbosity mode.
        steps: Total number of steps (batches of samples)
            before declaring `predict_loop` finished.
            Ignored with the default value of `None`.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during evaluation.
            See [callbacks](/callbacks).

    # Returns
        Array of predictions (if the model has a single output)
        or list of arrays of predictions
        (if the model has multiple outputs).
    """
    num_samples = check_num_samples(ins,
                                    batch_size=batch_size,
                                    steps=steps,
                                    steps_name='steps')

    if steps is not None:
        count_param = 'pred_steps'
    else:
        count_param = 'pred_samples'

    _callbacks = []
    if verbose == 1:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_param=count_param,
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
        'batch_size': batch_size,
        'pred_steps': steps,
        'pred_samples': num_samples,
        'verbose': verbose
    })
    # callbacks.on_train_begin()  # todo
    callback_model.stop_predicting = False
    for cbk in callbacks:
        cbk.validation_data = ins

    indices_for_conversion_to_dense = []
    for i in range(len(model._feed_inputs)):
        if issparse(ins[i]) and not K.is_sparse(model._feed_inputs[i]):
            indices_for_conversion_to_dense.append(i)

    if steps is not None:
        # Step-based predictions.
        # Since we do not know how many samples
        # we will see, we cannot pre-allocate
        # the returned Numpy arrays.
        # Instead, we store one array per batch seen
        # and concatenate them upon returning.
        unconcatenated_outs = []
        for step in range(steps):
            batch_logs = {}
            batch_logs['batch'] = step
            batch_logs['size'] = 1
            callbacks.on_predict_batch_begin(step, batch_logs)

            batch_outs = model.predict_function(ins)
            batch_outs = to_list(batch_outs)

            callbacks.on_predict_batch_end(step, batch_logs)

            if step == 0:
                for batch_out in batch_outs:
                    unconcatenated_outs.append([])
            for i, batch_out in enumerate(batch_outs):
                unconcatenated_outs[i].append(batch_out)
            if callback_model.stop_predicting:
                break
        concatenated_outs = [np.concatenate(out, axis=0) for out in unconcatenated_outs]
        return unpack_singleton(concatenated_outs)
    else:
        # Sample-based predictions.
        outs = []
        batches = make_batches(num_samples, batch_size)
        index_array = np.arange(num_samples)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            # todo: wrap in try? (see fit_loop)
            if ins and isinstance(ins[-1], float):
                # Do not slice the training phase flag.
                ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = slice_arrays(ins, batch_ids)
            batch_logs = {}
            batch_logs['batch'] = batch_index
            batch_logs['size'] = len(batch_ids)

            callbacks.on_predict_batch_begin(batch_index, batch_logs)
            for i in indices_for_conversion_to_dense:
                ins_batch[i] = ins_batch[i].toarray()

            batch_outs = model.predict_function(ins_batch)
            batch_outs = to_list(batch_outs)

            callbacks.on_predict_batch_end(batch_index, batch_logs)

            if batch_index == 0:
                # Pre-allocate the results arrays.
                for batch_out in batch_outs:
                    shape = (num_samples,) + batch_out.shape[1:]
                    outs.append(np.zeros(shape, dtype=batch_out.dtype))
            for i, batch_out in enumerate(batch_outs):
                outs[i][batch_start:batch_end] = batch_out
            if callback_model.stop_predicting:
                break
        return unpack_singleton(outs)


def evaluate_loop(model, ins, batch_size=None, verbose=0, steps=None, callbacks=None):
    """Abstract method to loop over some data in batches.

    # Arguments
        model: Keras model instance.
        ins: list of tensors to be fed to the test function.
        batch_size: integer batch size or `None`.
        verbose: verbosity mode.
        steps: Total number of steps (batches of samples)
            before declaring predictions finished.
            Ignored with the default value of `None`.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during evaluation.
            See [callbacks](/callbacks).

    # Returns
        Scalar loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.
    """

    if hasattr(model, 'metrics'):
        for m in model.stateful_metric_functions:
            m.reset_states()
        stateful_metric_indices = [
            i for i, name in enumerate(model.metrics_names)
            if str(name) in model.stateful_metric_names]
    else:
        stateful_metric_indices = []

    num_samples = check_num_samples(ins,
                                    batch_size=batch_size,
                                    steps=steps,
                                    steps_name='steps')

    if steps is not None:
        count_param = 'val_steps'
    else:
        count_param = 'val_samples'

    outs = []
    out_labels = ['val_' + n for n in model.metrics_names]

    _callbacks = []
    if verbose == 1:
        _callbacks.append(
            cbks.ProgbarLogger(
                count_param=count_param,
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
        'batch_size': batch_size,
        'val_steps': steps,
        'val_samples': num_samples,
        'val_metrics': out_labels,
        'verbose': verbose
    })
    # callbacks.on_train_begin()  # todo
    callback_model.stop_evaluating = False
    for cbk in callbacks:
        cbk.validation_data = ins

    # To prevent a slowdown,
    # we find beforehand the arrays that need conversion.
    feed = (model._feed_inputs +
            model._feed_targets +
            model._feed_sample_weights)
    indices_for_conversion_to_dense = []
    for i in range(len(feed)):
        if issparse(ins[i]) and not K.is_sparse(feed[i]):
            indices_for_conversion_to_dense.append(i)

    if steps is not None:
        for step in range(steps):
            batch_logs = {}
            batch_logs['batch'] = step
            batch_logs['size'] = 1

            callbacks.on_evaluate_batch_begin(step, batch_logs)

            batch_outs = model.test_function(ins)

            batch_outs = to_list(batch_outs)
            for l, o in zip(out_labels, batch_outs):
                batch_logs[l] = o

            callbacks.on_evaluate_batch_end(step, batch_logs)

            if step == 0:
                for _ in enumerate(batch_outs):
                    outs.append(0.)
            for i, batch_out in enumerate(batch_outs):
                if i in stateful_metric_indices:
                    outs[i] = float(batch_out)
                else:
                    outs[i] += batch_out
            if callback_model.stop_evaluating:
                break
        for i in range(len(outs)):
            if i not in stateful_metric_indices:
                outs[i] /= steps
    else:
        batches = make_batches(num_samples, batch_size)
        index_array = np.arange(num_samples)
        for batch_index, (batch_start, batch_end) in enumerate(batches):
            batch_ids = index_array[batch_start:batch_end]
            # todo: wrap in try? (see fit_loop)
            if isinstance(ins[-1], float):
                # Do not slice the training phase flag.
                ins_batch = slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
            else:
                ins_batch = slice_arrays(ins, batch_ids)
            batch_logs = {}
            batch_logs['batch'] = batch_index
            batch_logs['size'] = len(batch_ids)

            callbacks.on_evaluate_batch_begin(batch_index, batch_logs)
            for i in indices_for_conversion_to_dense:
                ins_batch[i] = ins_batch[i].toarray()

            batch_outs = model.test_function(ins_batch)

            batch_outs = to_list(batch_outs)
            for l, o in zip(out_labels, batch_outs):
                batch_logs[l] = o

            callbacks.on_evaluate_batch_end(batch_index, batch_logs)

            if batch_index == 0:
                for _ in enumerate(batch_outs):
                    outs.append(0.)
            for i, batch_out in enumerate(batch_outs):
                if i in stateful_metric_indices:
                    outs[i] = batch_out
                else:
                    outs[i] += batch_out * len(batch_ids)
            if callback_model.stop_evaluating:
                break
        for i in range(len(outs)):
            if i not in stateful_metric_indices:
                outs[i] /= num_samples
    return unpack_singleton(outs)
