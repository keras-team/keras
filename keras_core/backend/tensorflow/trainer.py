import contextlib
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context as tf_context

from keras_core import callbacks as callbacks_module
from keras_core import optimizers as optimizers_module
from keras_core.trainers import trainer as base_trainer
from keras_core.trainers.data_adapters import data_adapter_utils
from keras_core.trainers.epoch_iterator import EpochIterator


class TensorFlowTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None

        # Model must be created under scope of DistStrat it will be trained
        # with.
        if tf.distribute.has_strategy():
            self._distribute_strategy = tf.distribute.get_strategy()
        else:
            self._distribute_strategy = None

        self._distribute_reduction_method = None

    @property
    def distribute_strategy(self):
        return self._distribute_strategy or tf.distribute.get_strategy()

    @property
    def distribute_reduction_method(self):
        return self._distribute_reduction_method or "auto"

    @distribute_reduction_method.setter
    def distribute_reduction_method(self, value):
        self._distribute_reduction_method = value

    def train_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)

        # Forward pass
        with tf.GradientTape() as tape:
            if self._call_has_training_arg():
                y_pred = self(x, training=True)
            else:
                y_pred = self(x)
            loss = self.compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
            )
        self._loss_tracker.update_state(loss)

        # Compute gradients
        # TODO: move value conversion to TF
        if self.trainable_weights:
            trainable_weights = [v.value for v in self.trainable_weights]
            gradients = tape.gradient(loss, trainable_weights)

            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        else:
            warnings.warn("The model does not have any trainable weights.")

        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg():
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        loss = self.compute_loss(
            x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
        )
        self._loss_tracker.update_state(loss)
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def predict_step(self, data):
        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        if self._call_has_training_arg():
            y_pred = self(x, training=False)
        else:
            y_pred = self(x)
        return y_pred

    def make_train_function(self, force=False):
        # TODO: support tf.distribute and steps_per_execution.
        if self.train_function is not None and not force:
            return self.train_function

        def one_step_on_data(data):
            """Runs a single training step on a batch of data."""
            return self.train_step(data)

        if not self.run_eagerly and self.jit_compile:
            one_step_on_data = tf.function(
                one_step_on_data, jit_compile=True, reduce_retracing=True
            )

        def one_step_on_iterator(iterator):
            """Runs a single training step given a Dataset iterator."""
            data = next(iterator)
            outputs = self.distribute_strategy.run(
                one_step_on_data, args=(data,)
            )
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        if not self.run_eagerly:
            train_function = tf.function(
                one_step_on_iterator, reduce_retracing=True
            )
        else:
            train_function = one_step_on_iterator
        self.train_function = train_function

    def make_test_function(self, force=False):
        # TODO: support tf.distribute and steps_per_execution.
        if self.test_function is not None and not force:
            return self.test_function

        def one_step_on_data(data):
            """Runs a single test step on a batch of data."""
            return self.test_step(data)

        if not self.run_eagerly and self.jit_compile:
            one_step_on_data = tf.function(
                one_step_on_data, jit_compile=True, reduce_retracing=True
            )

        def one_step_on_iterator(iterator):
            """Runs a single test step given a Dataset iterator."""
            data = next(iterator)
            outputs = self.distribute_strategy.run(
                one_step_on_data, args=(data,)
            )
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        if not self.run_eagerly:
            test_function = tf.function(
                one_step_on_iterator, reduce_retracing=True
            )
        else:
            test_function = one_step_on_iterator
        self.test_function = test_function

    def make_predict_function(self, force=False):
        # TODO: support tf.distribute and steps_per_execution.
        if self.predict_function is not None and not force:
            return self.predict_function

        def one_step_on_data(data):
            """Runs a predict test step on a batch of data."""
            return self.predict_step(data)

        if not self.run_eagerly and self.jit_compile:
            one_step_on_data = tf.function(
                one_step_on_data, jit_compile=True, reduce_retracing=True
            )

        def one_step_on_iterator(iterator):
            """Runs a single predict step given a Dataset iterator."""
            data = next(iterator)
            outputs = self.distribute_strategy.run(
                one_step_on_data, args=(data,)
            )
            outputs = reduce_per_replica(
                outputs,
                self.distribute_strategy,
                reduction=self.distribute_reduction_method,
            )
            return outputs

        if not self.run_eagerly:
            predict_function = tf.function(
                one_step_on_iterator, reduce_retracing=True
            )
        else:
            predict_function = one_step_on_iterator
        self.predict_function = predict_function

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
        if not self.compiled:
            raise ValueError(
                "You must call `compile()` before calling `fit()`."
            )
        # TODO: respect compiled trainable state
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for TF/numpy/jax arrays.
            (
                x,
                y,
                sample_weight,
            ), validation_data = data_adapter_utils.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(validation_data)

        # Create an iterator that yields batches for one epoch.
        epoch_iterator = TFEpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            distribute_strategy=self.distribute_strategy,
        )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=epochs,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.stop_training = False
        self.make_train_function()
        callbacks.on_train_begin()
        training_logs = None
        logs = None
        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            with epoch_iterator.catch_stop_iteration():
                for step, iterator in epoch_iterator.enumerate_epoch():
                    callbacks.on_train_batch_begin(step)
                    logs = self.train_function(iterator)
                    callbacks.on_train_batch_end(step, logs)
                    if self.stop_training:
                        break

            # Override with model metrics instead of last step logs
            epoch_logs = self._pythonify_logs(self.get_metrics_result())

            # Run validation.
            if validation_data and self._should_eval(epoch, validation_freq):
                # Create EpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = TFEpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        distribute_strategy=self.distribute_strategy,
                    )
                val_logs = self.evaluate(
                    x=val_x,
                    y=val_y,
                    sample_weight=val_sample_weight,
                    batch_size=validation_batch_size or batch_size,
                    steps=validation_steps,
                    callbacks=callbacks,
                    return_dict=True,
                    _use_cached_eval_dataset=True,
                )
                val_logs = {
                    "val_" + name: val for name, val in val_logs.items()
                }
                epoch_logs.update(self._pythonify_logs(val_logs))

            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break

        if (
            isinstance(self.optimizer, optimizers_module.Optimizer)
            and epochs > 0
        ):
            self.optimizer.finalize_variable_values(self.trainable_weights)

        # If _eval_epoch_iterator exists, delete it after all epochs are done.
        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        return self.history

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        # TODO: respect compiled trainable state
        use_cached_eval_dataset = kwargs.pop("_use_cached_eval_dataset", False)
        if kwargs:
            raise ValueError(f"Arguments not recognized: {kwargs}")

        if use_cached_eval_dataset:
            epoch_iterator = self._eval_epoch_iterator
        else:
            # Create an iterator that yields batches of input/target data.
            epoch_iterator = TFEpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                distribute_strategy=self.distribute_strategy,
            )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.make_test_function()
        callbacks.on_test_begin()
        logs = None
        self.reset_metrics()
        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator.enumerate_epoch():
                callbacks.on_test_batch_begin(step)
                logs = self.test_function(iterator)
                callbacks.on_test_batch_end(step, logs)
        logs = self._pythonify_logs(self.get_metrics_result())
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        # Create an iterator that yields batches of input data.
        epoch_iterator = TFEpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            distribute_strategy=self.distribute_strategy,
        )

        # Container that configures and calls callbacks.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                verbose=verbose,
                epochs=1,
                steps=epoch_iterator.num_batches,
                model=self,
            )

        self.make_predict_function()
        callbacks.on_predict_begin()
        outputs = None
        with epoch_iterator.catch_stop_iteration():
            for step, iterator in epoch_iterator.enumerate_epoch():
                callbacks.on_predict_batch_begin(step)
                batch_outputs = self.predict_function(iterator)
                if outputs is None:
                    outputs = tf.nest.map_structure(
                        lambda batch_output: [batch_output],
                        batch_outputs,
                    )
                else:
                    tf.__internal__.nest.map_structure_up_to(
                        batch_outputs,
                        lambda output, batch_output: output.append(
                            batch_output
                        ),
                        outputs,
                        batch_outputs,
                    )
                callbacks.on_predict_batch_end(step, {"outputs": batch_outputs})
        callbacks.on_predict_end()
        return tf.__internal__.nest.map_structure_up_to(
            batch_outputs, np.concatenate, outputs
        )


class TFEpochIterator(EpochIterator):
    def __init__(self, distribute_strategy=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._distribute_strategy = distribute_strategy
        self._steps_seen = 0

    def enumerate_epoch(self):
        if self.steps_per_epoch:
            if not self._current_iterator:
                self._current_iterator = iter(
                    self._distribute_strategy.experimental_distribute_dataset(
                        self.data_adapter.get_tf_dataset()))
            for step in range(self.steps_per_epoch):
                yield step, self._current_iterator
        else:
            iterator = iter(
                self._distribute_strategy.experimental_distribute_dataset(
                    self.data_adapter.get_tf_dataset()))
            if self.num_batches:
                for step in range(self.num_batches):
                    yield step, iterator
            else:
                step = -1
                while True:
                    step += 1
                    self._steps_seen = step + 1
                    yield step, iterator
        self.data_adapter.on_epoch_end()

    def tf_sync(self):
        tf_context.async_wait()

    @contextlib.contextmanager
    def catch_stop_iteration(self):
        """Catches errors when an iterator runs out of data."""
        try:
            yield
            self.tf_sync()
        except (StopIteration, tf.errors.OutOfRangeError):
            if self._num_batches is None:
                self._num_batches = self._steps_seen
            warnings.warn(
                "Your input ran out of data; interrupting training. "
                "Make sure that your dataset or generator can generate "
                "at least `steps_per_epoch * epochs` batches. "
                "You may need to use the `.repeat()` "
                "function when building your dataset.",
                stacklevel=2,
            )
            self._current_iterator = None
            self.data_adapter.on_epoch_end()


def reduce_per_replica(values, strategy, reduction):
    """Attempt to reduce the structure `values` to single values.

    Given `values` (a `tf.Tensor` or a `PerReplica` structure),
    which represents the values across all the replicas, `reduce_per_replica`
    attempts to "reduce" those values and returns the corresponding structure
    that represents only single values.

    Currently, `reduce_per_replica` is only used for reducing the metric results
    from `tf.distribute.Strategy.run()`. Depending on the underlying
    `Strategy` implementation, `values` may be a `PerReplica` object,
     which can be thought of as a collection of values across the replicas,
    or a `tf.Tensor`, if the strategy has already conducted the reduction
    for the downstream library.

    There are five possible outcomes of reduction:

    1) if the `values` is a structure of simple `tf.Tensor`s, meaning that
       reduction is not actually needed, `reduce_per_replica` returns the
       structure as-is.
    2) else, if `reduction="auto"`, then the best reduction strategy is
       chosen based on the current environment. This should only be used
       for training cases (`fit()`).
    3) else, if `reduction="first"`, then `reduce_per_replica`
       returns the values of the first replica. This is used in the case of
       training and evaluation, where `values` is expected to hold the same
       value across the replicas as a result of `Strategy`'s synchronization
       across the replicas.
       `reduce_per_replica` does not synchronize the values.
    4) else, if `reduction="sum"`, then `reduce_per_replica` returns the sum
       of values for all replicas. This may be used in the custom training loop
       case, where each replica contain different values which are not
       synchronized.
    5) else, if `reduction="concat"`, then `reduce_per_replica`
       returns the concatenation of the values across the replicas, along the
       axis of dimension 0. This is used in the inference case (`predict()`).

    Args:
      values: Structure of `PerReplica` objects or `tf.Tensor`s. `tf.Tensor`s
        are returned as-is.
      strategy: `tf.distribute.Strategy` object.
      reduction: One of `"auto"`, `"first"`, `"concat"`, or `"sum"`.
        `"auto"` will select `"first"` when used under a TPUStrategy, or
        `"sum"` otherwise.

    Returns:
      Structure of `Tensor`s, representing the result of reduction.

    Raises:
      ValueError: if the reduction method is not supported.
    """

    if reduction == "auto":
        reduction = "sum"  # Ignore TPU strategy which should default to "first"

    def _reduce(v):
        """Reduce a single `PerReplica` object."""
        if _collective_all_reduce_multi_worker(strategy):
            if reduction == "concat":
                return _multi_worker_concat(v, strategy)
            elif reduction == "sum":
                return strategy.reduce("SUM", v, axis=None)

        if not _is_per_replica_instance(v):
            return v
        elif reduction == "first":
            return strategy.experimental_local_results(v)[0]
        elif reduction == "concat":
            if _is_tpu_multi_host(strategy):
                return _tpu_multi_host_concat(v, strategy)
            else:
                return concat(strategy.experimental_local_results(v))
        elif reduction == "sum":
            return tf.reduce_sum(strategy.experimental_local_results(v))
        else:
            raise ValueError(
                '`reduction` must be "first", "concat", "sum", or "auto". '
                f"Received: reduction={reduction}."
            )

    return tf.nest.map_structure(_reduce, values)


def _multi_worker_concat(v, strategy):
    """Order PerReplica objects for CollectiveAllReduceStrategy and concat."""
    replicas = strategy.gather(v, axis=0)
    # v might not have the same shape on different replicas
    if _is_per_replica_instance(v):
        shapes = tf.concat(
            [
                tf.expand_dims(tf.shape(single_value)[0], axis=0)
                for single_value in v.values
            ],
            axis=0,
        )
        all_shapes = strategy.gather(shapes, axis=0)
    else:
        # v is a tensor. This may happen when, say, we have 2x1 multi-worker.
        all_shapes = strategy.gather(
            tf.expand_dims(tf.shape(v)[0], axis=0), axis=0
        )

    replicas = tf.split(
        replicas,
        num_or_size_splits=all_shapes,
        num=strategy.num_replicas_in_sync,
    )
    ordered_replicas = []
    num_replicas_per_worker = len(strategy.extended.worker_devices)
    for replica_id in range(num_replicas_per_worker):
        ordered_replicas += replicas[replica_id::num_replicas_per_worker]
    return concat(ordered_replicas)


def concat(tensors, axis=0):
    """Concats `tensor`s along `axis`."""
    if isinstance(tensors[0], tf.SparseTensor):
        return tf.sparse.concat(axis=axis, sp_inputs=tensors)
    elif _is_scalar(tensors[0]):
        return tf.stack(tensors, axis=axis)
    else:
        return tf.concat(tensors, axis=axis)


def _tpu_multi_host_concat(v, strategy):
    """Correctly order TPU PerReplica objects."""
    replicas = strategy.experimental_local_results(v)
    # When distributed datasets are created from Tensors / NumPy,
    # TPUStrategy.experimental_distribute_dataset shards data in
    # (Replica, Host) order, and TPUStrategy.experimental_local_results returns
    # it in (Host, Replica) order.
    num_replicas_per_host = strategy.extended.num_replicas_per_host
    ordered_replicas = []
    for replica_id in range(num_replicas_per_host):
        ordered_replicas += replicas[replica_id::num_replicas_per_host]
    return concat(ordered_replicas)


def _collective_all_reduce_multi_worker(strategy):
    return (
        isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy)
    ) and strategy.extended._in_multi_worker_mode()


def _is_per_replica_instance(obj):
    return isinstance(obj, tf.distribute.DistributedValues) and isinstance(
        obj, tf.__internal__.CompositeTensor
    )


def _is_scalar(x):
    return isinstance(x, (tf.Tensor, tf.Variable)) and x.shape.rank == 0


def _is_tpu_multi_host(strategy):
    return _is_tpu_strategy(strategy) and strategy.extended.num_hosts > 1


def _is_tpu_strategy(strategy):
    return _is_tpu_strategy_class(strategy.__class__)


def _is_tpu_strategy_class(clz):
    def is_tpu_strat(k):
        return k.__name__.startswith("TPUStrategy")

    if is_tpu_strat(clz):
        return True
    return any(map(_is_tpu_strategy_class, clz.__bases__))
