import copy

import numpy as np
import tensorflow as tf

from keras_core import callbacks as callbacks_module
from keras_core import optimizers as optimizers_module
from keras_core.trainers import data_adapter
from keras_core.trainers import trainer


class TensorFlowTrainer(trainer.Trainer):
    def train_step(self, data):
        # TODO: should be
        # x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        x, y = data
        sample_weight = None

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compute_loss(
                x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
            )

        # Compute gradients
        trainable_weights = self.trainable_weights
        gradients = tape.gradient(loss, trainable_weights)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        return self.compute_metrics(x, y, y_pred, sample_weight=sample_weight)

    def test_step(self, data):
        raise NotImplementedError

    def predict_step(self, data):
        raise NotImplementedError

    def make_train_function(self):
        raise NotImplementedError

    def make_test_function(self):
        raise NotImplementedError

    def make_predict_function(self):
        raise NotImplementedError

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
        # TODO: respect compiled trainable state
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for `Tensor` and `NumPy` input.
            (
                x,
                y,
                sample_weight,
            ), validation_data = data_adapter.train_validation_split(
                (x, y, sample_weight), validation_split=validation_split
            )

        if validation_data:
            (
                val_x,
                val_y,
                val_sample_weight,
            ) = data_adapter.unpack_x_y_sample_weight(validation_data)

        # Creates a `tf.data.Dataset` and handles batch and epoch iteration.
        data_handler = data_adapter.get_data_handler(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            initial_epoch=initial_epoch,
            epochs=epochs,
            shuffle=shuffle,
            class_weight=class_weight,
            model=self,
            steps_per_execution=self._steps_per_execution,
        )

        # Container that configures and calls `tf.keras.Callback`s.
        if not isinstance(callbacks, callbacks_module.CallbackList):
            callbacks = callbacks_module.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=verbose != 0,
                model=self,
                verbose=verbose,
                epochs=epochs,
                steps=data_handler.inferred_steps,
            )

        self.stop_training = False
        self.train_function = self.make_train_function()
        callbacks.on_train_begin()
        training_logs = None
        logs = None
        for epoch, iterator in data_handler.enumerate_epochs():
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)
            with data_handler.catch_stop_iteration():
                for step in data_handler.steps():
                    callbacks.on_train_batch_begin(step)
                    logs = self.train_function(iterator)
                    end_step = step + data_handler.step_increment
                    callbacks.on_train_batch_end(end_step, logs)
                    if self.stop_training:
                        break

            logs = sync_to_numpy_or_python_type(logs)
            if logs is None:
                raise ValueError(
                    "Unexpected result of `train_function` "
                    "(Empty logs). This could be due to issues in input "
                    "pipeline that resulted in an empty dataset. "
                    "Otherwise, please use "
                    "`Model.compile(..., run_eagerly=True)`, or "
                    "`tf.config.run_functions_eagerly(True)` for more "
                    "information of where went wrong."
                )
            # Override with model metrics instead of last step logs
            logs = self._validate_and_get_metrics_result(logs)
            epoch_logs = copy.copy(logs)

            # Run validation.
            if validation_data and self._should_eval(epoch, validation_freq):
                if self._pss_evaluation_shards:
                    self._disallow_exact_eval_with_add_metrics()
                # Create data_handler for evaluation and cache it.
                if getattr(self, "_eval_data_handler", None) is None:
                    self._eval_data_handler = data_adapter.get_data_handler(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps_per_epoch=validation_steps,
                        initial_epoch=0,
                        epochs=1,
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
                epoch_logs.update(val_logs)

            callbacks.on_epoch_end(epoch, epoch_logs)
            training_logs = epoch_logs
            if self.stop_training:
                break

        if (
            isinstance(self.optimizer, optimizers_module.Optimizer)
            and epochs > 0
        ):
            self.optimizer.finalize_variable_values(self.trainable_variables)

        # If eval data_handler exists, delete it after all epochs are done.
        if getattr(self, "_eval_data_handler", None) is not None:
            del self._eval_data_handler
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
        raise NotImplementedError

    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        raise NotImplementedError


def sync_to_numpy_or_python_type(tensors):
    """Syncs and converts a structure of `Tensor`s to `NumPy` arrays or Python
    scalar types.

    For each tensor, it calls `tensor.numpy()`. If the result is a scalar value,
    it converts it to a Python type, such as a float or int, by calling
    `result.item()`.

    Numpy scalars are converted, as Python types are often more convenient to
    deal with. This is especially useful for bfloat16 Numpy scalars, which don't
    support as many operations as other Numpy values.

    Async strategies (such as `TPUStrategy` and `ParameterServerStrategy`) are
    forced to sync during this process.

    Args:
        tensors: A structure of tensors.

    Returns:
        `tensors`, but scalar tensors are converted to Python types and non-scalar
        tensors are converted to Numpy arrays.
    """
    if isinstance(tensors, tf.distribute.experimental.coordinator.RemoteValue):
        tensors = tensors.fetch()
    if isinstance(tensors, list) and isinstance(
        tensors[0], tf.distribute.experimental.coordinator.RemoteValue
    ):
        tensors = tf.nest.map_structure(lambda t: t.fetch(), tensors)

    def _to_single_numpy_or_python_type(t):
        # Don't turn ragged or sparse tensors to NumPy.
        if isinstance(t, tf.Tensor):
            t = t.numpy()
        # Strings, ragged and sparse tensors don't have .item(). Return them
        # as-is.
        if not isinstance(t, (np.ndarray, np.generic)):
            return t
        return t.item() if np.ndim(t) == 0 else t

    return tf.nest.map_structure(_to_single_numpy_or_python_type, tensors)
