import jax
import numpy as np
import tensorflow as tf  # for nest

from keras_core import backend
from keras_core import callbacks as callbacks_module
from keras_core import optimizers as optimizers_module
from keras_core.trainers import trainer as base_trainer
from keras_core.trainers.data_adapters import data_adapter_utils
from keras_core.trainers.epoch_iterator import EpochIterator


class JAXTrainer(base_trainer.Trainer):
    def compute_loss_and_updates(
        self, trainable_variables, non_trainable_variables, x, y, sample_weight
    ):
        """This method is stateless and is intended for use with jax.grad."""
        y_pred, non_trainable_variables = self.stateless_call(
            trainable_variables, non_trainable_variables, x
        )

        loss = self.compute_loss(x, y, y_pred, sample_weight)
        return loss, (y_pred, non_trainable_variables)

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
        epoch_iterator = EpochIterator(
            x=x,
            y=y,
            sample_weight=sample_weight,
            batch_size=batch_size,
            steps_per_epoch=steps_per_epoch,
            shuffle=shuffle,
            class_weight=class_weight,
            steps_per_execution=self.steps_per_execution,
        )

        compile_metrics_unbuilt = (
            self._compile_metrics is not None
            and not self._compile_metrics.built
        )
        if not self.built or compile_metrics_unbuilt:
            # Build the model on one batch of data.
            for _, data in epoch_iterator.enumerate_epoch(return_type="np"):
                data = data[0]
                (
                    x,
                    y,
                    sample_weight,
                ) = data_adapter_utils.unpack_x_y_sample_weight(data)
                # Build model
                with backend.StatelessScope():
                    y_pred = self(x)
                    if compile_metrics_unbuilt:
                        # Build metrics
                        self.compute_metrics(
                            x, y, y_pred, sample_weight=sample_weight
                        )
                break
        if not self.optimizer.built:
            # Build optimizer
            self.optimizer.build(self.trainable_variables)

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

        grad_fn = jax.value_and_grad(
            self.compute_loss_and_updates, has_aux=True
        )

        def _train_step(state, data):
            data = data[0]
            (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                metrics_variables,
            ) = state
            x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(
                data
            )
            (loss, (y_pred, non_trainable_variables)), grads = grad_fn(
                trainable_variables,
                non_trainable_variables,
                x,
                y,
                sample_weight,
            )

            (
                trainable_variables,
                optimizer_variables,
            ) = self.optimizer.stateless_apply(
                grads, trainable_variables, optimizer_variables
            )

            with backend.StatelessScope(
                state_mapping=[
                    (ref_v, v)
                    for ref_v, v in zip(
                        self.metrics_variables, metrics_variables
                    )
                ]
            ) as scope:
                logs = self.compute_metrics(x, y, y_pred, sample_weight)
                self._loss_tracker.update_state(loss)

            new_metrics_variables = []
            for ref_v in self.metrics_variables:
                new_v = scope.get_current_value(ref_v)
                if new_v is None:
                    new_v = ref_v.value
                new_metrics_variables.append(new_v)
            metrics_variables = new_metrics_variables

            state = (
                trainable_variables,
                non_trainable_variables,
                optimizer_variables,
                metrics_variables,
            )
            return logs, state

        def _train_multi_step(state, data):
            for single_step_data in data:
                logs, state = _train_step(state, [single_step_data])
            return logs, state

        if self.steps_per_execution > 1:
            _train_function = _train_multi_step
        else:
            _train_function = _train_step

        if not self.run_eagerly and self.jit_compile:

            @jax.jit
            def train_step(state, data):
                return _train_function(state, data)

        else:
            train_step = _train_function

        self.stop_training = False
        callbacks.on_train_begin()

        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)

            trainable_variables = self.trainable_variables
            non_trainable_variables = self.non_trainable_variables
            optimizer_variables = self.optimizer.variables
            metrics_variables = self.metrics_variables

            for step, data in epoch_iterator.enumerate_epoch(return_type="np"):
                # Callbacks
                callbacks.on_train_batch_begin(step)

                # Train step
                state = (
                    trainable_variables,
                    non_trainable_variables,
                    optimizer_variables,
                    metrics_variables,
                )
                logs, state = train_step(state, data)
                (
                    trainable_variables,
                    non_trainable_variables,
                    optimizer_variables,
                    metrics_variables,
                ) = state

                # Callbacks
                callbacks.on_train_batch_end(step, logs)
                if self.stop_training:
                    break

            # Update variable values
            # NOTE: doing this after each step would be a big performance
            # bottleneck.
            for ref_v, v in zip(self.trainable_variables, trainable_variables):
                ref_v.assign(v)
            for ref_v, v in zip(
                self.non_trainable_variables, non_trainable_variables
            ):
                ref_v.assign(v)
            for ref_v, v in zip(self.optimizer.variables, optimizer_variables):
                ref_v.assign(v)
            for ref_v, v in zip(self.metrics_variables, metrics_variables):
                ref_v.assign(v)

            # Override with model metrics instead of last step logs
            epoch_logs = self._pythonify_logs(self.get_metrics_result())

            # Run validation.
            if validation_data and self._should_eval(epoch, validation_freq):
                # Create EpochIterator for evaluation and cache it.
                if getattr(self, "_eval_epoch_iterator", None) is None:
                    self._eval_epoch_iterator = EpochIterator(
                        x=val_x,
                        y=val_y,
                        sample_weight=val_sample_weight,
                        batch_size=validation_batch_size or batch_size,
                        steps_per_execution=self.steps_per_execution,
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
            epoch_iterator = EpochIterator(
                x=x,
                y=y,
                sample_weight=sample_weight,
                batch_size=batch_size,
                steps_per_epoch=steps,
                shuffle=False,
                steps_per_execution=self.steps_per_execution,
            )

        if not self.built:
            # Build the model on one batch of data.
            for _, data in epoch_iterator.enumerate_epoch(return_type="np"):
                data = data[0]
                (
                    x,
                    y,
                    sample_weight,
                ) = data_adapter_utils.unpack_x_y_sample_weight(data)
                # Build model
                y_pred = self(x)
                # Build metrics
                self.compute_metrics(x, y, y_pred, sample_weight)
                self.reset_metrics()
                break

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

        def _test_step(state, data):
            data = data[0]
            (
                trainable_variables,
                non_trainable_variables,
                metrics_variables,
            ) = state
            x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(
                data
            )
            loss, (
                y_pred,
                non_trainable_variables,
            ) = self.compute_loss_and_updates(
                trainable_variables,
                non_trainable_variables,
                x,
                y,
                sample_weight,
            )

            with backend.StatelessScope(
                state_mapping=[
                    (ref_v, v)
                    for ref_v, v in zip(
                        self.metrics_variables, metrics_variables
                    )
                ]
            ) as scope:
                logs = self.compute_metrics(x, y, y_pred, sample_weight)
                self._loss_tracker.update_state(loss)

            new_metrics_variables = []
            for ref_v in self.metrics_variables:
                new_v = scope.get_current_value(ref_v)
                if new_v is None:
                    new_v = ref_v.value
                new_metrics_variables.append(new_v)
            metrics_variables = new_metrics_variables

            state = (
                trainable_variables,
                non_trainable_variables,
                metrics_variables,
            )
            return logs, state

        def _test_multi_step(state, data):
            for single_step_data in data:
                logs, state = _test_step(state, [single_step_data])
            return logs, state

        if self.steps_per_execution > 1:
            _test_function = _test_multi_step
        else:
            _test_function = _test_step

        if not self.run_eagerly and self.jit_compile:

            @jax.jit
            def test_step(state, data):
                return _test_function(state, data)

        else:
            test_step = _test_function

        callbacks.on_test_begin()
        logs = None
        self.reset_metrics()

        trainable_variables = self.trainable_variables
        non_trainable_variables = self.non_trainable_variables
        metrics_variables = self.metrics_variables

        for step, data in epoch_iterator.enumerate_epoch(return_type="np"):
            callbacks.on_test_batch_begin(step)

            state = (
                trainable_variables,
                non_trainable_variables,
                metrics_variables,
            )
            logs, state = test_step(state, data)
            # Note that trainable variables are not returned since they're
            # immutable here.
            _, non_trainable_variables, metrics_variables = state

            callbacks.on_test_batch_end(step, logs)

        for ref_v, v in zip(
            self.non_trainable_variables, non_trainable_variables
        ):
            # I wouldn't recommend modifying non-trainable model state
            # during evaluate(), but it's allowed.
            ref_v.assign(v)
        for ref_v, v in zip(self.metrics_variables, metrics_variables):
            ref_v.assign(v)
        logs = self._pythonify_logs(self.get_metrics_result())
        callbacks.on_test_end(logs)

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        # Create an iterator that yields batches of input data.
        epoch_iterator = EpochIterator(
            x=x,
            batch_size=batch_size,
            steps_per_epoch=steps,
            shuffle=False,
            steps_per_execution=self.steps_per_execution,
        )

        if not self.built:
            # Build the model on one batch of data.
            for _, data in epoch_iterator.enumerate_epoch(return_type="np"):
                # Build model
                self(data[0])
                break

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

        def _predict_step(trainable_variables, non_trainable_variables, data):
            outputs, _ = self.stateless_call(
                trainable_variables, non_trainable_variables, data[0]
            )
            return outputs

        def _predict_multi_step(
            trainable_variables, non_trainable_variables, data
        ):
            outputs = _predict_step(
                trainable_variables, non_trainable_variables, data[:1]
            )
            for single_step_data in data[1:]:
                step_outputs = _predict_step(
                    trainable_variables,
                    non_trainable_variables,
                    [single_step_data],
                )
                outputs = tf.nest.map_structure(
                    lambda t1, t2: jax.numpy.concatenate([t1, t2]),
                    outputs,
                    step_outputs,
                )
            return outputs

        if self.steps_per_execution > 1:
            _predict_function = _predict_multi_step
        else:
            _predict_function = _predict_step

        if not self.run_eagerly and self.jit_compile:

            @jax.jit
            def predict_step(
                trainable_variables, non_trainable_variables, data
            ):
                return _predict_function(
                    trainable_variables, non_trainable_variables, data
                )

        else:
            predict_step = _predict_function

        callbacks.on_predict_begin()

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tf.nest.map_structure(
                    lambda batch_output: [batch_output],
                    batch_outputs,
                )
            else:
                tf.__internal__.nest.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs,
                    batch_outputs,
                )
            return outputs

        trainable_variables = self.trainable_variables
        non_trainable_variables = self.non_trainable_variables
        outputs = None
        for step, x in epoch_iterator.enumerate_epoch(return_type="np"):
            callbacks.on_predict_batch_begin(step)
            batch_outputs = predict_step(
                trainable_variables, non_trainable_variables, x
            )
            outputs = append_to_outputs(batch_outputs, outputs)
            callbacks.on_predict_batch_end(step, {"outputs": batch_outputs})
        callbacks.on_predict_end()
        return tf.__internal__.nest.map_structure_up_to(
            batch_outputs, np.concatenate, outputs
        )
