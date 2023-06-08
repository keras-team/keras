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
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        x,
        y,
        sample_weight,
        training=False,
    ):
        """This method is stateless and is intended for use with jax.grad."""
        kwargs = {}
        if self._call_has_training_arg():
            kwargs["training"] = training
        y_pred, non_trainable_variables = self.stateless_call(
            trainable_variables, non_trainable_variables, x, **kwargs
        )

        loss = self.compute_loss(x, y, y_pred, sample_weight)
        return loss, (y_pred, non_trainable_variables)

    def _eager_build(self, data_batch):
        compile_metrics_unbuilt = (
            self._compile_metrics is not None
            and not self._compile_metrics.built
        )
        if not self.built or compile_metrics_unbuilt:
            # Build the model on one batch of data.
            (
                x,
                y,
                sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(data_batch)
            # Build model
            with backend.StatelessScope():
                y_pred = self(x)
                if compile_metrics_unbuilt:
                    # Build metrics
                    self.compute_metrics(
                        x, y, y_pred, sample_weight=sample_weight
                    )
        if self.optimizer is not None and not self.optimizer.built:
            # Build optimizer
            self.optimizer.build(self.trainable_variables)

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        grad_fn = jax.value_and_grad(
            self.compute_loss_and_updates, has_aux=True
        )

        def one_train_step(state, data):
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
                training=True,
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
                self._loss_tracker.update_state(loss)
                logs = self.compute_metrics(x, y, y_pred, sample_weight)

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

        def multi_train_steps(state, data):
            for single_step_data in data:
                logs, state = one_train_step(state, [single_step_data])
            return logs, state

        if self.steps_per_execution > 1:
            train_step = multi_train_steps
        else:
            train_step = one_train_step

        if not self.run_eagerly and self.jit_compile:

            @jax.jit
            def compiled_train_step(state, data):
                return train_step(state, data)

            self.train_function = compiled_train_step

        else:
            self.train_function = train_step

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        def one_test_step(state, data):
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
                training=False,
            )

            with backend.StatelessScope(
                state_mapping=[
                    (ref_v, v)
                    for ref_v, v in zip(
                        self.metrics_variables, metrics_variables
                    )
                ]
            ) as scope:
                self._loss_tracker.update_state(loss)
                logs = self.compute_metrics(x, y, y_pred, sample_weight)

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

        def multi_test_steps(state, data):
            for single_step_data in data:
                logs, state = one_test_step(state, [single_step_data])
            return logs, state

        if self.steps_per_execution > 1:
            test_step = multi_test_steps
        else:
            test_step = one_test_step

        if not self.run_eagerly and self.jit_compile:

            @jax.jit
            def compiled_test_step(state, data):
                return test_step(state, data)

            self.test_function = compiled_test_step

        else:
            self.test_function = test_step

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        def one_predict_step(
            trainable_variables, non_trainable_variables, data
        ):
            kwargs = {}
            if self._call_has_training_arg():
                kwargs["training"] = False
            outputs, _ = self.stateless_call(
                trainable_variables, non_trainable_variables, data[0], **kwargs
            )
            return outputs

        def multi_predict_steps(
            trainable_variables, non_trainable_variables, data
        ):
            outputs = one_predict_step(
                trainable_variables, non_trainable_variables, data[:1]
            )
            for single_step_data in data[1:]:
                step_outputs = one_predict_step(
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
            predict_step = multi_predict_steps
        else:
            predict_step = one_predict_step

        if not self.run_eagerly and self.jit_compile:

            @jax.jit
            def compiled_predict_step(
                trainable_variables, non_trainable_variables, data
            ):
                return predict_step(
                    trainable_variables, non_trainable_variables, data
                )

            self.predict_function = compiled_predict_step

        else:
            self.predict_function = predict_step

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
        self._assert_compile_called("fit")
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

        needs_building = (
            not self.built
            or not self.optimizer.built
            or (
                self._compile_metrics is not None
                and not self._compile_metrics.built
            )
        )
        if needs_building:
            # Build the model on one batch of data.
            for _, data in epoch_iterator.enumerate_epoch(return_type="np"):
                data_batch = data[0]
                self._eager_build(data_batch)
                break

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

        self.make_train_function()
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
                logs, state = self.train_function(state, data)
                (
                    trainable_variables,
                    non_trainable_variables,
                    optimizer_variables,
                    metrics_variables,
                ) = state

                # Setting _jax_state enables callbacks to force a state sync
                # if they need to.
                self._jax_state = {
                    "trainable_variables": trainable_variables,
                    "non_trainable_variables": non_trainable_variables,
                    "optimizer_variables": optimizer_variables,
                    "metrics_variables": metrics_variables,
                }

                # Callbacks
                callbacks.on_train_batch_end(step, logs)
                if self.stop_training:
                    break

            # Reattach state to model variables.
            # NOTE: doing this after each step would be a big performance
            # bottleneck.
            self.jax_state_sync()

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
        self._jax_state = None
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
        self._assert_compile_called("evaluate")
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
                data_batch = data[0]
                self._eager_build(data_batch)
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

        self.make_test_function()
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
            logs, state = self.test_function(state, data)
            # Note that trainable variables are not returned since they're
            # immutable here.
            _, non_trainable_variables, metrics_variables = state

            # Setting _jax_state enables callbacks to force a state sync
            # if they need to.
            self._jax_state = {
                # I wouldn't recommend modifying non-trainable model state
                # during evaluate(), but it's allowed.
                "non_trainable_variables": non_trainable_variables,
                "metrics_variables": metrics_variables,
            }
            callbacks.on_test_batch_end(step, logs)

        # Reattach state back to model.
        self.jax_state_sync()

        logs = self._pythonify_logs(self.get_metrics_result())
        callbacks.on_test_end(logs)
        self._jax_state = None
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
                with backend.StatelessScope():
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

        self.make_predict_function()
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
            batch_outputs = self.predict_function(
                trainable_variables, non_trainable_variables, x
            )
            outputs = append_to_outputs(batch_outputs, outputs)
            callbacks.on_predict_batch_end(step, {"outputs": batch_outputs})
        callbacks.on_predict_end()
        return tf.__internal__.nest.map_structure_up_to(
            batch_outputs, np.concatenate, outputs
        )

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        """Runs a single gradient update on a single batch of data.

        Args:
            x: Input data. Must be array-like.
            y: Target data. Must be array-like.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape `(samples, sequence_length)`, to apply a different
                weight to every timestep of every sample.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) to apply to the model's loss for the samples
                from this class during training. This can be useful to tell the
                model to "pay more attention" to samples from an
                under-represented class. When `class_weight` is specified
                and targets have a rank of 2 or greater, either `y` must
                be one-hot encoded, or an explicit final dimension of 1
                must be included for sparse class labels.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric. If `False`,
                they are returned as a list.

        Returns:
            A scalar loss value (when no metrics and `return_dict=False`),
            a list of loss and metric values
            (if there are metrics and `return_dict=False`), or a dict of
            metric and loss values (if `return_dict=True`).
        """
        self._assert_compile_called("train_on_batch")
        if class_weight is not None:
            if sample_weight is not None:
                raise ValueError(
                    "Arguments `sample_weight` and `class_weight` "
                    "cannot be specified at the same time. "
                    f"Received: sample_weight={sample_weight}, "
                    f"class_weight={class_weight}"
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )
        data = (x, y, sample_weight)

        # Maybe build model
        self._eager_build(data)
        self.make_train_function()

        # Train step
        trainable_variables = self.trainable_variables
        non_trainable_variables = self.non_trainable_variables
        optimizer_variables = self.optimizer.variables
        metrics_variables = self.metrics_variables
        state = (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        )
        logs, state = self.train_function(state, [data])

        # State sync
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        self._jax_state = {
            "trainable_variables": trainable_variables,
            "non_trainable_variables": non_trainable_variables,
            "optimizer_variables": optimizer_variables,
            "metrics_variables": metrics_variables,
        }
        self.jax_state_sync()

        # Format return values
        logs = tf.nest.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        """Test the model on a single batch of samples.

        Args:
            x: Input data. Must be array-like.
            y: Target data. Must be array-like.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape `(samples, sequence_length)`, to apply a different
                weight to every timestep of every sample.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric. If `False`,
                they are returned as a list.

        Returns:
            A scalar loss value (when no metrics and `return_dict=False`),
            a list of loss and metric values
            (if there are metrics and `return_dict=False`), or a dict of
            metric and loss values (if `return_dict=True`).
        """
        self._assert_compile_called("test_on_batch")

        data = (x, y, sample_weight)
        # Maybe build model
        self._eager_build(data)
        self.make_test_function()

        # Test step
        trainable_variables = self.trainable_variables
        non_trainable_variables = self.non_trainable_variables
        metrics_variables = self.metrics_variables
        state = (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        )
        logs, state = self.test_function(state, [data])

        # State sync
        _, non_trainable_variables, metrics_variables = state
        self._jax_state = {
            "non_trainable_variables": non_trainable_variables,
            "metrics_variables": metrics_variables,
        }
        self.jax_state_sync()

        # Format return values.
        logs = tf.nest.map_structure(lambda x: np.array(x), logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        """Returns predictions for a single batch of samples.

        Args:
            x: Input data. It must be array-like.

        Returns:
            NumPy array(s) of predictions.
        """
        if not self.built:
            # Build model
            with backend.StatelessScope():
                self(x)

        self.make_predict_function()
        trainable_variables = self.trainable_variables
        non_trainable_variables = self.non_trainable_variables
        batch_outputs = self.predict_function(
            trainable_variables, non_trainable_variables, [x]
        )
        batch_outputs = tf.nest.map_structure(
            lambda x: np.array(x), batch_outputs
        )
        return batch_outputs

    def jax_state_sync(self):
        if not getattr(self, "_jax_state", None):
            return

        trainable_variables = self._jax_state.get("trainable_variables", None)
        non_trainable_variables = self._jax_state.get(
            "non_trainable_variables", None
        )
        optimizer_variables = self._jax_state.get("optimizer_variables", None)
        metrics_variables = self._jax_state.get("metrics_variables", None)
        if trainable_variables:
            for ref_v, v in zip(self.trainable_variables, trainable_variables):
                ref_v.assign(v)
        if non_trainable_variables:
            for ref_v, v in zip(
                self.non_trainable_variables, non_trainable_variables
            ):
                ref_v.assign(v)
        if optimizer_variables:
            for ref_v, v in zip(self.optimizer.variables, optimizer_variables):
                ref_v.assign(v)
        if metrics_variables:
            for ref_v, v in zip(self.metrics_variables, metrics_variables):
                ref_v.assign(v)
