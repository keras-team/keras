import mlx.core as mx
import numpy as np

from keras.src import backend
from keras.src import callbacks as callbacks_module
from keras.src import optimizers as optimizers_module
from keras.src import tree
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.mlx.core import is_tensor
from keras.src.trainers import trainer as base_trainer
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.epoch_iterator import EpochIterator
from keras.src.utils import traceback_utils


class MLXTrainer(base_trainer.Trainer):
    def __init__(self):
        super().__init__()
        self.train_function = None
        self.test_function = None
        self.predict_function = None
        self._mlx_state_synced = True

    def _data_to_mlx(self, data):
        def _transform(x):
            if isinstance(x, np.ndarray):
                return mx.array(x)
            else:
                return x

        return tree.map_structure(_transform, data)

    def mlx_state_sync(self):
        if not getattr(self, "_mlx_state", None):
            return

        trainable_variables = self._mlx_state.get("trainable_variables", None)
        non_trainable_variables = self._mlx_state.get(
            "non_trainable_variables", None
        )
        optimizer_variables = self._mlx_state.get("optimizer_variables", None)
        metrics_variables = self._mlx_state.get("metrics_variables", None)
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
        self._mlx_state_synced = True

    def _get_mlx_state(
        self,
        trainable_variables=False,
        non_trainable_variables=False,
        optimizer_variables=False,
        metrics_variables=False,
        purge_model_variables=False,
    ):
        state = []
        if trainable_variables:
            state.append([v.value for v in self.trainable_variables])
        if non_trainable_variables:
            state.append([v.value for v in self.non_trainable_variables])
        if optimizer_variables:
            state.append([v.value for v in self.optimizer.variables])
        if metrics_variables:
            state.append([v.value for v in self.metrics_variables])
        if purge_model_variables:
            self._purge_model_variables(
                trainable_variables=trainable_variables,
                non_trainable_variables=non_trainable_variables,
                optimizer_variables=optimizer_variables,
                metric_variables=metrics_variables,
            )
        return tuple(state)

    def _purge_model_variables(
        self,
        trainable_variables=False,
        non_trainable_variables=False,
        optimizer_variables=False,
        metric_variables=False,
    ):
        """Remove all the model variables so they can be garbage collected and
        the memory reclaimed by MLX.

        Very similar to JAX we have a stateless training function and keeping
        an extra reference to the model weights means the memory cannot be
        reclaimed by MLX.

        When finished updating the model variables by training, then we can
        reattach them to the model with `mlx_state_sync()`.
        """
        if trainable_variables:
            for v in self.trainable_variables:
                v._value = None
        if non_trainable_variables:
            for v in self.non_trainable_variables:
                v._value = None
        if optimizer_variables:
            for v in self.optimizer.variables:
                v._value = None
        if metric_variables:
            for v in self.metrics_variables:
                v._value = None

    def _pythonify_logs(self, logs):
        result = {}
        for key, value in sorted(logs.items()):
            if isinstance(value, dict):
                result.update(self._pythonify_logs(value))
            else:
                if (
                    isinstance(value, mx.array)
                    and value.size == 1
                    and value.dtype in (mx.float32, mx.float16, mx.bfloat16)
                ):
                    value = value.item()
                result[key] = value
        return result

    def get_metrics_result(self):
        """Same as the base class but doesn't pythonify the logs to allow for
        lazy computation."""
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

    def compute_loss_and_updates(
        self,
        trainable_variables,
        non_trainable_variables,
        metrics_variables,
        x,
        y,
        sample_weight,
        training=False,
        optimizer_variables=None,
    ):
        """Similar to the jax trainer, this method is stateless and is intended
        for use with mx.value_and_grad ."""
        kwargs = {}
        if self._call_has_training_arg:
            kwargs["training"] = training
        y_pred, non_trainable_variables, losses = self.stateless_call(
            trainable_variables,
            non_trainable_variables,
            x,
            return_losses=True,
            **kwargs,
        )
        if losses:
            # Make forward pass losses available to compute_loss.
            self._losses_override.clear()
            self._losses_override = losses

        loss, variables = self.stateless_compute_loss(
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
            x=x,
            y=y,
            y_pred=y_pred,
            sample_weight=sample_weight,
        )
        if losses:
            self._losses_override.clear()
        (trainable_variables, non_trainable_variables, metrics_variables) = (
            variables
        )
        unscaled_loss = loss
        if training and self.optimizer is not None:
            # Scale loss with a StatelessScope, to use an update scale variable.
            mapping = list(zip(self.optimizer.variables, optimizer_variables))
            with backend.StatelessScope(state_mapping=mapping):
                loss = self.optimizer.scale_loss(loss)

        return (
            loss,
            unscaled_loss,
            y_pred,
            non_trainable_variables,
            metrics_variables,
        )

    def train_step(self, state, data):
        data = self._data_to_mlx(data)
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        grad_fn = mx.value_and_grad(self.compute_loss_and_updates)

        if trainable_variables:
            (
                (
                    loss,
                    unscaled_loss,
                    y_pred,
                    non_trainable_variables,
                    metrics_variables,
                ),
                grads,
            ) = grad_fn(
                trainable_variables,
                non_trainable_variables,
                metrics_variables,
                x,
                y,
                sample_weight,
                training=True,
                optimizer_variables=optimizer_variables,
            )

            (
                trainable_variables,
                optimizer_variables,
            ) = self.optimizer.stateless_apply(
                optimizer_variables, grads, trainable_variables
            )
        else:
            (
                loss,
                unscaled_loss,
                y_pred,
                non_trainable_variables,
                metrics_variables,
            ) = self.compute_loss_and_updates(
                trainable_variables,
                non_trainable_variables,
                metrics_variables,
                x,
                y,
                sample_weight,
                training=True,
                optimizer_variables=optimizer_variables,
            )

        with backend.StatelessScope(
            state_mapping=[
                (ref_v, v)
                for ref_v, v in zip(self.metrics_variables, metrics_variables)
            ]
        ) as scope:
            self._loss_tracker.update_state(unscaled_loss)
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

    def test_step(self, state, data):
        data = self._data_to_mlx(data)
        (
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
        ) = state
        x, y, sample_weight = data_adapter_utils.unpack_x_y_sample_weight(data)
        (
            loss,
            unscaled_loss,
            y_pred,
            non_trainable_variables,
            metrics_variables,
        ) = self.compute_loss_and_updates(
            trainable_variables,
            non_trainable_variables,
            metrics_variables,
            x,
            y,
            sample_weight,
            training=False,
        )

        with backend.StatelessScope(
            state_mapping=[
                (ref_v, v)
                for ref_v, v in zip(self.metrics_variables, metrics_variables)
            ]
        ) as scope:
            self._loss_tracker.update_state(unscaled_loss)
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

    def predict_step(self, state, data):
        trainable_variables, non_trainable_variables = state
        kwargs = {}
        if self._call_has_training_arg:
            kwargs["training"] = False

        x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data)
        x = self._data_to_mlx(x)
        outputs, non_trainable_variables = self.stateless_call(
            trainable_variables, non_trainable_variables, x, **kwargs
        )
        return outputs, (trainable_variables, non_trainable_variables)

    def make_train_function(self, force=False):
        if self.train_function is not None and not force:
            return self.train_function

        def one_train_step(state, data):
            data = data[0]
            return self.train_step(state, data)

        def multi_train_steps(state, data):
            for single_step_data in data:
                logs, state = one_train_step(state, [single_step_data])
            return logs, state

        if self.steps_per_execution > 1:
            train_step = multi_train_steps
        else:
            train_step = one_train_step

        self.train_function = train_step

    def make_test_function(self, force=False):
        if self.test_function is not None and not force:
            return self.test_function

        def one_test_step(state, data):
            data = data[0]
            return self.test_step(state, data)

        def multi_test_steps(state, data):
            for single_step_data in data:
                logs, state = one_test_step(state, [single_step_data])
            return logs, state

        if self.steps_per_execution > 1:
            test_step = multi_test_steps
        else:
            test_step = one_test_step

        self.test_function = test_step

    def make_predict_function(self, force=False):
        if self.predict_function is not None and not force:
            return self.predict_function

        def one_predict_step(state, data):
            data = data[0]
            return self.predict_step(state, data)

        def multi_predict_steps(state, data):
            outputs, state = one_predict_step(state, data[:1])

            for single_step_data in data[1:]:
                step_outputs, state = one_predict_step(
                    state,
                    [single_step_data],
                )
                outputs = tree.map_structure(
                    lambda t1, t2: mx.concatenate([t1, t2]),
                    outputs,
                    step_outputs,
                )
            return outputs, state

        if self.steps_per_execution > 1:
            predict_step = multi_predict_steps
        else:
            predict_step = one_predict_step

        self.predict_function = predict_step

    def _symbolic_build(self, data_batch):
        model_unbuilt = not all(layer.built for layer in self._flatten_layers())
        compile_metrics_unbuilt = (
            hasattr(self, "_compile_metrics")
            and self._compile_metrics is not None
            and not self._compile_metrics.built
        )
        if model_unbuilt or compile_metrics_unbuilt:
            # Create symbolic tensors matching an input batch.

            def to_symbolic_input(v):
                if is_tensor(v):
                    return KerasTensor(v.shape, standardize_dtype(v.dtype))
                return v

            data_batch = tree.map_structure(to_symbolic_input, data_batch)
            (
                x,
                y,
                sample_weight,
            ) = data_adapter_utils.unpack_x_y_sample_weight(data_batch)
            # Build all model state with `backend.compute_output_spec`.
            try:
                y_pred = backend.compute_output_spec(self, x)
            except Exception as e:
                raise RuntimeError(
                    "Unable to automatically build the model. "
                    "Please build it yourself before calling "
                    "fit/evaluate/predict. "
                    "A model is 'built' when its variables have "
                    "been created and its `self.built` attribute "
                    "is True. Usually, calling the model on a batch "
                    "of data is the right way to build it.\n"
                    "Exception encountered:\n"
                    f"'{e}'"
                )
            if compile_metrics_unbuilt and y is not None:
                # Build all metric state with `backend.compute_output_spec`.
                backend.compute_output_spec(
                    self.compute_metrics,
                    x,
                    y,
                    y_pred,
                    sample_weight=sample_weight,
                )
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer.built
        ):
            # Build optimizer
            self.optimizer.build(self.trainable_variables)
        self._post_build()

    @traceback_utils.filter_traceback
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
        self._eval_epoch_iterator = None
        if validation_split and validation_data is None:
            # Create the validation data using the training data. Only supported
            # for TF/numpy/jax arrays.
            (
                (
                    x,
                    y,
                    sample_weight,
                ),
                validation_data,
            ) = array_slicing.train_validation_split(
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
            not all(layer.built for layer in self._flatten_layers())
            or not self.optimizer.built
            or (
                self._compile_metrics is not None
                and not self._compile_metrics.built
            )
        )
        if needs_building:
            # Build the model on one batch of data.
            for _, data in epoch_iterator.enumerate_epoch():
                data_batch = data[0]
                self._symbolic_build(data_batch)
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

        self.stop_training = False
        self.make_train_function()
        callbacks.on_train_begin()
        initial_epoch = self._initial_epoch or initial_epoch
        for epoch in range(initial_epoch, epochs):
            self.reset_metrics()
            callbacks.on_epoch_begin(epoch)

            self._mlx_state_synced = True
            for step, data in epoch_iterator.enumerate_epoch():
                # Callbacks
                callbacks.on_train_batch_begin(step)
                # Train step
                if self._mlx_state_synced:
                    # The state may have been synced by a callback.
                    state = self._get_mlx_state(
                        trainable_variables=True,
                        non_trainable_variables=True,
                        optimizer_variables=True,
                        metrics_variables=True,
                        purge_model_variables=True,
                    )
                    self._mlx_state_synced = False

                logs, state = self.train_function(state, data)
                mx.eval(logs, state)
                (
                    trainable_variables,
                    non_trainable_variables,
                    optimizer_variables,
                    metrics_variables,
                ) = state

                # Setting _mlx_state enables callbacks to force a state sync
                # if they need to.
                self._mlx_state = {
                    "trainable_variables": trainable_variables,
                    "non_trainable_variables": non_trainable_variables,
                    "optimizer_variables": optimizer_variables,
                    "metrics_variables": metrics_variables,
                }

                # Callbacks
                callbacks.on_train_batch_end(step, self._pythonify_logs(logs))
                if self.stop_training:
                    break

            # Reattach state to model variables.
            self.mlx_state_sync()

            # Override with model metrics instead of last step logs
            epoch_logs = self._pythonify_logs(
                self._get_metrics_result_or_logs(logs)
            )

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
                        steps_per_epoch=validation_steps,
                        shuffle=False,
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
            self.optimizer.finalize_variable_values(self.trainable_weights)

        # If _eval_epoch_iterator exists, delete it after all epochs are done.
        if getattr(self, "_eval_epoch_iterator", None) is not None:
            del self._eval_epoch_iterator
        callbacks.on_train_end(logs=training_logs)
        self._mlx_state = None
        return self.history

    @traceback_utils.filter_traceback
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

        if not all(layer.built for layer in self._flatten_layers()):
            # Build the model on one batch of data.
            for _, data in epoch_iterator.enumerate_epoch():
                data_batch = data[0]
                self._symbolic_build(data_batch)
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
        self.stop_evaluating = False
        callbacks.on_test_begin()
        logs = None
        self.reset_metrics()

        trainable_variables = [v.value for v in self.trainable_variables]
        non_trainable_variables = [
            v.value for v in self.non_trainable_variables
        ]
        metrics_variables = [v.value for v in self.metrics_variables]

        self._purge_model_variables(optimizer_variables=False)
        for step, data in epoch_iterator.enumerate_epoch():
            callbacks.on_test_batch_begin(step)

            state = (
                trainable_variables,
                non_trainable_variables,
                metrics_variables,
            )
            logs, state = self.test_function(state, data)
            mx.eval(logs, state)
            (
                trainable_variables,
                non_trainable_variables,
                metrics_variables,
            ) = state

            self._mlx_state = {
                # I wouldn't recommend modifying non-trainable model state
                # during evaluate(), but it's allowed.
                "trainable_variables": trainable_variables,
                "non_trainable_variables": non_trainable_variables,
                "metrics_variables": metrics_variables,
            }
            callbacks.on_test_batch_end(step, self._pythonify_logs(logs))
            if self.stop_evaluating:
                break

        self.mlx_state_sync()
        logs = self._pythonify_logs(self._get_metrics_result_or_logs(logs))
        callbacks.on_test_end(logs)
        self._mlx_state = None

        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    @traceback_utils.filter_traceback
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

        if not all(layer.built for layer in self._flatten_layers()):
            # Build the model on one batch of data.
            for _, data in epoch_iterator.enumerate_epoch():
                # Build model
                x, _, _ = data_adapter_utils.unpack_x_y_sample_weight(data[0])
                with backend.StatelessScope():
                    self(x)
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

        def append_to_outputs(batch_outputs, outputs):
            if outputs is None:
                outputs = tree.map_structure(
                    lambda batch_output: [batch_output],
                    batch_outputs,
                )
            else:
                tree.map_structure_up_to(
                    batch_outputs,
                    lambda output, batch_output: output.append(batch_output),
                    outputs,
                    batch_outputs,
                )
            return outputs

        self.make_predict_function()
        self.stop_predicting = False
        callbacks.on_predict_begin()

        self._mlx_state_synced = True
        outputs = None
        non_trainable_variables = None
        for step, data in epoch_iterator.enumerate_epoch():
            callbacks.on_predict_batch_begin(step)
            if self._mlx_state_synced:
                # The state may have been synced by a callback.
                state = self._get_mlx_state(
                    trainable_variables=True,
                    non_trainable_variables=True,
                )
                self._purge_model_variables(non_trainable_variables=True)
                self._mlx_state_synced = False
            else:
                state = (state[0], non_trainable_variables)
            batch_outputs, state = self.predict_function(state, data)
            mx.eval(batch_outputs, state)
            (trainable_variables, non_trainable_variables) = state
            outputs = append_to_outputs(batch_outputs, outputs)
            callbacks.on_predict_batch_end(step, {"outputs": batch_outputs})
            if self.stop_predicting:
                break
        self._mlx_state = {
            # I wouldn't recommend modifying non-trainable model state
            # during predict(), but it's allowed.
            "non_trainable_variables": non_trainable_variables,
        }
        self.mlx_state_sync()
        callbacks.on_predict_end()
        self._mlx_state = None
        outputs = tree.map_structure(
            backend.convert_to_numpy, outputs
        )  # TODO: This copies but we could avoid it
        return tree.map_structure_up_to(batch_outputs, np.concatenate, outputs)

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
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
        self._symbolic_build(data)
        self.make_train_function()

        state = self._get_mlx_state(
            trainable_variables=True,
            non_trainable_variables=True,
            optimizer_variables=True,
            metrics_variables=True,
            purge_model_variables=False,
        )
        self._mlx_state_synced = False
        logs, state = self.train_function(state, [data])
        mx.eval(logs, state)

        # State sync
        (
            trainable_variables,
            non_trainable_variables,
            optimizer_variables,
            metrics_variables,
        ) = state
        self._mlx_state = {
            "trainable_variables": trainable_variables,
            "non_trainable_variables": non_trainable_variables,
            "optimizer_variables": optimizer_variables,
            "metrics_variables": metrics_variables,
        }
        self.mlx_state_sync()

        logs = self._pythonify_logs(logs)
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
        self._assert_compile_called("test_on_batch")

        data = (x, y, sample_weight)

        # Maybe build model
        self._symbolic_build(data)
        self.make_test_function()

        # Test step
        state = self._get_mlx_state(
            trainable_variables=True,
            non_trainable_variables=True,
            metrics_variables=True,
            purge_model_variables=False,
        )
        self._mlx_state_synced = False
        logs, state = self.test_function(state, [data])
        mx.eval(logs, state)

        # State sync
        trainable_variables, non_trainable_variables, metrics_variables = state
        self._mlx_state = {
            "trainable_variables": trainable_variables,
            "non_trainable_variables": non_trainable_variables,
            "metrics_variables": metrics_variables,
        }
        self.mlx_state_sync()

        logs = self._pythonify_logs(logs)
        if return_dict:
            return logs
        return self._flatten_metrics_in_order(logs)

    def predict_on_batch(self, x):
        if not all(layer.built for layer in self._flatten_layers()):
            # Build model
            with backend.StatelessScope():
                self(x)
        self._symbolic_build(x)
        self.make_predict_function()
        state = self._get_mlx_state(
            trainable_variables=True,
            non_trainable_variables=True,
            metrics_variables=False,
            purge_model_variables=False,
        )
        self._mlx_state_synced = False
        batch_outputs, state = self.predict_function(state, [(x,)])
        mx.eval(batch_outputs, state)
        trainable_variables, non_trainable_variables = state
        self._mlx_state = {
            "non_trainable_variables": non_trainable_variables,
        }
        self.mlx_state_sync()
        # TODO: This copies but we could avoid it
        batch_outputs = tree.map_structure(
            backend.convert_to_numpy, batch_outputs
        )
        return batch_outputs
