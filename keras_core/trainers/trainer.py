import platform
import warnings

from keras_core import backend
from keras_core import metrics as metrics_module
from keras_core import operations as ops
from keras_core import optimizers
from keras_core.saving import serialization_lib
from keras_core.trainers.compile_utils import CompileLoss
from keras_core.trainers.compile_utils import CompileMetrics
from keras_core.utils import traceback_utils
from keras_core.utils import tracking


class Trainer:
    def __init__(self):
        self._lock = False
        self._run_eagerly = False
        self._jit_compile = None
        self.compiled = False
        self.steps_per_execution = 1

    @traceback_utils.filter_traceback
    @tracking.no_automatic_dependency_tracking
    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        loss_weights=None,
        metrics=None,
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=1,
        jit_compile="auto",
    ):
        self.optimizer = optimizers.get(optimizer)
        if loss is not None:
            self._compile_loss = CompileLoss(loss, loss_weights)
        else:
            self._compile_loss = None
        if metrics is not None:
            self._compile_metrics = CompileMetrics(metrics, weighted_metrics)
        else:
            self._compile_metrics = None
        if jit_compile == "auto":
            if model_supports_jit(self):
                jit_compile = True
            else:
                jit_compile = False
        if jit_compile and run_eagerly:
            jit_compile = False
            warnings.warn(
                "If `run_eagerly` is True, then `jit_compile` "
                "cannot also be True. Disabling `jit_compile`.",
                stacklevel=2,
            )
        self.jit_compile = jit_compile
        self.run_eagerly = run_eagerly
        self.stop_training = False
        self.compiled = True
        self._loss_tracker = metrics_module.Mean(name="loss")
        self.steps_per_execution = steps_per_execution

        self._compile_config = serialization_lib.SerializableDict(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
        )

    @property
    def jit_compile(self):
        if self._jit_compile is None:
            # Value was never set. Resolve it now.
            jit_compile = model_supports_jit(self)
            self._jit_compile = jit_compile
        return self._jit_compile

    @jit_compile.setter
    def jit_compile(self, value):
        self._jit_compile = value

    @property
    def run_eagerly(self):
        return self._run_eagerly

    @run_eagerly.setter
    def run_eagerly(self, value):
        self._run_eagerly = value

    @property
    def metrics(self):
        metrics = [self._loss_tracker]
        metrics.extend(self._metrics[:])
        if self._compile_metrics is not None and self._compile_metrics.built:
            metrics += [self._compile_metrics]
        return metrics

    @property
    def metrics_variables(self):
        vars = []
        for metric in self.metrics:
            vars.extend(metric.variables)
        return vars

    def reset_metrics(self):
        for m in self.metrics:
            m.reset_state()

    def compute_loss(self, x=None, y=None, y_pred=None, sample_weight=None):
        """Compute the total loss, validate it, and return it.

        Subclasses can optionally override this method to provide custom loss
        computation logic.

        Example:

        ```python
        class MyModel(Model):
          def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.loss_tracker = metrics.Mean(name='loss')

          def compute_loss(self, x, y, y_pred, sample_weight):
            loss = ops.means((y_pred - y) ** 2)
            loss += ops.sum(self.losses)
            self.loss_tracker.update_state(loss)
            return loss

          def reset_metrics(self):
            self.loss_tracker.reset_state()

          @property
          def metrics(self):
            return [self.loss_tracker]

        inputs = layers.Input(shape=(10,), name='my_input')
        outputs = layers.Dense(10)(inputs)
        model = MyModel(inputs, outputs)
        model.add_loss(ops.sum(outputs))

        optimizer = SGD()
        model.compile(optimizer, loss='mse', steps_per_execution=10)
        dataset = ...
        model.fit(dataset, epochs=2, steps_per_epoch=10)
        print(f"Custom loss: {model.loss_tracker.result()}")
        ```

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the model (output of `model(x)`)
            sample_weight: Sample weights for weighting the loss function.

        Returns:
            The total loss as a scalar tensor, or `None` if no loss results
            (which is the case when called by `Model.test_step`).
        """
        del x  # The default implementation does not use `x`.
        losses = []
        if self._compile_loss is not None:
            loss = self._compile_loss(y, y_pred, sample_weight)
            if loss is not None:
                losses.append(loss)
        for loss in self.losses:
            losses.append(ops.cast(loss, dtype=backend.floatx()))
        if len(losses) == 0:
            raise ValueError(
                "No loss to compute. Provide a `loss` argument in `compile()`."
            )
        if len(losses) == 1:
            total_loss = losses[0]
        else:
            total_loss = ops.sum(losses)
        return total_loss

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        """Update metric states and collect all metrics to be returned.

        Subclasses can optionally override this method to provide custom metric
        updating and collection logic.

        Example:

        ```python
        class MyModel(Sequential):
          def compute_metrics(self, x, y, y_pred, sample_weight):
            # This super call updates `self.compiled_metrics` and returns
            # results for all metrics listed in `self.metrics`.
            metric_results = super().compute_metrics(
                x, y, y_pred, sample_weight)

            # Note that `self.custom_metric` is not listed in `self.metrics`.
            self.custom_metric.update_state(x, y, y_pred, sample_weight)
            metric_results['custom_metric_name'] = self.custom_metric.result()
            return metric_results
        ```

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the model output of `model.call(x)`.
            sample_weight: Sample weights for weighting the loss function.

        Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end()`. Typically,
            the values of the metrics listed in `self.metrics` are returned.
            Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        del x  # The default implementation does not use `x`.
        if self._compile_metrics is not None:
            self._compile_metrics.update_state(y, y_pred, sample_weight)
        return self.get_metrics_result()

    def get_metrics_result(self):
        """Returns the model's metrics values as a dict.

        If any of the metric result is a dict (containing multiple metrics),
        each of them gets added to the top level returned dict of this method.

        Returns:
            A `dict` containing values of the metrics listed in `self.metrics`.
            Example:
            `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return self._pythonify_logs(return_metrics)

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
        raise NotImplementedError

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
        raise NotImplementedError

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
        raise NotImplementedError

    def predict_on_batch(self, x):
        """Returns predictions for a single batch of samples.

        Args:
            x: Input data. It must be array-like.

        Returns:
            NumPy array(s) of predictions.
        """
        raise NotImplementedError

    def get_compile_config(self):
        """Returns a serialized config with information for compiling the model.

        This method returns a config dictionary containing all the information
        (optimizer, loss, metrics, etc.) with which the model was compiled.

        Returns:
            A dict containing information for compiling the model.
        """
        if self.compiled and hasattr(self, "_compile_config"):
            return self._compile_config.serialize()

    def compile_from_config(self, config):
        """Compiles the model with the information given in config.

        This method uses the information in the config (optimizer, loss,
        metrics, etc.) to compile the model.

        Args:
            config: Dict containing information for compiling the model.
        """
        has_overridden_compile = self.__class__.compile != Trainer.compile
        if has_overridden_compile:
            warnings.warn(
                "`compile()` was not called as part of model loading "
                "because the model's `compile()` method is custom. "
                "All subclassed Models that have `compile()` "
                "overridden should also override "
                "`get_compile_config()` and `compile_from_config(config)`. "
                "Alternatively, you can "
                "call `compile()` manually after loading.",
                stacklevel=2,
            )
            return
        config = serialization_lib.deserialize_keras_object(config)
        self.compile(**config)
        if hasattr(self, "optimizer") and self.built:
            # Create optimizer variables.
            self.optimizer.build(self.trainable_variables)

    def _should_eval(self, epoch, validation_freq):
        epoch = epoch + 1  # one-index the user-facing epoch.
        if isinstance(validation_freq, int):
            return epoch % validation_freq == 0
        elif isinstance(validation_freq, list):
            return epoch in validation_freq
        else:
            raise ValueError(
                "Expected `validation_freq` to be a list or int. "
                f"Received: validation_freq={validation_freq} of the "
                f"type {type(validation_freq)}."
            )

    def _pythonify_logs(self, logs):
        result = {}
        for key, value in sorted(logs.items()):
            try:
                value = float(value)
            except:
                pass
            result[key] = value
        return result

    def _flatten_metrics_in_order(self, logs):
        """Turns `logs` dict into a list as per key order of `metrics_names`."""
        metric_names = [m.name for m in self.metrics]
        results = []
        for name in metric_names:
            if name in logs:
                results.append(logs[name])
        for key in sorted(logs.keys()):
            if key not in metric_names:
                results.append(logs[key])
        if len(results) == 1:
            return results[0]
        return results

    def _assert_compile_called(self, method_name=None):
        if not self.compiled:
            msg = "You must call `compile()` before "
            if metrics_module:
                msg += "using the model."
            else:
                msg += f"calling `{method_name}()`."
            raise ValueError(msg)


def model_supports_jit(model):
    if platform.system() == "Darwin" and "arm" in platform.processor().lower():
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            if tf.config.list_physical_devices("GPU"):
                return False
    if all(x.supports_jit for x in model._flatten_layers()):
        return True
    return False
