import warnings
from keras_core import backend
from keras_core import operations as ops
from keras_core import metrics as metrics_module
from keras_core.trainers.compile_utils import CompileLoss
from keras_core.trainers.compile_utils import CompileMetrics


class Trainer:
    def __init__(self):
        self._run_eagerly = False
        self._jit_compile = True
        self.compiled = False
        self.train_function = None
        self.test_function = None
        self.predict_function = None

    def compile(
        self,
        optimizer,
        loss=None,
        loss_weights=None,
        metrics=None,
        weighted_metrics=None,
        run_eagerly=False,
        jit_compile=True,
    ):
        # TODO: get from module
        self.optimizer = optimizer
        if loss is not None:
            self._compile_loss = CompileLoss(loss, loss_weights)
        else:
            self._compile_loss = None
        if metrics is not None:
            self._compile_metrics = CompileMetrics(metrics, weighted_metrics)
        else:
            self._compile_metrics = None
        if jit_compile and run_eagerly:
            jit_compile = False
            warnings.warn(
                "If `run_eagerly` is True, then `jit_compile` cannot also be True. "
                "Disabling `jit_compile`.",
                stacklevel=2,
            )
        self.jit_compile = jit_compile
        self.run_eagerly = run_eagerly
        self.stop_training = False
        self.compiled = True
        self._loss_tracker = metrics_module.Mean(name="loss")

    @property
    def jit_compile(self):
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
        for l in self.losses:
            losses.append(ops.cast(l, dtype=backend.floatx()))
        if len(losses) == 0:
            raise ValueError(
                "No loss to compute. Provide a `loss` argument in `compile()`."
            )
        if len(losses) == 1:
            total_loss = losses[0]
        else:
            total_loss = ops.sum(losses)
        self._loss_tracker.update_state(total_loss)
        return total_loss

    def compute_metrics(self, x, y, y_pred, sample_weight):
        """Update metric states and collect all metrics to be returned.

        Subclasses can optionally override this method to provide custom metric
        updating and collection logic.

        Example:

        ```python
        class MyModel(Sequential):
          def compute_metrics(self, x, y, y_pred, sample_weight):
            # This super call updates `self.compiled_metrics` and returns
            # results for all metrics listed in `self.metrics`.
            metric_results = super().compute_metrics(x, y, y_pred, sample_weight)

            # Note that `self.custom_metric` is not listed in `self.metrics`.
            self.custom_metric.update_state(x, y, y_pred, sample_weight)
            metric_results['custom_metric_name'] = self.custom_metric.result()
            return metric_results
        ```

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the model (output of `model.call(x)`)
            sample_weight: Sample weights for weighting the loss function.

        Returns:
            A `dict` containing values that will be passed to
            `tf.keras.callbacks.CallbackList.on_train_batch_end()`. Typically, the
            values of the metrics listed in `self.metrics` are returned. Example:
            `{'loss': 0.2, 'accuracy': 0.7}`.
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
        return return_metrics

    def train_step(self, data):
        raise NotImplementedError

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

    def get_compile_config(self):
        # TODO
        raise NotImplementedError

    def compile_from_config(self):
        # TODO
        raise NotImplementedError
