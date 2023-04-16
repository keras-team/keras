class Trainer:
    def compile(
        self,
        optimizer,
        loss=None,
        loss_weights=None,
        metrics=None,
        weighted_metrics=None,
        run_eagerly=False,
        jit_compile=False,
    ):
        # TODO: get from module
        self.optimizer = optimizer

        self._compile_loss = CompileLoss(loss, loss_weights)
        self._compile_metrics = CompileMetrics(metrics, weighted_metrics)

        # TODO: immediately build the loss/metrics if we have a graph network
        self.jit_compile = jit_compile
        self.run_eagerly = run_eagerly

    # @property
    # def num_inputs(self):
    #     """Return the number of input tensors expected to be taken by the first argument of `call`."""
    #     from keras_core.models.functional import Functional
    #     from keras_core.models.sequential import Sequential
    #     if isinstance(self, Functional):
    #         return len(self.inputs)
    #     elif isinstance(self, Sequential) and self._functional:
    #         return len(self._functional.inputs)
    #     # Subclassed model: unknown
    #     return None

    # @property
    # def num_outputs(self):
    #     """Return the number of output tensors expected to be returned by `call`."""
    #     from keras_core.models.functional import Functional
    #     from keras_core.models.sequential import Sequential
    #     if isinstance(self, Functional):
    #         return len(self.outputs)
    #     elif isinstance(self, Sequential) and self._functional:
    #         return len(self._functional.outputs)
    #     # Subclassed model: unknown
    #     return None

    # @property
    # def output_names(self):
    #     from keras_core.models.functional import Functional
    #     from keras_core.models.sequential import Sequential
    #     if isinstance(self, Functional):
    #         functional = self
    #     elif isinstance(self, Sequential) and self._functional:
    #         functional = self._functional
    #     else:
    #         # Subclassed model: names undefined
    #         return None
    #     if isinstance(functional._outputs_struct, dict):
    #         return sorted(functional._outputs_struct.keys())
    #     return [
    #         x._keras_history.operation.name for x in self._outputs_struct
    #     ]

    # @property
    # def input_names(self):
    #     from keras_core.models.functional import Functional
    #     from keras_core.models.sequential import Sequential
    #     if isinstance(self, Functional):
    #         functional = self
    #     elif isinstance(self, Sequential) and self._functional:
    #         functional = self._functional
    #     else:
    #         # Subclassed model: names undefined
    #         return None
    #     if isinstance(functional._inputs_struct, dict):
    #         return sorted(functional._inputs_struct.keys())
    #     return [
    #         x._keras_history.operation.name for x in self._inputs_struct
    #     ]

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
        # TODO: combine sub-metrics with compile metric
        return self._metrics

    def reset_metrics(self):
        for m in self.metrics:
            m.reset_state()

    def call(self, inputs):
        raise NotImplementedError

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
        print(f"My custom loss: {model.loss_tracker.result()}")
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
        return self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses
        )

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
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
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

    def fit(self, x, y=None):
        raise NotImplementedError

    def evaluate(self, x, y=None):
        raise NotImplementedError

    def predict(self, x, y=None):
        raise NotImplementedError

    def get_compile_config(self):
        return {}

    def compile_from_config(self):
        pass
