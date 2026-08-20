from keras.src import backend
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src.api_export import keras_export
from keras.src.saving.keras_saveable import KerasSaveable
from keras.src.utils.naming import auto_name
from keras.src.utils.tracking import Tracker


@keras_export(["keras.Metric", "keras.metrics.Metric"])
class Metric(KerasSaveable):
    """Encapsulates metric logic and state.

    Args:
        name: Optional name for the metric instance.
        dtype: The dtype of the metric's computations. Defaults to `None`, which
            means using `keras.backend.floatx()`. `keras.backend.floatx()` is a
            `"float32"` unless set to different value
            (via `keras.backend.set_floatx()`). If a `keras.DTypePolicy` is
            provided, then the `compute_dtype` will be utilized.

    Example:

    ```python
    m = SomeMetric(...)
    for input in ...:
        m.update_state(input)
    print('Final result: ', m.result())
    ```

    Usage with `compile()` API:

    ```python
    model = keras.Sequential()
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer=keras.optimizers.RMSprop(0.01),
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=[keras.metrics.CategoricalAccuracy()])

    data = np.random.random((1000, 32))
    labels = np.random.random((1000, 10))

    model.fit(data, labels, epochs=10)
    ```

    To be implemented by subclasses:

    * `__init__()`: All state variables should be created in this method by
      calling `self.add_variable()` like: `self.var = self.add_variable(...)`
    * `update_state()`: Has all updates to the state variables like:
      `self.var.assign(...)`.
    * `result()`: Computes and returns a scalar value or a dict of scalar values
      for the metric from the state variables.

    Example subclass implementation:

    ```python
    class BinaryTruePositives(Metric):

        def __init__(self, name='binary_true_positives', **kwargs):
            super().__init__(name=name, **kwargs)
            self.true_positives = self.add_variable(
                shape=(),
                initializer='zeros',
                name='true_positives'
            )

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = ops.cast(y_true, "bool")
            y_pred = ops.cast(y_pred, "bool")

            values = ops.logical_and(
                ops.equal(y_true, True), ops.equal(y_pred, True))
            values = ops.cast(values, self.dtype)
            if sample_weight is not None:
                sample_weight = ops.cast(sample_weight, self.dtype)
                sample_weight = ops.broadcast_to(
                    sample_weight, ops.shape(values)
                )
                values = ops.multiply(values, sample_weight)
            self.true_positives.assign(self.true_positives + ops.sum(values))

        def result(self):
            return self.true_positives
    ```
    """

    def __init__(self, dtype=None, name=None):
        self.name = name or auto_name(self.__class__.__name__)
        self._dtype_policy = dtype_policies.get(dtype or backend.floatx())
        self._dtype = self._dtype_policy.compute_dtype
        self._metrics = []
        self._variables = []
        self._tracker = Tracker(
            {
                "variables": (
                    lambda x: isinstance(x, backend.Variable),
                    self._variables,
                ),
                "metrics": (lambda x: isinstance(x, Metric), self._metrics),
            }
        )

    def reset_state(self):
        """Reset all of the metric state variables.

        This function is called between epochs/steps,
        when a metric is evaluated during training.
        """
        for v in self.variables:
            v.assign(ops.zeros(v.shape, dtype=v.dtype))

    def update_state(self, *args, **kwargs):
        """Accumulate statistics for the metric."""
        raise NotImplementedError

    def stateless_update_state(self, metric_variables, *args, **kwargs):
        if len(metric_variables) != len(self.variables):
            raise ValueError(
                "Argument `metric_variables` must be a list of tensors "
                f"corresponding 1:1 to {self.__class__.__name__}().variables. "
                f"Received list with length {len(metric_variables)}, but "
                f"expected {len(self.variables)} variables."
            )
        # Gather variable mapping
        mapping = list(zip(self.variables, metric_variables))

        # Call in stateless scope
        with backend.StatelessScope(state_mapping=mapping) as scope:
            self.update_state(*args, **kwargs)

        # Gather updated variables
        metric_variables = []
        for v in self.variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                metric_variables.append(new_v)
            else:
                metric_variables.append(v)
        return metric_variables

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.
        """
        raise NotImplementedError

    def stateless_result(self, metric_variables):
        if len(metric_variables) != len(self.variables):
            raise ValueError(
                "Argument `metric_variables` must be a list of tensors "
                f"corresponding 1:1 to {self.__class__.__name__}().variables. "
                f"Received list with length {len(metric_variables)}, but "
                f"expected {len(self.variables)} variables."
            )
        # Gather variable mapping
        mapping = list(zip(self.variables, metric_variables))

        # Call in stateless scope
        with backend.StatelessScope(state_mapping=mapping):
            res = self.result()
        return res

    def stateless_reset_state(self):
        # Call in stateless scope
        with backend.StatelessScope() as scope:
            self.reset_state()

        # Gather updated variables
        metric_variables = []
        for v in self.variables:
            new_v = scope.get_current_value(v)
            if new_v is not None:
                metric_variables.append(new_v)
            else:
                metric_variables.append(v)
        return metric_variables

    @property
    def dtype(self):
        return self._dtype

    def _obj_type(self):
        return "Metric"

    def add_variable(
        self, shape, initializer, dtype=None, aggregation="sum", name=None
    ):
        self._check_super_called()
        with backend.name_scope(self.name.replace("/", ">"), caller=self):
            initializer = initializers.get(initializer)
            variable = backend.Variable(
                initializer=initializer,
                shape=shape,
                dtype=dtype,
                trainable=False,
                aggregation=aggregation,
                synchronization="on_read",
                name=name,
            )
        # Prevent double-tracking
        self._tracker.add_to_store("variables", variable)
        return variable

    def add_weight(self, shape=(), initializer=None, dtype=None, name=None):
        # Backwards compatibility alias
        return self.add_variable(
            shape=shape, initializer=initializer, dtype=dtype, name=name
        )

    @property
    def variables(self):
        variables = list(self._variables)
        for metric in self._metrics:
            variables.extend(metric.variables)
        return variables

    def __call__(self, *args, **kwargs):
        self._check_super_called()
        self.update_state(*args, **kwargs)
        return self.result()

    def get_config(self):
        """Return the serializable config of the metric."""
        return {"name": self.name, "dtype": self.dtype}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def __setattr__(self, name, value):
        # Track Variables, Layers, Metrics
        if hasattr(self, "_tracker"):
            value = self._tracker.track(value)
        return super().__setattr__(name, value)

    def _check_super_called(self):
        if not hasattr(self, "_tracker"):
            raise RuntimeError(
                "You forgot to call `super().__init__()` "
                "in the `__init__()` method. Go add it!"
            )

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name}>"

    def __str__(self):
        return self.__repr__()
