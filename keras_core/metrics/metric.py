from keras_core import backend
from keras_core.utils.tracking import Tracker
from keras_core.utils.naming import auto_name
from keras_core.api_export import keras_core_export


@keras_core_export(["keras_core.Metric", "keras_core.metrics.Metric"])
class Metric:
    def __init__(self, dtype=None, name=None):
        self.name = name or auto_name(self.__class__.__name__)
        self._dtype = dtype
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
            v.assign(0)

    def update_state(self, *args, **kwargs):
        """Accumulate statistics for the metric."""
        raise NotImplementedError

    def result(self):
        """Compute the current metric value.

        Returns:
            A scalar tensor, or a dictionary of scalar tensors.
        """
        raise NotImplementedError

    @property
    def dtype(self):
        return self._dtype

    def add_variable(self, shape, initializer, dtype=None, name=None):
        self._check_super_called()
        if callable(initializer):
            value = initializer(shape=shape, dtype=dtype)
        else:
            raise ValueError(f"Invalid initializer: {initializer}")
        variable = backend.Variable(
            value=value,
            dtype=dtype,
            trainable=False,
            name=name,
        )
        self._variables.append(variable)
        # Prevent double-tracking
        self._tracker.stored_ids["variables"].add(id(variable))
        return variable

    @property
    def variables(self):
        variables = self._variables[:]
        for metric in self._metrics:
            variables.extend(metric._variables)
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
