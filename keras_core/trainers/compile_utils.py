"""
## Simple case:

metrics = ["m1', "m2"]
outputs = x

## Nested list case:

metrics = [["m1', "m2"], ["m3"]]
outputs [x1, x2]

## Nested dict case:

metrics = {"out_1": ["m1', "m2"], "out_2": ["m3"]]
outputs {"out_1": x1, "out_2": x2}

## Process

Case 1: known symbolic outputs

-> Flatten metrics to standard order at build time
-> Flatten arrays to standard order at computation time

Case 2: only arrays are known

-> expect exact matches between arrays and metrics
-> Give array elements names based on dict keys or order,
used to uniquify public-facing metric names



General process:

1. At build time
- Check structure match
- Flatten everything



"""
from tensorflow import nest

from keras_core import backend
from keras_core import losses as losses_module
from keras_core import metrics as metrics_module
from keras_core import operations as ops
from keras_core.utils.naming import get_object_name


class MetricsList(metrics_module.Metric):
    def __init__(self, metrics):
        self.metrics = metrics

    def update_state(self, y_true, y_pred, sample_weight=None):
        for m in self.metrics:
            m.update_state(
                y_true, y_pred, sample_weight=sample_weight
            )

    def reset_state(self):
        for m in self.metrics:
            m.reset_state()

    def get_result(self):
        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError


def is_function_like(value):
    if value is None:
        return True
    if isinstance(value, str):
        return True
    if callable(value):
        return True
    return False


def is_binary_or_sparse_categorical(y_true, y_pred):
    y_t_rank = len(y_true.shape)
    y_p_rank = len(y_pred.shape)
    y_t_last_dim = y_true.shape[-1]
    y_p_last_dim = y_pred.shape[-1]

    is_binary = y_p_last_dim == 1
    is_sparse_categorical = (
        y_t_rank < y_p_rank or y_t_last_dim == 1 and y_p_last_dim > 1
    )
    return is_binary, is_sparse_categorical


def get_metric(identifier, y_true, y_pred):
    if identifier is None:
        return None  # Ok to have no metric for an output.

    # Convenience feature for selecting b/t binary, categorical,
    # and sparse categorical.
    if str(identifier).lower() not in ["accuracy", "acc"]:
        metric_obj = metrics_module.get(identifier)
    else:
        is_binary, is_sparse_categorical = is_binary_or_sparse_categorical(
            y_true, y_pred
        )
        if is_binary:
            metric_obj = metrics_module.binary_accuracy
        elif is_sparse_categorical:
            metric_obj = metrics_module.sparse_categorical_accuracy
        else:
            metric_obj = metrics_module.categorical_accuracy

    if not isinstance(metric_obj, metrics_module.Metric):
        if isinstance(identifier, str):
            metric_name = identifier
        else:
            metric_name = get_object_name(metric_obj)
        metric_obj = metrics_module.MeanMetricWrapper(
            metric_obj, name=metric_name
        )
    return metric_obj


def get_loss(identifier, y_true, y_pred):
    if identifier is None:
        return None  # Ok to have no loss for an output.

    # Convenience feature for selecting b/t binary, categorical,
    # and sparse categorical.
    if str(identifier).lower() not in ["crossentropy", "ce"]:
        loss_obj = losses_module.get(identifier)
    else:
        is_binary, is_sparse_categorical = is_binary_or_sparse_categorical(
            y_true, y_pred
        )
        if is_binary:
            loss_obj = losses_module.binary_crossentropy
        elif is_sparse_categorical:
            loss_obj = losses_module.sparse_categorical_crossentropy
        else:
            loss_obj = losses_module.categorical_crossentropy

    if not isinstance(loss_obj, losses_module.Loss):
        if isinstance(identifier, str):
            loss_name = identifier
        else:
            loss_name = get_object_name(loss_obj)
        loss_obj = losses_module.LossFunctionWrapper(loss_obj, name=loss_name)
    return loss_obj


class CompileMetrics(metrics_module.Metric):
    def __init__(self, metrics, weighted_metrics):
        if metrics and not isinstance(metrics, (list, tuple, dict)):
            raise ValueError(
                "Expected `metrics` argument to be a list, tuple, or dict. "
                f"Received instead: metrics={metrics} of type {type(metrics)}"
            )
        if weighted_metrics and not isinstance(
            weighted_metrics, (list, tuple, dict)
        ):
            raise ValueError(
                "Expected `weighted_metrics` argument to be a list, tuple, or dict. "
                f"Received instead: weighted_metrics={weighted_metrics} "
                f"of type {type(weighted_metrics)}"
            )
        self._user_metrics = metrics
        self._user_weighted_metrics = weighted_metrics
        self.built = False

    def build(self, y_true, y_pred):
        if isinstance(y_pred, dict):
            output_names = sorted(list(y_pred.keys()))
            num_outputs = len(output_names)
        elif isinstance(y_pred, (list, tuple)):
            output_names = None
            num_outputs = len(y_pred)
        else:
            output_names = None
            num_outputs = 1

        y_pred = nest.flatten(y_pred)

        metrics = self._user_metrics
        weighted_metrics = self._user_weighted_metrics
        if output_names and not num_outputs:
            num_outputs = len(output_names)

        self._flat_metrics = self._build_metrics_set(
            metrics,
            num_outputs,
            output_names,
            y_true,
            y_pred,
            argument_name="metrics",
        )
        self._flat_weighted_metrics = self._build_metrics_set(
            weighted_metrics,
            num_outputs,
            output_names,
            y_true,
            y_pred,
            argument_name="weighted_metrics",
        )
        self.built = True

    def _build_metrics_set(
        self, metrics, num_outputs, output_names, y_true, y_pred, argument_name
    ):
        flat_metrics = []
        if num_outputs == 1:
            if not metrics:
                flat_metrics.append(None)
            else:
                if not isinstance(metrics, list):
                    raise ValueError(
                        f"When there is only a single output, the `{argument_name}` argument "
                        "must be a list of metric objects. "
                        f"Received instead: {argument_name}={metrics} of type {type(metrics)}"
                    )
                if not all(is_function_like(m) for m in metrics):
                    raise ValueError(
                        f"Expected all entries in the `{argument_name}` list to be "
                        f"metric objects. Received instead: {argument_name}={metrics}"
                    )
                flat_metrics.append(
                    MetricsList([get_metric(m, y_true[0], y_pred[0]) for m in metrics if m is not None])
                )
        else:
            if isinstance(metrics, (list, tuple)):
                if len(metrics) != len(y_pred):
                    raise ValueError(
                        "For a model with multiple outputs, "
                        f"when providing the `{argument_name}` argument as a list, "
                        "it should have as many entries as the model has outputs. "
                        f"Received {argument_name}={metrics} of length {len(metrics)} "
                        f"whereas the model has {len(y_pred)} outputs."
                    )
                for mls, yt, yp in zip(metrics, y_true, y_pred):
                    if not all(is_function_like(e) for e in mls):
                        raise ValueError(
                            f"All entries in the sublists of the `{argument_name}` "
                            "list should be metric objects. "
                            f"Found the following sublist with unknown types: {mls}"
                        )
                    flat_metrics.append(
                        MetricsList([get_metric(m, yt, yp) for m in mls if m is not None])
                    )
            elif isinstance(metrics, dict):
                for name in metrics.keys():
                    if name not in output_names:
                        raise ValueError(
                            "In the dict argument `{argument_name}`, key "
                            f"'{name}' does not correspond to any model output. "
                            f"Received: {argument_name}={metrics}"
                        )
                    if not all(is_function_like(e) for e in metrics[name]):
                        raise ValueError(
                            f"All entries in the sublists of the `{argument_name}` "
                            "dict should be metric objects. "
                            f"At key '{name}', found the following sublist "
                            f"with unknown types: {metrics[name]}"
                        )
                for name, yt, yp in zip(output_names, y_true, y_pred):
                    if name in metrics:
                        flat_metrics.append(
                            MetricsList(
                                [get_metric(m, yt, yp) for m in metrics[name] if m is not None]
                            )
                        )
                    else:
                        flat_metrics.append(None)
        return flat_metrics

    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self.built:
            self.build(y_true, y_pred)
        y_true = nest.flatten(y_true)
        y_pred = nest.flatten(y_pred)
        for m, y_t, y_p in zip(self._flat_metrics, y_true, y_pred):
            if m:
                m.update_state(y_t, y_p)
        if sample_weight is not None:
            sample_weight = nest.flatten(sample_weight)
        else:
            sample_weight = [None for _ in range(len(y_true))]
        for m, y_t, y_p, s_w in zip(
            self._flat_weighted_metrics, y_true, y_pred, sample_weight
        ):
            if m:
                m.update_state(y_t, y_p, s_w)

    def reset_state(self):
        for m in self._flat_metrics:
            if m:
                m.reset_state()
        for m in self._flat_weighted_metrics:
            if m:
                m.reset_state()

    def result(self):
        results = {}
        unique_name_counters = {}
        for mls in self._flat_metrics:
            if not mls:
                continue
            for m in mls.metrics:
                if m.name not in unique_name_counters:
                    results[m.name] = m.result()
                    unique_name_counters[m.name] = 1
                else:
                    name = f"{m.name}_{unique_name_counters[m.name]}"
                    unique_name_counters[m.name] += 1
                    results[name] = m.result()

        for mls in self._flat_weighted_metrics:
            if not mls:
                continue
            for m in mls.metrics:
                if m.name not in unique_name_counters:
                    results[m.name] = m.result()
                    unique_name_counters[m.name] = 1
                else:
                    name = f"weighted_{m.name}"
                    if name not in unique_name_counters:
                        unique_name_counters[name] = 1
                    else:
                        name = f"{m.name}_{unique_name_counters[m.name]}"
                        unique_name_counters[m.name] += 1
                    results[name] = m.result()
        return results

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError


class CompileLoss(losses_module.Loss):
    def __init__(self, loss, loss_weights):
        self._user_loss = loss
        self._user_loss_weights = loss_weights

    def build(self, y_true, y_pred):
        # TODO
        pass

    def call(self, y_true, y_pred):
        if not self.built:
            self.build(y_true, y_pred)

        y_true = nest.flatten(y_true)
        y_pred = nest.flatten(y_pred)
        loss_values = []
        for loss, y_t, y_p, w in zip(
            self.losses, y_true, y_pred, self.loss_weights
        ):
            if loss:
                value = w * ops.cast(loss(y_t, y_p), dtype=backend.floatx())
                loss_values.append(value)

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
