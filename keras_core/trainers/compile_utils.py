from tensorflow import nest

from keras_core import backend
from keras_core import losses as losses_module
from keras_core import metrics as metrics_module
from keras_core import operations as ops
from keras_core.utils.naming import get_object_name


class MetricsList(metrics_module.Metric):
    def __init__(self, metrics, name="metrics_list"):
        super().__init__(name=name)
        self.metrics = metrics

    def update_state(self, y_true, y_pred, sample_weight=None):
        for m in self.metrics:
            m.update_state(y_true, y_pred, sample_weight=sample_weight)

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
    def __init__(self, metrics, weighted_metrics, name="compile_metric"):
        super().__init__(name=name)
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
        self.name = "compile_metrics"

    @property
    def variables(self):
        # Avoiding relying on implicit tracking since
        # CompileMetrics may be instantiated or built in a no tracking scope.
        if not self.built:
            return []
        vars = []
        for m in self._flat_metrics + self._flat_weighted_metrics:
            if m is not None:
                vars.extend(m.variables)
        return vars

    def build(self, y_true, y_pred):
        if isinstance(y_pred, dict):
            output_names = sorted(list(y_pred.keys()))
            num_outputs = len(output_names)
        elif isinstance(y_pred, (list, tuple)):
            num_outputs = len(y_pred)
            if all(hasattr(x, "_keras_history") for x in y_pred):
                output_names = [x._keras_history.operation.name for x in y_pred]
            else:
                output_names = None
        else:
            output_names = None
            num_outputs = 1

        y_pred = nest.flatten(y_pred)
        y_true = nest.flatten(y_true)

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
                        f"Received instead:\n{argument_name}={metrics} of type {type(metrics)}"
                    )
                if not all(is_function_like(m) for m in metrics):
                    raise ValueError(
                        f"Expected all entries in the `{argument_name}` list to be "
                        f"metric objects. Received instead:\n{argument_name}={metrics}"
                    )
                flat_metrics.append(
                    MetricsList(
                        [
                            get_metric(m, y_true[0], y_pred[0])
                            for m in metrics
                            if m is not None
                        ]
                    )
                )
        else:
            if isinstance(metrics, (list, tuple)):
                if len(metrics) != len(y_pred):
                    raise ValueError(
                        "For a model with multiple outputs, "
                        f"when providing the `{argument_name}` argument as a list, "
                        "it should have as many entries as the model has outputs. "
                        f"Received:\n{argument_name}={metrics}\nof length {len(metrics)} "
                        f"whereas the model has {len(y_pred)} outputs."
                    )
                if not all(isinstance(mls, list) for mls in metrics):
                    raise ValueError(
                        "For a model with multiple outputs, "
                        f"when providing the `{argument_name}` argument as a list, "
                        "each list entry should itself be a list (the list of metrics "
                        f"corresponding to that output). Received:\n{argument_name}={metrics}"
                    )
                for mls, yt, yp in zip(metrics, y_true, y_pred):
                    if not all(is_function_like(e) for e in mls):
                        raise ValueError(
                            f"All entries in the sublists of the `{argument_name}` "
                            "list should be metric objects. "
                            f"Found the following sublist with unknown types: {mls}"
                        )
                    flat_metrics.append(
                        MetricsList(
                            [
                                get_metric(m, yt, yp)
                                for m in mls
                                if m is not None
                            ]
                        )
                    )
            elif isinstance(metrics, dict):
                if output_names is None:
                    raise ValueError(
                        f"Argument `{argument_name}` can only be provided as a dict "
                        "when the model also returns a dict of outputs. Received "
                        f"{argument_name}={metrics}"
                    )
                for name in metrics.keys():
                    if name not in output_names:
                        raise ValueError(
                            f"In the dict argument `{argument_name}`, key "
                            f"'{name}' does not correspond to any model output. "
                            f"Received:\n{argument_name}={metrics}"
                        )
                    if not isinstance(metrics[name], list):
                        raise ValueError(
                            "For a model with multiple outputs, "
                            f"when providing the `{argument_name}` argument as a dict, "
                            "each dict entry should be a list (the list of metrics "
                            "corresponding to that output). "
                            f"At key '{name}', received invalid type:\n{metrics[name]}"
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
                                [
                                    get_metric(m, yt, yp)
                                    for m in metrics[name]
                                    if m is not None
                                ]
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
        if not self.built:
            return
        for m in self._flat_metrics:
            if m:
                m.reset_state()
        for m in self._flat_weighted_metrics:
            if m:
                m.reset_state()

    def result(self):
        if not self.built:
            raise ValueError(
                "Cannot get result() since the metric has not yet been built."
            )
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
                        index = unique_name_counters[name]
                        unique_name_counters[name] += 1
                        name = f"{name}_{index}"
                    results[name] = m.result()
        return results

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError


class CompileLoss(losses_module.Loss):
    def __init__(
        self, loss, loss_weights=None, reduction="sum_over_batch_size"
    ):
        if loss_weights and not isinstance(loss_weights, (list, tuple, dict)):
            raise ValueError(
                "Expected `loss_weights` argument to be a list, tuple, or dict. "
                f"Received instead: loss_weights={loss_weights} "
                f"of type {type(loss_weights)}"
            )
        self._user_loss = loss
        self._user_loss_weights = loss_weights
        self.built = False
        super().__init__(name="compile_loss", reduction=reduction)

    def build(self, y_true, y_pred):
        if isinstance(y_pred, dict):
            output_names = sorted(list(y_pred.keys()))
            num_outputs = len(output_names)
        elif isinstance(y_pred, (list, tuple)):
            num_outputs = len(y_pred)
            if all(hasattr(x, "_keras_history") for x in y_pred):
                output_names = [x._keras_history.operation.name for x in y_pred]
            else:
                output_names = None
        else:
            output_names = None
            num_outputs = 1

        y_pred = nest.flatten(y_pred)
        loss = self._user_loss
        loss_weights = self._user_loss_weights
        flat_losses = []
        flat_loss_weights = []
        if num_outputs == 1:
            if not is_function_like(loss):
                raise ValueError(
                    f"When there is only a single output, the `loss` argument "
                    "must be a callable. "
                    f"Received instead:\nloss={loss} of type {type(loss)}"
                )
            flat_losses.append(get_loss(loss, y_true, y_pred))
            if loss_weights:
                if not isinstance(loss_weights, float):
                    raise ValueError(
                        f"When there is only a single output, the `loss_weights` argument "
                        "must be a Python float. "
                        f"Received instead:\loss_weights={loss_weights} of type {type(loss_weights)}"
                    )
                flat_loss_weights.append(loss_weights)
            else:
                flat_loss_weights.append(1.0)
        elif isinstance(loss, (list, tuple)):
            if len(loss) != len(y_pred):
                raise ValueError(
                    "For a model with multiple outputs, "
                    f"when providing the `loss` argument as a list, "
                    "it should have as many entries as the model has outputs. "
                    f"Received:\nloss={loss}\nof length {len(loss)} "
                    f"whereas the model has {len(y_pred)} outputs."
                )
            if not all(is_function_like(e) for e in loss):
                raise ValueError(
                    "For a model with multiple outputs, "
                    f"when providing the `loss` argument as a list, "
                    "each list entry should be a callable (the loss function "
                    "corresponding to that output). "
                    f"Received: loss={loss}"
                )
            flat_losses = [
                get_loss(fn, y_true, y_pred) for fn in loss if fn is not None
            ]
            if loss_weights:
                if not isinstance(loss_weights, (list, tuple)):
                    raise ValueError(
                        "If the `loss` argument is provided as a list/tuple, "
                        "the `loss_weight` argument should also be provided as a list/tuple, "
                        f"of equal length. Received: loss_weights={loss_weights}"
                    )
                if len(loss_weights) != len(y_pred):
                    raise ValueError(
                        "For a model with multiple outputs, "
                        f"when providing the `loss_weights` argument as a list, "
                        "it should have as many entries as the model has outputs. "
                        f"Received:\loss_weights={loss_weights}\nof length {len(loss_weights)} "
                        f"whereas the model has {len(y_pred)} outputs."
                    )
                if not all(isinstance(e, float) for e in loss_weights):
                    raise ValueError(
                        "For a model with multiple outputs, "
                        f"when providing the `loss_weights` argument as a list, "
                        "each list entry should be a Python float (the weighting coefficient "
                        "corresponding to the loss for that output). "
                        f"Received: loss_weights={loss_weights}"
                    )
                flat_loss_weights = list(loss_weights)
            else:
                flat_loss_weights = [1.0 for _ in loss]
        elif isinstance(loss, dict):
            if output_names is None:
                raise ValueError(
                    f"Argument `loss` can only be provided as a dict "
                    "when the model also returns a dict of outputs. Received "
                    f"loss={loss}"
                )
            for name in loss.keys():
                if name not in output_names:
                    raise ValueError(
                        f"In the dict argument `loss`, key "
                        f"'{name}' does not correspond to any model output. "
                        f"Received:\nloss={loss}"
                    )
                if not is_function_like(loss[name]):
                    raise ValueError(
                        "For a model with multiple outputs, "
                        f"when providing the `loss` argument as a dict, "
                        "each dict entry should be a callable (the loss "
                        "function corresponding to that output). "
                        f"At key '{name}', received invalid type:\n{loss[name]}"
                    )
            for name, yt, yp in zip(output_names, y_true, y_pred):
                if name in loss:
                    if loss[name]:
                        flat_losses.append(get_metric(loss[name], yt, yp))
                    else:
                        flat_losses.append(None)
                else:
                    flat_losses.append(None)
            if loss_weights:
                if not isinstance(loss_weights, dict):
                    raise ValueError(
                        "If the `loss` argument is provided as a dict, "
                        "the `loss_weight` argument should also be provided as a dict. "
                        f"Received: loss_weights={loss_weights}"
                    )
                for name in loss_weights.keys():
                    if name not in output_names:
                        raise ValueError(
                            f"In the dict argument `loss_weights`, key "
                            f"'{name}' does not correspond to any model output. "
                            f"Received:\loss_weights={loss_weights}"
                        )
                    if not isinstance(loss_weights[name], float):
                        raise ValueError(
                            "For a model with multiple outputs, "
                            f"when providing the `loss_weights` argument as a dict, "
                            "each dict entry should be a Python float (the weighting coefficient "
                            "corresponding to the loss for that output). "
                            f"At key '{name}', received invalid type:\n{loss_weights[name]}"
                        )
                for name in output_names:
                    if name in loss_weights:
                        flat_loss_weights.append(loss_weights[name])
                    else:
                        flat_loss_weights.append(1.0)
        self.flat_losses = flat_losses
        self.flat_loss_weights = flat_loss_weights
        self.built = True

    def call(self, y_true, y_pred):
        if not self.built:
            self.build(y_true, y_pred)

        y_true = nest.flatten(y_true)
        y_pred = nest.flatten(y_pred)
        loss_values = []
        for loss, y_t, y_p, w in zip(
            self.flat_losses, y_true, y_pred, self.flat_loss_weights
        ):
            if loss:
                value = w * ops.cast(loss(y_t, y_p), dtype=backend.floatx())
                loss_values.append(value)
        if loss_values:
            total_loss = sum(loss_values)
            return total_loss
        return None

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
