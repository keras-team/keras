from keras.src import losses as losses_module
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src import tree
from keras.src.utils.naming import get_object_name
from keras.src.utils.tracking import Tracker


class MetricsList(metrics_module.Metric):
    def __init__(self, metrics, name="metrics_list", output_name=None):
        super().__init__(name=name)
        self.metrics = metrics
        self.output_name = output_name

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
            metric_obj = metrics_module.BinaryAccuracy(name=str(identifier))
        elif is_sparse_categorical:
            metric_obj = metrics_module.SparseCategoricalAccuracy(
                name=str(identifier)
            )
        else:
            metric_obj = metrics_module.CategoricalAccuracy(
                name=str(identifier)
            )

    if isinstance(identifier, str):
        metric_name = identifier
    else:
        metric_name = get_object_name(metric_obj)

    if not isinstance(metric_obj, metrics_module.Metric):
        metric_obj = metrics_module.MeanMetricWrapper(metric_obj)

    metric_obj.name = metric_name
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
    def __init__(
        self,
        metrics,
        weighted_metrics,
        name="compile_metric",
        output_names=None,
    ):
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
                "Expected `weighted_metrics` argument to be a list, tuple, or "
                f"dict. Received instead: weighted_metrics={weighted_metrics} "
                f"of type {type(weighted_metrics)}"
            )
        self._user_metrics = metrics
        self._user_weighted_metrics = weighted_metrics
        self.built = False
        self.name = "compile_metrics"
        self.output_names = output_names

    @property
    def metrics(self):
        if not self.built:
            return []
        metrics = []
        for m in self._flat_metrics + self._flat_weighted_metrics:
            if isinstance(m, MetricsList):
                metrics.extend(m.metrics)
            elif m is not None:
                metrics.append(m)
        return metrics

    @property
    def variables(self):
        # Avoiding relying on implicit tracking since
        # CompileMetrics may be instantiated or built in a no tracking scope.
        if not self.built:
            return []
        vars = []
        for m in self.metrics:
            if m is not None:
                vars.extend(m.variables)
        return vars

    def build(self, y_true, y_pred):
        if self.output_names:
            output_names = self.output_names
        elif isinstance(y_pred, dict):
            output_names = sorted(list(y_pred.keys()))
        elif isinstance(y_pred, (list, tuple)):
            num_outputs = len(y_pred)
            if all(hasattr(x, "_keras_history") for x in y_pred):
                output_names = [x._keras_history.operation.name for x in y_pred]
            else:
                output_names = None
        else:
            output_names = None
            num_outputs = 1
        if output_names:
            num_outputs = len(output_names)

        y_pred = self._flatten_y(y_pred)
        y_true = self._flatten_y(y_true)

        metrics = self._user_metrics
        weighted_metrics = self._user_weighted_metrics
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
        if isinstance(metrics, dict):
            for name in metrics.keys():
                if name not in output_names:
                    raise ValueError(
                        f"In the dict argument `{argument_name}`, key "
                        f"'{name}' does not correspond to any model "
                        f"output. Received:\n{argument_name}={metrics}"
                    )
        if num_outputs == 1:
            if not metrics:
                flat_metrics.append(None)
            else:
                if isinstance(metrics, dict):
                    metrics = tree.flatten(metrics)
                if not isinstance(metrics, list):
                    metrics = [metrics]
                if not all(is_function_like(m) for m in metrics):
                    raise ValueError(
                        f"Expected all entries in the `{argument_name}` list "
                        f"to be metric objects. Received instead:\n"
                        f"{argument_name}={metrics}"
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
                        f"when providing the `{argument_name}` argument as a "
                        "list, it should have as many entries as the model has "
                        f"outputs. Received:\n{argument_name}={metrics}\nof "
                        f"length {len(metrics)} whereas the model has "
                        f"{len(y_pred)} outputs."
                    )
                for idx, (mls, yt, yp) in enumerate(
                    zip(metrics, y_true, y_pred)
                ):
                    if not isinstance(mls, list):
                        mls = [mls]
                    name = output_names[idx] if output_names else None
                    if not all(is_function_like(e) for e in mls):
                        raise ValueError(
                            f"All entries in the sublists of the "
                            f"`{argument_name}` list should be metric objects. "
                            f"Found the following sublist with unknown "
                            f"types: {mls}"
                        )
                    flat_metrics.append(
                        MetricsList(
                            [
                                get_metric(m, yt, yp)
                                for m in mls
                                if m is not None
                            ],
                            output_name=name,
                        )
                    )
            elif isinstance(metrics, dict):
                if output_names is None:
                    raise ValueError(
                        f"Argument `{argument_name}` can only be provided as a "
                        "dict when the model also returns a dict of outputs. "
                        f"Received {argument_name}={metrics}"
                    )
                for name in metrics.keys():
                    if not isinstance(metrics[name], list):
                        metrics[name] = [metrics[name]]
                    if not all(is_function_like(e) for e in metrics[name]):
                        raise ValueError(
                            f"All entries in the sublists of the "
                            f"`{argument_name}` dict should be metric objects. "
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
                                ],
                                output_name=name,
                            )
                        )
                    else:
                        flat_metrics.append(None)
        return flat_metrics

    def _flatten_y(self, y):
        if isinstance(y, dict) and self.output_names:
            result = []
            for name in self.output_names:
                if name in y:
                    result.append(y[name])
            return result
        return tree.flatten(y)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self.built:
            self.build(y_true, y_pred)
        y_true = self._flatten_y(y_true)
        y_pred = self._flatten_y(y_pred)
        for m, y_t, y_p in zip(self._flat_metrics, y_true, y_pred):
            if m:
                m.update_state(y_t, y_p)
        if sample_weight is not None:
            sample_weight = self._flatten_y(sample_weight)
            # For multi-outputs, repeat sample weights for n outputs.
            if len(sample_weight) < len(y_true):
                sample_weight = [sample_weight[0] for _ in range(len(y_true))]
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
                name = m.name
                if mls.output_name:
                    name = f"{mls.output_name}_{name}"
                if name not in unique_name_counters:
                    results[name] = m.result()
                    unique_name_counters[name] = 1
                else:
                    index = unique_name_counters[name]
                    unique_name_counters[name] += 1
                    name = f"{name}_{index}"
                    results[name] = m.result()

        for mls in self._flat_weighted_metrics:
            if not mls:
                continue
            for m in mls.metrics:
                name = m.name
                if mls.output_name:
                    name = f"{mls.output_name}_{name}"
                if name not in unique_name_counters:
                    results[name] = m.result()
                    unique_name_counters[name] = 1
                else:
                    name = f"weighted_{m.name}"
                    if mls.output_name:
                        name = f"{mls.output_name}_{name}"
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
        self,
        loss,
        loss_weights=None,
        reduction="sum_over_batch_size",
        output_names=None,
    ):
        if loss_weights and not isinstance(
            loss_weights, (list, tuple, dict, float)
        ):
            raise ValueError(
                "Expected `loss_weights` argument to be a float "
                "(single output case) or a list, tuple, or "
                "dict (multiple output case). "
                f"Received instead: loss_weights={loss_weights} "
                f"of type {type(loss_weights)}"
            )
        self._user_loss = loss
        self._user_loss_weights = loss_weights
        self.built = False
        self.output_names = output_names
        super().__init__(name="compile_loss", reduction=reduction)

        # Inferred by `y_pred` and `output_names`
        self.inferred_output_names = None

        # Use `Tracker` to track metrcis for individual losses.
        self._metrics = []
        self._tracker = Tracker(
            {
                "metrics": (
                    lambda x: isinstance(x, metrics_module.Metric),
                    self._metrics,
                )
            }
        )

    @property
    def metrics(self):
        return self._metrics

    @property
    def variables(self):
        vars = []
        for m in self.metrics:
            vars.extend(m.variables)
        return vars

    def build(self, y_true, y_pred):
        loss = self._user_loss
        loss_weights = self._user_loss_weights
        output_names = self._get_y_pred_output_names(y_pred)
        inferred_output_names = output_names or self.output_names

        if is_function_like(loss) and tree.is_nested(y_pred):
            # The model has multiple outputs but only one loss fn
            # was provided. Broadcast loss to all outputs.
            loss = tree.map_structure(lambda x: loss, y_pred)

        # Check and filter the keys.
        if isinstance(loss, dict):
            if inferred_output_names is None:
                raise ValueError(
                    "Argument `loss` can only be provided as a dict "
                    "when the model also returns a dict of outputs. "
                    f"Received loss={loss}"
                )
        filtered_y_pred_keys = []
        filtered_y_true_keys = []
        if isinstance(loss, dict):
            loss_keys = set(loss.keys())
            if inferred_output_names is not None:
                y_pred_keys = set(inferred_output_names)
                if len(loss_keys - y_pred_keys) > 0:
                    raise KeyError(
                        f"There are keys: {list(loss_keys - y_pred_keys)} in "
                        "the `loss` argument, but they can't be found in "
                        "the model's output (`y_pred`)."
                    )
                filtered_y_pred_keys.extend(list(y_pred_keys - loss_keys))
            if isinstance(y_true, dict):
                y_true_keys = set(y_true.keys())
                if len(loss_keys - y_true_keys) > 0:
                    raise KeyError(
                        f"There are keys: {list(loss_keys - y_true_keys)} in "
                        "the `loss` argument, but they can't be found in "
                        "`y` (`y_true`)."
                    )
                filtered_y_true_keys.extend(list(y_true_keys - loss_keys))
        filtered_y_pred_keys = set(filtered_y_pred_keys)
        filtered_y_true_keys = set(filtered_y_true_keys)

        # Filter unused inputs.
        y_true, y_pred = self._filter_unused_inputs(
            y_true,
            y_pred,
            filtered_y_true_keys,
            filtered_y_pred_keys,
            self.inferred_output_names,
        )

        # `loss` could be a plain function (or a `Loss` instance), a list, a
        # nested list, or a dict. However, in `call`, we want to iterate over
        # all losses, so we flatten them into a list regardless of their
        # original structure.
        flat_losses = tree.flatten(loss)
        if loss_weights is None:
            flat_loss_weights = [None] * len(flat_losses)
        else:
            flat_loss_weights = tree.flatten(loss_weights)
            for loss_weight in flat_loss_weights:
                if not isinstance(loss_weight, (int, float, type(None))):
                    raise TypeError(
                        "When providing the `loss_weights` argument, each "
                        "element should be a Python int, float (the weighting "
                        "coefficient corresponding to the loss for that "
                        "output) or `None`."
                        f"Received: loss_weights={loss_weights}"
                    )
            if len(flat_loss_weights) != len(flat_losses):
                raise ValueError(
                    "When providing the `loss_weights` argument, it should "
                    "have equal length of `loss` argument. "
                    f"Received: loss_weights length={len(flat_loss_weights)}, "
                    f"loss legnth={len(flat_losses)}"
                )

        y_true = tree.flatten(y_true)
        y_pred = tree.flatten(y_pred)
        if len(y_pred) != len(flat_losses):
            raise ValueError(
                "For a model with multiple outputs, "
                "when providing the `loss` argument as a list, "
                "it should have as many entries as the model has outputs. "
                f"Received:\nloss={loss}\nof length {len(flat_losses)} "
                f"whereas the model has {len(y_pred)} outputs."
            )

        # Get the real loss instances.
        flat_losses = [
            get_loss(identifier, _y_true, _y_pred)
            for identifier, _y_true, _y_pred in zip(flat_losses, y_true, y_pred)
        ]

        # Add `Mean` metric to the tracker for each loss.
        if len(flat_losses) > 1:
            for i, _loss in enumerate(flat_losses):
                if _loss is not None:
                    if inferred_output_names is not None and len(
                        inferred_output_names
                    ) == len(flat_losses):
                        name = inferred_output_names[i]
                    else:
                        name = _loss.name
                    name += "_loss"
                    self._tracker.add_to_store(
                        "metrics", metrics_module.Mean(name=name)
                    )

        self.flat_losses = flat_losses
        self.flat_loss_weights = flat_loss_weights
        self.filtered_y_true_keys = filtered_y_true_keys
        self.filtered_y_pred_keys = filtered_y_pred_keys
        self.inferred_output_names = inferred_output_names
        self.built = True

    def _get_y_pred_output_names(self, y_pred):
        if isinstance(y_pred, dict):
            output_names = sorted(y_pred.keys())
        else:
            y_pred = tree.flatten(y_pred)
            if all(hasattr(x, "_keras_history") for x in y_pred):
                output_names = [x._keras_history.operation.name for x in y_pred]
            else:
                output_names = None
        return output_names

    def _filter_unused_inputs(
        self,
        y_true,
        y_pred,
        filtered_y_true_keys,
        filtered_y_pred_keys,
        output_names,
    ):
        if len(filtered_y_true_keys) > 0:
            if isinstance(y_true, dict):
                for k in filtered_y_true_keys:
                    y_true.pop(k)
        if len(filtered_y_pred_keys) > 0:
            if isinstance(y_pred, dict):
                for k in filtered_y_pred_keys:
                    y_pred.pop(k)
            elif output_names is not None:
                y_pred = []
                for x, output_name in zip(tree.flatten(y_pred), output_names):
                    if output_name not in filtered_y_pred_keys:
                        y_pred.append(x)
        return y_true, y_pred

    def __call__(self, y_true, y_pred, sample_weight=None):
        with ops.name_scope(self.name):
            return self.call(y_true, y_pred, sample_weight)

    def call(self, y_true, y_pred, sample_weight=None):
        if not self.built:
            self.build(y_true, y_pred)

        # Filter unused inputs.
        y_true, y_pred = self._filter_unused_inputs(
            y_true,
            y_pred,
            self.filtered_y_true_keys,
            self.filtered_y_pred_keys,
            self.inferred_output_names,
        )

        # Flatten the inputs.
        y_true = tree.flatten(y_true)
        y_pred = tree.flatten(y_pred)
        if sample_weight is not None:
            sample_weight = tree.flatten(sample_weight)
            # For multi-outputs, repeat sample weights for n outputs.
            if len(sample_weight) < len(y_true):
                sample_weight = [sample_weight[0] for _ in range(len(y_true))]
        else:
            sample_weight = [None for _ in y_true]

        # We need to add a dummy `None` if the model has only a single output.
        metrics = [None] if len(self.metrics) == 0 else self.metrics

        # Iterate all losses in flat form.
        loss_values = []
        for loss_fn, y_t, y_p, loss_weight, sample_weight, metric in zip(
            self.flat_losses,
            y_true,
            y_pred,
            self.flat_loss_weights,
            sample_weight,
            metrics,
        ):
            if loss_fn:
                value = ops.cast(
                    loss_fn(y_t, y_p, sample_weight), dtype=self.dtype
                )
                if loss_weight is not None:
                    value = ops.multiply(value, loss_weight)
                loss_values.append(value)
                # Record individual losses.
                if metric:
                    metric.update_state(
                        value, sample_weight=tree.flatten(y_p)[0].shape[0]
                    )
        if loss_values:
            total_loss = sum(loss_values)
            return total_loss
        return None

    def get_config(self):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config):
        raise NotImplementedError
