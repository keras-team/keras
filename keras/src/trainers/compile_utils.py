from collections import OrderedDict
from collections import namedtuple

from keras.src import losses as losses_module
from keras.src import metrics as metrics_module
from keras.src import ops
from keras.src import tree
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.losses import loss as loss_module
from keras.src.utils.naming import get_object_name
from keras.src.utils.tracking import Tracker
from keras.src.utils.tracking import no_automatic_dependency_tracking


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
        raise ValueError("Expected metric, received `None`")

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


def get_metrics_list(metrics, y_true, y_pred, output_name=None):
    if metrics is None:
        return None
    if isinstance(metrics, (list, tuple)):
        return MetricsList(
            [get_metric(m, y_true, y_pred) for m in metrics],
            output_name=output_name,
        )
    else:
        return MetricsList(
            [get_metric(metrics, y_true, y_pred)], output_name=output_name
        )


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
    @no_automatic_dependency_tracking
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
        self._flat_metrics = self._build_metrics_set(
            self._user_metrics,
            y_true,
            y_pred,
            argument_name="metrics",
        )
        self._flat_weighted_metrics = self._build_metrics_set(
            self._user_weighted_metrics,
            y_true,
            y_pred,
            argument_name="weighted_metrics",
        )
        self.built = True

    def _build_metrics_set(self, metrics, y_true, y_pred, argument_name):
        num_outputs = len(tree.flatten(y_pred))

        if not metrics:
            return [None] * num_outputs

        output_names = None
        flat_metrics = None

        if num_outputs == 1:
            # Single output, all metrics apply to it, don't use `output_names`.
            flat_metrics = [tree.flatten(metrics)]
            output_names = [None]
        elif (
            isinstance(metrics, (list, tuple))
            and self.output_names
            and len(metrics) == num_outputs
        ):
            # `metrics` is a list with one entry per output.
            # Use the output names to name the metrics.
            output_names = self.output_names
            flat_metrics = metrics

        elif isinstance(metrics, dict) and len(metrics) <= num_outputs:
            # `metrics` is a dictionary with zero or one entry per output.
            keys = set(metrics.keys())
            if (
                isinstance(y_pred, dict)
                and len(y_pred) == num_outputs
                and keys <= set(y_pred.keys())
            ):
                # If the keys match the output keys, use that, but only if the
                # outputs are a flat dictionary (not deeply nested). Note that
                # we prefer these keys over the model output names.
                # Order `output_names` by the flattening order of `y_pred`.
                output_names = list(y_pred.keys())
                if not isinstance(y_pred, OrderedDict):
                    output_names.sort()
            elif self.output_names and keys <= set(self.output_names):
                # If the keys match the Functional output names, use that.
                # The flattening order of `y_pred` is given by `output_names`.
                output_names = self.output_names

            if output_names:
                # Flatten `metrics` with the correct flattening order.
                flat_metrics = [
                    metrics[name] if name in metrics else None
                    for name in output_names
                ]

        if output_names is not None:
            try:
                # Flat case: one output or list or dict of metrics.
                return [
                    get_metrics_list(m, yt, yp, n)
                    for m, yt, yp, n in zip(
                        flat_metrics,
                        tree.flatten(y_true),
                        tree.flatten(y_pred),
                        output_names,
                    )
                ]
            except ValueError as e:
                raise ValueError(
                    f"{e}\nReceived: {argument_name}={metrics}"
                ) from e

        try:
            # Deeply nested case: `metrics` must have the structure of `y_pred`.
            # Note that the tree API wants exact matches, lists and tuples are
            # not considered equivalent, so we have to turn them all to tuples.
            tuples_y_pred = tree.lists_to_tuples(y_pred)
            return tree.flatten(
                tree.map_structure_up_to(
                    tuples_y_pred,
                    get_metrics_list,
                    tree.lists_to_tuples(metrics),
                    tree.lists_to_tuples(y_true),
                    tuples_y_pred,
                )
            )
        except (ValueError, TypeError) as e:
            # A ValueError from `get_metrics_list` or a ValueError / TypeError
            # from `tree.map_structure_up_to` for mismatched structures.
            if self.output_names:
                raise ValueError(
                    f"{e}\nInvalid `{argument_name}`. `{argument_name}` should "
                    "contain metrics objects and either be a dict or a list "
                    "matching the output names of the functional model "
                    f"{self.output_names} or match the output structure of "
                    f"the model: {tree.map_structure(lambda _: 'X', y_pred)}.\n"
                    f"Received: {argument_name}={metrics}"
                ) from e
            else:
                raise ValueError(
                    f"{e}\nInvalid `{argument_name}`. `{argument_name}` should "
                    "contain metrics objects and match the output structure of "
                    f"the model: {tree.map_structure(lambda _: 'X', y_pred)}.\n"
                    f"Received: {argument_name}={metrics}"
                ) from e

    def update_state(self, y_true, y_pred, sample_weight=None):
        if not self.built:
            self.build(y_true, y_pred)
        y_true = tree.flatten(y_true)
        y_pred = tree.flatten(y_pred)
        for m, y_t, y_p in zip(self._flat_metrics, y_true, y_pred):
            if m:
                m.update_state(y_t, y_p)
        if sample_weight is not None:
            sample_weight = tree.flatten(sample_weight)
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
    Loss = namedtuple("Loss", ["path", "loss", "loss_weights", "name"])

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

        # Use `Tracker` to track metrics for individual losses.
        self._metrics = []
        self._tracker = Tracker(
            {
                "metrics": (
                    lambda x: isinstance(x, metrics_module.Metric),
                    self._metrics,
                )
            }
        )
        self._flat_losses = None
        self._y_pred_build_structure = None
        self._y_true_build_structure = None

    @property
    def metrics(self):
        return self._metrics

    @property
    def variables(self):
        vars = []
        for m in self.metrics:
            vars.extend(m.variables)
        return vars

    def _build_nested(self, y_true, y_pred, loss, output_names, current_path):
        flat_y_pred = tree.flatten(y_pred)
        if not tree.is_nested(loss):
            _loss = loss.loss
            if _loss is None:
                return
            loss_weight = loss.weight
            resolved_loss = get_loss(_loss, y_true, y_pred)
            name_path = current_path
            if not tree.is_nested(output_names):
                if output_names is not None:
                    output_name = output_names
                else:
                    output_name = resolved_loss.name
                if len(name_path) == 0:
                    name_path = (output_name,)
                elif isinstance(name_path[-1], int):
                    name_path = name_path[:-1] + (output_name,)
            name = "/".join([str(path) for path in name_path])
            if name == "":
                if isinstance(output_names, dict):
                    flat_output_names = list(output_names.keys())
                else:
                    flat_output_names = tree.flatten(output_names)
                name = "_".join(flat_output_names)
            self._flat_losses.append(
                CompileLoss.Loss(current_path, resolved_loss, loss_weight, name)
            )
            return
        elif (
            issubclass(type(loss), (list, tuple))
            and all([not tree.is_nested(_loss) for _loss in loss])
            and len(loss) == len(flat_y_pred)
        ):
            loss = tree.pack_sequence_as(y_pred, loss)
        elif issubclass(type(loss), (list, tuple)) and not isinstance(
            y_pred, type(loss)
        ):
            for _loss in loss:
                self._build_nested(
                    y_true,
                    y_pred,
                    _loss,
                    output_names,
                    current_path,
                )
            return

        if not tree.is_nested(loss):
            return self._build_nested(
                y_true, y_pred, loss, output_names, current_path
            )

        if not isinstance(loss, type(y_pred)):
            raise KeyError(
                f"The path: {current_path} in "
                "the `loss` argument, can't be found in "
                "the model's output (`y_pred`)."
            )

        # shallow traverse the loss config
        if isinstance(loss, dict):
            iterator = loss.items()

            def key_check_fn(key, objs):
                return all(
                    [isinstance(obj, dict) and key in obj for obj in objs]
                )

        elif issubclass(type(loss), (list, tuple)):
            iterator = enumerate(loss)

            def key_check_fn(key, objs):
                return all(
                    [
                        issubclass(type(obj), (list, tuple)) and key < len(obj)
                        for obj in objs
                    ]
                )

        else:
            raise TypeError(
                f"Unsupported type {type(loss)} in the `loss` configuration."
            )

        for key, _loss in iterator:
            if _loss is None:
                continue
            if not key_check_fn(key, (y_true, y_pred)):
                raise KeyError(
                    f"The path: {current_path + (key,)} in "
                    "the `loss` argument, can't be found in "
                    "either the model's output (`y_pred`) or in the "
                    "labels (`y_true`)."
                )

            self._build_nested(
                y_true[key],
                y_pred[key],
                _loss,
                output_names[key],
                current_path + (key,),
            )

    def build(self, y_true, y_pred):
        loss = self._user_loss
        loss_weights = self._user_loss_weights
        flat_output_names = self.output_names
        if (
            self.output_names
            and isinstance(self._user_loss, dict)
            and not isinstance(y_pred, dict)
        ):
            if set(self.output_names) == set(self._user_loss.keys()):
                loss = [self._user_loss[name] for name in self.output_names]
                if isinstance(self._user_loss_weights, dict):
                    loss_weights = [
                        self._user_loss_weights[name]
                        for name in self.output_names
                    ]
            else:
                raise ValueError(
                    f"Expected keys {self.output_names} in loss dict, but "
                    f"found loss.keys()={list(self._user_loss.keys())}"
                )

        # Pytree leaf container
        class WeightedLoss:
            def __new__(cls, loss, weight):
                if loss is None:
                    return None
                return object.__new__(cls)

            def __init__(self, loss, weight):
                self.loss = loss
                self.weight = weight

        # pack the losses and the weights together
        if loss_weights is not None:
            try:
                tree.assert_same_structure(loss, loss_weights)
            except ValueError:
                flat_loss_weights = tree.flatten(loss_weights)
                if len(tree.flatten(loss)) != len(flat_loss_weights):
                    raise ValueError(
                        f"`loss_weights` must match the number of losses, "
                        f"got {len(tree.flatten(loss))} losses "
                        f"and {len(loss_weights)} weights."
                    )
                loss_weights = tree.pack_sequence_as(loss, flat_loss_weights)
            loss = tree.map_structure(
                lambda _loss, _weight: WeightedLoss(_loss, _weight),
                loss,
                loss_weights,
            )
        else:
            loss = tree.map_structure(
                lambda _loss: WeightedLoss(_loss, None), loss
            )

        self._flat_losses = []

        if (
            isinstance(loss, dict)
            and issubclass(type(y_pred), (list, tuple))
            and set(loss.keys()) == set(flat_output_names)
            and len(y_pred) == len(flat_output_names)
        ):
            y_pred = {name: y_p for name, y_p in zip(flat_output_names, y_pred)}
            y_true = {name: y_t for name, y_t in zip(flat_output_names, y_true)}
        elif (
            isinstance(loss, dict)
            and not tree.is_nested(y_pred)
            and set(loss.keys()) == set(flat_output_names)
            and len(flat_output_names) == 1
        ):
            y_pred = {
                name: y_p for name, y_p in zip(flat_output_names, [y_pred])
            }
            y_true = {
                name: y_t for name, y_t in zip(flat_output_names, [y_true])
            }

        try:
            output_names = tree.pack_sequence_as(y_pred, flat_output_names)
        except:
            inferred_flat_output_names = self._get_y_pred_output_names(y_pred)
            output_names = tree.pack_sequence_as(
                y_pred, inferred_flat_output_names
            )

        if not tree.is_nested(loss):
            loss = tree.map_structure(lambda x: loss, y_pred)

        self._build_nested(y_true, y_pred, loss, output_names, ())

        # Add `Mean` metric to the tracker for each loss.
        if len(self._flat_losses) > 1:
            for _loss in self._flat_losses:
                name = f"{_loss.name}_loss"
                self._tracker.add_to_store(
                    "metrics", metrics_module.Mean(name=name)
                )

        self._y_pred_build_structure = tree.map_structure(
            lambda x: None, y_pred
        )
        self._y_true_build_structure = tree.map_structure(
            lambda x: None, y_true
        )
        self.built = True

    def _get_y_pred_output_names(self, y_pred):
        flat_y_pred = tree.flatten(y_pred)
        if all((isinstance(x, KerasTensor) for x in flat_y_pred)):
            output_names = []
            for tensor in flat_y_pred:
                if hasattr(tensor, "_keras_history"):
                    output_names.append(tensor._keras_history.operation.name)
                else:
                    output_names.append(tensor.name)
        else:
            output_names = [None] * len(flat_y_pred)
        return output_names

    def __call__(self, y_true, y_pred, sample_weight=None):
        with ops.name_scope(self.name):
            return self.call(y_true, y_pred, sample_weight)

    def call(self, y_true, y_pred, sample_weight=None):
        def resolve_path(path, object):
            for _path in path:
                object = object[_path]
            return object

        if not tree.is_nested(y_true) and not tree.is_nested(y_pred):
            # Fast path: single output case / no loss-tracking metric.
            if not self.built:
                self.build(y_true, y_pred)
            # Although we are in the fast path, we still need to iterate
            # through the losses to prevent the torch compiler from failing.
            loss_values = []
            for path, loss_fn, loss_weight, _ in self._flat_losses:
                y_t, y_p = (
                    resolve_path(path, y_true),
                    resolve_path(path, y_pred),
                )
                if sample_weight is not None and tree.is_nested(sample_weight):
                    _sample_weight = resolve_path(path, sample_weight)
                else:
                    _sample_weight = sample_weight
                value = ops.cast(
                    loss_fn(y_t, y_p, _sample_weight), dtype=self.dtype
                )
                if loss_weight is not None:
                    value = ops.multiply(value, loss_weight)
                loss_values.append(value)
            return loss_values[0]

        try:
            tree.assert_same_structure(y_pred, y_true)
        except ValueError:
            # Check case where y_true is either flat or leaf
            if (
                not tree.is_nested(y_true)
                and hasattr(y_pred, "__len__")
                and len(y_pred) == 1
            ):
                y_true = [y_true]

            # Check case where y_pred is list/tuple and y_true is dict
            elif isinstance(y_pred, (list, tuple)) and isinstance(y_true, dict):
                if set(self.output_names) == set(y_true.keys()):
                    y_true = [y_true[name] for name in self.output_names]

            try:
                y_true = tree.pack_sequence_as(y_pred, y_true)
            except:
                # Check case where y_true has the same structure but uses
                # different (but reconcilable) container types,
                # e.g `list` vs `tuple`.
                try:
                    tree.assert_same_paths(y_true, y_pred)
                    y_true = tree.pack_sequence_as(y_pred, tree.flatten(y_true))
                except:
                    try:
                        # Check case where loss is partially defined over y_pred
                        flat_y_true = tree.flatten(y_true)
                        flat_loss = tree.flatten(self._user_loss)
                        flat_loss_non_nones = [
                            (i, loss)
                            for i, loss in enumerate(flat_loss)
                            if loss is not None
                        ]
                        if len(flat_y_true) != len(flat_loss_non_nones):
                            raise ValueError(
                                "Internal error: the number of values in "
                                f"`y_true` ({len(flat_y_true)}) must match the "
                                "number of non-None values in `loss` "
                                f"({len(flat_loss_non_nones)})."
                            )
                        y_true = [None] * len(flat_loss)
                        for y_t, (i, loss) in zip(
                            flat_y_true, flat_loss_non_nones
                        ):
                            y_true[i] = y_t
                        y_true = tree.pack_sequence_as(self._user_loss, y_true)
                    except:
                        y_true_struct = tree.map_structure(
                            lambda _: "*", y_true
                        )
                        y_pred_struct = tree.map_structure(
                            lambda _: "*", y_pred
                        )
                        raise ValueError(
                            "y_true and y_pred have different structures.\n"
                            f"y_true: {y_true_struct}\n"
                            f"y_pred: {y_pred_struct}\n"
                        )

        if not self.built:
            self.build(y_true, y_pred)

        try:
            tree.assert_same_structure(self._y_pred_build_structure, y_pred)
        except ValueError:
            y_pred = tree.pack_sequence_as(
                self._y_pred_build_structure, tree.flatten(y_pred)
            )
        try:
            tree.assert_same_structure(self._y_true_build_structure, y_true)
        except ValueError:
            y_true = tree.pack_sequence_as(
                self._y_true_build_structure, tree.flatten(y_true)
            )

        # We need to add a dummy `None` if the model has only a single output.
        metrics = [None] if len(self.metrics) == 0 else self.metrics

        # Iterate all losses in flat form.
        loss_values = []

        for (path, loss_fn, loss_weight, _), metric in zip(
            self._flat_losses, metrics
        ):
            y_t, y_p = resolve_path(path, y_true), resolve_path(path, y_pred)
            if sample_weight is not None and tree.is_nested(sample_weight):
                _sample_weight = resolve_path(path, sample_weight)
            else:
                _sample_weight = sample_weight

            value = ops.cast(
                loss_fn(y_t, y_p, _sample_weight), dtype=self.dtype
            )
            # Record *unweighted* individual losses.
            if metric:
                metric.update_state(
                    loss_module.unscale_loss_for_distribution(value),
                    sample_weight=tree.flatten(y_p)[0].shape[0],
                )
            if loss_weight is not None:
                value = ops.multiply(value, loss_weight)
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
