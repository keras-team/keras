import functools
import re

from keras import Variable
from keras import device
from keras.src import layers
from keras.src import ops
from keras.src.backend import distribution_lib
from keras.src.distribution.tensor_parallel.tensor_layout import (
    split_tensor_for_parallelism,
)


class ShardedWeight:
    """Wrapper for a sharded Keras Variable providing a consistent interface.

    Attributes:
        regularizer: Placeholder for weight regularization logic.
    """

    def __init__(self, tensor_shard, name, trainable=True, device_id=None):
        """Initializes the ShardedWeight.

        Args:
            tensor_shard: The tensor slice belonging to this rank.
            name: The original variable name.
            trainable: Boolean indicating if the weight is trainable.
            device_id: Device identifier string for the variable placement.
        """
        dev_name = device_id if device_id else "UNKNOWN_DEVICE"

        safe_name = name.replace("/", "_").replace(":", "_")
        with device(dev_name):
            self._variable = Variable(
                initializer=tensor_shard, trainable=trainable, name=safe_name
            )
        self.regularizer = None

    @property
    def name(self):
        """Returns variable name."""
        return self._variable.name

    @property
    def trainable(self):
        """Returns trainability."""
        return self._variable.trainable

    @property
    def shape(self):
        """Returns sharded shape."""
        return self._variable.shape

    @property
    def dtype(self):
        """Returns data type."""
        return self._variable.dtype

    @property
    def variable(self):
        """Returns the internal Variable object."""
        return self._variable

    @property
    def value(self):
        """Returns the variable value."""
        return self._variable.value

    def numpy(self):
        """Returns numpy representation."""
        return self._variable.numpy()

    def num_elements(self):
        """Returns total element count."""
        return ops.size(self._variable)


class ParameterShardingStrategy:
    """Handles parameter-level sharding and configuration normalization."""

    def __init__(self, device_count, rank):
        """Initializes the strategy.

        Args:
            device_count: Total devices in the model axis.
            rank: Current device rank.
        """
        self.device_count = device_count
        self.rank = rank
        self.sharded_weights = {}
        self.original_weights = {}
        self.weight_mapping = {}
        self.sharded_weights_by_id = {}
        self.param_path_map = {}
        self._id_to_param_map = {}

    def shard_model_parameters(self, model, config, device_id):
        """Orchestrates the sharding of model parameters.

        Args:
            model: Original model instance.
            config: LayoutMap with sharding rules.
            device_id: Targeted device identifier.

        Returns:
            Tuple of (sharded_model, modified_parameters_set).
        """
        ParameterShardedModel = _define_parameter_sharded_model()

        self.param_path_map = {w.path: w for w in model.weights}
        for w in model.weights:
            ref_fn = getattr(w, "experimental_ref", None)
            ref = ref_fn() if ref_fn else w
            self._id_to_param_map[id(ref)] = (w.path, w)

        for layer in model._flatten_layers(recursive=True, include_self=True):
            if "Embedding" in layer.__class__.__name__:
                for attr in [
                    "embeddings",
                    "position_embeddings",
                    "_embeddings",
                ]:
                    var = getattr(layer, attr, None)
                    if var is not None:
                        vrf_fn = getattr(var, "experimental_ref", None)
                        vid = id(vrf_fn()) if vrf_fn else id(var)
                        if vid not in config.state_rules:
                            config.state_rules[vid] = functools.partial(
                                split_tensor_for_parallelism,
                                device_count=self.device_count,
                                dim=1,
                            )

        # Normalize configuration keys to string paths
        norm_rules = {}
        for pattern, action in list(config.state_rules.items()):
            if isinstance(pattern, int) and pattern in self._id_to_param_map:
                path, _ = self._id_to_param_map[pattern]
                norm_rules[path] = action
                del config.state_rules[pattern]
        config.state_rules.update(norm_rules)

        self._store_original_weights(model)
        modified = set()

        for pattern, action in config.state_rules.items():
            if callable(action):
                for name, param in self._find_matching_parameters(
                    model, pattern
                ):
                    pr_fn = getattr(param, "experimental_ref", None)
                    pid = id(pr_fn()) if pr_fn else id(param)

                    if pid in self.sharded_weights_by_id:
                        self.sharded_weights[name] = self.sharded_weights_by_id[
                            pid
                        ]
                        modified.add(name)
                        continue

                    shard = action(param, self.rank)
                    self.sharded_weights[name] = shard
                    self.sharded_weights_by_id[pid] = shard
                    self.weight_mapping[name] = {
                        "original": param.shape,
                        "sharded": shard.shape,
                    }
                    modified.add(name)

        sharded_model = ParameterShardedModel(model, self, config, device_id)
        return sharded_model, modified

    def _store_original_weights(self, model):
        """Backs up weights as numpy arrays."""
        for w in model.weights:
            if hasattr(w, "numpy"):
                self.original_weights[w.path] = w.numpy()

    def _find_matching_parameters(self, model, pattern):
        """Matches a pattern to model weights."""
        if isinstance(pattern, int):
            return (
                [self._id_to_param_map[pattern]]
                if pattern in self._id_to_param_map
                else []
            )
        if not isinstance(pattern, str):
            return []
        if pattern in self.param_path_map:
            return [(pattern, self.param_path_map[pattern])]
        suffix = "/" + pattern
        return [
            (p, w) for p, w in self.param_path_map.items() if p.endswith(suffix)
        ]


def _define_parameter_sharded_model():
    """Defines the wrapper model class dynamically."""
    from keras.src.models import Functional
    from keras.src.models import Model

    class ParameterShardedModel(Model):
        """Wrapper model implementing distributed forward pass logic."""

        def __init__(
            self, original_model, sharding_strategy, config, device_id
        ):
            """Initializes the model and caches mappings."""
            super().__init__(name=original_model.name)
            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self._device = device_id

            if not self.original_model.built and self.original_model.inputs:
                self.original_model.build(self.original_model.inputs[0].shape)

            self._build_and_cache_weights()
            self._layer_rules = self._cache_layer_rules()

        def _cache_layer_rules(self):
            """Pre-maps layers to communication rules."""
            rules = {}
            for layer in self.original_model.layers:
                lp = getattr(layer, "path", layer.name)
                if lp is None:
                    continue
                lp_s = str(lp)
                for pat, rule in self.config.output_rules.items():
                    if (
                        pat == lp_s
                        or lp_s.endswith("/" + pat)
                        or (isinstance(pat, str) and re.search(pat, lp_s))
                    ):
                        rules[id(layer)] = (
                            rule.get(0) if isinstance(rule, dict) else rule
                        )
                        break
            return rules

        def _build_and_cache_weights(self):
            """Merges sharded and original weights into a definitive list."""
            ws, sids = (
                [],
                set(self.sharding_strategy.sharded_weights_by_id.keys()),
            )
            for name, shard in self.sharding_strategy.sharded_weights.items():
                ws.append(ShardedWeight(shard, name, device_id=self._device))
            for w in self.original_model.weights:
                rf_fn = getattr(w, "experimental_ref", None)
                ref = rf_fn() if rf_fn else w
                if id(ref) not in sids:
                    ws.append(w)
            self._weights_list = ws

        @property
        def weights(self):
            """Returns model weights."""
            return self._weights_list

        def compute_output_shape(self, input_shape):
            """Returns original output shape."""
            return self.original_model.compute_output_shape(input_shape)

        def compute_output_spec(self, *args, **kwargs):
            """Returns original output spec."""
            if args:
                return self.original_model.compute_output_spec(args[0])
            return self.original_model.compute_output_spec(**kwargs)

        def call(self, inputs, training=None, mask=None):
            """Optimized forward pass using pre-calculated mappings."""
            cache = {}
            model_inps = self.original_model.inputs
            if isinstance(inputs, dict):
                for t in model_inps:
                    nm = getattr(t, "name", "")
                    val = inputs.get(nm, inputs.get(nm.split(":")[0]))
                    if val is not None:
                        cache[id(t)] = val
                        if nm:
                            cache[nm] = val
                            cache[nm.split(":")[0]] = val
            else:
                in_l = inputs if isinstance(inputs, (list, tuple)) else [inputs]
                for i, t in enumerate(model_inps):
                    if i < len(in_l):
                        val = in_l[i]
                        cache[id(t)] = val
                        nm = getattr(t, "name", "")
                        if nm:
                            cache[nm] = val
                            cache[nm.split(":")[0]] = val

            for layer in self.original_model.layers:
                if isinstance(layer, layers.InputLayer):
                    continue

                nodes = getattr(layer, "_inbound_nodes", [])
                if not nodes:
                    continue
                node = nodes[0]
                recon = []
                for sym in node.input_tensors:
                    val = cache.get(id(sym))
                    if val is None:
                        snm = getattr(sym, "name", "")
                        val = cache.get(snm, cache.get(snm.split(":")[0]))
                    if val is not None:
                        recon.append(val)

                if len(recon) == 0:
                    l_in = inputs
                elif isinstance(layer, (Functional, Model)):
                    nms = getattr(layer, "input_names", None)
                    if not nms:
                        in_t = getattr(layer, "inputs", [])
                        nms = [
                            getattr(x, "name", "").split(":")[0] for x in in_t
                        ]
                    l_in = (
                        dict(zip(nms, recon))
                        if nms and len(recon) == len(nms)
                        else recon
                    )
                else:
                    l_in = recon[0] if len(recon) == 1 else recon

                kwargs = {"training": training} if training is not None else {}
                node_args = getattr(node, "arguments", None)
                if node_args:
                    for k, v in (
                        getattr(node_args, "kwargs", {}) or {}
                    ).items():
                        if k != "training":
                            kwargs[k] = v

                out = layer(l_in, **kwargs)

                rule = self._layer_rules.get(id(layer))
                if rule:
                    out = (
                        rule(out)
                        if callable(rule)
                        else self._comm(out, layer.name, rule)
                    )

                for n in nodes:
                    syms = getattr(n, "output_tensors", [])
                    vals = out if isinstance(out, (list, tuple)) else [out]
                    for o_idx, (s, v) in enumerate(zip(syms, vals)):
                        cache[id(s)] = v
                        cache[("node", layer.name, 0, o_idx)] = v
                        snm = getattr(s, "name", "")
                        if snm:
                            cache[snm] = v
                            cache[snm.split(":")[0]] = v

            results = []
            for s_out in self.original_model.outputs:
                val = cache.get(id(s_out))
                if val is None:
                    hist = getattr(s_out, "_keras_history", None)
                    if hist and len(hist) >= 3:
                        val = cache.get(
                            (
                                "node",
                                getattr(hist[0], "name", ""),
                                hist[1],
                                hist[2],
                            )
                        )
                if val is None:
                    raise RuntimeError(f"Missing runtime value for: {s_out}.")
                results.append(val)

            return results[0] if len(results) == 1 else results

        def _comm(self, val, name, rule):
            """Internal communication wrapper."""
            if "sum" in rule or "allreduce" in rule:
                return distribution_lib.all_reduce(
                    val, op="sum", axis_name="model"
                )
            if "gather" in rule:
                parts = rule.split(" ")
                dim = (
                    int(parts[-1])
                    if len(parts) > 1 and parts[-1].lstrip("-").isdigit()
                    else -1
                )
                return distribution_lib.all_gather(
                    val, axis=dim, axis_name="model"
                )
            return val

        def get_config(self):
            """Returns model configuration."""
            return self.original_model.get_config()

        @classmethod
        def from_config(cls, config, custom_objects=None):
            """Rebuilds sharded model from config."""
            return cls(**config)

    return ParameterShardedModel


def make_parameter_sharded_model(module, config, rank, device_count, device_id):
    """Factory function to create a parameter-sharded model."""
    strat = ParameterShardingStrategy(device_count, rank)
    return strat.shard_model_parameters(module, config, device_id)
