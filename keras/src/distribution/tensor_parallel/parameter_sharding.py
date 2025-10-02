"""
Parameter-Level Sharding for Keras Tensor Parallel
This approach shards only the weights/parameters without rebuilding the model structure.
Works with ANY Keras model including KerasNLP models.
"""

import logging
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np

from keras.src.distribution.tensor_parallel.communications import (
    TensorParallelCommunicator,
)
from keras.src.distribution.tensor_parallel.config import ConfigKeras
from keras.src.distribution.tensor_parallel.state_action_keras import (
    StateActionKeras,
)

# --- 1. Define logger at the top ---
logger = logging.getLogger(__name__)


class ShardedWeight:
    def __init__(self, tensor_shard, name, trainable=True):
        # --- FIX: Local import to break circular dependency ---
        import keras

        self._variable = keras.Variable(
            initializer=tensor_shard, trainable=trainable, name=name
        )
        self.regularizer = None

    @property
    def name(self):
        """Returns the name of the underlying variable."""
        return self._variable.name

    @property
    def trainable(self):
        """Returns whether the variable is trainable."""
        return self._variable.trainable

    @property
    def shape(self):
        """Returns the shape of the variable."""
        return self._variable.shape

    @property
    def dtype(self):
        """Returns the dtype of the underlying variable."""
        return self._variable.dtype

    @property
    def variable(self):
        """Provides direct access to the underlying tf.Variable."""
        return self._variable

    def numpy(self):
        """Returns the value of the variable as a NumPy array."""
        return self._variable.numpy()

    def num_elements(self):
        """Returns the total number of elements in the tensor."""
        # --- FIX: Local import to break circular dependency ---
        import keras

        return keras.ops.size(self._variable)

    def __repr__(self):
        return (
            f"<ShardedWeight name='{self.name}' "
            f"shape={self.shape} trainable={self.trainable}>"
        )


class ParameterShardingStrategy:
    """
    Parameter-level sharding strategy that works with any Keras model.
    Instead of rebuilding the model, we shard only the weights and handle
    communication during forward/backward passes.
    """

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self.sharded_weights = {}
        self.original_weights = {}
        self.weight_mapping = {}
        self.sharded_weights_by_id = {}

    def shard_model_parameters(
        self,
        model: "Model",
        config: ConfigKeras,
        communicator: TensorParallelCommunicator,
    ) -> Tuple["Model", Set[str]]:
        """
        Shard model parameters without rebuilding the model structure.
        """
        # --- FIX: Get the Model class at runtime to break import cycle ---
        ParameterShardedModel = _define_parameter_sharded_model()

        print(f"🔧 Applying parameter-level sharding to {model.name}")

        self._store_original_weights(model)
        modified_parameters = set()

        for pattern, action in config.state_rules.items():
            if isinstance(action, StateActionKeras):
                matching_params = self._find_matching_parameters(model, pattern)

                for param_name, param in matching_params:
                    try:
                        param_id = id(param.experimental_ref())
                    except AttributeError:
                        param_id = id(param)  # Fallback

                    if param_id in self.sharded_weights_by_id:
                        self.sharded_weights[param_name] = (
                            self.sharded_weights_by_id[param_id]
                        )

                        existing_param_name = "unknown"
                        for name, shard in self.sharded_weights.items():
                            if shard is self.sharded_weights_by_id[param_id]:
                                existing_param_name = name
                                break

                        self.weight_mapping[param_name] = self.weight_mapping[
                            existing_param_name
                        ]
                        modified_parameters.add(param_name)
                        print(
                            f"   🔗 Tied {param_name} to existing shard from {existing_param_name}"
                        )
                        continue

                    sharded_param = action(param, self.rank)

                    self.sharded_weights[param_name] = sharded_param
                    self.sharded_weights_by_id[param_id] = sharded_param

                    self.weight_mapping[param_name] = {
                        "original_shape": param.shape,
                        "sharded_shape": sharded_param.shape,
                        "action": action,
                    }

                    modified_parameters.add(param_name)
                    print(
                        f"   ✅ Sharded {param_name}: {param.shape} -> {sharded_param.shape}"
                    )

        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            communicator=communicator,
            config=config,
        )

        print(
            f"🎯 Parameter sharding completed: {len(modified_parameters)} parameters sharded"
        )
        return sharded_model, modified_parameters

    def _store_original_weights(self, model: "Model"):
        """Store original weights for reference."""
        from keras.src import layers

        def find_weights_recursive(
            current_layer: layers.Layer, prefix: str = ""
        ):
            name = current_layer.name
            full_name = f"{prefix}.{name}" if prefix else name

            if hasattr(current_layer, "weights") and current_layer.weights:
                for weight in current_layer.weights:
                    cleaned_name = weight.name.split("/")[-1].split(":")[0]
                    param_name = f"{full_name}.{cleaned_name}"
                    self.original_weights[param_name] = weight.numpy()

            if hasattr(current_layer, "layers") and current_layer.layers:
                for sub_layer in current_layer.layers:
                    find_weights_recursive(sub_layer, full_name)

            for attr_name in dir(current_layer):
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue
                try:
                    attr = getattr(current_layer, attr_name)
                except Exception:
                    continue
                if isinstance(attr, layers.Layer) and attr is not current_layer:
                    find_weights_recursive(attr, full_name)
                elif isinstance(attr, (list, tuple)):
                    for item in attr:
                        if isinstance(item, layers.Layer):
                            find_weights_recursive(item, full_name)

        for layer in model.layers:
            find_weights_recursive(layer, prefix="")

    def _find_matching_parameters(
        self, model: "Model", pattern: str
    ) -> List[Tuple[str, Any]]:
        """
        Find parameters that match the given pattern using smart recursion.
        """
        from keras.src import layers

        matching_params = []
        processed_layers = set()

        def search_layer_recursive(
            current_layer: layers.Layer, prefix: str = ""
        ):
            if id(current_layer) in processed_layers:
                return
            processed_layers.add(id(current_layer))

            name = current_layer.name
            full_name = f"{prefix}.{name}" if prefix else name

            if hasattr(current_layer, "weights") and current_layer.weights:
                for weight in current_layer.weights:
                    cleaned_weight_name = weight.name.split("/")[-1].split(":")[
                        0
                    ]
                    param_name = f"{full_name}.{cleaned_weight_name}"

                    if re.match(pattern, param_name):
                        matching_params.append((param_name, weight))

            if hasattr(current_layer, "layers") and current_layer.layers:
                for sub_layer in current_layer.layers:
                    search_layer_recursive(sub_layer, full_name)

            for attr_name in dir(current_layer):
                if attr_name.startswith("__") and attr_name.endswith("__"):
                    continue

                try:
                    attr = getattr(current_layer, attr_name)
                except Exception:
                    continue

                if isinstance(attr, layers.Layer) and attr is not current_layer:
                    search_layer_recursive(attr, full_name)

                elif isinstance(attr, (list, tuple)):
                    for item in attr:
                        if isinstance(item, layers.Layer):
                            search_layer_recursive(item, full_name)

        for layer in model.layers:
            search_layer_recursive(layer, prefix="")

        return matching_params

    def get_sharded_weight(self, param_name: str) -> Optional[np.ndarray]:
        """Get sharded weight for a parameter."""
        if param_name in self.sharded_weights:
            return self.sharded_weights[param_name].numpy()
        return None

    def get_weight_info(self, param_name: str) -> Optional[Dict]:
        """Get information about a sharded weight."""
        return self.weight_mapping.get(param_name)


def _define_parameter_sharded_model():
    """
    Factory function to define and return the ParameterShardedModel class.
    This delays the import of keras.src.models.Model to break circular dependencies.
    """
    from keras.src.models import Model

    class ParameterShardedModel(Model):
        """
        Wrapper model that handles parameter sharding without rebuilding the structure.
        This preserves the original model's functionality while enabling tensor parallelism.
        """

        def __init__(
            self,
            original_model: Model,
            sharding_strategy: ParameterShardingStrategy,
            communicator: TensorParallelCommunicator,
            config: ConfigKeras,
        ):
            super().__init__()

            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self.communicator = communicator

            self._build_and_cache_weights()

            if original_model.inputs:
                self.build(original_model.inputs[0].shape)

            print("🚀 ParameterShardedModel created successfully")

        def _build_and_cache_weights(self):
            """
            Builds the list of trainable/non-trainable weights ONCE and caches it.
            """
            print("   - Building and caching the definitive weights list...")
            logger.debug("--- WEIGHT CACHE BUILDER ---")
            weights_list = []

            sharded_weight_ids = set(
                self.sharding_strategy.sharded_weights_by_id.keys()
            )

            for (
                param_name,
                sharded_tensor,
            ) in self.sharding_strategy.sharded_weights.items():
                weights_list.append(ShardedWeight(sharded_tensor, param_name))

            logger.debug(f"Added {len(weights_list)} sharded weights.")

            unsharded_count = 0
            for weight in self.original_model.weights:
                try:
                    weight_id = id(weight.experimental_ref())
                except AttributeError:
                    weight_id = id(weight)

                if weight_id not in sharded_weight_ids:
                    weights_list.append(weight)
                    unsharded_count += 1

            logger.debug(f"Added {unsharded_count} replicated weights.")

            self._weights_list = weights_list
            logger.debug("--- WEIGHT CACHE BUILD COMPLETE ---")
            logger.debug(f"Total weights in list: {len(self._weights_list)}")

        @property
        def weights(self):
            """
            Override weights property to return the cached list of sharded weights.
            """
            return self._weights_list

        def call(self, inputs, training=None, mask=None):
            """
            This is now the REAL forward pass. It must walk the model and apply communication.
            """
            from keras.src import layers

            tensor_cache = {}
            current_tensor = inputs

            for layer in self.original_model.layers:
                if isinstance(layer, layers.InputLayer):
                    continue

                if isinstance(layer, layers.Add):
                    try:
                        if "feedforward_output" in layer.name:
                            residual_source_name = layer.name.replace(
                                "feedforward_output", "self_attention_output"
                            )
                        elif "self_attention_output" in layer.name:
                            residual_source_name = layer.name.replace(
                                "self_attention_output", "input_layer_norm"
                            )
                        else:
                            residual_source_name = None

                        if (
                            residual_source_name
                            and residual_source_name in tensor_cache
                        ):
                            layer_inputs = [
                                current_tensor,
                                tensor_cache[residual_source_name],
                            ]
                        else:
                            layer_inputs = [current_tensor, current_tensor]
                    except Exception:
                        layer_inputs = [current_tensor, current_tensor]
                else:
                    layer_inputs = current_tensor

                if (
                    "attention_output" in layer.name
                    or "feedforward_output" in layer.name
                ):
                    tensor_cache[layer.name] = current_tensor

                current_tensor = layer(layer_inputs, training=training)

                layer_path = layer.path

                output_rule = None
                for pattern, rule in self.config.output_rules.items():
                    if re.search(pattern, layer_path):
                        output_rule = rule.get(0)
                        break

                if output_rule:
                    current_tensor = self._apply_communication(
                        current_tensor, layer.name, output_rule
                    )

            return current_tensor

        def _apply_communication(self, sharded_output, layer_name, rule):
            """Applies communication using the high-level communicator."""
            op_name = str(rule).lower()

            if "sum" in op_name or "allreduce" in op_name:
                logger.debug(
                    f"Applying Row-Parallel Forward (AllReduce) to {layer_name}"
                )
                return self.communicator.forward_row_parallel(
                    sharded_output, op="sum", axis_name="model"
                )

            elif "gather" in op_name:
                try:
                    dim = int(op_name.split(" ")[-1])
                except (ValueError, IndexError):
                    dim = -1
                logger.debug(
                    f"Applying Column-Parallel Forward (AllGather dim={dim}) to {layer_name}"
                )
                return self.communicator.forward_column_parallel(
                    sharded_output, dim=dim, axis_name="model"
                )

            elif hasattr(rule, "__call__"):
                logger.debug(f"Applying custom rule {rule} to {layer_name}")
                return rule(sharded_output)

            else:
                return sharded_output

        def get_config(self):
            """Get model configuration."""
            return self.original_model.get_config()

        @classmethod
        def from_config(cls, config, custom_objects=None):
            """Create model from config."""
            return cls(**config)

    return ParameterShardedModel


# --- FIX: Removed the class definition from the module level ---
# ParameterShardedModel = _define_parameter_sharded_model()


def make_parameter_sharded_model(
    module: "Model", config: ConfigKeras, rank: int, world_size: int
) -> Tuple["Model", Set[str]]:
    """
    Create a parameter-sharded version of a Keras model.
    """
    communicator = TensorParallelCommunicator(world_size=world_size, rank=rank)
    sharding_strategy = ParameterShardingStrategy(world_size, rank)

    sharded_model, modified_parameters = (
        sharding_strategy.shard_model_parameters(module, config, communicator)
    )

    return sharded_model, modified_parameters


def apply_parameter_sharding_to_existing_model(
    model: "Model", config: ConfigKeras, rank: int, world_size: int
) -> "Model":
    """
    Apply parameter sharding to an existing model without creating a new one.
    """
    print(f"🔧 Applying parameter sharding to existing model: {model.name}")

    sharding_strategy = ParameterShardingStrategy(world_size, rank)
    for pattern, action in config.state_rules.items():
        if isinstance(action, StateActionKeras):
            matching_params = sharding_strategy._find_matching_parameters(
                model, pattern
            )

            for param_name, param in matching_params:
                try:
                    param_id = id(param.experimental_ref())
                except AttributeError:
                    param_id = id(param)

                if param_id in sharding_strategy.sharded_weights_by_id:
                    sharding_strategy.sharded_weights[param_name] = (
                        sharding_strategy.sharded_weights_by_id[param_id]
                    )
                    existing_param_name = next(
                        k
                        for k, v in sharding_strategy.sharded_weights.items()
                        if v
                        is sharding_strategy.sharded_weights_by_id[param_id]
                    )
                    sharding_strategy.weight_mapping[param_name] = (
                        sharding_strategy.weight_mapping[existing_param_name]
                    )
                    print(
                        f"   🔗 Tied {param_name} to existing shard from {existing_param_name}"
                    )
                    continue

                sharded_param = action(param, rank)

                sharding_strategy.sharded_weights[param_name] = sharded_param
                sharding_strategy.sharded_weights_by_id[param_id] = (
                    sharded_param
                )

                sharding_strategy.weight_mapping[param_name] = {
                    "original_shape": param.shape,
                    "sharded_shape": sharded_param.shape,
                    "action": action,
                }

                print(
                    f"   ✅ Sharded {param_name}: {param.shape} -> {sharded_param.shape}"
                )

    model._tensor_parallel_sharding = sharding_strategy

    print("🎯 Parameter sharding applied to existing model")
    return model
