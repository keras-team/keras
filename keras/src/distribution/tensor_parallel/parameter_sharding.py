import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from keras.src.distribution.tensor_parallel.communications import (
    TensorParallelCommunicator,
)
from keras.src.distribution.tensor_parallel.config import ConfigKeras
from keras.src.distribution.tensor_parallel.state_action_keras import (
    StateActionKeras,
)


class ShardedWeight:
    def __init__(self, tensor_shard, name, trainable=True):
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
        model,
        config: ConfigKeras,
        communicator: TensorParallelCommunicator,
        device_id: Any,
    ):
        """
        Shard model parameters without rebuilding the model structure.
        """
        ParameterShardedModel = _define_parameter_sharded_model()

        self._store_original_weights(model)
        modified_parameters = set()

        for pattern, action in config.state_rules.items():
            if isinstance(action, StateActionKeras):
                matching_params = self._find_matching_parameters(model, pattern)

                for param_name, param in matching_params:
                    try:
                        param_id = id(param.experimental_ref())
                    except AttributeError:
                        param_id = id(param)

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

        sharded_model = ParameterShardedModel(
            original_model=model,
            sharding_strategy=self,
            communicator=communicator,
            config=config,
            device_id=device_id,
        )

        return sharded_model, modified_parameters

    def _store_original_weights(self, model):
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
        self, model, pattern: str
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

        search_layer_recursive(model, prefix="")

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
    """
    from keras.src.models import Model

    class ParameterShardedModel(Model):
        def __init__(
            self,
            original_model: Model,
            sharding_strategy: ParameterShardingStrategy,
            communicator: TensorParallelCommunicator,
            config: ConfigKeras,
            device_id: Any,
        ):
            super().__init__()

            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self.communicator = communicator
            self._device = device_id

            self._build_and_cache_weights()

            if original_model.inputs:
                self.build(original_model.inputs[0].shape)

        @property
        def device(self):
            return self._device

        def train_step(self, data):
            """
            Custom train_step for the parameter-sharded model.

            This override includes a gradient synchronization (all-reduce) step,
            which is essential for the backward pass in tensor parallelism.
            """
            import tensorflow as tf

            import keras

            x, y, sample_weight = keras.utils.unpack_x_y_sample_weight(data)

            with tf.GradientTape() as tape:
                y_pred = self(x, training=True)
                loss = self.compute_loss(
                    x=x, y=y, y_pred=y_pred, sample_weight=sample_weight
                )

            trainable_vars = self.trainable_variables
            gradients = tape.gradient(loss, trainable_vars)

            synced_gradients = self.communicator.all_reduce(
                gradients, op="sum", axis_name="model"
            )
            self.optimizer.apply_gradients(
                zip(synced_gradients, trainable_vars)
            )

            self.compiled_metrics.update_state(y, y_pred, sample_weight)

            return {m.name: m.result() for m in self.metrics}

        def _build_and_cache_weights(self):
            weights_list = []

            sharded_weight_ids = set(
                self.sharding_strategy.sharded_weights_by_id.keys()
            )

            for (
                param_name,
                sharded_tensor,
            ) in self.sharding_strategy.sharded_weights.items():
                weights_list.append(ShardedWeight(sharded_tensor, param_name))

            unsharded_count = 0
            for weight in self.original_model.weights:
                try:
                    weight_id = id(weight.experimental_ref())
                except AttributeError:
                    weight_id = id(weight)

                if weight_id not in sharded_weight_ids:
                    weights_list.append(weight)
                    unsharded_count += 1

            self._weights_list = weights_list

        @property
        def weights(self):
            return self._weights_list

        def call(self, inputs, training=None, mask=None):
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
                return self.communicator.forward_row_parallel(
                    sharded_output, op="sum", axis_name="model"
                )

            elif "gather" in op_name:
                try:
                    dim = int(op_name.split(" ")[-1])
                except (ValueError, IndexError):
                    dim = -1
                return self.communicator.forward_column_parallel(
                    sharded_output, dim=dim, axis_name="model"
                )

            elif hasattr(rule, "__call__"):
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


def make_parameter_sharded_model(
    module, config: ConfigKeras, rank: int, world_size: int, device_id: Any
):
    """
    Create a parameter-sharded version of a Keras model.
    """
    communicator = TensorParallelCommunicator(world_size=world_size, rank=rank)
    sharding_strategy = ParameterShardingStrategy(world_size, rank)

    sharded_model, modified_parameters = (
        sharding_strategy.shard_model_parameters(
            module, config, communicator, device_id
        )
    )

    return sharded_model, modified_parameters


def apply_parameter_sharding_to_existing_model(
    model, config: ConfigKeras, rank: int, world_size: int
):
    """
    Apply parameter sharding to an existing model without creating a new one.
    """

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

    model._tensor_parallel_sharding = sharding_strategy

    return model
