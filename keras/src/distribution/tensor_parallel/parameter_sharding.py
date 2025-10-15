import re

from keras.src.backend import distributed_backend


class ShardedWeight:
    """A wrapper for a sharded Keras Variable to provide a consistent interface.

    This class wraps a tensor shard in a Keras Variable, making it compatible
    with the Keras ecosystem. It exposes common variable properties like name,
    shape, and trainable status.
    """

    def __init__(self, tensor_shard, name, trainable=True):
        """Initializes the ShardedWeight.

        Args:
            tensor_shard: The tensor piece (shard) to be managed by this weight.
            name (str): The name for the underlying Keras Variable.
            trainable (bool, optional): Whether the variable is trainable.
                Defaults to True.
        """
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
        """Provides direct access to the underlying Keras Variable."""
        return self._variable

    @property
    def value(self):
        """Returns the value of the underlying variable."""
        return self._variable.value

    def numpy(self):
        """Returns the value of the variable as a NumPy array."""
        return self._variable.numpy()

    def num_elements(self):
        """Returns the total number of elements in the tensor."""
        import keras

        return keras.ops.size(self._variable)

    def __repr__(self):
        """Returns a string representation of the ShardedWeight."""
        return (
            f"<ShardedWeight name='{self.name}' "
            f"shape={self.shape} trainable={self.trainable}>"
        )


class ParameterShardingStrategy:
    """Implements parameter-level sharding for a Keras model.

    This strategy shards a model's weights according to a provided configuration
    without altering the model's architecture. It identifies weights
    that match specific patterns, applies sharding actions to them, and stores
    the mapping between original and sharded weights.
    """

    def __init__(self, world_size, rank):
        """Initializes the ParameterShardingStrategy.

        Args:
            world_size (int): The total number of devices in distributed setup.
            rank (int): The rank of the current device.
        """
        self.world_size = world_size
        self.rank = rank
        self.sharded_weights = {}
        self.original_weights = {}
        self.weight_mapping = {}
        self.sharded_weights_by_id = {}

    def shard_model_parameters(self, model, config, device_id):
        """Shards model parameters based on a layout configuration.

        This method iterates through the rules in configuration, finds matching
        parameters in the model, and applies the specified sharding action. It
        then returns a `ParameterShardedModel` wrapper that uses these sharded
        weights.

        Args:
            model (keras.Model): The Keras model to be sharded.
            config (LayoutMap): A configuration object specifying which weights
                to shard and how.
            device_id: The device identifier for the current process.

        Returns:
            tuple: A tuple containing:
                - ParameterShardedModel: The wrapped model with sharded
                    parameters.
                - set: A set of names of the parameters that were modified.
        """
        ParameterShardedModel = _define_parameter_sharded_model()

        self._store_original_weights(model)
        modified_parameters = set()

        for pattern, action in config.state_rules.items():
            if hasattr(action, "__call__"):
                matching_params = self._find_matching_parameters(model, pattern)

                for param_name, param in matching_params:
                    if hasattr(param, "experimental_ref"):
                        param_id = id(param.experimental_ref())
                    else:
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
            config=config,
            device_id=device_id,
        )

        return sharded_model, modified_parameters

    def _store_original_weights(self, model):
        """Recursively finds and stores the original weights of a model."""
        from keras.src import layers

        def find_weights_recursive(current_layer, prefix=""):
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

                attr = getattr(current_layer, attr_name)

                if isinstance(attr, layers.Layer) and attr is not current_layer:
                    find_weights_recursive(attr, full_name)
                elif isinstance(attr, (list, tuple)):
                    for item in attr:
                        if isinstance(item, layers.Layer):
                            find_weights_recursive(item, full_name)

        for layer in model.layers:
            find_weights_recursive(layer, prefix="")

    def _find_matching_parameters(self, model, pattern):
        """Finds model parameters that match a given regex pattern.

        Args:
            model (keras.Model): The model to search within.
            pattern (str): The regex pattern to match against parameter names.

        Returns:
            list: A list of tuples, where each tuple contains the full parameter
                  name and the corresponding weight object.
        """
        from keras.src import layers

        matching_params = []
        processed_layers = set()

        def search_layer_recursive(current_layer, prefix=""):
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

                attr = getattr(current_layer, attr_name)

                if isinstance(attr, layers.Layer) and attr is not current_layer:
                    search_layer_recursive(attr, full_name)
                elif isinstance(attr, (list, tuple)):
                    for item in attr:
                        if isinstance(item, layers.Layer):
                            search_layer_recursive(item, full_name)

        search_layer_recursive(model, prefix="")
        return matching_params


def _define_parameter_sharded_model():
    """Factory function to define and return the ParameterShardedModel class.

    This approach avoids circular import dependencies by defining the class
    that inherits from `keras.src.models.Model` inside a function.

    Returns:
        The ParameterShardedModel class definition.
    """
    from keras.src.models import Model

    class ParameterShardedModel(Model):
        """A wrapper model that manages sharded parameters for tensor
        parallelism.

        This model wraps an existing Keras model, preserving its original
        architecture. It overrides the `weights` property and the `call` method
        to handle sharded weights and insert the necessary communication
        collectives (e.g., AllReduce, AllGather) during the forward pass.
        """

        def __init__(
            self, original_model, sharding_strategy, config, device_id
        ):
            """Initializes the ParameterShardedModel.

            Args:
                original_model: The original, unsharded Keras model.
                sharding_strategy: The strategy object
                    that contains the sharded weights and mappings.
                config (LayoutMap): The sharding configuration.
                device_id: The device identifier for the current process.
            """
            super().__init__()
            self.original_model = original_model
            self.sharding_strategy = sharding_strategy
            self.config = config
            self._device = device_id
            self._build_and_cache_weights()
            if original_model.inputs:
                self.build(original_model.inputs[0].shape)

        @property
        def device(self):
            """Returns the device ID associated with this model shard."""
            return self._device

        def _build_and_cache_weights(self):
            """Constructs and caches the definitive list of model weights.

            This method combines newly created `ShardedWeight` objects with any
            original weights that were not sharded (i.e., replicated weights).
            This combined list is then cached to be returned by the `weights`
            property, ensuring the optimizer sees all trainable parameters.
            """
            weights_list = []
            sharded_weight_ids = set(
                self.sharding_strategy.sharded_weights_by_id.keys()
            )

            for (
                param_name,
                sharded_tensor,
            ) in self.sharding_strategy.sharded_weights.items():
                weights_list.append(ShardedWeight(sharded_tensor, param_name))

            for weight in self.original_model.weights:
                if hasattr(weight, "experimental_ref"):
                    weight_id = id(weight.experimental_ref())
                else:
                    weight_id = id(weight)

                if weight_id not in sharded_weight_ids:
                    weights_list.append(weight)

            self._weights_list = weights_list

        @property
        def weights(self):
            """Overrides the base property to return the cached list of weights.

            This list includes both the custom `ShardedWeight` objects and any
            unsharded (replicated) weights from the original model.
            """
            return self._weights_list

        def call(self, inputs, training=None, mask=None):
            """Executes the forward pass of the model with sharded parameters.

            This method manually reconstructs the forward pass of original
            model's computation graph. It propagates tensors from one layer to
            the next, and after layer's computation, it checks if communication
            collective needs to be applied to the output tensor based on the
            sharding configuration.

            Args:
                inputs: Input tensor(s).
                training (bool, optional): Indicates whether the model is in
                    training mode. Defaults to None.
                mask: Mask tensor(s). Defaults to None.

            Returns:
                The final output tensor(s) of the model.
            """
            from keras.src import layers

            tensor_cache = {}

            if isinstance(inputs, dict):
                for inp_tensor in self.original_model.inputs:
                    tensor_cache[id(inp_tensor)] = inputs[inp_tensor.name]
            else:
                tensor_cache[id(self.original_model.inputs[0])] = inputs

            for layer in self.original_model.layers:
                if isinstance(layer, layers.InputLayer):
                    continue

                layer_inputs = []
                for node in layer._inbound_nodes:
                    for symbolic_input_tensor in node.input_tensors:
                        layer_inputs.append(
                            tensor_cache[id(symbolic_input_tensor)]
                        )

                if len(layer_inputs) == 1:
                    layer_inputs = layer_inputs[0]

                current_tensor = layer(layer_inputs, training=training)
                tensor_cache[id(layer.output)] = current_tensor

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
                    tensor_cache[id(layer.output)] = current_tensor

            final_outputs = []
            for symbolic_output in self.original_model.outputs:
                final_outputs.append(tensor_cache[id(symbolic_output)])

            if len(final_outputs) == 1:
                return final_outputs[0]
            return final_outputs

        def _apply_communication(self, sharded_output, layer_name, rule_str):
            """Applies a collective communication operation to a tensor.

            This method uses the distributed backend to perform operations like
            AllReduce (for summing partial results in row-parallel layouts) or
            AllGather (for combining results in column-parallel layouts).

            Args:
                sharded_output: The tensor to apply the communication op to.
                layer_name (str): The name of the layer producing the output.
                rule_str (str): A string from config describing the operation
                    (e.g., 'allreduce sum', 'allgather -1').

            Returns:
                The tensor after the communication operation has been applied.
            """
            comm_ops = distributed_backend.get_communication_ops()

            if "sum" in rule_str or "allreduce" in rule_str:
                return comm_ops["all_reduce"](
                    sharded_output, op="sum", axis_name="model"
                )
            elif "gather" in rule_str:
                parts = rule_str.split(" ")
                last_part = parts[-1]
                if len(parts) > 1 and (
                    last_part.isdigit()
                    or (last_part.startswith("-") and last_part[1:].isdigit())
                ):
                    dim = int(last_part)
                else:
                    dim = -1
                return comm_ops["all_gather"](
                    sharded_output, axis=dim, axis_name="model"
                )
            else:
                return sharded_output

        def get_config(self):
            """Returns the configuration of the original model."""
            return self.original_model.get_config()

        @classmethod
        def from_config(cls, config, custom_objects=None):
            """Creates a model from its configuration."""
            return cls(**config)

    return ParameterShardedModel


def make_parameter_sharded_model(module, config, rank, world_size, device_id):
    """Creates a parameter-sharded version of a Keras model.

    This is the main entry point for applying parameter sharding. It initializes
    the sharding strategy and uses it to transform the given model.

    Args:
        module (keras.Model): The Keras model to shard.
        config (LayoutMap): The configuration defining the sharding rules.
        rank (int): The rank of the current device.
        world_size (int): The total number of devices.
        device_id: The identifier for the current device.

    Returns:
        tuple: A tuple containing:
            - ParameterShardedModel: The new, sharded model wrapper.
            - set: A set of names of the parameters that were sharded.
    """
    sharding_strategy = ParameterShardingStrategy(world_size, rank)
    sharded_model, modified_parameters = (
        sharding_strategy.shard_model_parameters(module, config, device_id)
    )
    return sharded_model, modified_parameters
