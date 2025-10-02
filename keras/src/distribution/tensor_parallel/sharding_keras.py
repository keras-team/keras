from typing import Any
from typing import Collection
from typing import Dict
from typing import List
from typing import Sequence

from keras.src.distribution.tensor_parallel.config import ConfigKeras


class ShardedKeras:
    """
    Manages sharded parameters for Keras models.
    """

    def __init__(
        self,
        model_shards,
        replicated_param_names: Collection[str],
        tensor_parallel_config: ConfigKeras,
        devices: Sequence[str],
        output_device_index: int,
    ):
        """
        Initialize the sharding manager.

        Args:
            model_shards: List of model shards
            replicated_param_names: Names of parameters that are replicated
            tensor_parallel_config: Tensor parallel configuration
            devices: List of device IDs
            output_device_index: Index of the output device
        """
        self.model_shards = model_shards
        self.replicated_param_names = set(replicated_param_names)
        self.tensor_parallel_config = tensor_parallel_config
        self.devices = devices
        self.output_device_index = output_device_index

    def get_shard_parameters(self, shard_index: int) -> Dict[str, Any]:
        """
        Get parameters for a specific shard.

        Args:
            shard_index: Index of the shard

        Returns:
            Dictionary of parameter names to values
        """
        if shard_index >= len(self.model_shards):
            raise ValueError(f"Shard index {shard_index} out of range")

        shard = self.model_shards[shard_index]
        params = {}

        for layer in shard.layers:
            name = layer.name
            if hasattr(layer, "weights") and layer.weights:
                for i, weight in enumerate(layer.weights):
                    param_name = f"{name}.weight_{i}"
                    params[param_name] = weight

        return params

    def get_all_parameters(self) -> List[Dict[str, Any]]:
        """
        Get parameters from all shards.

        Returns:
            List of parameter dictionaries for each shard
        """
        return [
            self.get_shard_parameters(i) for i in range(len(self.model_shards))
        ]

    def apply_sharding(self):
        """
        Apply sharding to the model parameters.
        """
        pass

    def unshard_parameters(self):
        """
        Unshard parameters back to their original form.
        """
        pass