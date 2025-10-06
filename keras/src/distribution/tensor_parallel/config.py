import dataclasses
from typing import Any
from typing import Dict
from typing import Sequence

from keras.src.distribution.tensor_parallel.communications import AllGatherKeras
from keras.src.distribution.tensor_parallel.communications import AllReduceKeras
from keras.src.distribution.tensor_parallel.communications import BroadcastKeras


def _create_ops_from_rules(
    rules: Dict[str, Any], world_size: int
) -> Dict[str, Any]:
    """Parses a rules dictionary to create collective op instances.

    This function iterates through a dictionary of rules. If it encounters a
    string identifier for a collective operation (e.g., "sum", "mean",
    "gather -1"), it replaces it with an instantiated Keras collective op
    object. Other values are passed through unchanged.

    Args:
        rules (Dict[str, Any]): The dictionary of rules to process.
        world_size (int): The total number of devices in the distributed setup.

    Returns:
        Dict[str, Any]: A new dictionary with string identifiers replaced by
        collective op instances.
    """
    processed_rules = {}
    for pattern, actions in rules.items():
        if not isinstance(actions, dict):
            processed_rules[pattern] = actions
            continue

        processed_rules[pattern] = {}
        for key, action in actions.items():
            if not isinstance(action, str):
                processed_rules[pattern][key] = action
                continue

            if action == "sum":
                op = AllReduceKeras(world_size, op="sum")
            elif action == "mean":
                op = AllReduceKeras(world_size, op="mean")
            elif action.startswith("gather"):
                dim = int(action.split(" ")[1]) if " " in action else -1
                op = AllGatherKeras(world_size, dim=dim)
            elif action == "broadcast":
                op = BroadcastKeras(world_size)
            else:
                op = action
            processed_rules[pattern][key] = op
    return processed_rules


@dataclasses.dataclass
class ConfigKeras:
    """A dataclass holding configuration for tensor parallelism in Keras.

    Attributes:
        state_rules (Dict[str, Any]): Rules governing how model state variables
            (e.g., weights) are handled across devices.
        output_rules (Dict[str, Any]): Rules governing how layer outputs are
            handled. These rules are processed by `create_collective_ops` to
            instantiate the necessary communication operations.
    """

    state_rules: Dict[str, Any]
    output_rules: Dict[str, Any]

    def create_collective_ops(self, devices: Sequence[str]):
        """Creates a new ConfigKeras instance with collective ops.

        This method processes the `output_rules` of the current instance,
        replacing string-based rule definitions with actual collective
        communication op objects required for distributed execution.

        Args:
            devices (Sequence[str]): A sequence of device strings (e.g.,
                ["/gpu:0", "/gpu:1"]), used to determine the world size.

        Returns:
            ConfigKeras: A new `ConfigKeras` object with the `output_rules`
            populated with instantiated collective op objects.
        """
        world_size = len(devices)
        new_output_rules = _create_ops_from_rules(self.output_rules, world_size)

        return dataclasses.replace(
            self,
            output_rules=new_output_rules,
        )
