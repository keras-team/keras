from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import List


class DistributedBackend(ABC):
    """
    Abstract Base Class for a distributed backend.

    This class defines the interface for backend-specific operations required
    for distributed training. Tensor conversions should be handled by the
    backend-agnostic `keras.ops.convert_to_tensor` function.
    """

    @abstractmethod
    def compute_gradients(
        self, loss: Any, trainable_vars: List[Any]
    ) -> List[Any]:
        """Compute gradients using the backend's automatic differentiation."""
        raise NotImplementedError

    @abstractmethod
    def apply_gradients(
        self,
        gradients: List[Any],
        trainable_vars: List[Any],
        learning_rate: float = 0.001,
    ) -> None:
        """Apply gradients to trainable variables."""
        raise NotImplementedError

    @abstractmethod
    def create_optimizer(self, optimizer_class: str, **kwargs):
        """Create an optimizer for the backend."""
        raise NotImplementedError

    @abstractmethod
    def get_device_info(self) -> dict:
        """Get information about available devices."""
        raise NotImplementedError

    @abstractmethod
    def is_multi_device_capable(self) -> bool:
        """Check if the backend supports multi-device operations."""
        raise NotImplementedError

    @abstractmethod
    def get_communication_ops(self) -> dict:
        """Get collective communication operations for the backend."""
        raise NotImplementedError
