# keras/src/backend/distributed/__init__.py

from .base import BaseDistributedBackend
from .factory import get_distributed_backend

__all__ = ["get_distributed_backend", "BaseDistributedBackend"]
