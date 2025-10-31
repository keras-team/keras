"""Utilities for distribution strategy with PyTorch backend."""


def process_id():
    """Return the current process ID for the distribution setting."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
        return 0
    except (ImportError, AttributeError):
        return 0
