"""Shared test utilities for PyTorch distributed tests."""

import os

import torch

from keras.src.backend.torch import distribution_lib


class TorchDistributedTestMixin:
    """Mixin that manages a single torch.distributed process group lifecycle.

    Use this mixin with ``testing.TestCase`` for any test class that needs
    ``torch.distributed`` (e.g. DTensor, DDP, etc.).  It:

    * Initialises the gloo process group **once** in ``setUpClass`` and
      destroys it **once** in ``tearDownClass``, avoiding the gloo
      ``TIME_WAIT`` hang that occurs when re-initialising per test.
    * Saves and restores all environment variables that
      ``distribution_lib.initialize()`` touches (``MASTER_ADDR``,
      ``MASTER_PORT``, ``WORLD_SIZE``, ``RANK``).

    Subclasses can override ``_master_port`` to use a different port
    (default ``"29500"``).
    """

    _master_port = "29500"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._saved_env = {}
        for key, value in [
            ("MASTER_ADDR", "localhost"),
            ("MASTER_PORT", cls._master_port),
            # distribution_lib.initialize() also sets these via setdefault.
            ("WORLD_SIZE", None),
            ("RANK", None),
        ]:
            cls._saved_env[key] = os.environ.get(key)
            if value is not None:
                os.environ[key] = value

        if not torch.distributed.is_initialized():
            distribution_lib.initialize(num_processes=1, process_id=0)

    @classmethod
    def tearDownClass(cls):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        for key, old_value in cls._saved_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

        super().tearDownClass()
