import os
from unittest import mock

import pytest
import torch
import torch.multiprocessing as mp

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import distribution_lib as backend_dlib


def _distributed_worker(rank, world_size, test_name):
    # Set standard environment variables for torch distributed
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize using the library's function
    backend_dlib.initialize()

    try:
        if test_name == "utils":
            if backend_dlib.num_processes() != world_size:
                raise AssertionError
            if backend_dlib.process_id() != rank:
                raise AssertionError
            # When initialized, get_device_count returns world_size for CPU
            if backend_dlib.get_device_count("cpu") != world_size:
                raise AssertionError
            devices = backend_dlib.list_devices("cpu")
            if len(devices) != world_size:
                raise AssertionError
            if devices[rank] != f"cpu:{rank}":
                raise AssertionError

        elif test_name == "to_backend_device":
            # Test default device
            device = backend_dlib.to_backend_device(None)
            expected_type = "cuda" if torch.cuda.is_available() else "cpu"
            if device.type != expected_type:
                raise AssertionError

    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(backend.backend() != "torch", reason="Torch specific.")
class TorchDistributionLibTest(testing.TestCase):
    def test_distributed_real(self):
        """Test multi-process distribution without mocks."""
        world_size = 2
        for test_name in ["utils", "to_backend_device"]:
            # mp.spawn handles process creation and joining
            mp.spawn(
                _distributed_worker,
                args=(world_size, test_name),
                nprocs=world_size,
                join=True,
            )

    def test_single_process_logic(self):
        """Test single-process logic using direct environment manipulation."""
        # Ensure we are not initialized
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

        # Save and clear env manually
        old_world = os.environ.get("WORLD_SIZE")
        old_rank = os.environ.get("RANK")
        if "WORLD_SIZE" in os.environ:
            del os.environ["WORLD_SIZE"]
        if "RANK" in os.environ:
            del os.environ["RANK"]

        try:
            self.assertEqual(backend_dlib.num_processes(), 1)
            self.assertEqual(backend_dlib.process_id(), 0)
            self.assertEqual(backend_dlib.to_backend_device("cpu").type, "cpu")
            self.assertEqual(
                backend_dlib.to_backend_device("meta").type, "meta"
            )
        finally:
            # Restore env
            if old_world:
                os.environ["WORLD_SIZE"] = old_world
            if old_rank:
                os.environ["RANK"] = old_rank

    def test_availability_mocking(self):
        """Mocks are reserved to simulate hardware we don't have."""
        # Test MPS path
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(backend_dlib.get_device_count(), 1)
            self.assertEqual(backend_dlib.list_devices()[0], "mps:0")

        # Mocking TPU requires patching the LazyModule to avoid ImportError
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
            mock.patch(
                "keras.src.backend.torch.distribution_lib._is_available",
                side_effect=lambda x: x == "tpu" or x == "cpu",
            ),
        ):
            mock_xla = mock.MagicMock()
            mock_xla.available = True
            mock_xla.runtime.global_device_count.return_value = 8
            with mock.patch("keras.src.utils.module_utils.torch_xla", mock_xla):
                self.assertEqual(backend_dlib.get_device_count("tpu"), 8)
                self.assertEqual(backend_dlib.get_device_count(), 8)

    def test_cuda_to_backend_device_with_local_rank(self):
        """Test LOCAL_RANK usage with minimal mocking of availability."""
        old_local_rank = os.environ.get("LOCAL_RANK")
        os.environ["LOCAL_RANK"] = "3"
        try:
            with mock.patch("torch.cuda.is_available", return_value=True):
                self.assertEqual(backend_dlib.to_backend_device("gpu").index, 3)
                self.assertEqual(
                    backend_dlib.to_backend_device("cuda:0").index, 0
                )
        finally:
            if old_local_rank:
                os.environ["LOCAL_RANK"] = old_local_rank
            else:
                if "LOCAL_RANK" in os.environ:
                    del os.environ["LOCAL_RANK"]
