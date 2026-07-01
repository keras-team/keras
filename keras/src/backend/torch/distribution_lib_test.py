import os
from unittest import mock

import pytest
import torch
import torch.multiprocessing as mp

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import distribution_lib as backend_dlib


def _distributed_worker(rank, world_size, test_name):
    os.environ.update(
        {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "12355",
            "RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )
    backend_dlib.initialize()
    try:
        if test_name == "utils":
            if backend_dlib.num_processes() != world_size:
                raise AssertionError
            if backend_dlib.process_id() != rank:
                raise AssertionError
            if backend_dlib.get_device_count("cpu") != world_size:
                raise AssertionError
            if backend_dlib.list_devices("cpu")[rank] != f"cpu:{rank}":
                raise AssertionError
        elif test_name == "to_backend_device":
            expected = "cuda" if torch.cuda.is_available() else "cpu"
            if backend_dlib.to_backend_device(None).type != expected:
                raise AssertionError
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


@pytest.mark.skipif(backend.backend() != "torch", reason="Torch specific.")
class TorchDistributionLibTest(testing.TestCase):
    def test_distributed_real(self):
        for test_name in ["utils", "to_backend_device"]:
            mp.spawn(
                _distributed_worker, args=(2, test_name), nprocs=2, join=True
            )

    def test_single_process_logic(self):
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(backend_dlib.num_processes(), 1)
            self.assertEqual(backend_dlib.process_id(), 0)
            self.assertEqual(backend_dlib.to_backend_device("cpu").type, "cpu")
            self.assertEqual(
                backend_dlib.to_backend_device("meta").type, "meta"
            )

    def test_initialize_logic(self):
        cases = [
            (True, None, None, None, False, None),
            (False, None, None, None, False, None),
            (False, 2, 1, "tcp://127.0.0.1:1234", True, "nccl"),
            (False, 2, 0, "127.0.0.1:1234", True, "nccl"),
            (False, 2, 1, None, True, "gloo"),
        ]
        for (
            is_init,
            num_proc,
            proc_id,
            addr,
            expected_init,
            expected_backend,
        ) in cases:
            with (
                mock.patch(
                    "torch.distributed.is_initialized", return_value=is_init
                ),
                mock.patch(
                    "torch.distributed.init_process_group"
                ) as mock_init_pg,
                mock.patch(
                    "torch.cuda.is_available",
                    return_value=(expected_backend == "nccl"),
                ),
                mock.patch("torch.cuda.set_device"),
                mock.patch.dict(
                    os.environ,
                    {"WORLD_SIZE": "2", "RANK": "1"}
                    if not num_proc and expected_init
                    else {},
                ),
            ):
                backend_dlib.initialize(
                    job_addresses=addr,
                    num_processes=num_proc,
                    process_id=proc_id,
                )
                if expected_init:
                    mock_init_pg.assert_called()
                    args = mock_init_pg.call_args[1]
                    self.assertEqual(args["backend"], expected_backend)
                    if addr:
                        self.assertIn("127.0.0.1:1234", args["init_method"])
                else:
                    mock_init_pg.assert_not_called()

        # Test comma separated job_addresses
        with (
            mock.patch("torch.distributed.is_initialized", return_value=False),
            mock.patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "0"}),
        ):
            with self.assertRaisesRegex(
                ValueError, "should only contain the coordinator address"
            ):
                backend_dlib.initialize(
                    job_addresses="10.0.0.1:1234,10.0.0.2:1234"
                )

    def test_device_counts_and_normalization(self):
        # Default device (gpu)
        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.device_count", return_value=4),
        ):
            self.assertEqual(backend_dlib.get_device_count(), 4)
            self.assertEqual(backend_dlib.get_device_count("cuda"), 4)
            self.assertEqual(
                backend_dlib.list_devices("cuda"),
                [f"gpu:{i}" for i in range(4)],
            )

        # MPS
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=True),
        ):
            self.assertEqual(backend_dlib.get_device_count(), 1)
            self.assertEqual(backend_dlib.list_devices()[0], "mps:0")

        # XPU
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
            mock.patch(
                "torch.xpu.is_available", return_value=True, create=True
            ),
            mock.patch("torch.xpu.device_count", return_value=2, create=True),
        ):
            self.assertEqual(backend_dlib.get_device_count(), 2)
            self.assertEqual(backend_dlib.get_device_count("xpu"), 2)

        # TPU
        mock_xla = mock.MagicMock(available=True)
        mock_xla.runtime.global_device_count.return_value = 8
        with (
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch("torch.backends.mps.is_available", return_value=False),
            mock.patch(
                "torch.xpu.is_available", return_value=False, create=True
            ),
            mock.patch("keras.src.utils.module_utils.torch_xla", mock_xla),
        ):
            self.assertEqual(backend_dlib.get_device_count(), 8)
            self.assertEqual(backend_dlib.get_device_count("tpu"), 8)

        # Non-available device
        self.assertEqual(backend_dlib.get_device_count("invalid"), 0)

    def test_distributed_state_and_to_device(self):
        with (
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=4),
            mock.patch("torch.distributed.get_rank", return_value=2),
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.dict(os.environ, {"LOCAL_RANK": "2"}),
        ):
            self.assertEqual(backend_dlib.num_processes(), 4)
            self.assertEqual(backend_dlib.process_id(), 2)
            self.assertEqual(backend_dlib.get_device_count("gpu"), 4)
            self.assertEqual(backend_dlib.to_backend_device(None).index, 2)
            self.assertEqual(backend_dlib.to_backend_device("gpu:1").index, 1)

        # to_backend_device branches
        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(backend_dlib.to_backend_device("gpu").type, "cpu")
            self.assertEqual(backend_dlib.to_backend_device(None).type, "cpu")

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch.dict(os.environ, {"LOCAL_RANK": "3"}),
        ):
            self.assertEqual(backend_dlib.to_backend_device("gpu").index, 3)
            self.assertEqual(backend_dlib.to_backend_device("cuda:0").index, 0)
            self.assertEqual(backend_dlib.to_backend_device(None).index, 3)
