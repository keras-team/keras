import os
from unittest import mock

import pytest
import torch

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import distribution_lib as backend_dlib


@pytest.mark.skipif(backend.backend() != "torch", reason="Torch specific.")
class TorchDistributionLibTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                "gloo", rank=0, world_size=1, init_method="tcp://127.0.0.1:0"
            )
        backend_dlib.initialize()

    def test_utils_and_init(self):
        with mock.patch("torch.distributed.is_initialized", return_value=False):
            self.assertEqual(backend_dlib.num_processes(), 1)
            self.assertEqual(backend_dlib.process_id(), 0)
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
                self.assertEqual(backend_dlib.get_device_count(), 2)
            with mock.patch.dict(os.environ, {}, clear=True):
                with (
                    mock.patch("torch.cuda.is_available", return_value=False),
                    mock.patch(
                        "torch.backends.mps.is_available", return_value=True
                    ),
                ):
                    self.assertEqual(backend_dlib.get_device_count(), 1)

                with (
                    mock.patch("torch.cuda.is_available", return_value=True),
                    mock.patch("torch.cuda.device_count", return_value=8),
                ):
                    self.assertEqual(backend_dlib.get_device_count(), 8)
                with (
                    mock.patch("torch.cuda.is_available", return_value=False),
                    mock.patch(
                        "torch.backends.mps.is_available", return_value=False
                    ),
                    mock.patch(
                        "keras.src.backend.torch.distribution_lib.hasattr",
                        return_value=False,
                    ),
                ):
                    self.assertEqual(backend_dlib.get_device_count(), 1)
                    self.assertEqual(
                        backend_dlib.get_device_count("invalid"), 0
                    )
                    for dev in ["gpu", "mps", "xpu", "tpu"]:
                        self.assertEqual(backend_dlib.get_device_count(dev), 0)

                    with (
                        mock.patch(
                            "keras.src.backend.torch.distribution_lib.hasattr",
                            side_effect=lambda o, n: n == "xpu"
                            if o == torch
                            else hasattr(o, n),
                            create=True,
                        ),
                        mock.patch(
                            "torch.xpu.is_available",
                            return_value=True,
                            create=True,
                        ),
                        mock.patch(
                            "torch.xpu.device_count",
                            return_value=2,
                            create=True,
                        ),
                    ):
                        self.assertEqual(backend_dlib.get_device_count(), 2)

                    mock_xla = mock.MagicMock(available=True)
                    mock_xla.runtime.global_device_count.return_value = 4
                    with mock.patch(
                        "keras.src.utils.module_utils.torch_xla", mock_xla
                    ):
                        self.assertEqual(
                            backend_dlib.get_device_count("tpu"), 4
                        )
                        self.assertEqual(backend_dlib.get_device_count(), 4)
                        mock_xla.available = False
                        self.assertEqual(backend_dlib.list_devices(), ["cpu:0"])
                self.assertEqual(
                    backend_dlib.to_backend_device("gpu").type, "cpu"
                )

        with mock.patch("torch.cuda.is_available", return_value=True):
            for d in ["cpu", "cpu:1", "meta", "gpu:0", "cuda:2"]:
                res = backend_dlib.to_backend_device(d)
                if d == "meta":
                    self.assertEqual(res.type, "meta")
                elif "cpu" in d:
                    self.assertEqual(res.type, "cpu")
                else:
                    self.assertEqual(res.type, "cuda")
                    if ":" in d:
                        self.assertEqual(res.index, int(d.split(":")[1]))

        with mock.patch("torch.cuda.is_available", return_value=False):
            self.assertEqual(backend_dlib.to_backend_device(None).type, "cpu")

        with (
            mock.patch("torch.cuda.is_available", return_value=True),
            mock.patch("torch.cuda.set_device") as mset,
            mock.patch("torch.distributed.init_process_group") as minit,
            mock.patch("torch.distributed.is_initialized", return_value=False),
            mock.patch.dict(os.environ, {"LOCAL_RANK": "1", "WORLD_SIZE": "2"}),
        ):
            backend_dlib.initialize()
            mset.assert_called_with(1)
            minit.assert_called_with(backend="nccl", rank=0, world_size=2)
            self.assertEqual(backend_dlib.to_backend_device("gpu").index, 1)

    def test_initialize_branches(self):
        with (
            mock.patch("torch.distributed.is_initialized", return_value=False),
            mock.patch("torch.distributed.init_process_group") as minit,
        ):
            backend_dlib.initialize()
            minit.assert_not_called()

            backend_dlib.initialize(num_processes=2, process_id=1)
            minit.assert_called_with(backend="gloo", rank=1, world_size=2)
            with mock.patch.dict(os.environ, {"RANK": "1", "WORLD_SIZE": "2"}):
                backend_dlib.initialize()
                minit.assert_called_with(backend="gloo", rank=1, world_size=2)
            with (
                mock.patch("torch.cuda.is_available", return_value=False),
                mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}),
            ):
                backend_dlib.initialize()
                minit.assert_called_with(backend="gloo", rank=0, world_size=2)
