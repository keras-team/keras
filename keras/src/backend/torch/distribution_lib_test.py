import os

import numpy as np
import pytest
import torch
from absl.testing import parameterized

from keras.src import backend
from keras.src import testing
from keras.src.backend.torch import core
from keras.src.backend.torch import distribution_lib
from keras.src.distribution.distribution_lib import DeviceMesh


@pytest.mark.skipif(backend.backend() != "torch", reason="Requires torch")
class TorchDistributionLibTest(testing.TestCase):
    def set_env(self, key, value):
        old = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
        self.addCleanup(
            lambda: (
                os.environ.update({key: old})
                if old is not None
                else os.environ.pop(key, None)
            )
        )

    def tearDown(self):
        super().tearDown()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    @parameterized.parameters(
        ({}, False, None),
        ({"WORLD_SIZE": "4"}, False, None),
        (
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29502",
                "WORLD_SIZE": "1",
                "RANK": "0",
            },
            True,
            None,
        ),
        ({}, False, "cuda"),
        ({}, False, "gpu"),
    )
    def test_get_device_count(self, env, init, device_type):
        for k, v in env.items():
            self.set_env(k, v)
        if init:
            torch.distributed.init_process_group(
                backend="gloo", rank=0, world_size=1
            )

        res = distribution_lib.get_device_count(device_type)

        if torch.distributed.is_initialized() or "WORLD_SIZE" in os.environ:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            actual_device_type = core._parse_device_input(
                core.get_device()
            ).split(":")[0]
            if device_type in (None, "cpu", actual_device_type) or (
                device_type == "gpu" and actual_device_type == "cuda"
            ):
                self.assertEqual(res, world_size)
            else:
                self.assertEqual(res, 0)
        else:
            resolved_device_type = core._parse_device_input(
                device_type or core.get_device()
            ).split(":")[0]
            if resolved_device_type == "cuda":
                self.assertEqual(res, torch.cuda.device_count())
            elif resolved_device_type in ("cpu", "mps"):
                self.assertEqual(res, 1)
            elif resolved_device_type == "xpu":
                self.assertEqual(res, torch.xpu.device_count())
            else:
                self.assertEqual(res, 0)

    @parameterized.parameters(
        (None, {}, False, True),
        ("cpu", {}, False, False),
        (
            "gpu",
            {"WORLD_SIZE": "4", "KERAS_TORCH_DEVICE": "gpu"},
            False,
            False,
        ),
    )
    def test_list_devices(self, device_type, env, init, default):
        for k, v in env.items():
            self.set_env(k, v)
        if init:
            torch.distributed.init_process_group(
                backend="gloo", rank=0, world_size=1
            )

        if default:
            devices = distribution_lib.list_devices()
            self.assertTrue(
                any(
                    devices[0].startswith(s)
                    for s in ["gpu:", "cpu:", "mps:", "xpu:"]
                )
            )
        else:
            devices = distribution_lib.list_devices(device_type)
            resolved_device_type = core._parse_device_input(
                device_type or core.get_device()
            ).split(":")[0]
            display_type = (
                "gpu"
                if resolved_device_type == "cuda"
                else resolved_device_type
            )
            count = distribution_lib.get_device_count(device_type)
            expected = [f"{display_type}:{i}" for i in range(count)]
            self.assertEqual(devices, expected)

    @parameterized.parameters(
        ({}, False, 1),
        ({"WORLD_SIZE": "4"}, False, 4),
        (
            {
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29504",
                "WORLD_SIZE": "1",
                "RANK": "0",
            },
            True,
            1,
        ),
    )
    def test_num_processes_and_id(self, env, init, expected):
        for k, v in env.items():
            self.set_env(k, v)
        if init:
            torch.distributed.init_process_group(
                backend="gloo", rank=0, world_size=1
            )

        self.assertEqual(distribution_lib.num_processes(), expected)
        self.assertEqual(distribution_lib.process_id(), int(env.get("RANK", 0)))

    @parameterized.parameters(
        (
            "127.0.0.1:29506",
            1,
            0,
            {"MASTER_ADDR": None, "MASTER_PORT": None},
            {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29506"},
        ),
        (
            "127.0.0.1",
            1,
            0,
            {"MASTER_ADDR": None, "MASTER_PORT": "29507"},
            {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29507"},
        ),
        (
            None,
            1,
            0,
            {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29508"},
            {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29508"},
        ),
    )
    def test_initialize(self, addr, nproc, pid, initial, expected):
        if torch.distributed.is_initialized():
            self.skipTest("torch.distributed already initialized")

        for k, v in initial.items():
            self.set_env(k, v)

        distribution_lib.initialize(addr, nproc, pid)
        self.addCleanup(
            lambda: (
                torch.distributed.destroy_process_group()
                if torch.distributed.is_initialized()
                else None
            )
        )
        self.assertTrue(torch.distributed.is_initialized())

        for k, v in expected.items():
            self.assertEqual(os.environ.get(k), v)

    @parameterized.parameters(
        ("cpu", "cpu"),
        ("gpu", "cuda"),
        (None, None),
    )
    def test_get_device_type(self, k_dev, expected):
        if k_dev:
            self.set_env("KERAS_TORCH_DEVICE", k_dev)

        res = core._parse_device_input(core.get_device()).split(":")[0]
        self.assertEqual(res, core.get_device().split(":")[0])

    @parameterized.parameters(
        ("cpu", {}, "cpu", None),
        ("gpu", {}, "cuda", None),
        (torch.device("cuda:0"), {}, "cuda", 0),
        ("cuda", {"LOCAL_RANK": "2"}, "cuda", 2),
        ("cpu:0", {}, "cpu", None),
        (torch.device("cpu"), {}, "cpu", None),
    )
    def test_to_backend_device(self, inp, env, etype, eidx):
        for k, v in env.items():
            self.set_env(k, v)

        if (
            isinstance(inp, torch.device)
            and inp.type == "cuda"
            and not torch.cuda.is_available()
        ):
            self.skipTest("No CUDA")

        if inp == "gpu" and not torch.cuda.is_available():
            self.skipTest("No CUDA")

        dev = distribution_lib._to_backend_device(inp)
        self.assertEqual(dev.type, etype)
        if eidx is not None:
            self.assertEqual(dev.index, eidx)

    @parameterized.parameters(
        (np.array(["cpu:0"]).reshape(1), "cpu"),
        (np.array(["gpu:0"]).reshape(1), "cuda"),
    )
    def test_to_backend_mesh(self, devs, etype):
        if etype == "cuda" and not torch.cuda.is_available():
            self.skipTest("No CUDA")

        if not torch.distributed.is_initialized():
            self.set_env("MASTER_ADDR", "localhost")
            self.set_env("MASTER_PORT", "29509")
            distribution_lib.initialize(num_processes=1, process_id=0)
            self.addCleanup(
                lambda: (
                    torch.distributed.destroy_process_group()
                    if torch.distributed.is_initialized()
                    else None
                )
            )

        mesh = DeviceMesh(shape=(1,), axis_names=["x"], devices=devs)
        backend_mesh = distribution_lib._to_backend_mesh(mesh)

        from torch.distributed.device_mesh import DeviceMesh as TorchDeviceMesh

        self.assertIsInstance(backend_mesh, TorchDeviceMesh)
        self.assertEqual(backend_mesh.device_type, etype)
        self.assertEqual(backend_mesh.mesh_dim_names, ("x",))
