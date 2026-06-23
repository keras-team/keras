import os
from unittest import mock

import numpy as np
import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor import Replicate
from torch.distributed.tensor import Shard

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.backend.torch import core as torch_core
from keras.src.backend.torch import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib as dist_lib


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

    def test_ops_and_conversions(self):
        mesh = dist_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        tm = backend_dlib._to_backend_mesh(mesh)
        self.assertIs(backend_dlib._to_backend_mesh(tm), tm)
        self.assertIsNone(backend_dlib._to_backend_layout(None))
        layout, t = dist_lib.TensorLayout(["data"], mesh), torch.randn(4)
        self.assertIsInstance(
            backend_dlib._to_backend_layout(layout).placements[0], Shard
        )
        self.assertIsInstance(
            backend_dlib._to_backend_layout(
                mock.MagicMock(axes=None, device_mesh=mesh)
            ).placements[0],
            Replicate,
        )

        self.assertIs(backend_dlib.distribute_tensor(t, None), t)
        mt = mock.MagicMock(spec=torch.Tensor, device=torch.device("cpu"))
        backend_dlib.distribute_tensor(mt, "cpu")
        mt.to.assert_not_called()
        backend_dlib.distribute_tensor(mt, "meta")
        mt.to.assert_called()

        self.assertIs(backend_dlib.distribute_data_input(t, None, "b"), t)
        self.assertIs(backend_dlib.distribute_data_input(t, "cpu", "b"), t)

        with dist_lib.ModelParallel(
            layout_map=dist_lib.LayoutMap(mesh)
        ).scope():
            self.assertIsInstance(
                backend_dlib.distribute_data_input(t, layout, "b"), DTensor
            )
            self.assertIsInstance(
                backend_dlib.distribute_data_input(
                    np.ones(4, "f"), layout, "b"
                ),
                DTensor,
            )
            mt = torch.randn(4, device="meta")
            self.assertIs(
                backend_dlib.distribute_data_input(mt, layout, "b"), mt
            )
            self.assertIsInstance(
                torch_core.convert_to_tensor(np.ones(4, "f")), DTensor
            )

    def test_unbind_strategy(self):
        mesh = dist_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        with dist_lib.ModelParallel(
            layout_map=dist_lib.LayoutMap(mesh)
        ).scope():
            dt = backend_dlib.distribute_tensor(
                torch.randn(4, 2), dist_lib.TensorLayout(["data", None], mesh)
            )
            for st in torch.unbind(dt, 0):
                self.assertIsInstance(st, DTensor)

        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import distribute_tensor

        tm = init_device_mesh("cpu", (1,))
        t = torch.randn(4, 4)

        class MockSchema:
            args_schema = None

        class MockStrat:
            output_spec = None

        ms, mstrat = MockSchema(), MockStrat()
        ms.args_schema = [mock.MagicMock(mesh=tm, strategies=[mstrat]), 0]

        mstrat.output_spec = distribute_tensor(t, tm, [Shard(1)])._spec
        strat = backend_dlib._unbind_op_strategy(ms)
        self.assertIsInstance(
            strat.strategies[0].output_specs[0].placements[0], Shard
        )

        ms.args_schema[1] = -1
        strat = backend_dlib._unbind_op_strategy(ms)
        self.assertIsInstance(
            strat.strategies[0].input_specs[0].placements[0], Replicate
        )

    def test_e2e_building(self):
        mesh = dist_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        layout_map = dist_lib.LayoutMap(mesh)
        layout_map[".*kernel"] = dist_lib.TensorLayout([None, "data"])
        dist = dist_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="data"
        )
        with dist.scope():
            dense = layers.Dense(8)
            dense.build((16, 16))
            self.assertIsInstance(dense.kernel.value.data, DTensor)
        backend_dlib.set_distribution(dist)
        backend_dlib.set_distribution(None)

    def test_register_strategies_exception(self):
        with mock.patch(
            "torch.distributed.tensor._ops.register_op_strategy",
            side_effect=AttributeError,
        ):
            backend_dlib._STRATEGIES_REGISTERED = False
            backend_dlib._register_distributed_strategies()
            self.assertFalse(backend_dlib._STRATEGIES_REGISTERED)

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

    def test_dot_product_attention_dtensor(self):
        from keras.src.backend.torch import nn as torch_nn

        mesh = dist_lib.DeviceMesh((1,), ["data"], ["cpu:0"])
        with dist_lib.ModelParallel(
            layout_map=dist_lib.LayoutMap(mesh)
        ).scope():
            q = torch.randn(2, 4, 3, 8)
            k = torch.randn(2, 4, 3, 8)
            v = torch.randn(2, 4, 3, 8)
            mask = torch.ones((2, 3, 4, 4), dtype=torch.bool)

            q_dt = backend_dlib.distribute_tensor(
                q, dist_lib.TensorLayout(["data", None, None, None], mesh)
            )
            k_dt = backend_dlib.distribute_tensor(
                k, dist_lib.TensorLayout(["data", None, None, None], mesh)
            )
            v_dt = backend_dlib.distribute_tensor(
                v, dist_lib.TensorLayout(["data", None, None, None], mesh)
            )
            mask_dt = backend_dlib.distribute_tensor(
                mask, dist_lib.TensorLayout(["data", None, None, None], mesh)
            )

            output = torch_nn.dot_product_attention(
                q_dt, k_dt, v_dt, mask=mask_dt
            )
            self.assertIsInstance(output, DTensor)
            self.assertEqual(output.shape, q.shape)
