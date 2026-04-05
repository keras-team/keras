import os
import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from keras.src import testing
from keras.src import layers
from keras.src import models
from keras.src import backend
from keras.src.backend.torch import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib

@pytest.mark.skipif(backend.backend() != "torch", reason="Only for Torch backend")
class TorchDistributedTestCase(testing.TestCase):
    def setUp(self):
        super().setUp()
        self.world_size = 2
    @staticmethod
    def _worker_wrapper(rank, world_size, test_fn):
        import torch.distributed as dist

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["KERAS_TORCH_DEVICE"] = "cpu"
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        backend_name = "gloo"

        dist.init_process_group(
            backend=backend_name, rank=rank, world_size=world_size
        )
        test_fn(rank, world_size)
        if dist.is_initialized():
            dist.destroy_process_group()

    def run_distributed(self, test_fn, world_size=None):
        world_size = world_size or self.world_size
        mp.spawn(
            TorchDistributedTestCase._worker_wrapper,
            args=(world_size, test_fn),
            nprocs=world_size,
            join=True,
        )

class TorchDeviceDiscoveryTest(TorchDistributedTestCase):
    @staticmethod
    def _list_devices_test(rank, world_size):
        devices = distribution_lib.list_devices()
        assert len(devices) == world_size
        for i in range(world_size):
            assert f":{i}" in devices[i]

    @staticmethod
    def _get_device_count_test(rank, world_size):
        assert backend_dlib.get_device_count() == world_size

    def test_list_devices(self):
        self.run_distributed(TorchDeviceDiscoveryTest._list_devices_test)

    def test_get_device_count(self):
        self.run_distributed(TorchDeviceDiscoveryTest._get_device_count_test)

class TorchProcessManagementTest(TorchDistributedTestCase):
    @staticmethod
    def _num_processes_test(rank, world_size):
        assert backend_dlib.num_processes() == world_size

    @staticmethod
    def _process_id_test(rank, world_size):
        assert backend_dlib.process_id() == rank

    def test_num_processes(self):
        self.run_distributed(TorchProcessManagementTest._num_processes_test)

    def test_process_id(self):
        self.run_distributed(TorchProcessManagementTest._process_id_test)

class TorchDeviceMeshMappingTest(TorchDistributedTestCase):
    @staticmethod
    def _to_backend_mesh_test(rank, world_size):
        axis_names = ["data"]
        mesh = distribution_lib.DeviceMesh((world_size,), axis_names)
        torch_mesh = backend_dlib._to_backend_mesh(mesh)
        from torch.distributed.device_mesh import DeviceMesh
        assert isinstance(torch_mesh, DeviceMesh)
        assert torch_mesh.mesh.shape == (world_size,)
        assert torch_mesh.mesh_dim_names == ("data",)

    @staticmethod
    def _to_backend_layout_test(rank, world_size):
        mesh = distribution_lib.DeviceMesh((world_size,), ["data"])
        layout = distribution_lib.TensorLayout(["data", None], mesh)
        torch_mesh, placements = backend_dlib._to_backend_layout(layout)
        from torch.distributed.tensor import Shard
        assert len(placements) == 1
        assert isinstance(placements[0], Shard)
        assert placements[0].dim == 0

    def test_to_backend_mesh(self):
        self.run_distributed(TorchDeviceMeshMappingTest._to_backend_mesh_test)

    def test_to_backend_layout(self):
        self.run_distributed(TorchDeviceMeshMappingTest._to_backend_layout_test)

class TorchTensorDistributionTest(TorchDistributedTestCase):
    @staticmethod
    def _distribute_tensor_test(rank, world_size):
        from torch.distributed.tensor import DTensor
        mesh = distribution_lib.DeviceMesh((world_size,), ["data"])
        layout = distribution_lib.TensorLayout(["data", None], mesh)
        distribution = distribution_lib.ModelParallel(
            device_mesh=mesh, layout_map=distribution_lib.LayoutMap(mesh)
        )
        with distribution.scope():
            data = torch.randn(world_size * 2, 4)
            distributed_tensor = backend_dlib.distribute_tensor(data, layout)
            assert isinstance(distributed_tensor, DTensor)
            assert distributed_tensor.shape == (world_size * 2, 4)
            assert distributed_tensor.to_local().shape == (2, 4)

    @staticmethod
    def _distribute_data_input_test(rank, world_size):
        from torch.distributed.tensor import DTensor
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        layout = distribution_lib.TensorLayout(["batch", None], mesh)
        local_data = torch.arange(rank * 4, (rank + 1) * 4).reshape(2, 2).float()
        distribution = distribution_lib.ModelParallel(
            device_mesh=mesh, layout_map=distribution_lib.LayoutMap(mesh),
            batch_dim_name="batch"
        )
        with distribution.scope():
            distributed_data = backend_dlib.distribute_data_input(
                local_data, layout, "batch"
            )
            assert isinstance(distributed_data, DTensor)
            assert distributed_data.shape == (world_size * 2, 2)
            assert torch.allclose(distributed_data.to_local(), local_data)

    def test_distribute_tensor(self):
        self.run_distributed(TorchTensorDistributionTest._distribute_tensor_test)

    def test_distribute_data_input(self):
        self.run_distributed(TorchTensorDistributionTest._distribute_data_input_test)

class TorchVariableDistributionAwarenessTest(TorchDistributedTestCase):
    @staticmethod
    def _variable_distribution_awareness_test(rank, world_size):
        mesh = distribution_lib.DeviceMesh((world_size,), ["model"])
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout([None, "model"])
        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch"
        )
        with distribution.scope():
            layer = layers.Dense(world_size * 4)
            layer.build((8, 8))
        from torch.distributed.tensor import DTensor
        assert isinstance(layer.kernel.value, DTensor)
        assert layer.kernel.value.to_local().shape == (8, 4)

    def test_variable_distribution_awareness(self):
        self.run_distributed(TorchVariableDistributionAwarenessTest._variable_distribution_awareness_test)

class TorchTrainerArchitectureTest(TorchDistributedTestCase):
    @staticmethod
    def _e2e_data_parallel_fit_test(rank, world_size):
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        distribution = distribution_lib.DataParallel(device_mesh=mesh, auto_shard_dataset=False)
        with distribution.scope():
            model = models.Sequential([
                layers.Input(shape=(8,)),
                layers.Dense(4),
                layers.Dense(2)
            ])
            model.compile(optimizer="adam", loss="mse", metrics=["mae"])
            x = np.random.randn(4, 8).astype("float32")
            y = np.random.randn(4, 2).astype("float32")
            model.fit(x, y, epochs=1, batch_size=2)
            metrics = model.evaluate(x, y, batch_size=2)
            assert len(metrics) > 0

    @staticmethod
    def _e2e_model_parallel_fit_test(rank, world_size):
        mesh = distribution_lib.DeviceMesh((1, world_size), ["batch", "model"])
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout([None, "model"])
        distribution = distribution_lib.ModelParallel(
            layout_map=layout_map, batch_dim_name="batch", auto_shard_dataset=False
        )
        with distribution.scope():
            model = models.Sequential([
                layers.Input(shape=(8,)),
                layers.Dense(world_size * 4),
                layers.Dense(2)
            ])
            model.compile(optimizer="adam", loss="mse")
            x = np.random.randn(4, 8).astype("float32")
            y = np.random.randn(4, 2).astype("float32")
            model.fit(x, y, epochs=1, batch_size=2)

    @staticmethod
    def _keras_module_wrapper_test(rank, world_size):
        from keras.src.backend.torch.trainer import _KerasModuleWrapper
        model = models.Sequential([
            layers.Input(shape=(8,)),
            layers.Dense(4),
            layers.Dense(2)
        ])
        model.build()
        wrapper = _KerasModuleWrapper(model)
        assert len(list(wrapper.parameters())) == 4
        x = torch.randn(2, 8)
        y = wrapper(x)
        assert y.shape == (2, 2)

    def test_e2e_data_parallel_fit(self):
        self.run_distributed(TorchTrainerArchitectureTest._e2e_data_parallel_fit_test)

    def test_e2e_model_parallel_fit(self):
        self.run_distributed(TorchTrainerArchitectureTest._e2e_model_parallel_fit_test)

    def test_keras_module_wrapper(self):
        self.run_distributed(TorchTrainerArchitectureTest._keras_module_wrapper_test, world_size=1)

class TorchDataLoadingTest(TorchDistributedTestCase):
    @staticmethod
    def _dataloader_distributed_sampler_test(rank, world_size):
        from torch.utils.data import DataLoader, TensorDataset
        from keras.src.trainers.data_adapters.torch_data_loader_adapter import TorchDataLoaderAdapter
        from torch.utils.data.distributed import DistributedSampler
        dataset = TensorDataset(torch.randn(world_size * 4, 8), torch.randn(world_size * 4, 2))
        dataloader = DataLoader(dataset, batch_size=2)
        distribution = distribution_lib.DataParallel()
        with distribution.scope():
            adapter = TorchDataLoaderAdapter(dataloader)
            new_dataloader = adapter.get_torch_dataloader()
            assert isinstance(new_dataloader.sampler, DistributedSampler)
            assert new_dataloader.sampler.num_replicas == world_size
            assert new_dataloader.sampler.rank == rank

    def test_dataloader_distributed_sampler(self):
        self.run_distributed(TorchDataLoadingTest._dataloader_distributed_sampler_test)

class TorchMetricAggregationTest(TorchDistributedTestCase):
    @staticmethod
    def _sync_metrics_test(rank, world_size):
        from keras.src import metrics
        m = metrics.MeanSquaredError()
        mesh = distribution_lib.DeviceMesh((world_size,), ["batch"])
        distribution = distribution_lib.DataParallel(device_mesh=mesh)
        with distribution.scope():
            model = models.Sequential([layers.Dense(1, input_shape=(1,))])
            model.compile(metrics=[m])
            model.compute_metrics(torch.zeros(1, 1), torch.ones(1, 1), torch.zeros(1, 1))
            m.update_state(torch.ones(1, 1), torch.tensor([[1.0 + np.sqrt(rank)]]))
            model._sync_metrics()
        
        expected_total = 3.0
        expected_count = 4.0
        assert abs(float(m.variables[0].value) - expected_total) < 1e-5
        assert abs(float(m.variables[1].value) - expected_count) < 1e-5

    def test_sync_metrics(self):
        self.run_distributed(TorchMetricAggregationTest._sync_metrics_test)
