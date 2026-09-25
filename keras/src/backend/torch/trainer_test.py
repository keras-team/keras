import os

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import metrics
from keras.src import models
from keras.src import optimizers
from keras.src import testing
from keras.src.backend.torch.distributed_test_utils import (
    TorchDistributedTestMixin,
)
from keras.src.backend.torch.trainer import TorchEpochIterator
from keras.src.backend.torch.trainer import _distribute_data
from keras.src.distribution import distribution_lib as dist_lib
from keras.src.distribution.distribution_lib import DataParallel
from keras.src.distribution.distribution_lib import DeviceMesh
from keras.src.distribution.distribution_lib import LayoutMap
from keras.src.distribution.distribution_lib import ModelParallel


class SimpleModel(models.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(1)

    def call(self, x):
        return self.dense(x)


class TrainingAwareModel(models.Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(1)
        self.dropout = layers.Dropout(0.5)

    def call(self, x, training=False):
        x = self.dense(x)
        return self.dropout(x, training=training)


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
@pytest.mark.no_pytest_xdist
class TorchTrainerDistributionTest(TorchDistributedTestMixin, testing.TestCase):
    _master_port = "29515"

    @parameterized.named_parameters(
        ("base", SimpleModel, False),
        ("with_distribution", SimpleModel, True),
        ("with_training_arg", TrainingAwareModel, False),
    )
    def test_torch_trainer_ddp(self, model_class, use_distribution):
        if use_distribution:
            mesh = DeviceMesh(
                shape=(1,),
                axis_names=["batch"],
                devices=np.array([dist_lib.list_devices()[0]]),
            )
            distribution = DataParallel(device_mesh=mesh)
            dist_lib.set_distribution(distribution)
            self.addCleanup(lambda: dist_lib.set_distribution(None))

        model = model_class()
        model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mae"])

        x = np.ones((10, 10), dtype="float32")
        y = np.ones((10, 1), dtype="float32")

        # Fit should trigger DDP wrapping
        model.fit(x, y, epochs=1, batch_size=2, verbose=0)

        # Verify ddp_model is created
        self.assertTrue(hasattr(model, "ddp_model"))

        # Evaluate should also work
        model.evaluate(x, y, verbose=0)

        # Predict should also work
        y_pred = model.predict(x, verbose=0)
        self.assertEqual(y_pred.shape, (10, 1))

        # Test on_batch methods with numpy inputs
        model.train_on_batch(x, y)
        model.test_on_batch(x, y)
        y_pred_batch = model.predict_on_batch(x)
        self.assertEqual(y_pred_batch.shape, (10, 1))

    def test_metrics_distributed_multi_process(self):
        # Use real multi-process distribution to hit the metrics logic
        mp.spawn(
            _distributed_metrics_worker,
            args=(),
            nprocs=2,
            join=True,
        )

    def test_get_metrics_result_model_parallel(self):
        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["batch"],
            devices=np.array([dist_lib.list_devices()[0]]),
        )
        distribution = ModelParallel(layout_map=LayoutMap(mesh))
        dist_lib.set_distribution(distribution)
        self.addCleanup(lambda: dist_lib.set_distribution(None))

        model = SimpleModel()
        mean_metric = metrics.MeanAbsoluteError()
        model.compile(optimizer="sgd", loss="mse", metrics=[mean_metric])

        # Initialize metrics
        x = np.ones((2, 10), dtype="float32")
        y = np.ones((2, 1), dtype="float32")
        model.train_on_batch(x, y)

        results = model.get_metrics_result()

        self.assertIn("mean_absolute_error", results)

    def test_model_parallel_data_distribution(self):
        from torch.distributed.tensor import DTensor

        mesh = DeviceMesh(
            shape=(1,),
            axis_names=["batch"],
            devices=np.array([dist_lib.list_devices()[0]]),
        )
        distribution = ModelParallel(layout_map=LayoutMap(mesh))
        dist_lib.set_distribution(distribution)
        self.addCleanup(lambda: dist_lib.set_distribution(None))

        x = np.ones((4, 2), dtype="float32")
        y = np.ones((4, 1), dtype="float32")
        distributed_data = _distribute_data((x, y, None))

        self.assertIsInstance(distributed_data[0], DTensor)
        self.assertIsInstance(distributed_data[1], DTensor)
        self.assertIsNone(distributed_data[2])

        iterator = TorchEpochIterator(x=x, y=y, batch_size=2)
        batches = list(iterator._get_iterator())
        self.assertLen(batches, 2)
        for batch in batches:
            self.assertIsInstance(batch[0], DTensor)
            self.assertIsInstance(batch[1], DTensor)


def _distributed_metrics_worker(rank):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29519"
    # Use gloo for CPU-based multi-process testing
    torch.distributed.init_process_group(
        backend="gloo", rank=rank, world_size=2
    )

    model = SimpleModel()
    model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mae"])

    x = np.ones((2, 10), dtype="float32")
    y = np.ones((2, 1), dtype="float32")

    # Run one step
    model.train_on_batch(x, y)

    # Trigger metrics aggregation
    results = model.get_metrics_result()

    # Basic verification
    if "mae" not in results or "loss" not in results:
        raise RuntimeError(f"Metrics missing from results: {results}")

    torch.distributed.destroy_process_group()
