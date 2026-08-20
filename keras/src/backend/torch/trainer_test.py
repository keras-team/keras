import os

import numpy as np
import pytest
import torch
import torch.multiprocessing as mp
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import optimizers
from keras.src import testing
from keras.src.backend.torch import distribution_lib
from keras.src.distribution import distribution_lib as dist_lib
from keras.src.distribution.distribution_lib import DataParallel
from keras.src.distribution.distribution_lib import DeviceMesh


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
class TorchTrainerDistributionTest(testing.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        if not torch.distributed.is_initialized():
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29515"
            distribution_lib.initialize(num_processes=1, process_id=0)

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
        os.environ.pop("MASTER_ADDR", None)
        os.environ.pop("MASTER_PORT", None)

    @parameterized.named_parameters(
        ("base", SimpleModel, False),
        ("with_distribution", SimpleModel, True),
        ("with_training_arg", TrainingAwareModel, False),
    )
    def test_torch_trainer_ddp(self, model_class, use_distribution):
        if use_distribution:
            mesh = DeviceMesh(
                shape=(1,), axis_names=["batch"], devices=np.array(["cpu:0"])
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
