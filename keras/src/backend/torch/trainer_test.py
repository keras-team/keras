import os

import numpy as np
import pytest
import torch

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import optimizers
from keras.src import testing
from keras.src.backend.torch import distribution_lib
from keras.src.distribution import distribution_lib as dist_lib
from keras.src.distribution.distribution_lib import DataParallel
from keras.src.distribution.distribution_lib import DeviceMesh


@pytest.mark.skipif(
    backend.backend() != "torch", reason="Requires torch backend"
)
class TorchTrainerDistributionTest(testing.TestCase):
    def set_env(self, key, value):
        os.environ[key] = value
        self.addCleanup(lambda: os.environ.pop(key, None))

    def test_torch_trainer_ddp(self):
        if not torch.distributed.is_initialized():
            self.set_env("MASTER_ADDR", "localhost")
            self.set_env("MASTER_PORT", "29515")
            distribution_lib.initialize(num_processes=1, process_id=0)
            self.addCleanup(
                lambda: (
                    torch.distributed.destroy_process_group()
                    if torch.distributed.is_initialized()
                    else None
                )
            )

        class SimpleModel(models.Model):
            def __init__(self):
                super().__init__()
                self.dense = layers.Dense(1)

            def call(self, x):
                return self.dense(x)

        model = SimpleModel()
        model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mae"])

        x = np.ones((10, 10), dtype="float32")
        y = np.ones((10, 1), dtype="float32")

        # Fit should trigger DDP wrapping
        model.fit(x, y, epochs=1, batch_size=2, verbose=0)

        # Verify ddp_model is created
        self.assertTrue(hasattr(model, "ddp_model"))

        # Evaluate should also work
        model.evaluate(x, y, verbose=0)

    def test_torch_trainer_ddp_with_distribution(self):
        if not torch.distributed.is_initialized():
            self.set_env("MASTER_ADDR", "localhost")
            self.set_env("MASTER_PORT", "29517")
            distribution_lib.initialize(num_processes=1, process_id=0)
            self.addCleanup(
                lambda: (
                    torch.distributed.destroy_process_group()
                    if torch.distributed.is_initialized()
                    else None
                )
            )

        # Set Keras distribution
        mesh = DeviceMesh(
            shape=(1,), axis_names=["batch"], devices=np.array(["cpu:0"])
        )
        distribution = DataParallel(device_mesh=mesh)
        dist_lib.set_distribution(distribution)
        self.addCleanup(lambda: dist_lib.set_distribution(None))

        class SimpleModel(models.Model):
            def __init__(self):
                super().__init__()
                self.dense = layers.Dense(1)

            def call(self, x):
                return self.dense(x)

        model = SimpleModel()
        model.compile(optimizer=optimizers.Adam(), loss="mse", metrics=["mae"])

        x = np.ones((10, 10), dtype="float32")
        y = np.ones((10, 1), dtype="float32")

        # Fit should trigger DDP wrapping with specific process group logic
        model.fit(x, y, epochs=1, batch_size=2, verbose=0)

        # Verify ddp_model is created
        self.assertTrue(hasattr(model, "ddp_model"))
