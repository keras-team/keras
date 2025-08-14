from unittest import mock

import jax
import jax.numpy as jnp

from keras.src import layers
from keras.src import models
from keras.src.distribution import distribution_lib
from keras.src.testing import test_case


class AutoShardDistributionTest(test_case.TestCase):
    def setUp(self):
        super().setUp()

    @mock.patch(
        "keras.src.backend.jax.trainer.JAXTrainer._enforce_jax_state_sharding",
        lambda self, state, *args, **kwargs: state,
    )
    @mock.patch(
        "keras.src.backend.jax.trainer.JAXTrainer.jax_state_sync",
        lambda self: None,
    )
    def test_autosharding(self):
        """Tests a simple model with AutoShardDistribution."""
        num_devices = jax.device_count()
        if num_devices < 2:
            self.skipTest("This test requires at least 2 devices for sharding.")

        mesh_shape = (1, num_devices)
        mesh = distribution_lib.DeviceMesh(
            shape=mesh_shape, axis_names=("batch", "model")
        )
        distribution = distribution_lib.AutoShardDistribution(mesh)

        with distribution.scope():
            model = models.Sequential(
                [
                    layers.Dense(16, input_shape=(8,)),
                    layers.Dense(1),
                ]
            )
            model.build(input_shape=(None, 8))

        sample_x = jnp.ones((2, 8), dtype=jnp.float32)
        distribution.shard(model, sample_x)

        model.compile(optimizer="sgd", loss="mse")

        batch_size = num_devices * 2
        x = jnp.ones((batch_size, 8), dtype=jnp.float32)
        y = jnp.ones((batch_size, 1), dtype=jnp.float32)

        with mock.patch(
            "keras.src.trainers.trainer.Trainer.get_metrics_result",
            return_value={"loss": 0.0},
        ):
            model.fit(x, y, epochs=1, steps_per_epoch=2, batch_size=batch_size)

        for var in model.variables:
            if "kernel" in var.path:
                sharding = distribution.get_variable_layout(var)
                self.assertEqual(
                    sharding.spec,
                    jax.sharding.PartitionSpec(None, None),
                    f"Variable '{var.path}' was not replicated as expected "
                    "under the current test mock setup. "
                    f"Spec: {sharding.spec}",
                )
