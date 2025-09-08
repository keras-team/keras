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

    def test_tensor_is_shared_and_accessible(self):
        num_devices = jax.device_count()
        if num_devices < 2:
            self.skipTest("This test requires at least 2 devices for sharding.")

        mesh_shape = (1, num_devices)
        axis_names = ("batch", "model")
        mesh = distribution_lib.DeviceMesh(
            shape=mesh_shape, axis_names=axis_names
        )
        distribution = distribution_lib.AutoShardDistribution(mesh)

        with distribution.scope():
            model = models.Sequential(
                [layers.Dense(4, use_bias=False, input_shape=(8,))]
            )
            model.build(input_shape=(None, 8))

        kernel_var = model.variables[0]
        self.assertIn("kernel", kernel_var.path)

        kernel_layout = distribution.get_variable_layout(kernel_var)
        self.assertEqual(
            kernel_layout.spec,
            jax.sharding.PartitionSpec(None, None),
            "Kernel variable should be replicated across all devices.",
        )

        def access_on_device(dummy_arg):
            tensor_sum = jnp.sum(kernel_var.value)
            device_id = jax.lax.axis_index("model")

            return tensor_sum + device_id.astype(kernel_var.dtype)

        with distribution.scope():
            pmapped_access_fn = jax.pmap(
                access_on_device,
                axis_name="model",
            )

            dummy_input = jnp.zeros(num_devices)
            results = pmapped_access_fn(dummy_input)

        expected_sum = jnp.sum(kernel_var.value)
        for i in range(num_devices):
            self.assertAllClose(
                results[i],
                expected_sum + i,
                msg=f"Device {i} did not compute the correct value.",
            )

        self.assertEqual(len(results.devices()), num_devices)