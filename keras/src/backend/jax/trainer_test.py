import warnings

import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend import distribution_lib as backend_dlib
from keras.src.distribution import distribution_lib


class JAXTrainerTest(testing.TestCase, parameterized.TestCase):
    def _skip_if_not_distributed(self):
        if backend.backend() != "jax":
            self.skipTest("Requires JAX backend")
        if len(backend_dlib.list_devices()) < 2:
            self.skipTest("Requires at least 2 devices")

    def _make_distribution(self, dist_type):
        if dist_type == "data_parallel":
            return distribution_lib.DataParallel()
        devices = backend_dlib.list_devices()
        n = len(devices)
        mesh = distribution_lib.DeviceMesh((n,), ["model"], devices)
        layout_map = distribution_lib.LayoutMap(mesh)
        layout_map[".*dense.*kernel"] = distribution_lib.TensorLayout(
            [None, "model"]
        )
        layout_map[".*dense.*bias"] = distribution_lib.TensorLayout(["model"])
        return distribution_lib.ModelParallel(layout_map=layout_map)

    # ----------------------------------------------------------------
    # Mixed-sharding warning tests
    # ----------------------------------------------------------------
    @parameterized.named_parameters(
        {"testcase_name": "DataParallel", "dist_type": "data_parallel"},
        {"testcase_name": "ModelParallel", "dist_type": "model_parallel"},
    )
    def test_warns_when_model_built_outside_scope(self, dist_type):
        """Model built outside distribution -> mixed warning on compile."""
        self._skip_if_not_distributed()
        import jax

        n = len(backend_dlib.list_devices())
        units = n * max(1, 4 // n)
        dist = self._make_distribution(dist_type)

        # Model created outside any distribution scope — weights are local.
        model = models.Sequential([layers.Dense(units, input_shape=(16,))])

        for w in model.weights:
            self.assertIsInstance(
                w.value.sharding, jax.sharding.SingleDeviceSharding
            )

        inputs = np.random.normal(size=(8, 16)).astype("float32")
        labels = np.random.normal(size=(8, units)).astype("float32")

        with dist.scope():
            model.compile(loss="mse", optimizer="adam")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model._symbolic_build(data_batch=(inputs[:2], labels[:2]))
                model._get_state_sharding_spec()

            mixed = [w for w in caught if "mix of local" in str(w.message)]
            self.assertGreater(
                len(mixed),
                0,
                "Expected a mixed-sharding warning but none was raised",
            )
            msg = str(mixed[0].message)
            self.assertIn("SingleDeviceSharding", msg)
            self.assertIn("set_distribution", msg)

    @parameterized.named_parameters(
        {"testcase_name": "DataParallel", "dist_type": "data_parallel"},
        {"testcase_name": "ModelParallel", "dist_type": "model_parallel"},
    )
    def test_no_warning_when_model_built_inside_scope(self, dist_type):
        """Model built inside distribution scope -> no warning."""
        self._skip_if_not_distributed()

        n = len(backend_dlib.list_devices())
        units = n * max(1, 4 // n)
        dist = self._make_distribution(dist_type)

        # Model created inside scope — weights get proper sharding.
        with dist.scope():
            model = models.Sequential([layers.Dense(units, input_shape=(16,))])

        inputs = np.random.normal(size=(8, 16)).astype("float32")
        labels = np.random.normal(size=(8, units)).astype("float32")

        with dist.scope():
            model.compile(loss="mse", optimizer="adam")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model._symbolic_build(data_batch=(inputs[:2], labels[:2]))
                model._get_state_sharding_spec()

            mixed = [w for w in caught if "mix of local" in str(w.message)]
            self.assertEqual(
                len(mixed),
                0,
                "Unexpected mixed-sharding warning when model is "
                "built inside scope",
            )
