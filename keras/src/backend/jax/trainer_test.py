import warnings

import jax
import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.backend import distribution_lib as backend_dlib
from keras.src.backend.jax import trainer as jax_trainer
from keras.src.distribution import distribution_lib


@pytest.mark.skipif(backend.backend() != "jax", reason="JAX only")
@pytest.mark.multi_device
class JAXTrainerTest(testing.TestCase, parameterized.TestCase):
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

    @parameterized.named_parameters(
        {"testcase_name": "DataParallel", "dist_type": "data_parallel"},
        {"testcase_name": "ModelParallel", "dist_type": "model_parallel"},
    )
    def test_train_on_batch(self, dist_type):
        n = len(backend_dlib.list_devices())
        units = n * max(1, 4 // n)
        dist = self._make_distribution(dist_type)

        with dist.scope():
            model = models.Sequential([layers.Dense(units, input_shape=(16,))])
            model.compile(loss="mse", optimizer="adam")

            inputs = np.random.normal(size=(8, 16)).astype("float32")
            labels = np.random.normal(size=(8, units)).astype("float32")
            sw = np.random.uniform(size=(8,)).astype("float32")

            # With sample weight.
            model.train_on_batch(x=inputs, y=labels, sample_weight=sw)
            model.test_on_batch(x=inputs, y=labels, sample_weight=sw)

            # Without sample weight.
            model.train_on_batch(x=inputs, y=labels)
            model.test_on_batch(x=inputs, y=labels)
            model.predict_on_batch(x=inputs)
            model.fit(x=inputs, y=labels, epochs=1, verbose=0)
            model.evaluate(x=inputs, y=labels, verbose=0)

    @parameterized.named_parameters(
        {"testcase_name": "DataParallel", "dist_type": "data_parallel"},
        {"testcase_name": "ModelParallel", "dist_type": "model_parallel"},
    )
    def test_jax_epoch_iterator_with_none_elements(self, dist_type):
        def generator():
            yield (np.ones((16, 32)), None)

        with self._make_distribution(dist_type).scope():
            iterator = jax_trainer.JAXEpochIterator(
                x=generator(), steps_per_epoch=1
            )

            epoch_iter = iterator._get_iterator()
            batch = next(epoch_iter)

        self.assertIsNone(batch[1])
        self.assertIsNotNone(batch[0])

    @parameterized.named_parameters(
        {"testcase_name": "spe_2", "steps_per_execution": 2},
        {"testcase_name": "spe_4", "steps_per_execution": 4},
    )
    def test_steps_per_execution_numeric_equivalence(self, steps_per_execution):
        np.random.seed(1337)
        x = np.random.normal(size=(64, 8)).astype("float32")
        y = np.random.normal(size=(64, 2)).astype("float32")

        init_weights = [
            np.random.normal(size=(8, 2)).astype("float32"),
            np.zeros((2,)).astype("float32"),
        ]

        # Model with steps_per_execution=1
        model_1 = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model_1.compile(
            loss="mse",
            optimizer="sgd",
            steps_per_execution=1,
            jit_compile=True,
        )
        model_1.set_weights(init_weights)
        h1 = model_1.fit(x, y, batch_size=8, epochs=10, verbose=0)

        # Model with steps_per_execution > 1
        model_spe = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model_spe.compile(
            loss="mse",
            optimizer="sgd",
            steps_per_execution=steps_per_execution,
            jit_compile=True,
        )
        model_spe.set_weights(init_weights)
        h_spe = model_spe.fit(x, y, batch_size=8, epochs=10, verbose=0)

        self.assertAllClose(
            h1.history["loss"], h_spe.history["loss"], rtol=1e-4, atol=1e-4
        )
        for w1, w2 in zip(model_1.get_weights(), model_spe.get_weights()):
            self.assertAllClose(w1, w2, rtol=1e-4, atol=1e-4)
        self.assertAllClose(
            model_1.predict(x, batch_size=8, verbose=0),
            model_spe.predict(x, batch_size=8, verbose=0),
            rtol=1e-4,
            atol=1e-4,
        )
        self.assertAllClose(
            model_1.evaluate(x, y, batch_size=8, verbose=0),
            model_spe.evaluate(x, y, batch_size=8, verbose=0),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_steps_per_execution_remainder_batches(self):
        x = np.random.normal(size=(40, 8)).astype("float32")
        y = np.random.normal(size=(40, 2)).astype("float32")

        model = models.Sequential([layers.Dense(2, input_shape=(8,))])
        model.compile(
            loss="mse",
            optimizer="adam",
            steps_per_execution=3,
            jit_compile=True,
        )
        history = model.fit(x, y, batch_size=4, epochs=2, verbose=0)
        self.assertLen(history.history["loss"], 2)
        preds = model.predict(x, batch_size=4, verbose=0)
        self.assertEqual(preds.shape, (40, 2))
        eval_loss = model.evaluate(x, y, batch_size=4, verbose=0)
        self.assertIsInstance(eval_loss, float)
