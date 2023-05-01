import numpy as np

from keras_core import metrics
from keras_core import testing


class KLDivergenceTest(testing.TestCase):
    def setup(self):
        self.y_pred = np.asarray(
            [0.4, 0.9, 0.12, 0.36, 0.3, 0.4], dtype=np.float32
        ).reshape((2, 3))
        self.y_true = np.asarray(
            [0.5, 0.8, 0.12, 0.7, 0.43, 0.8], dtype=np.float32
        ).reshape((2, 3))

        self.batch_size = 2
        self.expected_results = np.multiply(
            self.y_true, np.log(self.y_true / self.y_pred)
        )

    def test_config(self):
        k_obj = metrics.KLDivergence(name="kld", dtype="int32")
        self.assertEqual(k_obj.name, "kld")
        self.assertEqual(k_obj._dtype, "int32")

        k_obj2 = metrics.KLDivergence.from_config(k_obj.get_config())
        self.assertEqual(k_obj2.name, "kld")
        self.assertEqual(k_obj2._dtype, "int32")

    def test_unweighted(self):
        self.setup()
        k_obj = metrics.KLDivergence()

        k_obj.update_state(self.y_true, self.y_pred)
        result = k_obj.result()
        expected_result = np.sum(self.expected_results) / self.batch_size
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_weighted(self):
        self.setup()
        k_obj = metrics.KLDivergence()

        sample_weight = np.asarray([1.2, 3.4], dtype=np.float32).reshape((2, 1))
        result = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

        sample_weight = np.asarray(
            [1.2, 1.2, 1.2, 3.4, 3.4, 3.4], dtype=np.float32
        ).reshape((2, 3))
        expected_result = np.multiply(self.expected_results, sample_weight)
        expected_result = np.sum(expected_result) / (1.2 + 3.4)
        self.assertAllClose(result, expected_result, atol=1e-3)


class PoissonTest(testing.TestCase):
    def setup(self):
        self.y_pred = np.asarray([1, 9, 2, 5, 2, 6], dtype=np.float32).reshape(
            (2, 3)
        )
        self.y_true = np.asarray([4, 8, 12, 8, 1, 3], dtype=np.float32).reshape(
            (2, 3)
        )
        self.batch_size = 6
        self.expected_results = self.y_pred - np.multiply(
            self.y_true, np.log(self.y_pred)
        )

    def test_config(self):
        poisson_obj = metrics.Poisson(name="poisson", dtype="float32")
        self.assertEqual(poisson_obj.name, "poisson")
        self.assertEqual(poisson_obj._dtype, "float32")

        poisson_obj2 = metrics.Poisson.from_config(poisson_obj.get_config())
        self.assertEqual(poisson_obj2.name, "poisson")
        self.assertEqual(poisson_obj2._dtype, "float32")

    def test_unweighted(self):
        self.setup()
        poisson_obj = metrics.Poisson()
        poisson_obj.update_state(self.y_true, self.y_pred)

        result = poisson_obj.result()
        expected_result = np.sum(self.expected_results) / self.batch_size
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_weighted(self):
        self.setup()
        poisson_obj = metrics.Poisson()
        sample_weight = np.asarray([1.2, 3.4], dtype=np.float32).reshape((2, 1))

        result = poisson_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        sample_weight = np.asarray(
            [1.2, 1.2, 1.2, 3.4, 3.4, 3.4], dtype=np.float32
        ).reshape((2, 3))
        expected_result = np.multiply(self.expected_results, sample_weight)
        expected_result = np.sum(expected_result) / np.sum(sample_weight)
        self.assertAllClose(result, expected_result, atol=1e-3)
