import numpy as np

from keras import metrics
from keras import testing


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
        self.run_class_serialization_test(metrics.Poisson(name="poisson"))

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


class BinaryCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            metrics.BinaryCrossentropy(
                name="bce", dtype="int32", label_smoothing=0.2
            )
        )

    def test_unweighted(self):
        bce_obj = metrics.BinaryCrossentropy()
        y_true = np.array([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.array([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
        result = bce_obj(y_true, y_pred)
        self.assertAllClose(result, 3.9855, atol=1e-3)

    def test_unweighted_with_logits(self):
        bce_obj = metrics.BinaryCrossentropy(from_logits=True)

        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        y_pred = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        result = bce_obj(y_true, y_pred)
        self.assertAllClose(result, 3.333, atol=1e-3)

    def test_weighted(self):
        bce_obj = metrics.BinaryCrossentropy()
        y_true = np.array([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.array([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
        sample_weight = np.array([1.5, 2.0])
        result = bce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(result, 3.4162, atol=1e-3)

    def test_weighted_from_logits(self):
        bce_obj = metrics.BinaryCrossentropy(from_logits=True)
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        y_pred = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        sample_weight = np.array([2.0, 2.5])
        result = bce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(result, 3.7037, atol=1e-3)

    def test_label_smoothing(self):
        logits = np.array(((10.0, -10.0, -10.0)))
        y_true = np.array(((1, 0, 1)))
        label_smoothing = 0.1
        bce_obj = metrics.BinaryCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        result = bce_obj(y_true, logits)
        expected_value = (10.0 + 5.0 * label_smoothing) / 3.0
        self.assertAllClose(expected_value, result, atol=1e-3)


class CategoricalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            metrics.CategoricalCrossentropy(
                name="cce", dtype="int32", label_smoothing=0.2
            )
        )

    def test_unweighted(self):
        cce_obj = metrics.CategoricalCrossentropy()
        y_true = np.array([[0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        result = cce_obj(y_true, y_pred)
        self.assertAllClose(result, 1.176, atol=1e-3)

    def test_unweighted_from_logits(self):
        cce_obj = metrics.CategoricalCrossentropy(from_logits=True)

        y_true = np.array([[0, 1, 0], [0, 0, 1]])
        logits = np.array([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
        result = cce_obj(y_true, logits)
        self.assertAllClose(result, 3.5011, atol=1e-3)

    def test_weighted(self):
        cce_obj = metrics.CategoricalCrossentropy()

        y_true = np.array([[0, 1, 0], [0, 0, 1]])
        y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        sample_weight = np.array([1.5, 2.0])
        result = cce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(result, 1.338, atol=1e-3)

    def test_weighted_from_logits(self):
        cce_obj = metrics.CategoricalCrossentropy(from_logits=True)

        y_true = np.array([[0, 1, 0], [0, 0, 1]])
        logits = np.array([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
        sample_weight = np.array([1.5, 2.0])
        result = cce_obj(y_true, logits, sample_weight=sample_weight)
        self.assertAllClose(result, 4.0012, atol=1e-3)

    def test_label_smoothing(self):
        y_true = np.array([[0, 1, 0], [0, 0, 1]])
        logits = np.array([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
        label_smoothing = 0.1
        cce_obj = metrics.CategoricalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose(loss, 3.667, atol=1e-3)


class SparseCategoricalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            metrics.SparseCategoricalCrossentropy(name="scce", dtype="int32")
        )

    def test_unweighted(self):
        scce_obj = metrics.SparseCategoricalCrossentropy()

        y_true = np.array([1, 2])
        y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        result = scce_obj(y_true, y_pred)
        self.assertAllClose(result, 1.176, atol=1e-3)

    def test_unweighted_from_logits(self):
        scce_obj = metrics.SparseCategoricalCrossentropy(from_logits=True)

        y_true = np.array([1, 2])
        logits = np.array([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
        result = scce_obj(y_true, logits)
        self.assertAllClose(result, 3.5011, atol=1e-3)

    def test_weighted(self):
        scce_obj = metrics.SparseCategoricalCrossentropy()

        y_true = np.array([1, 2])
        y_pred = np.array([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
        sample_weight = np.array([1.5, 2.0])
        result = scce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(result, 1.338, atol=1e-3)

    def test_weighted_from_logits(self):
        scce_obj = metrics.SparseCategoricalCrossentropy(from_logits=True)

        y_true = np.array([1, 2])
        logits = np.array([[1, 9, 0], [1, 8, 1]], dtype=np.float32)
        sample_weight = np.array([1.5, 2.0])
        result = scce_obj(y_true, logits, sample_weight=sample_weight)
        self.assertAllClose(result, 4.0012, atol=1e-3)
