import numpy as np

from keras_core import testing
from keras_core.metrics import regression_metrics as metrics


class MeanSquaredErrorTest(testing.TestCase):
    def test_config(self):
        # TODO
        pass

    def test_unweighted(self):
        mse_obj = metrics.MeanSquaredError()
        y_true = np.array(
            [[0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1]]
        )
        y_pred = np.array(
            [[0, 0, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]]
        )

        mse_obj.update_state(y_true, y_pred)
        result = mse_obj.result()
        self.assertAllClose(0.5, result, atol=1e-5)

    def test_weighted(self):
        mse_obj = metrics.MeanSquaredError()
        y_true = np.array(
            [[0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1]]
        )
        y_pred = np.array(
            [[0, 0, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]]
        )
        sample_weight = np.array([1.0, 1.5, 2.0, 2.5])
        result = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.54285, result, atol=1e-5)


class CosineSimilarityTest(testing.TestCase):
    def l2_norm(self, x, axis):
        epsilon = 1e-12
        square_sum = np.sum(np.square(x), axis=axis, keepdims=True)
        x_inv_norm = 1 / np.sqrt(np.maximum(square_sum, epsilon))
        return np.multiply(x, x_inv_norm)

    def setup(self, axis=1):
        self.np_y_true = np.asarray([[1, 9, 2], [-5, -2, 6]], dtype=np.float32)
        self.np_y_pred = np.asarray([[4, 8, 12], [8, 1, 3]], dtype=np.float32)

        y_true = self.l2_norm(self.np_y_true, axis)
        y_pred = self.l2_norm(self.np_y_pred, axis)
        self.expected_loss = np.sum(np.multiply(y_true, y_pred), axis=(axis,))

        self.y_true = self.np_y_true
        self.y_pred = self.np_y_pred

    def test_config(self):
        cosine_obj = metrics.CosineSimilarity(
            axis=2, name="my_cos", dtype="int32"
        )
        self.assertEqual(cosine_obj.name, "my_cos")
        self.assertEqual(cosine_obj.dtype, "int32")

        # Check save and restore config
        cosine_obj2 = metrics.CosineSimilarity.from_config(
            cosine_obj.get_config()
        )
        self.assertEqual(cosine_obj2.name, "my_cos")
        self.assertEqual(cosine_obj2._dtype, "int32")

    def test_unweighted(self):
        self.setup()
        cosine_obj = metrics.CosineSimilarity()
        loss = cosine_obj(self.y_true, self.y_pred)
        expected_loss = np.mean(self.expected_loss)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_weighted(self):
        self.setup()
        cosine_obj = metrics.CosineSimilarity()
        sample_weight = np.asarray([1.2, 3.4])
        loss = cosine_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        expected_loss = np.sum(self.expected_loss * sample_weight) / np.sum(
            sample_weight
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_axis(self):
        self.setup(axis=1)
        cosine_obj = metrics.CosineSimilarity(axis=1)
        loss = cosine_obj(self.y_true, self.y_pred)
        expected_loss = np.mean(self.expected_loss)
        self.assertAlmostEqual(loss, expected_loss, 3)


class MeanAbsoluteErrorTest(testing.TestCase):
    def test_config(self):
        mae_obj = metrics.MeanAbsoluteError(name="my_mae", dtype="int32")
        self.assertEqual(mae_obj.name, "my_mae")
        self.assertEqual(mae_obj._dtype, "int32")

        # Check save and restore config
        mae_obj2 = metrics.MeanAbsoluteError.from_config(mae_obj.get_config())
        self.assertEqual(mae_obj2.name, "my_mae")
        self.assertEqual(mae_obj2._dtype, "int32")

    def test_unweighted(self):
        mae_obj = metrics.MeanAbsoluteError()
        y_true = np.array(
            [[0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1]]
        )
        y_pred = np.array(
            [[0, 0, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]]
        )

        mae_obj.update_state(y_true, y_pred)
        result = mae_obj.result()
        self.assertAllClose(0.5, result, atol=1e-5)

    def test_weighted(self):
        mae_obj = metrics.MeanAbsoluteError()
        y_true = np.array(
            [[0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1]]
        )
        y_pred = np.array(
            [[0, 0, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]]
        )
        sample_weight = np.array([1.0, 1.5, 2.0, 2.5])
        result = mae_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.54285, result, atol=1e-5)


class MeanAbsolutePercentageErrorTest(testing.TestCase):
    def test_config(self):
        mape_obj = metrics.MeanAbsolutePercentageError(
            name="my_mape", dtype="int32"
        )
        self.assertEqual(mape_obj.name, "my_mape")
        self.assertEqual(mape_obj._dtype, "int32")

        # Check save and restore config
        mape_obj2 = metrics.MeanAbsolutePercentageError.from_config(
            mape_obj.get_config()
        )
        self.assertEqual(mape_obj2.name, "my_mape")
        self.assertEqual(mape_obj2._dtype, "int32")

    def test_unweighted(self):
        mape_obj = metrics.MeanAbsolutePercentageError()
        y_true = np.array(
            [[0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1]]
        )
        y_pred = np.array(
            [
                [0, 0, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype="float32",
        )

        result = mape_obj(y_true, y_pred)
        self.assertAllClose(35e7, result, atol=1e-5)

    def test_weighted(self):
        mape_obj = metrics.MeanAbsolutePercentageError()
        y_true = np.array(
            [[0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1]]
        )
        y_pred = np.array(
            [
                [0, 0, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [0, 1, 0, 1, 0],
                [1, 1, 1, 1, 1],
            ],
            dtype="float32",
        )

        sample_weight = np.array([1.0, 1.5, 2.0, 2.5])
        result = mape_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(40e7, result, atol=1e-5)


class MeanSquaredLogarithmicErrorTest(testing.TestCase):
    def test_config(self):
        msle_obj = metrics.MeanSquaredLogarithmicError(
            name="my_msle", dtype="int32"
        )
        self.assertEqual(msle_obj.name, "my_msle")
        self.assertEqual(msle_obj._dtype, "int32")

        # Check save and restore config
        msle_obj2 = metrics.MeanSquaredLogarithmicError.from_config(
            msle_obj.get_config()
        )
        self.assertEqual(msle_obj2.name, "my_msle")
        self.assertEqual(msle_obj2._dtype, "int32")

    def test_unweighted(self):
        msle_obj = metrics.MeanSquaredLogarithmicError()
        y_true = np.array(
            [[0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1]]
        )
        y_pred = np.array(
            [[0, 0, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]]
        )

        msle_obj.update_state(y_true, y_pred)
        result = msle_obj.result()
        self.assertAllClose(0.24022, result, atol=1e-5)

    def test_weighted(self):
        msle_obj = metrics.MeanSquaredLogarithmicError()
        y_true = np.array(
            [[0, 1, 0, 1, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 0], [0, 0, 0, 0, 1]]
        )
        y_pred = np.array(
            [[0, 0, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0], [1, 1, 1, 1, 1]]
        )
        sample_weight = np.array([1.0, 1.5, 2.0, 2.5])
        result = msle_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.26082, result, atol=1e-5)


class LogCoshErrorTest(testing.TestCase):
    def setup(self):
        y_true = np.asarray([[1, 9, 2], [-5, -2, 6]], dtype=np.float32)
        y_pred = np.asarray([[4, 8, 12], [8, 1, 3]], dtype=np.float32)

        self.batch_size = 6
        error = y_pred - y_true
        self.expected_results = np.log((np.exp(error) + np.exp(-error)) / 2)

        self.y_pred = y_pred
        self.y_true = y_true

    def test_config(self):
        logcosh_obj = metrics.LogCoshError(name="logcosh", dtype="int32")
        self.assertEqual(logcosh_obj.name, "logcosh")
        self.assertEqual(logcosh_obj._dtype, "int32")

    def test_unweighted(self):
        self.setup()
        logcosh_obj = metrics.LogCoshError()

        logcosh_obj.update_state(self.y_true, self.y_pred)
        result = logcosh_obj.result()
        expected_result = np.sum(self.expected_results) / self.batch_size
        self.assertAllClose(result, expected_result, atol=1e-3)

    def test_weighted(self):
        self.setup()
        logcosh_obj = metrics.LogCoshError(dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        result = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )

        sample_weight = np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape(
            (2, 3)
        )
        expected_result = np.multiply(self.expected_results, sample_weight)
        expected_result = np.sum(expected_result) / np.sum(sample_weight)
        self.assertAllClose(result, expected_result, atol=1e-3)
