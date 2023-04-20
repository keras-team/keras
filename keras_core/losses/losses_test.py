import numpy as np

from keras_core import operations as ops
from keras_core import testing
from keras_core.losses import losses


class MeanSquaredErrorTest(testing.TestCase):
    def test_config(self):
        # TODO
        pass

    def test_all_correct_unweighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[4, 8, 12], [8, 1, 3]])
        loss = mse_obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 49.5)

    def test_scalar_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 113.85)

    def test_sample_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 767.8 / 6)

    def test_timestep_weighted(self):
        # TODO
        pass

    def test_zero_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0)

    def test_invalid_sample_weight(self):
        # TODO
        pass

    def test_no_reduction(self):
        mse_obj = losses.MeanSquaredError(reduction=None)
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, [84.3333, 143.3666])

    def test_sum_reduction(self):
        mse_obj = losses.MeanSquaredError(reduction="sum")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 227.69998)


class MeanAbsoluteErrorTest(testing.TestCase):
    def test_config(self):
        # TODO
        pass

    def test_all_correct_unweighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[4, 8, 12], [8, 1, 3]])
        loss = mae_obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 5.5)

    def test_scalar_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 12.65)

    def test_sample_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = mae_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 81.4 / 6)

    def test_timestep_weighted(self):
        # TODO
        pass

    def test_zero_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0)

    def test_invalid_sample_weight(self):
        # TODO
        pass

    def test_no_reduction(self):
        mae_obj = losses.MeanAbsoluteError(reduction=None)
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, [10.7333, 14.5666])

    def test_sum_reduction(self):
        mae_obj = losses.MeanAbsoluteError(reduction="sum")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 25.29999)


class MeanAbsolutePercentageErrorTest(testing.TestCase):
    def test_config(self):
        # TODO
        pass

    def test_all_correct_unweighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[4, 8, 12], [8, 1, 3]])
        loss = mape_obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 211.8518, 3)

    def test_scalar_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 487.259, 3)

    def test_sample_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = mape_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 422.8888, 3)

    def test_timestep_weighted(self):
        # TODO
        pass

    def test_zero_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_no_reduction(self):
        mape_obj = losses.MeanAbsolutePercentageError(reduction=None)
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, [621.8518, 352.6666])

class MeanSquaredLogarithmicErrorTest(testing.TestCase):
    def test_config(self):
        # TODO
        pass

    def test_unweighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = msle_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 1.4370, 3)

    def test_scalar_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = msle_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 3.3051, 3)

    def test_sample_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        sample_weight = np.array([[1.2], [3.4]])
        loss = msle_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 3.7856, 3)

    def test_timestep_weighted(self):
        # TODO
        pass

    def test_zero_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = msle_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)
