import numpy as np
import pytest

from keras.src import backend
from keras.src import testing
from keras.src.losses import losses


class MeanSquaredErrorTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(losses.MeanSquaredError(name="mymse"))

    def test_base_function_reduction(self):
        mse_fn = losses.mean_squared_error
        y_true = np.array([4, 8, 12])
        y_pred = np.array([[3], [0], [1]])
        loss = mse_fn(y_true, y_pred)
        self.assertEqual(backend.shape(loss), (3,))

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
        mse_obj = losses.MeanSquaredError()
        y_true = np.asarray([1, 9, 2, -5, -2, 6]).reshape(2, 3, 1)
        y_pred = np.asarray([4, 8, 12, 8, 1, 3]).reshape(2, 3, 1)
        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
        loss = mse_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        self.assertAlmostEqual(loss, 97.833336)

    def test_zero_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0)

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

    def test_dtype_arg(self):
        mse_obj = losses.MeanSquaredError(dtype="bfloat16")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mse_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class MeanAbsoluteErrorTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.MeanAbsoluteError(name="myname")
        )

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
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.asarray([1, 9, 2, -5, -2, 6]).reshape(2, 3, 1)
        y_pred = np.asarray([4, 8, 12, 8, 1, 3]).reshape(2, 3, 1)
        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
        loss = mae_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        self.assertAlmostEqual(loss, 13.833333)

    def test_zero_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0)

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

    def test_dtype_arg(self):
        mae_obj = losses.MeanAbsoluteError(dtype="bfloat16")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mae_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class MeanAbsolutePercentageErrorTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.MeanAbsolutePercentageError(name="mymape")
        )

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
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = np.asarray([1, 9, 2, -5, -2, 6]).reshape(2, 3, 1)
        y_pred = np.asarray([4, 8, 12, 8, 1, 3]).reshape(2, 3, 1)
        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
        loss = mape_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        self.assertAlmostEqual(loss, 694.4444)

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

    def test_dtype_arg(self):
        mape_obj = losses.MeanAbsolutePercentageError(dtype="bfloat16")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = mape_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class MeanSquaredLogarithmicErrorTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.MeanSquaredLogarithmicError(name="mysloge")
        )

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
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.asarray([1, 9, 2, -5, -2, 6]).reshape(2, 3, 1)
        y_pred = np.asarray([4, 8, 12, 8, 1, 3]).reshape(2, 3, 1)
        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3))
        loss = msle_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        self.assertAlmostEqual(loss, 2.647374)

    def test_zero_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = msle_obj(y_true, y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_dtype_arg(self):
        msle_obj = losses.MeanSquaredLogarithmicError(dtype="bfloat16")
        y_true = np.array([[1, 9, 2], [-5, -2, 6]])
        y_pred = np.array([[4, 8, 12], [8, 1, 3]], dtype="float32")
        loss = msle_obj(y_true, y_pred, sample_weight=2.3)
        self.assertDType(loss, "bfloat16")


class HingeTest(testing.TestCase):
    def test_unweighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.Hinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 1.3, 3)

        # Reduction = "sum"
        hinge_obj = losses.Hinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 2.6, 3)

        # Reduction = None
        hinge_obj = losses.Hinge(reduction=None)
        loss = hinge_obj(y_true, y_pred)
        self.assertAllClose(loss, [1.1, 1.5])

        # Bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            losses.Hinge(reduction="abc")

    def test_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = [1, 0]

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.Hinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.55, 3)

        # Reduction = "sum"
        hinge_obj = losses.Hinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.1, 3)

        # Reduction = None
        hinge_obj = losses.Hinge(reduction=None)
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, [1.1, 0.0])

    def test_zero_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = 0.0

        hinge_obj = losses.Hinge()
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss, 0.0)

    def test_dtype_arg(self):
        hinge_obj = losses.Hinge(dtype="bfloat16")
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        loss = hinge_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class SquaredHingeTest(testing.TestCase):
    def test_unweighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.SquaredHinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 1.86, 3)

        # Reduction = "sum"
        hinge_obj = losses.SquaredHinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 3.72, 3)

        # Reduction = None
        hinge_obj = losses.SquaredHinge(reduction=None)
        loss = hinge_obj(y_true, y_pred)
        self.assertAllClose(loss, [1.46, 2.26])

        # Bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            losses.SquaredHinge(reduction="abc")

    def test_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = [1, 0]

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.SquaredHinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.73, 3)

        # Reduction = "sum"
        hinge_obj = losses.SquaredHinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.46, 3)

        # Reduction = None
        hinge_obj = losses.SquaredHinge(reduction=None)
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, [1.46, 0.0])

    def test_zero_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = 0.0

        hinge_obj = losses.SquaredHinge()
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss, 0.0)

    def test_dtype_arg(self):
        hinge_obj = losses.SquaredHinge(dtype="bfloat16")
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        loss = hinge_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class CategoricalHingeTest(testing.TestCase):
    def test_unweighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.CategoricalHinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 1.4, 3)

        # Reduction = "sum"
        hinge_obj = losses.CategoricalHinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 2.8, 3)

        # Reduction = None
        hinge_obj = losses.CategoricalHinge(reduction=None)
        loss = hinge_obj(y_true, y_pred)
        self.assertAllClose(loss, [1.2, 1.6])

        # Bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            losses.CategoricalHinge(reduction="abc")

    def test_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = [1, 0]

        # Reduction = "sum_over_batch_size"
        hinge_obj = losses.CategoricalHinge(reduction="sum_over_batch_size")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.6, 3)

        # Reduction = "sum"
        hinge_obj = losses.CategoricalHinge(reduction="sum")
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.2, 3)

        # Reduction = None
        hinge_obj = losses.CategoricalHinge(reduction=None)
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, [1.2, 0.0])

    def test_zero_weighted(self):
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        sample_weight = 0.0

        hinge_obj = losses.CategoricalHinge()
        loss = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(loss, 0.0)

    def test_dtype_arg(self):
        hinge_obj = losses.CategoricalHinge(dtype="bfloat16")
        y_true = np.array([[0.0, 1.0], [0.0, 0.0]])
        y_pred = np.array([[0.6, 0.4], [0.4, 0.6]])
        loss = hinge_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


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
        cosine_obj = losses.CosineSimilarity(
            axis=2, reduction="sum", name="cosine_loss"
        )
        self.assertEqual(cosine_obj.name, "cosine_loss")
        self.assertEqual(cosine_obj.reduction, "sum")
        config = cosine_obj.get_config()
        self.assertEqual(config, {"name": "cosine_loss", "reduction": "sum"})

    def test_unweighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        loss = cosine_obj(self.y_true, self.y_pred)
        expected_loss = -np.mean(self.expected_loss)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        sample_weight = 2.3
        loss = cosine_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        expected_loss = -np.mean(self.expected_loss * sample_weight)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_sample_weighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        sample_weight = np.asarray([1.2, 3.4])
        loss = cosine_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        expected_loss = -np.mean(self.expected_loss * sample_weight)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        np_y_true = self.np_y_true.reshape((2, 3, 1))
        np_y_pred = self.np_y_pred.reshape((2, 3, 1))
        sample_weight = np.asarray([3, 6, 5, 0, 4, 2]).reshape((2, 3))

        y_true = self.l2_norm(np_y_true, 2)
        y_pred = self.l2_norm(np_y_pred, 2)
        expected_loss = np.sum(np.multiply(y_true, y_pred), axis=(2,))

        y_true = np_y_true
        y_pred = np_y_pred
        loss = cosine_obj(y_true, y_pred, sample_weight=sample_weight)

        expected_loss = -np.mean(expected_loss * sample_weight)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity()
        loss = cosine_obj(self.y_true, self.y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_axis(self):
        self.setup(axis=1)
        cosine_obj = losses.CosineSimilarity(axis=1)
        loss = cosine_obj(self.y_true, self.y_pred)
        expected_loss = -np.mean(self.expected_loss)
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_dtype_arg(self):
        self.setup()
        cosine_obj = losses.CosineSimilarity(dtype="bfloat16")
        loss = cosine_obj(self.y_true, self.y_pred)
        self.assertDType(loss, "bfloat16")


class HuberLossTest(testing.TestCase):
    def huber_loss(self, y_true, y_pred, delta=1.0):
        error = y_pred - y_true
        abs_error = np.abs(error)

        quadratic = np.minimum(abs_error, delta)
        linear = np.subtract(abs_error, quadratic)
        return np.add(
            np.multiply(0.5, np.multiply(quadratic, quadratic)),
            np.multiply(delta, linear),
        )

    def setup(self, delta=1.0):
        self.np_y_pred = np.array([[0.9, 0.2, 0.2], [0.8, 0.4, 0.6]])
        self.np_y_true = np.array([[1.0, 0.0, 1.0], [1.0, 0.0, 0.0]])

        self.batch_size = 6
        self.expected_losses = self.huber_loss(
            self.np_y_true, self.np_y_pred, delta
        )

        self.y_pred = self.np_y_pred
        self.y_true = self.np_y_true

    def test_config(self):
        h_obj = losses.Huber(reduction="sum", name="huber")
        self.assertEqual(h_obj.name, "huber")
        self.assertEqual(h_obj.reduction, "sum")
        config = h_obj.get_config()
        self.assertEqual(config, {"name": "huber", "reduction": "sum"})

    def test_all_correct(self):
        self.setup()
        h_obj = losses.Huber()
        loss = h_obj(self.y_true, self.y_true)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        self.setup()
        h_obj = losses.Huber()
        loss = h_obj(self.y_true, self.y_pred)
        actual_loss = np.sum(self.expected_losses) / self.batch_size
        self.assertAlmostEqual(loss, actual_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        h_obj = losses.Huber()
        sample_weight = 2.3
        loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        actual_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, actual_loss, 3)

        # Verify we get the same output when the same input is given
        loss_2 = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, loss_2, 3)

    def test_sample_weighted(self):
        self.setup()
        h_obj = losses.Huber()
        sample_weight = np.array([[1.2], [3.4]])

        loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        actual_loss = np.multiply(
            self.expected_losses,
            np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)),
        )
        actual_loss = np.sum(actual_loss) / self.batch_size
        self.assertAlmostEqual(loss, actual_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        h_obj = losses.Huber()
        y_pred = self.np_y_pred.reshape((2, 3, 1))
        y_true = self.np_y_true.reshape((2, 3, 1))
        expected_losses = self.huber_loss(y_true, y_pred)

        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3, 1))
        loss = h_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        actual_loss = np.multiply(expected_losses, sample_weight)
        actual_loss = np.sum(actual_loss) / self.batch_size
        self.assertAlmostEqual(loss, actual_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        h_obj = losses.Huber()
        sample_weight = 0
        loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_non_default_delta(self):
        self.setup(delta=0.8)
        h_obj = losses.Huber(delta=0.8)
        sample_weight = 2.3
        loss = h_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        actual_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, actual_loss, 3)

    def test_dtype_arg(self):
        self.setup()
        h_obj = losses.Huber(dtype="bfloat16")
        loss = h_obj(self.y_true, self.y_pred)
        self.assertDType(loss, "bfloat16")


class LogCoshTest(testing.TestCase):
    def setup(self):
        y_true = np.asarray([[1, 9, 2], [-5, -2, 6]], dtype=np.float32)
        y_pred = np.asarray([[4, 8, 12], [8, 1, 3]], dtype=np.float32)

        self.batch_size = 6
        error = y_pred - y_true
        self.expected_losses = np.log((np.exp(error) + np.exp(-error)) / 2)

        self.y_true = y_true
        self.y_pred = y_pred

    def test_config(self):
        logcosh_obj = losses.LogCosh(reduction="sum", name="logcosh_loss")
        self.assertEqual(logcosh_obj.name, "logcosh_loss")
        self.assertEqual(logcosh_obj.reduction, "sum")
        config = logcosh_obj.get_config()
        self.assertEqual(config, {"name": "logcosh_loss", "reduction": "sum"})

    def test_unweighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()

        loss = logcosh_obj(self.y_true, self.y_pred)
        expected_loss = np.sum(self.expected_losses) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()
        sample_weight = 2.3

        loss = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        expected_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

        # Verify we get the same output when the same input is given
        loss_2 = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        self.assertAlmostEqual(loss, loss_2, 3)

    def test_sample_weighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()

        sample_weight = np.asarray([1.2, 3.4])
        loss = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )

        expected_loss = np.multiply(
            self.expected_losses,
            np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)),
        )
        expected_loss = np.sum(expected_loss) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()
        y_true = np.asarray([1, 9, 2, -5, -2, 6]).reshape(2, 3, 1)
        y_pred = np.asarray([4, 8, 12, 8, 1, 3]).reshape(2, 3, 1)
        error = y_pred - y_true
        expected_losses = np.log((np.exp(error) + np.exp(-error)) / 2)
        sample_weight = np.array([3, 6, 5, 0, 4, 2]).reshape((2, 3, 1))

        loss = logcosh_obj(
            y_true,
            y_pred,
            sample_weight=sample_weight,
        )
        expected_loss = (
            np.sum(expected_losses * sample_weight) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        logcosh_obj = losses.LogCosh()
        sample_weight = 0
        loss = logcosh_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_dtype_arg(self):
        self.setup()
        logcosh_obj = losses.LogCosh(dtype="bfloat16")
        loss = logcosh_obj(self.y_true, self.y_pred)
        self.assertDType(loss, "bfloat16")


class KLDivergenceTest(testing.TestCase):
    def setup(self):
        self.y_pred = np.asarray(
            [0.4, 0.9, 0.12, 0.36, 0.3, 0.4], dtype=np.float32
        ).reshape((2, 3))
        self.y_true = np.asarray(
            [0.5, 0.8, 0.12, 0.7, 0.43, 0.8], dtype=np.float32
        ).reshape((2, 3))

        self.batch_size = 2
        self.expected_losses = np.multiply(
            self.y_true, np.log(self.y_true / self.y_pred)
        )

    def test_config(self):
        k_obj = losses.KLDivergence(reduction="sum", name="kld")
        self.assertEqual(k_obj.name, "kld")
        self.assertEqual(k_obj.reduction, "sum")

    def test_unweighted(self):
        self.setup()
        k_obj = losses.KLDivergence()

        loss = k_obj(self.y_true, self.y_pred)
        expected_loss = np.sum(self.expected_losses) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        k_obj = losses.KLDivergence()
        sample_weight = 2.3

        loss = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        expected_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

        # Verify we get the same output when the same input is given
        loss_2 = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, loss_2, 3)

    def test_sample_weighted(self):
        self.setup()
        k_obj = losses.KLDivergence()
        sample_weight = np.asarray([1.2, 3.4], dtype=np.float32).reshape((2, 1))
        loss = k_obj(self.y_true, self.y_pred, sample_weight=sample_weight)

        expected_loss = np.multiply(
            self.expected_losses,
            np.asarray(
                [1.2, 1.2, 1.2, 3.4, 3.4, 3.4], dtype=np.float32
            ).reshape(2, 3),
        )
        expected_loss = np.sum(expected_loss) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        k_obj = losses.KLDivergence()
        y_true = self.y_true.reshape(2, 3, 1)
        y_pred = self.y_pred.reshape(2, 3, 1)
        sample_weight = np.asarray([3, 6, 5, 0, 4, 2]).reshape(2, 3)
        expected_losses = np.sum(
            np.multiply(y_true, np.log(y_true / y_pred)), axis=-1
        )
        loss = k_obj(y_true, y_pred, sample_weight=sample_weight)

        num_timesteps = 3
        expected_loss = np.sum(expected_losses * sample_weight) / (
            self.batch_size * num_timesteps
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        k_obj = losses.KLDivergence()
        loss = k_obj(self.y_true, self.y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_dtype_arg(self):
        self.setup()
        k_obj = losses.KLDivergence(dtype="bfloat16")
        loss = k_obj(self.y_true, self.y_pred)
        self.assertDType(loss, "bfloat16")


class PoissonTest(testing.TestCase):
    def setup(self):
        self.y_pred = np.asarray([1, 9, 2, 5, 2, 6], dtype=np.float32).reshape(
            (2, 3)
        )
        self.y_true = np.asarray([4, 8, 12, 8, 1, 3], dtype=np.float32).reshape(
            (2, 3)
        )

        self.batch_size = 6
        self.expected_losses = self.y_pred - np.multiply(
            self.y_true, np.log(self.y_pred)
        )

    def test_config(self):
        poisson_obj = losses.Poisson(reduction="sum", name="poisson")
        self.assertEqual(poisson_obj.name, "poisson")
        self.assertEqual(poisson_obj.reduction, "sum")

    def test_unweighted(self):
        self.setup()
        poisson_obj = losses.Poisson()

        loss = poisson_obj(self.y_true, self.y_pred)
        expected_loss = np.sum(self.expected_losses) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_scalar_weighted(self):
        self.setup()
        poisson_obj = losses.Poisson()
        sample_weight = 2.3
        loss = poisson_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        expected_loss = (
            sample_weight * np.sum(self.expected_losses) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)
        self.assertAlmostEqual(loss, expected_loss, 3)

        # Verify we get the same output when the same input is given
        loss_2 = poisson_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )
        self.assertAlmostEqual(loss, loss_2, 3)

    def test_sample_weighted(self):
        self.setup()
        poisson_obj = losses.Poisson()

        sample_weight = np.asarray([1.2, 3.4]).reshape((2, 1))
        loss = poisson_obj(
            self.y_true, self.y_pred, sample_weight=sample_weight
        )

        expected_loss = np.multiply(
            self.expected_losses,
            np.asarray([1.2, 1.2, 1.2, 3.4, 3.4, 3.4]).reshape((2, 3)),
        )
        expected_loss = np.sum(expected_loss) / self.batch_size
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_timestep_weighted(self):
        self.setup()
        poisson_obj = losses.Poisson()
        y_true = self.y_true.reshape(2, 3, 1)
        y_pred = self.y_pred.reshape(2, 3, 1)
        sample_weight = np.asarray([3, 6, 5, 0, 4, 2]).reshape(2, 3, 1)
        expected_losses = y_pred - np.multiply(y_true, np.log(y_pred))

        loss = poisson_obj(
            y_true,
            y_pred,
            sample_weight=np.asarray(sample_weight).reshape((2, 3)),
        )
        expected_loss = (
            np.sum(expected_losses * sample_weight) / self.batch_size
        )
        self.assertAlmostEqual(loss, expected_loss, 3)

    def test_zero_weighted(self):
        self.setup()
        poisson_obj = losses.Poisson()
        loss = poisson_obj(self.y_true, self.y_pred, sample_weight=0)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_dtype_arg(self):
        self.setup()
        poisson_obj = losses.Poisson(dtype="bfloat16")
        loss = poisson_obj(self.y_true, self.y_pred)
        self.assertDType(loss, "bfloat16")


class BinaryCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.BinaryCrossentropy(name="bce", axis=-1)
        )

    def test_all_correct_unweighted(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
        bce_obj = losses.BinaryCrossentropy()
        loss = bce_obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0)

        # Test with logits.
        logits = np.array(
            [
                [10.0, -10.0, -10.0],
                [-10.0, 10.0, -10.0],
                [-10.0, -10.0, 10.0],
            ]
        )
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
        y_pred = np.array(
            [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.1, 0.2, 0.7]], dtype="float32"
        )
        bce_obj = losses.BinaryCrossentropy()
        loss = bce_obj(y_true, y_pred)
        self.assertAllClose(loss, 0.20046903)

        y_true = np.array([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.array([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
        bce_obj = losses.BinaryCrossentropy()
        loss = bce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 3.98559)

        # Test with logits.
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        logits = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 3.3333)

    def test_scalar_weighted(self):
        bce_obj = losses.BinaryCrossentropy()
        y_true = np.array([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.array([1, 1, 1, 0], dtype="float32").reshape([2, 2])
        loss = bce_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 9.1668)

        # Test with logits.
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        logits = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits, sample_weight=2.3)
        self.assertAlmostEqual(loss, 7.666)

    def test_sample_weighted(self):
        bce_obj = losses.BinaryCrossentropy()
        y_true = np.array([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.array([1, 1, 1, 0], dtype="float32").reshape([2, 2])
        sample_weight = np.array([1.2, 3.4]).reshape((2, 1))
        loss = bce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 4.7827)

        # Test with logits.
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        logits = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        weights = np.array([4, 3])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits, sample_weight=weights)
        self.assertAlmostEqual(loss, 10.0)

    def test_no_reduction(self):
        y_true = np.array([[1, 0, 1], [0, 1, 1]])
        logits = np.array([[10.0, -10.0, 10.0], [10.0, 10.0, -10.0]])
        bce_obj = losses.BinaryCrossentropy(from_logits=True, reduction=None)
        loss = bce_obj(y_true, logits)
        self.assertAllClose(loss, [0.0, 6.666], atol=1e-3)

    def test_label_smoothing(self):
        logits = np.array([[10.0, -10.0, -10.0]])
        y_true = np.array([[1, 0, 1]])
        label_smoothing = 0.1
        bce_obj = losses.BinaryCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = bce_obj(y_true, logits)
        expected_value = (10.0 + 5.0 * label_smoothing) / 3.0
        self.assertAlmostEqual(loss, expected_value)

    def test_shape_mismatch(self):
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]]
        )
        cce_obj = losses.BinaryCrossentropy()
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            cce_obj(y_true, y_pred)

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Torch doesn't support bfloat16 for BinaryCrossentropy",
    )
    def test_dtype_arg(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
        y_pred = np.array(
            [[0.9, 0.1, 0.2], [0.3, 0.8, 0.1], [0.1, 0.2, 0.7]], dtype="float32"
        )
        bce_obj = losses.BinaryCrossentropy(dtype="bfloat16")
        loss = bce_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class CategoricalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.CategoricalCrossentropy(name="cce", axis=-1)
        )

    def test_all_correct_unweighted(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.CategoricalCrossentropy()
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0)

        # Test with logits.
        logits = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0)

    def test_unweighted(self):
        cce_obj = losses.CategoricalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.3239)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0573)

    def test_scalar_weighted(self):
        cce_obj = losses.CategoricalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.7449)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.1317)

    def test_sample_weighted(self):
        cce_obj = losses.CategoricalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        sample_weight = np.array([[1.2], [3.4], [5.6]]).reshape((3, 1))
        loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.0696)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.31829)

    def test_no_reduction(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalCrossentropy(
            from_logits=True, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose((0.001822, 0.000459, 0.169846), loss)

    def test_label_smoothing(self):
        logits = np.array([[100.0, -100.0, -100.0]])
        y_true = np.array([[1, 0, 0]])
        label_smoothing = 0.1
        cce_obj = losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = cce_obj(y_true, logits)
        expected_value = 400.0 * label_smoothing / 3.0
        self.assertAlmostEqual(loss, expected_value)

    def test_label_smoothing_ndarray(self):
        logits = np.asarray([[100.0, -100.0, -100.0]])
        y_true = np.asarray([[1, 0, 0]])
        label_smoothing = 0.1
        cce_obj = losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = cce_obj(y_true, logits)
        expected_value = 400.0 * label_smoothing / 3.0
        self.assertAlmostEqual(loss, expected_value)

    def test_shape_mismatch(self):
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]]
        )

        cce_obj = losses.CategoricalCrossentropy()
        with self.assertRaisesRegex(ValueError, "must have the same shape"):
            cce_obj(y_true, y_pred)

    def test_dtype_arg(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.CategoricalCrossentropy(dtype="bfloat16")
        loss = cce_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class SparseCategoricalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.SparseCategoricalCrossentropy(name="scce")
        )

    def test_all_correct_unweighted(self):
        y_true = np.array([[0], [1], [2]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.SparseCategoricalCrossentropy()
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0, 3)

        # Test with logits.
        logits = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        cce_obj = losses.SparseCategoricalCrossentropy()
        y_true = np.array([0, 1, 2])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.3239, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0573, 3)

    def test_scalar_weighted(self):
        cce_obj = losses.SparseCategoricalCrossentropy()
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.7449, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.1317, 3)

    def test_sample_weighted(self):
        cce_obj = losses.SparseCategoricalCrossentropy()
        y_true = np.array([[0], [1], [2]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        sample_weight = np.array([[1.2], [3.4], [5.6]]).reshape((3, 1))
        loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 1.0696, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.31829, 3)

    def test_no_reduction(self):
        y_true = np.array([[0], [1], [2]])
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose((0.001822, 0.000459, 0.169846), loss)

    def test_ignore_class(self):
        y_true = np.array([[-1, 2]])
        logits = np.array([[[0.854, 0.698, 0.598], [0.088, 0.86, 0.018]]])
        cce_obj = losses.SparseCategoricalCrossentropy(
            from_logits=True, ignore_class=-1, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose([[0.0, 1.480129]], loss)

        y_true = np.array([[[-1], [2]]])
        logits = np.array([[[0.854, 0.698, 0.598], [0.088, 0.86, 0.018]]])
        cce_obj = losses.SparseCategoricalCrossentropy(
            from_logits=True, ignore_class=-1, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose([[0.0, 1.480129]], loss)

    def test_dtype_arg(self):
        y_true = np.array([[0], [1], [2]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.SparseCategoricalCrossentropy(dtype="bfloat16")
        loss = cce_obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class BinaryFocalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.BinaryFocalCrossentropy(name="bfce")
        )

    def test_all_correct_unweighted(self):
        y_true = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype="float32",
        )
        obj = losses.BinaryFocalCrossentropy(gamma=1.5)
        loss = obj(y_true, y_true)
        self.assertAlmostEqual(loss, 0.0, 3)

        # Test with logits.
        logits = np.array(
            [
                [100.0, -100.0, -100.0],
                [-100.0, 100.0, -100.0],
                [-100.0, -100.0, 100.0],
            ]
        )
        obj = losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=True)
        loss = obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        obj = losses.BinaryFocalCrossentropy(gamma=2.0)
        loss = obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.268, 3)

        # Test with logits.
        y_true = np.array([[1, 1, 0], [0, 1, 0]], dtype="float32")
        logits = np.array([[1.5, -2.7, 2.9], [-3.8, 1.2, -4.5]])
        obj = losses.BinaryFocalCrossentropy(gamma=3.0, from_logits=True)
        loss = obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.799, 3)

    def test_scalar_weighted(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        obj = losses.BinaryFocalCrossentropy(gamma=2.0)
        loss = obj(y_true, y_pred, sample_weight=1.23)
        self.assertAlmostEqual(loss, 0.3296, 3)

        # Test with logits.
        y_true = np.array([[1, 1, 0], [0, 1, 0]], dtype="float32")
        logits = np.array([[1.5, -2.7, 2.9], [-3.8, 1.2, -4.5]])
        obj = losses.BinaryFocalCrossentropy(gamma=3.0, from_logits=True)
        loss = obj(y_true, logits, sample_weight=3.21)
        self.assertAlmostEqual(loss, 2.565, 3)

    def test_sample_weighted(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        sample_weight = np.array([1.2, 3.4]).reshape((2, 1))
        obj = losses.BinaryFocalCrossentropy(gamma=2.0)
        loss = obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.34415, 3)

        # Test with logits.
        y_true = np.array([[1, 1, 0], [0, 1, 0]], dtype="float32")
        logits = np.array([[1.5, -2.7, 2.9], [-3.8, 1.2, -4.5]])
        obj = losses.BinaryFocalCrossentropy(gamma=3.0, from_logits=True)
        loss = obj(y_true, logits, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.95977, 3)

    def test_no_reduction(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        obj = losses.BinaryFocalCrossentropy(
            gamma=2.0,
            reduction=None,
        )
        loss = obj(y_true, y_pred)
        self.assertAllClose(loss, (0.515547, 0.020513))

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Torch doesn't support bfloat16 for BinaryFocalCrossentropy",
    )
    def test_dtype_arg(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([0.9, 0.8, 0.7, 0.2], dtype=np.float32).reshape(
            [2, 2]
        )
        obj = losses.BinaryFocalCrossentropy(dtype="bfloat16")
        loss = obj(y_true, y_pred)
        self.assertDType(loss, "bfloat16")


class CategoricalFocalCrossentropyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(
            losses.CategoricalFocalCrossentropy(name="cfce")
        )

    def test_all_correct_unweighted(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="int64")
        y_pred = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype="float32",
        )
        cce_obj = losses.CategoricalFocalCrossentropy(alpha=0.25, gamma=2.0)
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.0, 3)

        # Test with logits.
        logits = np.array(
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
        )
        cce_obj = losses.CategoricalFocalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.0, 3)

    def test_unweighted(self):
        cce_obj = losses.CategoricalFocalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred)
        self.assertAlmostEqual(loss, 0.02059, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalFocalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        self.assertAlmostEqual(loss, 0.000345, 3)

    def test_scalar_weighted(self):
        cce_obj = losses.CategoricalFocalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        loss = cce_obj(y_true, y_pred, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.047368, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalFocalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=2.3)
        self.assertAlmostEqual(loss, 0.000794, 4)

    def test_sample_weighted(self):
        cce_obj = losses.CategoricalFocalCrossentropy()
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = np.array(
            [[0.9, 0.05, 0.05], [0.5, 0.89, 0.6], [0.05, 0.01, 0.94]],
            dtype="float32",
        )
        sample_weight = np.array([[1.2], [3.4], [5.6]]).reshape((3, 1))
        loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.06987, 3)

        # Test with logits.
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalFocalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=sample_weight)
        self.assertAlmostEqual(loss, 0.001933, 3)

    def test_no_reduction(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        logits = np.array([[8.0, 1.0, 1.0], [0.0, 9.0, 1.0], [2.0, 3.0, 5.0]])
        cce_obj = losses.CategoricalFocalCrossentropy(
            from_logits=True, reduction=None
        )
        loss = cce_obj(y_true, logits)
        self.assertAllClose(
            (1.5096224e-09, 2.4136547e-11, 1.0360638e-03),
            loss,
        )

    def test_label_smoothing(self):
        logits = np.array([[4.9, -0.5, 2.05]])
        y_true = np.array([[1, 0, 0]])
        label_smoothing = 0.1

        cce_obj = losses.CategoricalFocalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing
        )
        loss = cce_obj(y_true, logits)

        expected_value = 0.06685
        self.assertAlmostEqual(loss, expected_value, 3)

    def test_dtype_arg(self):
        logits = np.array([[4.9, -0.5, 2.05]])
        y_true = np.array([[1, 0, 0]])
        cce_obj = losses.CategoricalFocalCrossentropy(
            from_logits=True, dtype="bfloat16"
        )
        loss = cce_obj(y_true, logits)
        self.assertDType(loss, "bfloat16")


class CTCTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(losses.CTC(name="myctc"))

    def test_correctness(self):
        logits = (np.arange(24).reshape((2, 4, 3)).astype("float32") - 12) / 100
        y_true = np.array(([[1, 2, 1, 0], [1, 2, 0, 2]]))
        output = losses.CTC()(y_true, logits)
        self.assertAllClose(output, 2.448645)

    def test_dtype_arg(self):
        logits = (np.arange(24).reshape((2, 4, 3)).astype("float32") - 12) / 100
        y_true = np.array(([[1, 2, 1, 0], [1, 2, 0, 2]]))
        output = losses.CTC(dtype="bfloat16")(y_true, logits)
        self.assertDType(output, "bfloat16")


class DiceTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(losses.Dice(name="mydice"))

    def test_correctness(self):
        y_true = np.array(([[1, 2], [1, 2]]))
        y_pred = np.array(([[4, 1], [6, 1]]))
        output = losses.Dice()(y_true, y_pred)
        self.assertAllClose(output, -0.55555546)

    def test_binary_segmentation(self):
        y_true = np.array(
            ([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        )
        y_pred = np.array(
            ([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 1]])
        )
        output = losses.Dice()(y_true, y_pred)
        self.assertAllClose(output, 0.77777773)

    def test_binary_segmentation_with_axis(self):
        y_true = np.array(
            [[[[1.0], [1.0]], [[0.0], [0.0]]], [[[1.0], [1.0]], [[0.0], [0.0]]]]
        )
        y_pred = np.array(
            [[[[0.0], [1.0]], [[0.0], [1.0]]], [[[0.4], [0.0]], [[0.0], [0.9]]]]
        )
        output = losses.Dice(axis=(1, 2, 3), reduction=None)(y_true, y_pred)
        self.assertAllClose(output, [0.5, 0.75757575])

    def test_dtype_arg(self):
        y_true = np.array(([[1, 2], [1, 2]]))
        y_pred = np.array(([[4, 1], [6, 1]]))
        output = losses.Dice(dtype="bfloat16")(y_true, y_pred)
        self.assertDType(output, "bfloat16")


class TverskyTest(testing.TestCase):
    def test_config(self):
        self.run_class_serialization_test(losses.Tversky(name="mytversky"))

    def test_correctness(self):
        y_true = np.array(([[1, 2], [1, 2]]))
        y_pred = np.array(([[4, 1], [6, 1]]))
        output = losses.Tversky()(y_true, y_pred)
        self.assertAllClose(output, -0.55555546)

    def test_correctness_custom_coefficients(self):
        y_true = np.array(([[1, 2], [1, 2]]))
        y_pred = np.array(([[4, 1], [6, 1]]))
        output = losses.Tversky(alpha=0.2, beta=0.8)(y_true, y_pred)
        self.assertAllClose(output, -0.29629636)

    def test_binary_segmentation(self):
        y_true = np.array(
            ([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        )
        y_pred = np.array(
            ([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 1]])
        )
        output = losses.Tversky()(y_true, y_pred)
        self.assertAllClose(output, 0.77777773)

    def test_binary_segmentation_custom_coefficients(self):
        y_true = np.array(
            ([[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
        )
        y_pred = np.array(
            ([[0, 1, 0, 1], [1, 0, 1, 1], [0, 1, 0, 1], [1, 0, 1, 1]])
        )
        output = losses.Tversky(alpha=0.2, beta=0.8)(y_true, y_pred)
        self.assertAllClose(output, 0.7916667)

    def test_dtype_arg(self):
        y_true = np.array(([[1, 2], [1, 2]]))
        y_pred = np.array(([[4, 1], [6, 1]]))
        output = losses.Tversky(dtype="bfloat16")(y_true, y_pred)
        self.assertDType(output, "bfloat16")
