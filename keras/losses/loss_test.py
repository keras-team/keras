import numpy as np
import pytest

from keras import backend
from keras import losses as losses_module
from keras import ops
from keras import testing
from keras.losses.loss import Loss


class ExampleLoss(Loss):
    def call(self, y_true, y_pred):
        return (y_true - y_pred) ** 2


class LossTest(testing.TestCase):
    def test_reduction(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        # No reduction
        loss_fn = ExampleLoss(reduction=None)
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose((y_true - y_pred) ** 2, loss)

        # sum
        loss_fn = ExampleLoss(reduction="sum")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(np.sum((y_true - y_pred) ** 2), loss)

        # sum_over_batch_size
        loss_fn = ExampleLoss(reduction="sum_over_batch_size")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(np.sum((y_true - y_pred) ** 2) / 4, loss)

        # bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            ExampleLoss(reduction="abc")

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_mask(self):
        mask = np.array([True, False, True, True])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        masked_y_true = np.array([1.0, 1.0, 0.0])
        masked_y_pred = np.array([0.1, 0.3, 0.4])

        mask = ops.convert_to_tensor(mask)
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        y_pred._keras_mask = mask

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            np.sum((masked_y_true - masked_y_pred) ** 2) / 3, loss
        )

        # no reduction
        loss_fn = ExampleLoss(reduction=None)
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        expected = (y_true - y_pred) ** 2
        expected = ops.where(mask, expected, ops.zeros_like(expected))
        self.assertAllClose(expected, loss)

        # sum reduction
        loss_fn = ExampleLoss(reduction="sum")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(np.sum((masked_y_true - masked_y_pred) ** 2), loss)

        # Test edge case where everything is masked.
        loss_fn = ExampleLoss()
        y_pred._keras_mask = np.array([False, False, False, False])
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(loss, 0)  # No NaN.

    def test_sample_weight(self):
        sample_weight = np.array([0.4, 0.3, 0.2, 0.1])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            np.sum(sample_weight * (y_true - y_pred) ** 2) / 4, loss
        )

        # no reduction
        loss_fn = ExampleLoss(reduction=None)
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(sample_weight * (y_true - y_pred) ** 2, loss)

        # sum reduction
        loss_fn = ExampleLoss(reduction="sum")
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            ops.sum(sample_weight * (y_true - y_pred) ** 2), loss
        )

        # Test edge case where every weight is 0.
        loss_fn = ExampleLoss()
        sample_weight = np.array([0.0, 0.0, 0.0, 0.0])
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(loss, 0)  # No NaN.

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_mask_and_sample_weight(self):
        sample_weight = np.array([0.4, 0.3, 0.2, 0.1])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])
        mask = np.array([True, False, True, True])

        masked_sample_weight = np.array([0.4, 0.2, 0.1])
        masked_y_true = np.array([1.0, 1.0, 0.0])
        masked_y_pred = np.array([0.1, 0.3, 0.4])

        mask = ops.convert_to_tensor(mask)
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        y_pred._keras_mask = mask

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            np.sum(masked_sample_weight * (masked_y_true - masked_y_pred) ** 2)
            / 3,
            loss,
        )

        # ensure the result is the same if `y_pred` has masked nans.
        y_pred_with_nans = ops.where(
            mask, y_pred, ops.full_like(y_pred, np.nan)
        )
        y_pred_with_nans._keras_mask = mask
        loss_with_y_pred_nans = loss_fn(
            y_true, y_pred_with_nans, sample_weight=sample_weight
        )
        self.assertEqual(
            backend.standardize_dtype(loss_with_y_pred_nans.dtype), "float32"
        )
        self.assertAllClose(loss, loss_with_y_pred_nans)

        # ensure the result is the same if `sample_weights` has masked nans.
        sample_weight_with_nans = ops.where(
            mask, sample_weight, ops.full_like(sample_weight, np.nan)
        )
        loss_with_sample_weight_nans = loss_fn(
            y_true, y_pred, sample_weight=sample_weight_with_nans
        )
        self.assertEqual(
            backend.standardize_dtype(loss_with_sample_weight_nans.dtype),
            "float32",
        )
        self.assertAllClose(loss, loss_with_sample_weight_nans)

        # reduction is None
        loss_fn = ExampleLoss(reduction="none")
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            ops.cast(mask, sample_weight.dtype)
            * sample_weight
            * (y_true - y_pred) ** 2,
            loss,
        )

        # reduction is 'sum'
        loss_fn = ExampleLoss(reduction="sum")
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            ops.sum(
                ops.cast(mask, sample_weight.dtype)
                * sample_weight
                * (y_true - y_pred) ** 2
            ),
            loss,
        )

    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_mask_and_sample_weight_rank2(self):
        # check loss of inputs with duplicate rows doesn't change
        sample_weight = np.array([0.4, 0.3, 0.2, 0.1])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])
        mask = np.array([True, False, True, True])

        mask = ops.convert_to_tensor(mask)
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        y_pred._keras_mask = mask

        loss_fn = ExampleLoss()
        rank1_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)

        # duplicate rows
        mask = ops.tile(ops.expand_dims(mask, axis=0), (2, 1))
        y_true = ops.tile(ops.expand_dims(y_true, axis=0), (2, 1))
        y_pred = ops.tile(ops.expand_dims(y_pred, axis=0), (2, 1))
        sample_weight = ops.tile(ops.expand_dims(sample_weight, axis=0), (2, 1))
        y_pred._keras_mask = mask
        rank2_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(rank1_loss, rank2_loss)

    # @testing.parametrize(
    #     "uprank", ["mask", "sample_weight", "y_true", "y_pred"])
    # TODO: use parameterization decorator
    @pytest.mark.skipif(
        backend.backend() == "numpy",
        reason="Numpy backend does not support masking.",
    )
    def test_rank_adjustment(self):
        for uprank in ["mask", "sample_weight", "ys"]:
            sample_weight = np.array([0.4, 0.3, 0.2, 0.1])
            y_true = np.array([1.0, 0.0, 1.0, 0.0])
            y_pred = np.array([0.1, 0.2, 0.3, 0.4])
            mask = np.array([True, False, True, True])

            if uprank == "mask":
                mask = np.expand_dims(mask, -1)
            elif uprank == "sample_weight":
                sample_weight = np.expand_dims(sample_weight, -1)
            elif uprank == "ys":
                y_true = np.expand_dims(y_true, -1)
                y_pred = np.expand_dims(y_pred, -1)

            masked_sample_weight = np.array([0.4, 0.2, 0.1])
            masked_y_true = np.array([1.0, 1.0, 0.0])
            masked_y_pred = np.array([0.1, 0.3, 0.4])

            mask = ops.convert_to_tensor(mask)
            y_true = ops.convert_to_tensor(y_true)
            y_pred = ops.convert_to_tensor(y_pred)
            y_pred._keras_mask = mask

            loss_fn = ExampleLoss()
            loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
            self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
            self.assertAllClose(
                np.sum(
                    masked_sample_weight * (masked_y_true - masked_y_pred) ** 2
                )
                / 3,
                loss,
            )

    def test_mixed_dtypes(self):
        sample_weight = np.array([0.4, 0.3, 0.2, 0.1], dtype="float64")
        y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype="int32")
        y_pred = np.array([0.1, 0.2, 0.3, 0.4], dtype="float32")

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            np.sum(sample_weight * (y_true - y_pred) ** 2) / 4,
            loss,
        )

    def test_get_method(self):
        loss = losses_module.get("mse")
        self.assertEqual(loss, losses_module.mean_squared_error)

        loss = losses_module.get(None)
        self.assertEqual(loss, None)

        with self.assertRaises(ValueError):
            losses_module.get("typo")

    def test_dtype_arg(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype="float32")
        y_pred = np.array([0.1, 0.2, 0.3, 0.4], dtype="float32")

        # Note: we use float16 and not float64 to test this because
        # JAX will map float64 to float32.
        loss_fn = ExampleLoss(dtype="float16")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float16")
