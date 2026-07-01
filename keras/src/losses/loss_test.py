import pickle

import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import dtype_policies
from keras.src import losses as losses_module
from keras.src import ops
from keras.src import testing
from keras.src import tree
from keras.src.losses import loss as _loss_mod
from keras.src.losses.loss import Loss
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
from keras.src.losses.losses import MeanSquaredError


def _to_np(t):
    """Convert tensor or array to numpy for numeric comparison."""
    try:
        import torch as _torch

        if isinstance(t, _torch.Tensor):
            return t.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(t)


def _slow_call_mse(loss_fn, y_true, y_pred, sample_weight=None):
    """Replicate slow-path behaviour unconditionally (no fast-path shortcut)."""
    in_mask = backend.get_keras_mask(y_pred)
    with ops.name_scope(loss_fn.name):
        yp = tree.map_structure(
            lambda x: ops.convert_to_tensor(x, dtype=loss_fn.dtype), y_pred
        )
        yt = tree.map_structure(
            lambda x: ops.convert_to_tensor(x, dtype=loss_fn.dtype), y_true
        )
        losses = loss_fn.call(yt, yp)
        out_mask = backend.get_keras_mask(losses)
        if in_mask is not None and out_mask is not None:
            mask = in_mask & out_mask
        elif in_mask is not None:
            mask = in_mask
        elif out_mask is not None:
            mask = out_mask
        else:
            mask = None
        return _loss_mod.reduce_weighted_values(
            losses,
            sample_weight=sample_weight,
            mask=mask,
            reduction=loss_fn.reduction,
            dtype=loss_fn.dtype,
        )


class ExampleLoss(Loss):
    def call(self, y_true, y_pred):
        return (y_true - y_pred) ** 2


class LossTest(testing.TestCase):
    def setUp(self):
        super().setUp()
        self._global_dtype_policy = dtype_policies.dtype_policy.dtype_policy()
        self._floatx = backend.floatx()

    def tearDown(self):
        super().tearDown()
        dtype_policies.dtype_policy.set_dtype_policy(self._global_dtype_policy)
        backend.set_floatx(self._floatx)

    def test_squeeze_or_expand(self):
        x1 = ops.ones((3,))
        x2 = ops.ones((3, 1))
        x1, x2 = squeeze_or_expand_to_same_rank(x1, x2)
        self.assertEqual(ops.shape(x1), (3, 1))
        self.assertEqual(ops.shape(x2), (3, 1))

        x1 = ops.ones((3, 2))
        x2 = ops.ones((3, 2, 1))
        x1, x2 = squeeze_or_expand_to_same_rank(x1, x2)
        self.assertEqual(ops.shape(x1), (3, 2))
        self.assertEqual(ops.shape(x2), (3, 2))

        x1 = ops.ones((3,))
        x2 = ops.ones((3, 1))
        x2, x1 = squeeze_or_expand_to_same_rank(x2, x1)
        self.assertEqual(ops.shape(x1), (3, 1))
        self.assertEqual(ops.shape(x2), (3, 1))

        x1 = ops.ones((3, 2))
        x2 = ops.ones((3, 2, 1))
        x2, x1 = squeeze_or_expand_to_same_rank(x2, x1)
        self.assertEqual(ops.shape(x1), (3, 2))
        self.assertEqual(ops.shape(x2), (3, 2))

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

        # sum_over_batch_size or mean
        loss_fn = ExampleLoss(reduction="sum_over_batch_size")
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(np.sum((y_true - y_pred) ** 2) / 4, loss)

        # bad reduction
        with self.assertRaisesRegex(ValueError, "Invalid value for argument"):
            ExampleLoss(reduction="abc")

    def test_mask(self):
        mask = np.array([True, False, True, True])
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.1, 0.2, 0.3, 0.4])

        masked_y_true = np.array([1.0, 1.0, 0.0])
        masked_y_pred = np.array([0.1, 0.3, 0.4])

        mask = ops.convert_to_tensor(mask)
        y_true = ops.convert_to_tensor(y_true)
        y_pred = ops.convert_to_tensor(y_pred)
        backend.set_keras_mask(y_pred, mask)

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            np.sum((masked_y_true - masked_y_pred) ** 2) / 3, loss
        )

        # Test edge case where everything is masked.
        mask = np.array([False, False, False, False])
        backend.set_keras_mask(y_pred, mask)
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

        # Test edge case where every weight is 0.
        sample_weight = np.array([0.0, 0.0, 0.0, 0.0])
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(loss, 0)  # No NaN.

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
        backend.set_keras_mask(y_pred, mask)

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            np.sum(masked_sample_weight * (masked_y_true - masked_y_pred) ** 2)
            / 3,
            loss,
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
        backend.set_keras_mask(y_pred, mask)

        loss_fn = ExampleLoss()
        rank1_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)

        # duplicate rows
        mask = ops.tile(ops.expand_dims(mask, axis=0), (2, 1))
        y_true = ops.tile(ops.expand_dims(y_true, axis=0), (2, 1))
        y_pred = ops.tile(ops.expand_dims(y_pred, axis=0), (2, 1))
        sample_weight = ops.tile(ops.expand_dims(sample_weight, axis=0), (2, 1))
        backend.set_keras_mask(y_pred, mask)
        rank2_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(rank1_loss, rank2_loss)

    @parameterized.named_parameters(
        ("mask", "mask"),
        ("sample_weight", "sample_weight"),
        ("ys", "ys"),
    )
    def test_rank_adjustment(self, uprank):
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
        backend.set_keras_mask(y_pred, mask)

        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
        self.assertEqual(backend.standardize_dtype(loss.dtype), "float32")
        self.assertAllClose(
            np.sum(masked_sample_weight * (masked_y_true - masked_y_pred) ** 2)
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

    def test_pickle(self):
        loss = losses_module.get("mse")
        loss = pickle.loads(pickle.dumps(loss))
        self.assertEqual(loss, losses_module.mean_squared_error)

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
        self.assertDType(loss, "float16")

        # Test DTypePolicy for `dtype` argument
        loss_fn = ExampleLoss(dtype=dtype_policies.DTypePolicy("mixed_float16"))
        loss = loss_fn(y_true, y_pred)
        self.assertDType(loss, "float16")

        # `dtype` setter should raise AttributeError
        with self.assertRaises(AttributeError):
            loss_fn.dtype = "bfloat16"

    def test_default_dtype(self):
        y_true = np.array([1.0, 0.0, 1.0, 0.0], dtype="float32")
        y_pred = np.array([0.1, 0.2, 0.3, 0.4], dtype="float32")

        # Defaults to `keras.config.floatx()` not global `dtype_policy`
        dtype_policies.dtype_policy.set_dtype_policy("mixed_float16")
        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred)
        self.assertDType(loss, "float32")

        backend.set_floatx("float16")
        loss_fn = ExampleLoss()
        loss = loss_fn(y_true, y_pred)
        self.assertDType(loss, backend.floatx())

    def test_mse_fast_path_torch_reductions(self):
        """Fast path taken for both-torch inputs; numerics match slow path."""
        if backend.backend() != "torch":
            self.skipTest("torch backend only")
        import torch

        for reduction in ["sum_over_batch_size", "mean", "sum", "none", None]:
            mse = MeanSquaredError(reduction=reduction)
            yt = torch.tensor([1.0, 2.0, 3.0, 4.0])
            yp = torch.tensor([1.1, 1.9, 3.2, 3.8])
            fast = mse(yt, yp)
            slow = _slow_call_mse(mse, yt, yp)
            self.assertAllClose(
                _to_np(fast),
                _to_np(slow),
                rtol=1e-5,
                atol=1e-5,
                msg=f"reduction={reduction}",
            )

    def test_mse_numpy_ytrue_torch_ypred_fallback(self):
        """numpy y_true falls through; result must match slow path."""
        if backend.backend() != "torch":
            self.skipTest("torch backend only")
        import torch

        mse = MeanSquaredError()
        yt_np = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        yp_t = torch.tensor([1.1, 1.9, 3.2, 3.8])
        result = mse(yt_np, yp_t)
        ref = _slow_call_mse(mse, yt_np, yp_t)
        self.assertAllClose(_to_np(result), _to_np(ref), rtol=1e-5, atol=1e-5)
        val = float(_to_np(result))
        self.assertTrue(np.isfinite(val), f"Loss is not finite: {val}")

    def test_mse_sample_weight_uses_slow_path(self):
        """sample_weight != None bypasses fast path; result must be correct."""
        if backend.backend() != "torch":
            self.skipTest("torch backend only")
        import torch

        mse = MeanSquaredError()
        yt = torch.tensor([1.0, 2.0, 3.0, 4.0])
        yp = torch.tensor([1.1, 1.9, 3.2, 3.8])
        sw = torch.tensor([1.0, 2.0, 1.0, 2.0])
        result = mse(yt, yp, sample_weight=sw)
        ref = _slow_call_mse(mse, yt, yp, sample_weight=sw)
        self.assertAllClose(_to_np(result), _to_np(ref), rtol=1e-5, atol=1e-5)

    def test_mse_numeric_equivalence_all_cases(self):
        """Allclose for both-torch, numpy+torch, and sample_weight cases."""
        if backend.backend() != "torch":
            self.skipTest("torch backend only")
        import torch

        yt_t = torch.tensor([0.5, 1.5, 2.5, 3.5])
        yp_t = torch.tensor([0.6, 1.4, 2.7, 3.3])
        yt_np = yt_t.numpy()
        sw = torch.tensor([1.0, 2.0, 1.0, 2.0])
        cases = {
            "both_torch": (yt_t, yp_t, None),
            "numpy_ytrue_torch_ypred": (yt_np, yp_t, None),
            "sample_weight": (yt_t, yp_t, sw),
        }
        reductions = ["sum_over_batch_size", "mean", "sum", "none"]
        for desc, (yt, yp, sw_arg) in cases.items():
            r_list = (
                ["sum_over_batch_size"]
                if desc == "sample_weight"
                else reductions
            )
            for red in r_list:
                fn = MeanSquaredError(reduction=red)
                fast = (
                    fn(yt, yp)
                    if sw_arg is None
                    else fn(yt, yp, sample_weight=sw_arg)
                )
                ref = _slow_call_mse(fn, yt, yp, sample_weight=sw_arg)
                self.assertAllClose(
                    _to_np(fast),
                    _to_np(ref),
                    rtol=1e-5,
                    atol=1e-5,
                    msg=f"FAIL {desc} reduction={red}",
                )

    def test_mse_fast_path_actually_taken(self):
        """Fast path (sum_over_batch_size): torch.Tensor with correct value."""
        if backend.backend() != "torch":
            self.skipTest("torch backend only")
        import torch

        mse = MeanSquaredError(reduction="sum_over_batch_size")
        yt = torch.tensor([1.0, 2.0])
        yp = torch.tensor([1.5, 2.5])
        result = mse(yt, yp)
        self.assertIsInstance(result, torch.Tensor)
        self.assertAllClose(float(result), 0.25, rtol=1e-5, atol=1e-5)

    def test_fast_path_skipped_when_ypred_dtype_mismatch(self):
        """float16 y_pred falls through to slow path; result must be float32."""
        if backend.backend() != "torch":
            self.skipTest("torch backend only")
        import torch

        mse = MeanSquaredError(reduction="sum_over_batch_size")
        yt = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        yp = torch.tensor([1.1, 1.9, 3.2, 3.8], dtype=torch.float16)
        result = mse(yt, yp)
        self.assertEqual(backend.standardize_dtype(result.dtype), "float32")
        ref = _slow_call_mse(mse, yt, yp)
        self.assertAllClose(float(result), float(ref), rtol=1e-3, atol=1e-3)

    def test_fast_path_taken_when_dtypes_match(self):
        """Fast path IS taken when y_pred and y_true both match self.dtype."""
        if backend.backend() != "torch":
            self.skipTest("torch backend only")
        import torch

        mse = MeanSquaredError(reduction="sum_over_batch_size")
        # both float32, matches self.dtype
        yt = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        yp = torch.tensor([1.1, 1.9, 3.2, 3.8], dtype=torch.float32)

        result = mse(yt, yp)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(backend.standardize_dtype(result.dtype), "float32")
        ref = _slow_call_mse(mse, yt, yp)
        self.assertAllClose(float(result), float(ref), rtol=1e-5, atol=1e-5)

    def test_fast_path_same_device(self):
        """Fast path: both tensors on same device, numerics match slow path."""
        if backend.backend() != "torch":
            self.skipTest("torch backend only")
        import torch  # noqa: I001
        from keras.src.backend.torch.core import get_device

        device = get_device()
        mse = MeanSquaredError(reduction="sum_over_batch_size")
        yt = torch.tensor([1.0, 2.0, 3.0, 4.0]).to(device)
        yp = torch.tensor([1.1, 1.9, 3.2, 3.8]).to(device)
        result = mse(yt, yp)
        ref = _slow_call_mse(mse, yt, yp)
        self.assertAllClose(_to_np(result), _to_np(ref), rtol=1e-5, atol=1e-5)

    def test_fast_path_skipped_cross_device(self):
        """Cross-device inputs fall through to slow path without crashing."""
        if backend.backend() != "torch":
            self.skipTest("torch backend only")
        import torch

        if not torch.cuda.is_available():
            self.skipTest("cuda not available")
        mse = MeanSquaredError(reduction="sum_over_batch_size")
        yt = torch.tensor([1.0, 2.0, 3.0, 4.0]).cpu()
        yp = torch.tensor([1.1, 1.9, 3.2, 3.8]).cuda()
        result = mse(yt, yp)
        ref = _slow_call_mse(mse, yt, yp)
        self.assertAllClose(_to_np(result), _to_np(ref), rtol=1e-5, atol=1e-5)
