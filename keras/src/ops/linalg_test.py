import numpy as np
from absl.testing import parameterized

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.ops import linalg
from keras.src.testing.test_utils import named_product


class LinalgOpsDynamicShapeTest(testing.TestCase):
    def test_cholesky(self):
        x = KerasTensor([None, 20, 20])
        out = linalg.cholesky(x)
        self.assertEqual(out.shape, (None, 20, 20))

        x = KerasTensor([None, None, 20])
        with self.assertRaises(ValueError):
            linalg.cholesky(x)

        x = KerasTensor([None, 20, 15])
        with self.assertRaises(ValueError):
            linalg.cholesky(x)

    def test_det(self):
        x = KerasTensor([None, 20, 20])
        out = linalg.det(x)
        self.assertEqual(out.shape, (None,))

        x = KerasTensor([None, None, 20])
        with self.assertRaises(ValueError):
            linalg.det(x)

        x = KerasTensor([None, 20, 15])
        with self.assertRaises(ValueError):
            linalg.det(x)

    def test_eig(self):
        x = KerasTensor([None, 20, 20])
        w, v = linalg.eig(x)
        self.assertEqual(w.shape, (None, 20))
        self.assertEqual(v.shape, (None, 20, 20))

    def test_eigh(self):
        x = KerasTensor([None, 20, 20])
        w, v = linalg.eigh(x)
        self.assertEqual(w.shape, (None, 20))
        self.assertEqual(v.shape, (None, 20, 20))

    def test_inv(self):
        x = KerasTensor([None, 20, 20])
        out = linalg.inv(x)
        self.assertEqual(out.shape, (None, 20, 20))

        x = KerasTensor([None, None, 20])
        with self.assertRaises(ValueError):
            linalg.inv(x)

        x = KerasTensor([None, 20, 15])
        with self.assertRaises(ValueError):
            linalg.inv(x)

    def test_lu_factor(self):
        if testing.jax_uses_gpu():
            self.skipTest("Skipping test with JAX + GPU due to temporary error")

        x = KerasTensor([None, 4, 3])
        lu, p = linalg.lu_factor(x)
        self.assertEqual(lu.shape, (None, 4, 3))
        self.assertEqual(p.shape, (None, 3))

        x = KerasTensor([None, 2, 3])
        lu, p = linalg.lu_factor(x)
        self.assertEqual(lu.shape, (None, 2, 3))
        self.assertEqual(p.shape, (None, 2))

    def test_norm(self):
        x = KerasTensor((None, 3))
        self.assertEqual(linalg.norm(x).shape, ())

        x = KerasTensor((None, 3, 3))
        self.assertEqual(linalg.norm(x, axis=1).shape, (None, 3))
        self.assertEqual(
            linalg.norm(x, axis=1, keepdims=True).shape, (None, 1, 3)
        )

    def test_qr(self):
        x = KerasTensor((None, 4, 3), dtype="float32")
        q, r = linalg.qr(x, mode="reduced")
        qref, rref = np.linalg.qr(np.ones((2, 4, 3)), mode="reduced")
        qref_shape = (None,) + qref.shape[1:]
        rref_shape = (None,) + rref.shape[1:]
        self.assertEqual(q.shape, qref_shape)
        self.assertEqual(r.shape, rref_shape)

        q, r = linalg.qr(x, mode="complete")
        qref, rref = np.linalg.qr(np.ones((2, 4, 3)), mode="complete")
        qref_shape = (None,) + qref.shape[1:]
        rref_shape = (None,) + rref.shape[1:]
        self.assertEqual(q.shape, qref_shape)
        self.assertEqual(r.shape, rref_shape)

    def test_qr_invalid_mode(self):
        # backend agnostic error message
        x = np.array([[1, 2], [3, 4]])
        invalid_mode = "invalid_mode"
        with self.assertRaisesRegex(
            ValueError, "Expected one of {'reduced', 'complete'}."
        ):
            linalg.qr(x, mode=invalid_mode)

    def test_solve(self):
        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20, 5])
        out = linalg.solve(a, b)
        self.assertEqual(out.shape, (None, 20, 5))

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20])
        out = linalg.solve(a, b)
        self.assertEqual(out.shape, (None, 20))

        a = KerasTensor([None, None, 20])
        b = KerasTensor([None, 20, 5])
        with self.assertRaises(ValueError):
            linalg.solve(a, b)

        a = KerasTensor([None, 20, 15])
        b = KerasTensor([None, 20, 5])
        with self.assertRaises(ValueError):
            linalg.solve(a, b)

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, None, 5])
        with self.assertRaises(ValueError):
            linalg.solve(a, b)

    def test_solve_triangular(self):
        if testing.jax_uses_gpu():
            self.skipTest("Skipping test with JAX + GPU due to temporary error")

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20, 5])
        out = linalg.solve_triangular(a, b)
        self.assertEqual(out.shape, (None, 20, 5))

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20])
        out = linalg.solve_triangular(a, b)
        self.assertEqual(out.shape, (None, 20))

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20, 5])
        out = linalg.solve_triangular(a, b, lower=True)
        self.assertEqual(out.shape, (None, 20, 5))

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20])
        out = linalg.solve_triangular(a, b, lower=True)
        self.assertEqual(out.shape, (None, 20))

        a = KerasTensor([None, 20, 15])
        b = KerasTensor([None, 20, 5])
        with self.assertRaises(ValueError):
            linalg.solve_triangular(a, b)

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, None, 5])
        with self.assertRaises(ValueError):
            linalg.solve_triangular(a, b)

    def test_svd(self):
        x = KerasTensor((None, 3, 2))
        u, s, v = linalg.svd(x)
        self.assertEqual(u.shape, (None, 3, 3))
        self.assertEqual(s.shape, (None, 2))
        self.assertEqual(v.shape, (None, 2, 2))

        u, s, v = linalg.svd(x, full_matrices=False)
        self.assertEqual(u.shape, (None, 3, 2))
        self.assertEqual(s.shape, (None, 2))
        self.assertEqual(v.shape, (None, 2, 2))

        s = linalg.svd(x, compute_uv=False)
        self.assertEqual(s.shape, (None, 2))


class LinalgOpsStaticShapeTest(testing.TestCase):
    def test_cholesky(self):
        x = KerasTensor([4, 3, 3])
        out = linalg.cholesky(x)
        self.assertEqual(out.shape, (4, 3, 3))

        x = KerasTensor([10, 20, 15])
        with self.assertRaises(ValueError):
            linalg.cholesky(x)

    def test_det(self):
        x = KerasTensor([4, 3, 3])
        out = linalg.det(x)
        self.assertEqual(out.shape, (4,))

        x = KerasTensor([10, 20, 15])
        with self.assertRaises(ValueError):
            linalg.det(x)

    def test_eig(self):
        x = KerasTensor([4, 3, 3])
        w, v = linalg.eig(x)
        self.assertEqual(w.shape, (4, 3))
        self.assertEqual(v.shape, (4, 3, 3))

        x = KerasTensor([10, 20, 15])
        with self.assertRaises(ValueError):
            linalg.eig(x)

    def test_eigh(self):
        x = KerasTensor([4, 3, 3])
        w, v = linalg.eigh(x)
        self.assertEqual(w.shape, (4, 3))
        self.assertEqual(v.shape, (4, 3, 3))

        x = KerasTensor([10, 20, 15])
        with self.assertRaises(ValueError):
            linalg.eigh(x)

    def test_inv(self):
        x = KerasTensor([4, 3, 3])
        out = linalg.inv(x)
        self.assertEqual(out.shape, (4, 3, 3))

        x = KerasTensor([10, 20, 15])
        with self.assertRaises(ValueError):
            linalg.inv(x)

    def test_lu_factor(self):
        if testing.jax_uses_gpu():
            self.skipTest("Skipping test with JAX + GPU due to temporary error")

        x = KerasTensor([10, 4, 3])
        lu, p = linalg.lu_factor(x)
        self.assertEqual(lu.shape, (10, 4, 3))
        self.assertEqual(p.shape, (10, 3))

        x = KerasTensor([10, 2, 3])
        lu, p = linalg.lu_factor(x)
        self.assertEqual(lu.shape, (10, 2, 3))
        self.assertEqual(p.shape, (10, 2))

    def test_norm(self):
        x = KerasTensor((10, 3))
        self.assertEqual(linalg.norm(x).shape, ())

        x = KerasTensor((10, 3, 3))
        self.assertEqual(linalg.norm(x, axis=1).shape, (10, 3))
        self.assertEqual(
            linalg.norm(x, axis=1, keepdims=True).shape, (10, 1, 3)
        )

    def test_qr(self):
        x = KerasTensor((4, 3), dtype="float32")
        q, r = linalg.qr(x, mode="reduced")
        qref, rref = np.linalg.qr(np.ones((4, 3)), mode="reduced")
        self.assertEqual(q.shape, qref.shape)
        self.assertEqual(r.shape, rref.shape)

        q, r = linalg.qr(x, mode="complete")
        qref, rref = np.linalg.qr(np.ones((4, 3)), mode="complete")
        self.assertEqual(q.shape, qref.shape)
        self.assertEqual(r.shape, rref.shape)

        with self.assertRaises(ValueError):
            linalg.qr(x, mode="invalid")

    def test_solve(self):
        a = KerasTensor([4, 3, 3])
        b = KerasTensor([4, 3, 5])
        out = linalg.solve(a, b)
        self.assertEqual(out.shape, (4, 3, 5))

        a = KerasTensor([4, 3, 3])
        b = KerasTensor([4, 3])
        out = linalg.solve(a, b)
        self.assertEqual(out.shape, (4, 3))

        a = KerasTensor([10, 20, 15])
        b = KerasTensor([10, 20, 5])
        with self.assertRaises(ValueError):
            linalg.solve(a, b)

        a = KerasTensor([20, 20])
        b = KerasTensor([])
        with self.assertRaises(ValueError):
            linalg.solve(a, b)

    def test_solve_triangular(self):
        if testing.jax_uses_gpu():
            self.skipTest("Skipping test with JAX + GPU due to temporary error")

        a = KerasTensor([4, 3, 3])
        b = KerasTensor([4, 3, 5])
        out = linalg.solve_triangular(a, b)
        self.assertEqual(out.shape, (4, 3, 5))

        a = KerasTensor([4, 3, 3])
        b = KerasTensor([4, 3])
        out = linalg.solve_triangular(a, b)
        self.assertEqual(out.shape, (4, 3))

        a = KerasTensor([10, 20, 15])
        b = KerasTensor([10, 20, 5])
        with self.assertRaises(ValueError):
            linalg.solve_triangular(a, b)

    def test_svd(self):
        x = KerasTensor((10, 3, 2))
        u, s, v = linalg.svd(x)
        self.assertEqual(u.shape, (10, 3, 3))
        self.assertEqual(s.shape, (10, 2))
        self.assertEqual(v.shape, (10, 2, 2))

        u, s, v = linalg.svd(x, full_matrices=False)
        self.assertEqual(u.shape, (10, 3, 2))
        self.assertEqual(s.shape, (10, 2))
        self.assertEqual(v.shape, (10, 2, 2))

        s = linalg.svd(x, compute_uv=False)
        self.assertEqual(s.shape, (10, 2))


class LinalgOpsCorrectnessTest(testing.TestCase, parameterized.TestCase):

    def test_cholesky(self):
        x = np.random.rand(4, 3, 3).astype("float32")
        with self.assertRaises(ValueError):
            linalg.cholesky(x)
        x_psd = x @ x.transpose((0, 2, 1)) + 1e-5 * np.eye(3)
        out = linalg.cholesky(x_psd)
        self.assertAllClose(out, np.linalg.cholesky(x_psd), atol=1e-4)

    def test_det(self):
        x = np.random.rand(4, 3, 3)
        out = linalg.det(x)
        self.assertAllClose(out, np.linalg.det(x), atol=1e-5)

        with self.assertRaises(ValueError):
            x = np.random.rand(4, 3, 4)
            linalg.det(x)

    def test_eig(self):
        x = np.random.rand(2, 3, 3)
        x = x @ x.transpose((0, 2, 1))
        if backend.backend() == "jax":
            import jax

            if jax.default_backend() == "gpu":
                # eig not implemented for jax on gpu backend
                with self.assertRaises(NotImplementedError):
                    linalg.eig(x)
                return
        w, v = map(ops.convert_to_numpy, linalg.eig(x))
        x_reconstructed = (v * w[..., None, :]) @ v.transpose((0, 2, 1))
        self.assertAllClose(x_reconstructed, x, atol=1e-4)

    def test_eigh(self):
        x = np.random.rand(2, 3, 3)
        x = x @ x.transpose((0, 2, 1))
        w, v = map(ops.convert_to_numpy, linalg.eigh(x))
        x_reconstructed = (v * w[..., None, :]) @ v.transpose((0, 2, 1))
        self.assertAllClose(x_reconstructed, x, atol=1e-4)

    def test_inv(self):
        x = np.random.rand(4, 3, 3)
        x_inv = ops.convert_to_numpy(linalg.inv(x))
        x_reconstructed = x @ x_inv
        # high tolerance due to numerical instability
        self.assertAllClose(
            x_reconstructed, np.repeat(np.eye(3)[None], 4, 0), atol=1e-3
        )

    def test_lu_factor(self):
        if testing.jax_uses_gpu():
            self.skipTest("Skipping test with JAX + GPU due to temporary error")

        def _pivot_matrix(pivots, n):
            p_matrix = np.eye(n)
            for i, p in enumerate(pivots):
                identity = np.eye(n, n)
                q = identity[i, :].copy()
                identity[i, :] = identity[p, :]
                identity[p, :] = q
                p_matrix = np.dot(p_matrix, identity)
            return p_matrix

        def _reconstruct(lu, pivots, m, n):
            lower = np.tril(lu[:, : min(m, n)], -1) + np.eye(m, min(m, n))
            upper = np.triu(lu[: min(m, n)])

            # pivots are defined differently in tensorflow
            # compared to the other backends
            if backend.backend() == "tensorflow":
                p_matrix = np.eye(m)[pivots]
            else:
                p_matrix = _pivot_matrix(pivots, m)
            out = p_matrix @ lower @ upper
            return out

        m, n = 4, 4
        x = np.random.rand(m, n)
        lu, pivots = map(ops.convert_to_numpy, linalg.lu_factor(x))
        x_reconstructed = _reconstruct(lu, pivots, m, n)
        self.assertAllClose(x_reconstructed, x, atol=1e-5)

        m, n = 4, 3
        x = np.random.rand(m, n)
        if backend.backend() == "tensorflow":
            with self.assertRaises(ValueError):
                linalg.lu_factor(x)
        else:
            lu, pivots = map(ops.convert_to_numpy, linalg.lu_factor(x))
            x_reconstructed = _reconstruct(lu, pivots, m, n)
            self.assertAllClose(x_reconstructed, x, atol=1e-5)

        # batched case
        m, n = 3, 4
        x = np.random.rand(2, m, n)
        if backend.backend() == "tensorflow":
            with self.assertRaises(ValueError):
                linalg.lu_factor(x)
        else:
            lu, pivots = map(ops.convert_to_numpy, linalg.lu_factor(x))
            for i in range(2):
                self.assertAllClose(
                    _reconstruct(lu[i], pivots[i], m, n), x[i], atol=1e-5
                )

    @parameterized.named_parameters(
        named_product(
            ndim=[1, 2],
            ord=[None, "fro", "nuc", -np.inf, -2, -1, 0, 1, 2, np.inf, 3],
            axis=[None, 1, -1, (0, 1)],
            keepdims=[False, True],
        )
    )
    def test_norm(self, ndim, ord, axis, keepdims):
        if ndim == 1:
            x = np.random.random((5,)).astype("float32")
        else:
            x = np.random.random((5, 6)).astype("float32")

        vector_norm = (ndim == 1) or isinstance(axis, int)

        axis_out_of_bounds = ndim == 1 and (
            axis == 1 or isinstance(axis, tuple)
        )
        expected_error = None
        # when an out of bounds axis triggers an IndexError on torch is complex
        if (
            axis_out_of_bounds
            and (not isinstance(axis, tuple) or ord is None)
            and ord not in ("fro", "nuc")
        ):
            expected_error = IndexError
        elif (
            axis_out_of_bounds
            or (vector_norm and isinstance(axis, tuple))  # inv. axis for vector
            or (vector_norm and ord in ("fro", "nuc"))  # invalid ord for vector
            or (not vector_norm and ord in (0, 3))  # invalid ord for matrix
        ):
            expected_error = RuntimeError

        if expected_error is not None:
            # Non-torch backends always throw a ValueError
            expected_error = (
                expected_error if backend.backend() == "torch" else ValueError
            )
            with self.assertRaises(expected_error):
                linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
            return
        output = linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        expected_result = np.linalg.norm(
            x, ord=ord, axis=axis, keepdims=keepdims
        )
        self.assertAllClose(output, expected_result, atol=1e-5)

    def test_qr(self):
        x = np.random.random((4, 5))
        q, r = linalg.qr(x, mode="reduced")
        qref, rref = np.linalg.qr(x, mode="reduced")
        self.assertAllClose(qref, q)
        self.assertAllClose(rref, r)

        q, r = linalg.qr(x, mode="complete")
        qref, rref = np.linalg.qr(x, mode="complete")
        self.assertAllClose(qref, q)
        self.assertAllClose(rref, r)

    def test_solve(self):
        x1 = np.array([[1, 2], [4, 5]], dtype="float32")
        x2 = np.array([[2, 4], [8, 10]], dtype="float32")
        output = linalg.solve(x1, x2)
        expected_result = np.array([[2, 0], [0, 2]], dtype="float32")
        self.assertAllClose(output, expected_result)

    def test_solve_triangular(self):
        if testing.jax_uses_gpu():
            self.skipTest("Skipping test with JAX + GPU due to temporary error")

        # 2d-case
        x1 = np.array([[1, 2], [0, 5]], dtype="float32")
        x2 = np.array([2, 10], dtype="float32")
        output = linalg.solve_triangular(x1, x2, lower=True)
        expected_result = np.array([2, 2], dtype="float32")
        self.assertAllClose(output, expected_result)

        output = linalg.solve_triangular(x1, x2, lower=False)
        expected_result = np.array([-2, 2], dtype="float32")
        self.assertAllClose(output, expected_result)

        # batched case
        x1 = np.array([[[1, 2], [0, 5]], [[1, 2], [0, 5]]], dtype="float32")
        x2 = np.array([[2, 10], [2, 10]], dtype="float32")
        output = linalg.solve_triangular(x1, x2, lower=True)
        expected_result = np.array([[2, 2], [2, 2]], dtype="float32")
        self.assertAllClose(output, expected_result)

    def test_svd(self):
        x = np.random.rand(4, 30, 20).astype("float32")
        u, s, vh = linalg.svd(x)
        x_reconstructed = (u[..., :, : s.shape[-1]] * s[..., None, :]) @ vh[
            ..., : s.shape[-1], :
        ]
        # High tolerance due to numerical instability
        self.assertAllClose(x_reconstructed, x, atol=1e-3)

    @parameterized.named_parameters(
        ("b_rank_1", 1, None),
        ("b_rank_2", 2, None),
        ("rcond", 1, 1e-3),
    )
    def test_lstsq(self, b_rank, rcond):
        a = np.random.random((5, 7)).astype("float32")
        a_symb = backend.KerasTensor((5, 7))
        if b_rank == 1:
            b = np.random.random((5,)).astype("float32")
            b_symb = backend.KerasTensor((5,))
        else:
            b = np.random.random((5, 4)).astype("float32")
            b_symb = backend.KerasTensor((5, 4))
        out = linalg.lstsq(a, b, rcond=rcond)
        ref_out = np.linalg.lstsq(a, b, rcond=rcond)[0]
        self.assertAllClose(out, ref_out, atol=1e-5)

        out_symb = linalg.lstsq(a_symb, b_symb)
        self.assertEqual(out_symb.shape, out.shape)


class QrOpTest(testing.TestCase):
    def test_qr_init_mode_reduced(self):
        qr_op = linalg.Qr(mode="reduced")
        self.assertIsNotNone(qr_op)

    def test_qr_init_mode_complete(self):
        qr_op = linalg.Qr(mode="complete")
        self.assertIsNotNone(qr_op)

    def test_qr_init_invalid_mode(self):
        invalid_mode = "invalid_mode"
        expected_error = (
            r"`mode` argument value not supported. "
            r"Expected one of \{'reduced', 'complete'\}. "
            f"Received: mode={invalid_mode}"
        )
        with self.assertRaisesRegex(ValueError, expected_error):
            linalg.Qr(mode=invalid_mode)

    def test_compute_output_spec_low_rank(self):
        qr_op = linalg.Qr(mode="reduced")
        low_rank_input = np.random.rand(3)
        with self.assertRaisesRegex(
            ValueError, r"Input should have rank >= 2. Received: .*"
        ):
            qr_op.compute_output_spec(low_rank_input)

    def test_compute_output_spec_undefined_dimensions(self):
        qr_op = linalg.Qr(mode="reduced")
        undefined_dim_input = KerasTensor(shape=(None, 4), dtype="float32")
        with self.assertRaisesRegex(
            ValueError,
            r"Input should have its last 2 dimensions "
            r"fully-defined. Received: .*",
        ):
            qr_op.compute_output_spec(undefined_dim_input)

    def test_qr_call_mode_reduced(self):
        qr_op = linalg.Qr(mode="reduced")
        test_input = np.random.rand(10, 10)
        q, r = qr_op.call(test_input)
        self.assertEqual(q.shape, (10, 10))
        self.assertEqual(r.shape, (10, 10))

    def test_qr_call_mode_complete(self):
        qr_op = linalg.Qr(mode="complete")
        test_input = np.random.rand(10, 10)
        q, r = qr_op.call(test_input)
        self.assertEqual(q.shape, (10, 10))
        self.assertEqual(r.shape, (10, 10))
