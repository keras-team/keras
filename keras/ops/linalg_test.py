
import numpy as np
from absl.testing import parameterized

from keras import backend
from keras import testing
from keras.backend.common.keras_tensor import KerasTensor
from keras.ops import linalg
from keras.testing.test_utils import named_product


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

        x = KerasTensor([None, None, 20])
        with self.assertRaises(linalg.LinalgError):
            linalg.eig(x)

        x = KerasTensor([None, 20, 15])
        with self.assertRaises(linalg.LinalgError):
            linalg.eig(x)

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
        with self.assertRaises(linalg.LinalgError):
            linalg.solve_triangular(a, b)

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, None, 5])
        with self.assertRaises(linalg.LinalgError):
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
        with self.assertRaises(linalg.LinalgError):
            linalg.eig(x)


    def test_inv(self):
        x = KerasTensor([4, 3, 3])
        out = linalg.inv(x)
        self.assertEqual(out.shape, (4, 3, 3))

        x = KerasTensor([10, 20, 15])
        with self.assertRaises(ValueError):
            linalg.inv(x)

    def test_lu_factor(self):
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
        with self.assertRaises(linalg.LinalgError):
            linalg.cholesky(x)
        x_psd = x @ x.transpose((0, 2, 1)) + 1e-5 * np.eye(3)
        out = linalg.cholesky(x_psd)
        self.assertAllClose(out, np.linalg.cholesky(x_psd), atol=1e-4)

    def test_det(self):
        x = np.random.rand(4, 3, 3)
        out = linalg.det(x)
        self.assertAllClose(out, np.linalg.det(x), atol=1e-5)

    def test_eig(self):
        x = np.random.rand(4, 3, 3)
        x = x @ x.transpose((0, 2, 1))
        w, v = linalg.eig(x)
        w_ref, v_ref = np.linalg.eig(x)
        self.assertAllClose(w, w_ref, atol=1e-3)

    def test_inv(self):
        x = np.random.rand(4, 3, 3)
        out = linalg.inv(x)
        self.assertAllClose(out, np.linalg.inv(x), atol=1e-5)


    def test_lu_factor(self):
        from scipy.linalg import lu_factor
        # 2d-case
        x = np.random.rand(3, 3)
        lu, p = linalg.lu_factor(x)
        lu_ref, p_ref = lu_factor(x)
        self.assertAllClose(lu, lu_ref, atol=1e-5)
        self.assertAllClose(p, p_ref, atol=1e-5)

        # batched case
        x = np.random.rand(4, 3, 3)
        lu, p = linalg.lu_factor(x)
        for i in range(4):
            lu_ref, p_ref = lu_factor(x[i])
            self.assertAllClose(lu[i], lu_ref, atol=1e-5)
            self.assertAllClose(p[i], p_ref, atol=1e-5)        

    @parameterized.named_parameters(
        named_product(
            ord=[None, "fro", "nuc", -np.inf, -2, -1, 0, 1, 2, np.inf, 3],
            axis=[None, 1, -1],
            keepdims=[False, True],
        )
    )
    def test_norm_vectors(self, ord, axis, keepdims):
        if axis is None:
            x = np.random.random((5,))
        else:
            x = np.random.random((5, 6))
        if ord in ("fro", "nuc"):
            error = RuntimeError if backend.backend() == "torch" else ValueError
            with self.assertRaises(error):
                linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
            return
        output = linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)
        expected_result = np.linalg.norm(
            x, ord=ord, axis=axis, keepdims=keepdims
        )
        self.assertAllClose(output, expected_result)

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
        x = np.random.rand(4, 30, 20)
        u, s, vh = linalg.svd(x)
        x_reconstructed = (u[...,:,:s.shape[-1]] * s[...,None,:]) @ vh[..., :s.shape[-1],:]
        self.assertAllClose(x_reconstructed, x, atol=1e-4)