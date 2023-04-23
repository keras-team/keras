import numpy as np

from keras_core import operations as ops
from keras_core import testing
from keras_core.metrics import hinge_metrics


class HingeTest(testing.TestCase):
    def test_config(self):
        hinge_obj = hinge_metrics.Hinge(name="hinge", dtype="int32")
        self.assertEqual(hinge_obj.name, "hinge")
        self.assertEqual(hinge_obj._dtype, "int32")
        # TODO: Check save and restore config

    def test_unweighted(self):
        hinge_obj = hinge_metrics.Hinge()
        y_true = np.array([[0.0, 1.0, 0.0, 1.0], [0.0, 0.0, 1.0, 1.0]])
        y_pred = np.array([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        hinge_obj.update_state(y_true, y_pred)
        result = hinge_obj.result()
        self.assertAllClose(0.506, result, atol=1e-3)

    def test_weighted(self):
        hinge_obj = hinge_metrics.Hinge()
        y_true = np.array([[-1, 1, -1, 1], [-1, -1, 1, 1]])
        y_pred = np.array([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        sample_weight = np.array([1.5, 2.0])
        result = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.493, result, atol=1e-3)


class SquaredHingeTest(testing.TestCase):
    def test_config(self):
        sq_hinge_obj = hinge_metrics.SquaredHinge(
            name="squared_hinge", dtype="int32"
        )
        self.assertEqual(sq_hinge_obj.name, "squared_hinge")
        self.assertEqual(sq_hinge_obj._dtype, "int32")
        # TODO: Check save and restore config

    def test_unweighted(self):
        sq_hinge_obj = hinge_metrics.SquaredHinge()
        y_true = np.array([[0, 1, 0, 1], [0, 0, 1, 1]], dtype="float32")
        y_pred = np.array([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        sq_hinge_obj.update_state(y_true, y_pred)
        result = sq_hinge_obj.result()
        self.assertAllClose(0.364, result, atol=1e-3)

    def test_weighted(self):
        sq_hinge_obj = hinge_metrics.SquaredHinge()
        y_true = np.array([[-1, 1, -1, 1], [-1, -1, 1, 1]], dtype="float32")
        y_pred = np.array([[-0.3, 0.2, -0.1, 1.6], [-0.25, -1.0, 0.5, 0.6]])
        sample_weight = np.array([1.5, 2.0])
        result = sq_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.347, result, atol=1e-3)


class CategoricalHingeTest(testing.TestCase):
    def test_config(self):
        cat_hinge_obj = hinge_metrics.CategoricalHinge(
            name="cat_hinge", dtype="int32"
        )
        self.assertEqual(cat_hinge_obj.name, "cat_hinge")
        self.assertEqual(cat_hinge_obj._dtype, "int32")
        # TODO: Check save and restore config

    def test_unweighted(self):
        cat_hinge_obj = hinge_metrics.CategoricalHinge()
        y_true = np.array(
            (
                (0, 1, 0, 1, 0),
                (0, 0, 1, 1, 1),
                (1, 1, 1, 1, 0),
                (0, 0, 0, 0, 1),
            ),
            dtype="float32",
        )
        y_pred = np.array(
            (
                (0, 0, 1, 1, 0),
                (1, 1, 1, 1, 1),
                (0, 1, 0, 1, 0),
                (1, 1, 1, 1, 1),
            ),
            dtype="float32",
        )
        cat_hinge_obj.update_state(y_true, y_pred)
        result = cat_hinge_obj.result()
        self.assertAllClose(0.5, result, atol=1e-5)

    def test_weighted(self):
        cat_hinge_obj = hinge_metrics.CategoricalHinge()
        y_true = np.array(
            (
                (0, 1, 0, 1, 0),
                (0, 0, 1, 1, 1),
                (1, 1, 1, 1, 0),
                (0, 0, 0, 0, 1),
            ),
            dtype="float32",
        )
        y_pred = np.array(
            (
                (0, 0, 1, 1, 0),
                (1, 1, 1, 1, 1),
                (0, 1, 0, 1, 0),
                (1, 1, 1, 1, 1),
            ),
            dtype="float32",
        )
        sample_weight = np.array((1.0, 1.5, 2.0, 2.5))
        result = cat_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        self.assertAllClose(0.5, result, atol=1e-5)
