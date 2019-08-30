"""Tests for Keras metrics classes."""
import pytest
import numpy as np

from keras import metrics
from keras import backend as K


if K.backend() != 'tensorflow':
    # Need TensorFlow to use metric.__call__
    pytestmark = pytest.mark.skip


class TestSum(object):

    def test_sum(self):
        m = metrics.Sum(name='my_sum', dtype='float32')

        # check config
        assert m.name == 'my_sum'
        assert m.stateful
        assert m.dtype == 'float32'
        assert len(m.weights) == 1

        # check initial state
        assert K.eval(m.total) == 0

        # check __call__
        assert K.eval(m(100.0)) == 100
        assert K.eval(m.total) == 100

        # check update_state() and result() + state accumulation + tensor input
        result = m([1, 5])
        assert np.isclose(K.eval(result), 106)
        assert K.eval(m.total) == 106  # 100 + 1 + 5

        # check reset_states()
        m.reset_states()
        assert K.eval(m.total) == 0

    def test_sum_with_sample_weight(self):
        m = metrics.Sum(dtype='float64')
        assert m.dtype == 'float64'

        # check scalar weight
        result_t = m(100, sample_weight=0.5)
        assert K.eval(result_t) == 50
        assert K.eval(m.total) == 50

        # check weights not scalar and weights rank matches values rank
        result_t = m([1, 5], sample_weight=[1, 0.2])
        result = K.eval(result_t)
        assert np.isclose(result, 52.)  # 50 + 1 + 5 * 0.2
        assert np.isclose(K.eval(m.total), 52.)

        # check weights broadcast
        result_t = m([1, 2], sample_weight=0.5)
        assert np.isclose(K.eval(result_t), 53.5)  # 52 + 0.5 + 1
        assert np.isclose(K.eval(m.total), 53.5)

        # check weights squeeze
        result_t = m([1, 5], sample_weight=[[1], [0.2]])
        assert np.isclose(K.eval(result_t), 55.5)  # 53.5 + 1 + 1
        assert np.isclose(K.eval(m.total), 55.5)

        # check weights expand
        result_t = m([[1], [5]], sample_weight=[1, 0.2])
        assert np.isclose(K.eval(result_t), 57.5, 2)  # 55.5 + 1 + 1
        assert np.isclose(K.eval(m.total), 57.5, 1)

        # check values reduced to the dimensions of weight
        result_t = m([[[1., 2.], [3., 2.], [0.5, 4.]]], sample_weight=[0.5])
        result = np.round(K.eval(result_t), decimals=2)
        # result = (prev: 57.5) + 0.5 + 1 + 1.5 + 1 + 0.25 + 2
        assert np.isclose(result, 63.75, 2)
        assert np.isclose(K.eval(m.total), 63.75, 2)


class TestMean(object):

    def test_mean(self):
        m = metrics.Mean(name='my_mean')

        # check config
        assert m.name == 'my_mean'
        assert m.stateful
        assert m.dtype == 'float32'
        assert len(m.weights) == 2

        # check initial state
        assert K.eval(m.total) == 0
        assert K.eval(m.count) == 0

        # check __call__()
        assert K.eval(m(100)) == 100
        assert K.eval(m.total) == 100
        assert K.eval(m.count) == 1

        # check update_state() and result()
        result = m([1, 5])
        assert np.isclose(K.eval(result), 106. / 3)
        assert K.eval(m.total) == 106  # 100 + 1 + 5
        assert K.eval(m.count) == 3

        # check reset_states()
        m.reset_states()
        assert K.eval(m.total) == 0
        assert K.eval(m.count) == 0

        # Check save and restore config
        m2 = metrics.Mean.from_config(m.get_config())
        assert m2.name == 'my_mean'
        assert m2.stateful
        assert m2.dtype == 'float32'
        assert len(m2.weights) == 2

    def test_mean_with_sample_weight(self):
        m = metrics.Mean(dtype='float64')
        assert m.dtype == 'float64'

        # check scalar weight
        result_t = m(100, sample_weight=0.5)
        assert K.eval(result_t) == 50. / 0.5
        assert K.eval(m.total) == 50
        assert K.eval(m.count) == 0.5

        # check weights not scalar and weights rank matches values rank
        result_t = m([1, 5], sample_weight=[1, 0.2])
        result = K.eval(result_t)
        assert np.isclose(result, 52. / 1.7)
        assert np.isclose(K.eval(m.total), 52)  # 50 + 1 + 5 * 0.2
        assert np.isclose(K.eval(m.count), 1.7)  # 0.5 + 1.2

        # check weights broadcast
        result_t = m([1, 2], sample_weight=0.5)
        assert np.isclose(K.eval(result_t), 53.5 / 2.7, rtol=3)
        assert np.isclose(K.eval(m.total), 53.5, rtol=3)  # 52 + 0.5 + 1
        assert np.isclose(K.eval(m.count), 2.7, rtol=3)  # 1.7 + 0.5 + 0.5

        # check weights squeeze
        result_t = m([1, 5], sample_weight=[[1], [0.2]])
        assert np.isclose(K.eval(result_t), 55.5 / 3.9, rtol=3)
        assert np.isclose(K.eval(m.total), 55.5, rtol=3)  # 53.5 + 1 + 1
        assert np.isclose(K.eval(m.count), 3.9, rtol=3)  # 2.7 + 1.2

        # check weights expand
        result_t = m([[1], [5]], sample_weight=[1, 0.2])
        assert np.isclose(K.eval(result_t), 57.5 / 5.1, rtol=3)
        assert np.isclose(K.eval(m.total), 57.5, rtol=3)  # 55.5 + 1 + 1
        assert np.isclose(K.eval(m.count), 5.1, rtol=3)  # 3.9 + 1.2

    def test_multiple_instances(self):
        m = metrics.Mean()
        m2 = metrics.Mean()

        assert m.name == 'mean'
        assert m2.name == 'mean'

        # check initial state
        assert K.eval(m.total) == 0
        assert K.eval(m.count) == 0
        assert K.eval(m2.total) == 0
        assert K.eval(m2.count) == 0

        # check __call__()
        assert K.eval(m(100)) == 100
        assert K.eval(m.total) == 100
        assert K.eval(m.count) == 1
        assert K.eval(m2.total) == 0
        assert K.eval(m2.count) == 0

        assert K.eval(m2([63, 10])) == 36.5
        assert K.eval(m2.total) == 73
        assert K.eval(m2.count) == 2
        assert K.eval(m.result()) == 100
        assert K.eval(m.total) == 100
        assert K.eval(m.count) == 1


class TestAccuracy(object):

    def test_accuracy(self):
        acc_obj = metrics.Accuracy(name='my_acc')

        # check config
        assert acc_obj.name == 'my_acc'
        assert acc_obj.stateful
        assert len(acc_obj.weights) == 2
        assert acc_obj.dtype == 'float32'

        # verify that correct value is returned
        result = K.eval(acc_obj([[1], [2], [3], [4]], [[1], [2], [3], [4]]))
        assert result == 1  # 2/2

        # Check save and restore config
        a2 = metrics.Accuracy.from_config(acc_obj.get_config())
        assert a2.name == 'my_acc'
        assert a2.stateful
        assert len(a2.weights) == 2
        assert a2.dtype, 'float32'

        # check with sample_weight
        result_t = acc_obj([[2], [1]], [[2], [0]], sample_weight=[[0.5], [0.2]])
        result = K.eval(result_t)
        assert np.isclose(result, 4.5 / 4.7, atol=1e-3)

    def test_binary_accuracy(self):
        acc_obj = metrics.BinaryAccuracy(name='my_acc')

        # check config
        assert acc_obj.name == 'my_acc'
        assert acc_obj.stateful
        assert len(acc_obj.weights) == 2
        assert acc_obj.dtype == 'float32'

        # verify that correct value is returned
        result_t = acc_obj([[1], [0]], [[1], [0]])
        result = K.eval(result_t)
        assert result == 1  # 2/2

        # check y_pred squeeze
        result_t = acc_obj([[1], [1]], [[[1]], [[0]]])
        result = K.eval(result_t)
        assert np.isclose(result, 3. / 4., atol=1e-3)

        # check y_true squeeze
        result_t = acc_obj([[[1]], [[1]]], [[1], [0]])
        result = K.eval(result_t)
        assert np.isclose(result, 4. / 6., atol=1e-3)

        # check with sample_weight
        result_t = acc_obj([[1], [1]], [[1], [0]], [[0.5], [0.2]])
        result = K.eval(result_t)
        assert np.isclose(result, 4.5 / 6.7, atol=1e-3)

    def test_binary_accuracy_threshold(self):
        acc_obj = metrics.BinaryAccuracy(threshold=0.7)
        result_t = acc_obj([[1], [1], [0], [0]], [[0.9], [0.6], [0.4], [0.8]])
        result = K.eval(result_t)
        assert np.isclose(result, 0.5, atol=1e-3)

    def test_categorical_accuracy(self):
        acc_obj = metrics.CategoricalAccuracy(name='my_acc')

        # check config
        assert acc_obj.name == 'my_acc'
        assert acc_obj.stateful
        assert len(acc_obj.weights) == 2
        assert acc_obj.dtype == 'float32'

        # verify that correct value is returned
        result_t = acc_obj([[0, 0, 1], [0, 1, 0]],
                           [[0.1, 0.1, 0.8], [0.05, 0.95, 0]])
        result = K.eval(result_t)
        assert result == 1  # 2/2

        # check with sample_weight
        result_t = acc_obj([[0, 0, 1], [0, 1, 0]],
                           [[0.1, 0.1, 0.8], [0.05, 0, 0.95]],
                           [[0.5], [0.2]])
        result = K.eval(result_t)
        assert np.isclose(result, 2.5 / 2.7, atol=1e-3)  # 2.5/2.7

    def test_sparse_categorical_accuracy(self):
        acc_obj = metrics.SparseCategoricalAccuracy(name='my_acc')

        # check config
        assert acc_obj.name == 'my_acc'
        assert acc_obj.stateful
        assert len(acc_obj.weights) == 2
        assert acc_obj.dtype == 'float32'

        # verify that correct value is returned
        result_t = acc_obj([[2], [1]],
                           [[0.1, 0.1, 0.8],
                           [0.05, 0.95, 0]])
        result = K.eval(result_t)
        assert result == 1  # 2/2

        # check with sample_weight
        result_t = acc_obj([[2], [1]],
                           [[0.1, 0.1, 0.8], [0.05, 0, 0.95]],
                           [[0.5], [0.2]])
        result = K.eval(result_t)
        assert np.isclose(result, 2.5 / 2.7, atol=1e-3)

    def test_sparse_categorical_accuracy_mismatched_dims(self):
        acc_obj = metrics.SparseCategoricalAccuracy(name='my_acc')

        # check config
        assert acc_obj.name == 'my_acc'
        assert acc_obj.stateful
        assert len(acc_obj.weights) == 2
        assert acc_obj.dtype == 'float32'

        # verify that correct value is returned
        result_t = acc_obj([2, 1], [[0.1, 0.1, 0.8], [0.05, 0.95, 0]])
        result = K.eval(result_t)
        assert result == 1  # 2/2

        # check with sample_weight
        result_t = acc_obj([2, 1], [[0.1, 0.1, 0.8], [0.05, 0, 0.95]],
                           [[0.5], [0.2]])
        result = K.eval(result_t)
        assert np.isclose(result, 2.5 / 2.7, atol=1e-3)


class TestMeanSquaredErrorTest(object):

    def test_config(self):
        mse_obj = metrics.MeanSquaredError(name='my_mse', dtype='int32')
        assert mse_obj.name == 'my_mse'
        assert mse_obj.dtype == 'int32'

        # Check save and restore config
        mse_obj2 = metrics.MeanSquaredError.from_config(mse_obj.get_config())
        assert mse_obj2.name == 'my_mse'
        assert mse_obj2.dtype == 'int32'

    def test_unweighted(self):
        mse_obj = metrics.MeanSquaredError()
        y_true = ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                  (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        y_pred = ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                  (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))

        result = mse_obj(y_true, y_pred)
        np.isclose(0.5, K.eval(result), atol=1e-5)

    def test_weighted(self):
        mse_obj = metrics.MeanSquaredError()
        y_true = ((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                  (1, 1, 1, 1, 0), (0, 0, 0, 0, 1))
        y_pred = ((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                  (0, 1, 0, 1, 0), (1, 1, 1, 1, 1))
        sample_weight = (1., 1.5, 2., 2.5)
        result = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        np.isclose(0.54285, K.eval(result), atol=1e-5)


class TestHinge(object):

    def test_config(self):
        hinge_obj = metrics.Hinge(name='hinge', dtype='int32')
        assert hinge_obj.name == 'hinge'
        assert hinge_obj.dtype == 'int32'

        # Check save and restore config
        hinge_obj2 = metrics.Hinge.from_config(hinge_obj.get_config())
        assert hinge_obj2.name == 'hinge'
        assert hinge_obj2.dtype == 'int32'

    def test_unweighted(self):
        hinge_obj = metrics.Hinge()
        y_true = K.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
        y_pred = K.constant([[-0.3, 0.2, -0.1, 1.6],
                             [-0.25, -1., 0.5, 0.6]])

        result = hinge_obj(y_true, y_pred)
        assert np.allclose(0.506, K.eval(result), atol=1e-3)

    def test_weighted(self):
        hinge_obj = metrics.Hinge()
        y_true = K.constant([[-1, 1, -1, 1], [-1, -1, 1, 1]])
        y_pred = K.constant([[-0.3, 0.2, -0.1, 1.6],
                             [-0.25, -1., 0.5, 0.6]])
        sample_weight = K.constant([1.5, 2.])

        result = hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(0.493, K.eval(result), atol=1e-3)


class TestSquaredHinge(object):

    def test_config(self):
        sq_hinge_obj = metrics.SquaredHinge(name='sq_hinge', dtype='int32')
        assert sq_hinge_obj.name == 'sq_hinge'
        assert sq_hinge_obj.dtype == 'int32'

        # Check save and restore config
        sq_hinge_obj2 = metrics.SquaredHinge.from_config(
            sq_hinge_obj.get_config())
        assert sq_hinge_obj2.name == 'sq_hinge'
        assert sq_hinge_obj2.dtype == 'int32'

    def test_unweighted(self):
        sq_hinge_obj = metrics.SquaredHinge()
        y_true = K.constant([[0, 1, 0, 1], [0, 0, 1, 1]])
        y_pred = K.constant([[-0.3, 0.2, -0.1, 1.6],
                             [-0.25, -1., 0.5, 0.6]])

        result = sq_hinge_obj(y_true, y_pred)
        assert np.allclose(0.364, K.eval(result), atol=1e-3)

    def test_weighted(self):
        sq_hinge_obj = metrics.SquaredHinge()
        y_true = K.constant([[-1, 1, -1, 1], [-1, -1, 1, 1]])
        y_pred = K.constant([[-0.3, 0.2, -0.1, 1.6],
                             [-0.25, -1., 0.5, 0.6]])
        sample_weight = K.constant([1.5, 2.])

        result = sq_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(0.347, K.eval(result), atol=1e-3)


class TestCategoricalHinge(object):

    def test_config(self):
        cat_hinge_obj = metrics.CategoricalHinge(
            name='cat_hinge', dtype='int32')
        assert cat_hinge_obj.name == 'cat_hinge'
        assert cat_hinge_obj.dtype == 'int32'

        # Check save and restore config
        cat_hinge_obj2 = metrics.CategoricalHinge.from_config(
            cat_hinge_obj.get_config())
        assert cat_hinge_obj2.name == 'cat_hinge'
        assert cat_hinge_obj2.dtype == 'int32'

    def test_unweighted(self):
        cat_hinge_obj = metrics.CategoricalHinge()
        y_true = K.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                             (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
        y_pred = K.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                             (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))

        result = cat_hinge_obj(y_true, y_pred)
        assert np.allclose(0.5, K.eval(result), atol=1e-5)

    def test_weighted(self):
        cat_hinge_obj = metrics.CategoricalHinge()
        y_true = K.constant(((0, 1, 0, 1, 0), (0, 0, 1, 1, 1),
                             (1, 1, 1, 1, 0), (0, 0, 0, 0, 1)))
        y_pred = K.constant(((0, 0, 1, 1, 0), (1, 1, 1, 1, 1),
                             (0, 1, 0, 1, 0), (1, 1, 1, 1, 1)))
        sample_weight = K.constant((1., 1.5, 2., 2.5))
        result = cat_hinge_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(0.5, K.eval(result), atol=1e-5)


class TestTopKCategoricalAccuracy(object):

    def test_config(self):
        a_obj = metrics.TopKCategoricalAccuracy(name='topkca', dtype='int32')
        assert a_obj.name == 'topkca'
        assert a_obj.dtype == 'int32'

        a_obj2 = metrics.TopKCategoricalAccuracy.from_config(a_obj.get_config())
        assert a_obj2.name == 'topkca'
        assert a_obj2.dtype == 'int32'

    def test_correctness(self):
        a_obj = metrics.TopKCategoricalAccuracy()
        y_true = [[0, 0, 1], [0, 1, 0]]
        y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]

        result = a_obj(y_true, y_pred)
        assert 1 == K.eval(result)  # both the samples match

        # With `k` < 5.
        a_obj = metrics.TopKCategoricalAccuracy(k=1)
        result = a_obj(y_true, y_pred)
        assert 0.5 == K.eval(result)  # only sample #2 matches

        # With `k` > 5.
        y_true = ([[0, 0, 1, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0]])
        y_pred = [[0.5, 0.9, 0.1, 0.7, 0.6, 0.5, 0.4],
                  [0.05, 0.95, 0, 0, 0, 0, 0]]
        a_obj = metrics.TopKCategoricalAccuracy(k=6)
        result = a_obj(y_true, y_pred)
        assert 0.5 == K.eval(result)  # only 1 sample matches.

    def test_weighted(self):
        a_obj = metrics.TopKCategoricalAccuracy(k=2)
        y_true = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        y_pred = [[0, 0.9, 0.1], [0, 0.9, 0.1], [0, 0.9, 0.1]]
        sample_weight = (1.0, 0.0, 1.0)
        result = a_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(1.0, K.eval(result), atol=1e-5)


class TestSparseTopKCategoricalAccuracy(object):

    def test_config(self):
        a_obj = metrics.SparseTopKCategoricalAccuracy(
            name='stopkca', dtype='int32')
        assert a_obj.name == 'stopkca'
        assert a_obj.dtype == 'int32'

        a_obj2 = metrics.SparseTopKCategoricalAccuracy.from_config(
            a_obj.get_config())
        assert a_obj2.name == 'stopkca'
        assert a_obj2.dtype == 'int32'

    def test_correctness(self):
        a_obj = metrics.SparseTopKCategoricalAccuracy()
        y_true = [2, 1]
        y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]

        result = a_obj(y_true, y_pred)
        assert 1 == K.eval(result)  # both the samples match

        # With `k` < 5.
        a_obj = metrics.SparseTopKCategoricalAccuracy(k=1)
        result = a_obj(y_true, y_pred)
        assert 0.5 == K.eval(result)  # only sample #2 matches

        # With `k` > 5.
        y_pred = [[0.5, 0.9, 0.1, 0.7, 0.6, 0.5, 0.4],
                  [0.05, 0.95, 0, 0, 0, 0, 0]]
        a_obj = metrics.SparseTopKCategoricalAccuracy(k=6)
        result = a_obj(y_true, y_pred)
        assert 0.5 == K.eval(result)  # only 1 sample matches.

    def test_weighted(self):
        a_obj = metrics.SparseTopKCategoricalAccuracy(k=2)
        y_true = [1, 0, 2]
        y_pred = [[0, 0.9, 0.1], [0, 0.9, 0.1], [0, 0.9, 0.1]]
        sample_weight = (1.0, 0.0, 1.0)
        result = a_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(1.0, K.eval(result), atol=1e-5)
