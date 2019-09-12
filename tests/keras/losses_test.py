import pytest
import numpy as np

import keras
from keras import losses
from keras import backend as K
from keras.utils import losses_utils
from keras.utils.generic_utils import custom_object_scope


all_functions = [losses.mean_squared_error,
                 losses.mean_absolute_error,
                 losses.mean_absolute_percentage_error,
                 losses.mean_squared_logarithmic_error,
                 losses.squared_hinge,
                 losses.hinge,
                 losses.categorical_crossentropy,
                 losses.binary_crossentropy,
                 losses.kullback_leibler_divergence,
                 losses.poisson,
                 losses.cosine_proximity,
                 losses.logcosh,
                 losses.categorical_hinge]
all_classes = [
    losses.Hinge,
    losses.SquaredHinge,
    losses.CategoricalHinge,
    losses.Poisson,
    losses.LogCosh,
    losses.KLDivergence,
    losses.Huber,
    # losses.SparseCategoricalCrossentropy,
    losses.BinaryCrossentropy,
    losses.MeanSquaredLogarithmicError,
    losses.MeanAbsolutePercentageError,
    losses.MeanAbsoluteError,
    losses.MeanSquaredError,
]


class MSE_MAE_loss(object):
    """Loss function with internal state, for testing serialization code."""

    def __init__(self, mse_fraction):
        self.mse_fraction = mse_fraction

    def __call__(self, y_true, y_pred, sample_weight=None):
        return (self.mse_fraction * losses.mse(y_true, y_pred) +
                (1 - self.mse_fraction) * losses.mae(y_true, y_pred))

    def get_config(self):
        return {'mse_fraction': self.mse_fraction}


class TestLossFunctions(object):

    @pytest.mark.parametrize('loss_fn', all_functions)
    def test_objective_shapes_3d(self, loss_fn):
        y_a = K.variable(np.random.random((5, 6, 7)))
        y_b = K.variable(np.random.random((5, 6, 7)))
        objective_output = loss_fn(y_a, y_b)
        assert K.eval(objective_output).shape == (5, 6)

    @pytest.mark.parametrize('loss_fn', all_functions)
    def test_objective_shapes_2d(self, loss_fn):
        y_a = K.variable(np.random.random((6, 7)))
        y_b = K.variable(np.random.random((6, 7)))
        objective_output = loss_fn(y_a, y_b)
        assert K.eval(objective_output).shape == (6,)

    def test_cce_one_hot(self):
        y_a = K.variable(np.random.randint(0, 7, (5, 6)))
        y_b = K.variable(np.random.random((5, 6, 7)))
        objective_output = losses.sparse_categorical_crossentropy(y_a, y_b)
        assert K.eval(objective_output).shape == (5, 6)

        y_a = K.variable(np.random.randint(0, 7, (6,)))
        y_b = K.variable(np.random.random((6, 7)))
        assert K.eval(losses.sparse_categorical_crossentropy(y_a, y_b)).shape == (6,)

    def test_categorical_hinge(self):
        y_pred = K.variable(np.array([[0.3, 0.2, 0.1],
                                      [0.1, 0.2, 0.7]]))
        y_true = K.variable(np.array([[0, 1, 0],
                                      [1, 0, 0]]))
        expected_loss = ((0.3 - 0.2 + 1) + (0.7 - 0.1 + 1)) / 2.0
        loss = K.eval(losses.categorical_hinge(y_true, y_pred))
        assert np.isclose(expected_loss, np.mean(loss))

    def test_sparse_categorical_crossentropy(self):
        y_pred = K.variable(np.array([[0.3, 0.6, 0.1],
                                      [0.1, 0.2, 0.7]]))
        y_true = K.variable(np.array([1, 2]))
        expected_loss = - (np.log(0.6) + np.log(0.7)) / 2
        loss = K.eval(losses.sparse_categorical_crossentropy(y_true, y_pred))
        assert np.isclose(expected_loss, np.mean(loss))

    def test_sparse_categorical_crossentropy_4d(self):
        y_pred = K.variable(np.array([[[[0.7, 0.1, 0.2],
                                        [0.0, 0.3, 0.7],
                                        [0.1, 0.1, 0.8]],
                                       [[0.3, 0.7, 0.0],
                                        [0.3, 0.4, 0.3],
                                        [0.2, 0.5, 0.3]],
                                       [[0.8, 0.1, 0.1],
                                        [1.0, 0.0, 0.0],
                                        [0.4, 0.3, 0.3]]]]))
        y_true = K.variable(np.array([[[0, 1, 0],
                                       [2, 1, 0],
                                       [2, 2, 1]]]))
        expected_loss = - (np.log(0.7) + np.log(0.3) + np.log(0.1) +
                           np.log(K.epsilon()) + np.log(0.4) + np.log(0.2) +
                           np.log(0.1) + np.log(K.epsilon()) + np.log(0.3)) / 9
        loss = K.eval(losses.sparse_categorical_crossentropy(y_true, y_pred))
        assert np.isclose(expected_loss, np.mean(loss))

    def test_serializing_loss_class(self):
        orig_loss_class = MSE_MAE_loss(0.3)
        with custom_object_scope({'MSE_MAE_loss': MSE_MAE_loss}):
            serialized = losses.serialize(orig_loss_class)

        with custom_object_scope({'MSE_MAE_loss': MSE_MAE_loss}):
            deserialized = losses.deserialize(serialized)
        assert isinstance(deserialized, MSE_MAE_loss)
        assert deserialized.mse_fraction == 0.3

    def test_serializing_model_with_loss_class(self, tmpdir):
        model_filename = str(tmpdir / 'custom_loss.hdf')

        with custom_object_scope({'MSE_MAE_loss': MSE_MAE_loss}):
            loss = MSE_MAE_loss(0.3)
            inputs = keras.layers.Input((2,))
            outputs = keras.layers.Dense(1, name='model_output')(inputs)
            model = keras.models.Model(inputs, outputs)
            model.compile(optimizer='sgd', loss={'model_output': loss})
            model.fit(np.random.rand(256, 2), np.random.rand(256, 1))
            model.save(model_filename)

        with custom_object_scope({'MSE_MAE_loss': MSE_MAE_loss}):
            loaded_model = keras.models.load_model(model_filename)
            loaded_model.predict(np.random.rand(128, 2))

    def test_loss_wrapper(self):
        loss_fn = losses.get('mse')
        mse_obj = losses.LossFunctionWrapper(loss_fn, name=loss_fn.__name__)

        assert mse_obj.name == 'mean_squared_error'
        assert (mse_obj.reduction == losses_utils.Reduction.SUM_OVER_BATCH_SIZE)

        y_true = K.constant([[1., 9.], [2., 5.]])
        y_pred = K.constant([[4., 8.], [12., 3.]])
        sample_weight = K.constant([1.2, 0.5])
        loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)

        # mse = [((4 - 1)^2 + (8 - 9)^2) / 2, ((12 - 2)^2 + (3 - 5)^2) / 2]
        # mse = [5, 52]
        # weighted_mse = [5 * 1.2, 52 * 0.5] = [6, 26]
        # reduced_weighted_mse = (6 + 26) / 2 =
        np.allclose(K.eval(loss), 16, atol=1e-2)


skipif_not_tf = pytest.mark.skipif(
    K.backend() != 'tensorflow',
    reason='Need TensorFlow to __call__ a loss')


class TestLossClasses(object):

    @pytest.mark.parametrize('cls', all_classes)
    def test_objective_shapes_3d(self, cls):
        y_a = K.variable(np.random.random((5, 6, 7)))
        y_b = K.variable(np.random.random((5, 6, 7)))
        sw = K.variable(np.random.random((5, 6)))
        obj_fn = cls(name='test')
        objective_output = obj_fn(y_a, y_b, sample_weight=sw)
        assert K.eval(objective_output).shape == ()

    @pytest.mark.parametrize('cls', all_classes)
    def test_objective_shapes_2d(self, cls):
        y_a = K.variable(np.random.random((6, 7)))
        y_b = K.variable(np.random.random((6, 7)))
        sw = K.variable(np.random.random((6,)))
        obj_fn = cls(name='test')
        objective_output = obj_fn(y_a, y_b, sample_weight=sw)
        assert K.eval(objective_output).shape == ()


@skipif_not_tf
class TestMeanSquaredError:

    def test_config(self):
        mse_obj = losses.MeanSquaredError(
            reduction=losses_utils.Reduction.SUM, name='mse_1')
        assert mse_obj.name == 'mse_1'
        assert mse_obj.reduction == losses_utils.Reduction.SUM

    def test_all_correct_unweighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mse_obj(y_true, y_true)
        assert np.isclose(K.eval(loss), 0.0)

    def test_unweighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mse_obj(y_true, y_pred)
        assert np.isclose(K.eval(loss), 49.5, atol=1e-3)

    def test_scalar_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        assert np.isclose(K.eval(loss), 113.85, atol=1e-3)

    def test_sample_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        sample_weight = K.constant([1.2, 3.4], shape=(2, 1))
        loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.isclose(K.eval(loss), 767.8 / 6, atol=1e-3)

    def test_timestep_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3, 1))
        sample_weight = K.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
        loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.isclose(K.eval(loss), 97.833, atol=1e-3)

    def test_zero_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mse_obj(y_true, y_pred, sample_weight=0)
        assert np.isclose(K.eval(loss), 0.0)

    def test_invalid_sample_weight(self):
        mse_obj = losses.MeanSquaredError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3, 1))
        sample_weight = K.constant([3, 6, 5, 0], shape=(2, 2))
        with pytest.raises(Exception):
            mse_obj(y_true, y_pred, sample_weight=sample_weight)

    def test_no_reduction(self):
        mse_obj = losses.MeanSquaredError(
            reduction=losses_utils.Reduction.NONE)
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        assert np.allclose(K.eval(loss), [84.3333, 143.3666], atol=1e-3)

    def test_sum_reduction(self):
        mse_obj = losses.MeanSquaredError(
            reduction=losses_utils.Reduction.SUM)
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        assert np.isclose(K.eval(loss), 227.69998, atol=1e-3)


@skipif_not_tf
class TestMeanAbsoluteError(object):

    def test_config(self):
        mae_obj = losses.MeanAbsoluteError(
            reduction=losses_utils.Reduction.SUM, name='mae_1')
        assert mae_obj.name == 'mae_1'
        assert mae_obj.reduction == losses_utils.Reduction.SUM

    def test_all_correct_unweighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mae_obj(y_true, y_true)
        assert np.isclose(K.eval(loss), 0.0, atol=1e-3)

    def test_unweighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mae_obj(y_true, y_pred)
        assert np.isclose(K.eval(loss), 5.5, atol=1e-3)

    def test_scalar_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mae_obj(y_true, y_pred, sample_weight=2.3)
        assert np.isclose(K.eval(loss), 12.65, atol=1e-3)

    def test_sample_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        sample_weight = K.constant([1.2, 3.4], shape=(2, 1))
        loss = mae_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.isclose(K.eval(loss), 81.4 / 6, atol=1e-3)

    def test_timestep_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3, 1))
        sample_weight = K.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
        loss = mae_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.isclose(K.eval(loss), 13.833, atol=1e-3)

    def test_zero_weighted(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mae_obj(y_true, y_pred, sample_weight=0)
        assert np.isclose(K.eval(loss), 0.0, atol=1e-3)

    def test_invalid_sample_weight(self):
        mae_obj = losses.MeanAbsoluteError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3, 1))
        sample_weight = K.constant([3, 6, 5, 0], shape=(2, 2))
        with pytest.raises(Exception):
            mae_obj(y_true, y_pred, sample_weight=sample_weight)

    def test_no_reduction(self):
        mae_obj = losses.MeanAbsoluteError(
            reduction=losses_utils.Reduction.NONE)
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mae_obj(y_true, y_pred, sample_weight=2.3)
        assert np.allclose(K.eval(loss), [10.7333, 14.5666], atol=1e-3)

    def test_sum_reduction(self):
        mae_obj = losses.MeanAbsoluteError(
            reduction=losses_utils.Reduction.SUM)
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mae_obj(y_true, y_pred, sample_weight=2.3)
        assert np.isclose(K.eval(loss), 25.29999, atol=1e-3)


@skipif_not_tf
class TestMeanAbsolutePercentageError(object):

    def test_config(self):
        mape_obj = losses.MeanAbsolutePercentageError(
            reduction=losses_utils.Reduction.SUM, name='mape_1')
        assert mape_obj.name == 'mape_1'
        assert mape_obj.reduction == losses_utils.Reduction.SUM

    def test_all_correct_unweighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mape_obj(y_true, y_true)
        assert np.allclose(K.eval(loss), 0.0, atol=1e-3)

    def test_unweighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mape_obj(y_true, y_pred)
        assert np.allclose(K.eval(loss), 211.8518, atol=1e-3)

    def test_scalar_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mape_obj(y_true, y_pred, sample_weight=2.3)
        assert np.allclose(K.eval(loss), 487.259, atol=1e-3)

    def test_sample_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        sample_weight = K.constant([1.2, 3.4], shape=(2, 1))
        loss = mape_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(K.eval(loss), 422.8888, atol=1e-3)

    def test_timestep_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3, 1))
        sample_weight = K.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
        loss = mape_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(K.eval(loss), 694.4445, atol=1e-3)

    def test_zero_weighted(self):
        mape_obj = losses.MeanAbsolutePercentageError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mape_obj(y_true, y_pred, sample_weight=0)
        assert np.allclose(K.eval(loss), 0.0, atol=1e-3)

    def test_no_reduction(self):
        mape_obj = losses.MeanAbsolutePercentageError(
            reduction=losses_utils.Reduction.NONE)
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mape_obj(y_true, y_pred, sample_weight=2.3)
        assert np.allclose(K.eval(loss), [621.8518, 352.6666], atol=1e-3)


@skipif_not_tf
class TestMeanSquaredLogarithmicError(object):

    def test_config(self):
        msle_obj = losses.MeanSquaredLogarithmicError(
            reduction=losses_utils.Reduction .SUM, name='mape_1')
        assert msle_obj.name == 'mape_1'
        assert msle_obj.reduction == losses_utils.Reduction .SUM

    def test_unweighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = msle_obj(y_true, y_pred)
        assert np.allclose(K.eval(loss), 1.4370, atol=1e-3)

    def test_scalar_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = msle_obj(y_true, y_pred, sample_weight=2.3)
        assert np.allclose(K.eval(loss), 3.3051, atol=1e-3)

    def test_sample_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        sample_weight = K.constant([1.2, 3.4], shape=(2, 1))
        loss = msle_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(K.eval(loss), 3.7856, atol=1e-3)

    def test_timestep_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3, 1))
        sample_weight = K.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
        loss = msle_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.allclose(K.eval(loss), 2.6473, atol=1e-3)

    def test_zero_weighted(self):
        msle_obj = losses.MeanSquaredLogarithmicError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = msle_obj(y_true, y_pred, sample_weight=0)
        assert np.allclose(K.eval(loss), 0.0, atol=1e-3)


@skipif_not_tf
class TestBinaryCrossentropy(object):

    def test_config(self):
        bce_obj = losses.BinaryCrossentropy(
            reduction=losses_utils.Reduction.SUM, name='bce_1')
        assert bce_obj.name == 'bce_1'
        assert bce_obj.reduction == losses_utils.Reduction.SUM

    def test_all_correct_unweighted(self):
        y_true = K.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        bce_obj = losses.BinaryCrossentropy()
        loss = bce_obj(y_true, y_true)
        assert np.isclose(K.eval(loss), 0.0, atol=1e-3)

        # Test with logits.
        logits = K.constant([[100.0, -100.0, -100.0],
                             [-100.0, 100.0, -100.0],
                             [-100.0, -100.0, 100.0]])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits)
        assert np.isclose(K.eval(loss), 0.0, 3)

    def test_unweighted(self):
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
        bce_obj = losses.BinaryCrossentropy()
        loss = bce_obj(y_true, y_pred)

        # EPSILON = 1e-7, y = y_true, y` = y_pred, Y_MAX = 0.9999999
        # y` = clip(output, EPSILON, 1. - EPSILON)
        # y` = [Y_MAX, Y_MAX, Y_MAX, EPSILON]

        # Loss = -(y log(y` + EPSILON) + (1 - y) log(1 - y` + EPSILON))
        #      = [-log(Y_MAX + EPSILON), -log(1 - Y_MAX + EPSILON),
        #         -log(Y_MAX + EPSILON), -log(1)]
        #      = [0, 15.33, 0, 0]
        # Reduced loss = 15.33 / 4

        assert np.isclose(K.eval(loss), 3.833, atol=1e-3)

        # Test with logits.
        y_true = K.constant([[1., 0., 1.], [0., 1., 1.]])
        logits = K.constant([[100.0, -100.0, 100.0], [100.0, 100.0, -100.0]])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits)

        # Loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
        #            (where x = logits and z = y_true)
        #      = [((100 - 100 * 1 + log(1 + exp(-100))) +
        #          (0 + 100 * 0 + log(1 + exp(-100))) +
        #          (100 - 100 * 1 + log(1 + exp(-100))),
        #         ((100 - 100 * 0 + log(1 + exp(-100))) +
        #          (100 - 100 * 1 + log(1 + exp(-100))) +
        #          (0 + 100 * 1 + log(1 + exp(-100))))]
        #      = [(0 + 0 + 0) / 3, 200 / 3]
        # Reduced loss = (0 + 66.666) / 2

        assert np.isclose(K.eval(loss), 33.333, atol=1e-3)

    def test_scalar_weighted(self):
        bce_obj = losses.BinaryCrossentropy()
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
        loss = bce_obj(y_true, y_pred, sample_weight=2.3)

        # EPSILON = 1e-7, y = y_true, y` = y_pred, Y_MAX = 0.9999999
        # y` = clip(output, EPSILON, 1. - EPSILON)
        # y` = [Y_MAX, Y_MAX, Y_MAX, EPSILON]

        # Loss = -(y log(y` + EPSILON) + (1 - y) log(1 - y` + EPSILON))
        #      = [-log(Y_MAX + EPSILON), -log(1 - Y_MAX + EPSILON),
        #         -log(Y_MAX + EPSILON), -log(1)]
        #      = [0, 15.33, 0, 0]
        # Weighted loss = [0, 15.33 * 2.3, 0, 0]
        # Reduced loss = 15.33 * 2.3 / 4

        assert np.isclose(K.eval(loss), 8.817, atol=1e-3)

        # Test with logits.
        y_true = K.constant([[1, 0, 1], [0, 1, 1]])
        logits = K.constant([[100.0, -100.0, 100.0], [100.0, 100.0, -100.0]])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits, sample_weight=2.3)

        # Loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
        #            (where x = logits and z = y_true)
        # Loss = [(0 + 0 + 0) / 3, 200 / 3]
        # Weighted loss = [0 * 2.3, 66.666 * 2.3]
        # Reduced loss = (0 + 66.666 * 2.3) / 2

        assert np.isclose(K.eval(loss), 76.667, atol=1e-3)

    def test_sample_weighted(self):
        bce_obj = losses.BinaryCrossentropy()
        y_true = np.asarray([1, 0, 1, 0]).reshape([2, 2])
        y_pred = np.asarray([1, 1, 1, 0], dtype=np.float32).reshape([2, 2])
        sample_weight = K.constant([1.2, 3.4], shape=(2, 1))
        loss = bce_obj(y_true, y_pred, sample_weight=sample_weight)

        # EPSILON = 1e-7, y = y_true, y` = y_pred, Y_MAX = 0.9999999
        # y` = clip(output, EPSILON, 1. - EPSILON)
        # y` = [Y_MAX, Y_MAX, Y_MAX, EPSILON]

        # Loss = -(y log(y` + EPSILON) + (1 - y) log(1 - y` + EPSILON))
        #      = [-log(Y_MAX + EPSILON), -log(1 - Y_MAX + EPSILON),
        #         -log(Y_MAX + EPSILON), -log(1)]
        #      = [0, 15.33, 0, 0]
        # Reduced loss = 15.33 * 1.2 / 4

        assert np.isclose(K.eval(loss), 4.6, atol=1e-3)

        # Test with logits.
        y_true = K.constant([[1, 0, 1], [0, 1, 1]])
        logits = K.constant([[100.0, -100.0, 100.0], [100.0, 100.0, -100.0]])
        weights = K.constant([4, 3])
        bce_obj = losses.BinaryCrossentropy(from_logits=True)
        loss = bce_obj(y_true, logits, sample_weight=weights)

        # Loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
        #            (where x = logits and z = y_true)
        # Loss = [(0 + 0 + 0)/3, 200 / 3]
        # Weighted loss = [0 * 4, 66.666 * 3]
        # Reduced loss = (0 + 66.666 * 3) / 2

        assert np.isclose(K.eval(loss), 100, atol=1e-3)

    def test_no_reduction(self):
        y_true = K.constant([[1, 0, 1], [0, 1, 1]])
        logits = K.constant([[100.0, -100.0, 100.0], [100.0, 100.0, -100.0]])
        bce_obj = losses.BinaryCrossentropy(
            from_logits=True, reduction=losses_utils.Reduction.NONE)
        loss = bce_obj(y_true, logits)

        # Loss = max(x, 0) - x * z + log(1 + exp(-abs(x)))
        #            (where x = logits and z = y_true)
        # Loss = [(0 + 0 + 0)/3, (200)/3]

        assert np.allclose(K.eval(loss), (0., 66.6666), atol=1e-3)

    def test_label_smoothing(self):
        logits = K.constant([[100.0, -100.0, -100.0]])
        y_true = K.constant([[1, 0, 1]])
        label_smoothing = 0.1
        # Loss: max(x, 0) - x * z + log(1 + exp(-abs(x)))
        #            (where x = logits and z = y_true)
        # Label smoothing: z' = z * (1 - L) + 0.5L
        #                  1  = 1 - 0.5L
        #                  0  = 0.5L
        # Applying the above two fns to the given input:
        # (100 - 100 * (1 - 0.5 L)  + 0 +
        #  0   + 100 * (0.5 L)      + 0 +
        #  0   + 100 * (1 - 0.5 L)  + 0) * (1/3)
        #  = (100 + 50L) * 1/3
        bce_obj = losses.BinaryCrossentropy(
            from_logits=True, label_smoothing=label_smoothing)
        loss = bce_obj(y_true, logits)
        expected_value = (100.0 + 50.0 * label_smoothing) / 3.0
        assert np.isclose(K.eval(loss), expected_value, atol=1e-3)


@skipif_not_tf
class TestCategoricalCrossentropy(object):

    def test_config(self):
        cce_obj = losses.CategoricalCrossentropy(
            reduction=losses_utils.Reduction.SUM, name='bce_1')
        assert cce_obj.name == 'bce_1'
        assert cce_obj.reduction == losses_utils.Reduction.SUM

    def test_all_correct_unweighted(self):
        y_true = K.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = K.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        cce_obj = losses.CategoricalCrossentropy()
        loss = cce_obj(y_true, y_pred)
        assert np.isclose(K.eval(loss), 0.0, atol=1e-3)

        # Test with logits.
        logits = K.constant([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        assert np.isclose(K.eval(loss), 0.0, atol=1e-3)

    def test_unweighted(self):
        cce_obj = losses.CategoricalCrossentropy()
        y_true = K.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = K.constant(
            [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
        loss = cce_obj(y_true, y_pred)
        assert np.isclose(K.eval(loss), .3239, atol=1e-3)

        # Test with logits.
        logits = K.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        assert np.isclose(K.eval(loss), .05737, atol=1e-3)

    def test_scalar_weighted(self):
        cce_obj = losses.CategoricalCrossentropy()
        y_true = K.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = K.constant(
            [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
        loss = cce_obj(y_true, y_pred, sample_weight=2.3)
        assert np.isclose(K.eval(loss), .7449, atol=1e-3)

        # Test with logits.
        logits = K.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=2.3)
        assert np.isclose(K.eval(loss), .132, atol=1e-3)

    def test_sample_weighted(self):
        cce_obj = losses.CategoricalCrossentropy()
        y_true = K.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_pred = K.constant(
            [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
        sample_weight = K.constant([[1.2], [3.4], [5.6]], shape=(3, 1))
        loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.isclose(K.eval(loss), 1.0696, atol=1e-3)

        # Test with logits.
        logits = K.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
        cce_obj = losses.CategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=sample_weight)
        assert np.isclose(K.eval(loss), 0.31829, atol=1e-3)

    def test_no_reduction(self):
        y_true = K.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        logits = K.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
        cce_obj = losses.CategoricalCrossentropy(
            from_logits=True, reduction=losses_utils.Reduction.NONE)
        loss = cce_obj(y_true, logits)
        assert np.allclose(K.eval(loss), (0.001822, 0.000459, 0.169846), atol=1e-3)

    def test_label_smoothing(self):
        logits = K.constant([[100.0, -100.0, -100.0]])
        y_true = K.constant([[1, 0, 0]])
        label_smoothing = 0.1
        # Softmax Cross Entropy Loss: -\sum_i p_i \log q_i
        # where for a softmax activation
        # \log q_i = x_i - \log \sum_j \exp x_j
        #          = x_i - x_max - \log \sum_j \exp (x_j - x_max)
        # For our activations, [100, -100, -100]
        # \log ( exp(0) + exp(-200) + exp(-200) ) = 0
        # so our log softmaxes become: [0, -200, -200]
        # Label smoothing: z' = z * (1 - L) + L/n
        #                  1  = 1 - L + L/n
        #                  0  = L/n
        # Applying the above two fns to the given input:
        # -0 * (1 - L + L/n) + 200 * L/n + 200 * L/n = 400 L/n
        cce_obj = losses.CategoricalCrossentropy(
            from_logits=True, label_smoothing=label_smoothing)
        loss = cce_obj(y_true, logits)
        expected_value = 400.0 * label_smoothing / 3.0
        assert np.isclose(K.eval(loss), expected_value, atol=1e-3)


@skipif_not_tf
class TestSparseCategoricalCrossentropy(object):

    def test_config(self):
        cce_obj = losses.SparseCategoricalCrossentropy(
            reduction=losses_utils.Reduction.SUM, name='scc')
        assert cce_obj.name == 'scc'
        assert cce_obj.reduction == losses_utils.Reduction.SUM

    def test_all_correct_unweighted(self):
        y_true = K.constant([[0], [1], [2]])
        y_pred = K.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        cce_obj = losses.SparseCategoricalCrossentropy()
        loss = cce_obj(y_true, y_pred)
        assert np.isclose(K.eval(loss), 0.0, atol=1e-3)

        # Test with logits.
        logits = K.constant([[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        assert np.isclose(K.eval(loss), 0.0, atol=1e-3)

    def test_unweighted(self):
        cce_obj = losses.SparseCategoricalCrossentropy()
        y_true = K.constant([0, 1, 2])
        y_pred = K.constant(
            [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
        loss = cce_obj(y_true, y_pred)
        assert np.isclose(K.eval(loss), .3239, atol=1e-3)

        # Test with logits.
        logits = K.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits)
        assert np.isclose(K.eval(loss), .0573, atol=1e-3)

    def test_scalar_weighted(self):
        cce_obj = losses.SparseCategoricalCrossentropy()
        y_true = K.constant([[0], [1], [2]])
        y_pred = K.constant(
            [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
        loss = cce_obj(y_true, y_pred, sample_weight=2.3)
        assert np.isclose(K.eval(loss), .7449, atol=1e-3)

        # Test with logits.
        logits = K.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=2.3)
        assert np.isclose(K.eval(loss), .1317, atol=1e-3)

    def test_sample_weighted(self):
        cce_obj = losses.SparseCategoricalCrossentropy()
        y_true = K.constant([[0], [1], [2]])
        y_pred = K.constant(
            [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
        sample_weight = K.constant([[1.2], [3.4], [5.6]], shape=(3, 1))
        loss = cce_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.isclose(K.eval(loss), 1.0696, atol=1e-3)

        # Test with logits.
        logits = K.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
        cce_obj = losses.SparseCategoricalCrossentropy(from_logits=True)
        loss = cce_obj(y_true, logits, sample_weight=sample_weight)
        assert np.isclose(K.eval(loss), 0.31829, atol=1e-3)

    def test_no_reduction(self):
        y_true = K.constant([[0], [1], [2]])
        logits = K.constant([[8., 1., 1.], [0., 9., 1.], [2., 3., 5.]])
        cce_obj = losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=losses_utils.Reduction.NONE)
        loss = cce_obj(y_true, logits)
        assert np.allclose(K.eval(loss), (0.001822, 0.000459, 0.169846), atol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__])
