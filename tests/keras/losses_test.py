import pytest
import numpy as np

import keras
from keras import losses
from keras import backend as K
from keras.utils import losses_utils
from keras.utils.generic_utils import custom_object_scope


allobj = [losses.mean_squared_error,
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


class MSE_MAE_loss:
    """Loss function with internal state, for testing serialization code."""
    def __init__(self, mse_fraction):
        self.mse_fraction = mse_fraction

    def __call__(self, y_true, y_pred):
        return (self.mse_fraction * losses.mse(y_true, y_pred) +
                (1 - self.mse_fraction) * losses.mae(y_true, y_pred))

    def get_config(self):
        return {'mse_fraction': self.mse_fraction}


class TestLossFunctions:

    def test_objective_shapes_3d(self):
        y_a = K.variable(np.random.random((5, 6, 7)))
        y_b = K.variable(np.random.random((5, 6, 7)))
        for obj in allobj:
            objective_output = obj(y_a, y_b)
            assert K.eval(objective_output).shape == (5, 6)

    def test_objective_shapes_2d(self):
        y_a = K.variable(np.random.random((6, 7)))
        y_b = K.variable(np.random.random((6, 7)))
        for obj in allobj:
            objective_output = obj(y_a, y_b)
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
        assert np.isclose(K.eval(loss), 49.5, rtol=1e-3)

    def test_scalar_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        assert np.isclose(K.eval(loss), 113.85, rtol=1e-3)

    def test_sample_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        sample_weight = K.constant([1.2, 3.4], shape=(2, 1))
        loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.isclose(K.eval(loss), 767.8 / 6, rtol=1e-3)

    def test_timestep_weighted(self):
        mse_obj = losses.MeanSquaredError()
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3, 1))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3, 1))
        sample_weight = K.constant([3, 6, 5, 0, 4, 2], shape=(2, 3))
        loss = mse_obj(y_true, y_pred, sample_weight=sample_weight)
        assert np.isclose(K.eval(loss), 587 / 6, rtol=1e-3)

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
        assert np.allclose(K.eval(loss), [84.3333, 143.3666], rtol=1e-3)

    def test_sum_reduction(self):
        mse_obj = losses.MeanSquaredError(
            reduction=losses_utils.Reduction.SUM)
        y_true = K.constant([1, 9, 2, -5, -2, 6], shape=(2, 3))
        y_pred = K.constant([4, 8, 12, 8, 1, 3], shape=(2, 3))
        loss = mse_obj(y_true, y_pred, sample_weight=2.3)
        assert np.isclose(K.eval(loss), 227.69998, rtol=1e-3)


if __name__ == '__main__':
    pytest.main([__file__])
