import numpy as np
import pytest

from keras.models import Sequential
from keras.layers.core import Masking
from keras.layers import Dense
from keras.layers.wrappers import TimeDistributed
from keras.utils.test_utils import keras_test
from keras.engine.training import masked_metric
from keras import metrics
from keras import backend as K

import warnings


@keras_test
def test_masking():
    np.random.seed(1000)
    X = np.array([[[1], [1]],
                  [[0], [0]]])
    model = Sequential()
    model.add(Masking(mask_value=0, input_shape=(2, 1)))
    model.add(TimeDistributed(Dense(1, init='one')))
    model.compile(loss='mse', optimizer='sgd', metrics=['mse'])
    y = np.array([[[1], [1]],
                  [[1], [1]]])
    metric = model.evaluate(X, y)[1]
    assert metric == 0


@keras_test
def test_sparse_categorical_crossentropy():
    _masking_function_test('sparse_categorical_crossentropy', output_shape=(3, 4, 5), target_shape=(3, 4, 1), target_type=int)


@keras_test
def test_categorical_accuracy():
    _masking_function_test('categorical_accuracy', target_type=float)


@keras_test
def test_sparse_categorical_accuracy():
    _masking_function_test('sparse_categorical_accuracy', output_shape=(3, 4, 5), target_shape=(3, 4, 1), target_type=int)


@keras_test
def top_k_categorical_accuracy():
    _masking_function_test('top_k_categorical_accuracy', output_shape=(3, 4, 5), target_shape=(3, 4, 1), target_type=int)


@keras_test
def test_categorical_crossentropy():
    _masking_function_test('categorical_crossentropy', target_type=int)


@keras_test
def test_binary_crossentropy():
    _masking_function_test('binary_crossentropy', target_type=float)


@keras_test
def test_masking_binary_accuracy():
    _masking_function_test('binary_accuracy', target_type=int)


@keras_test
def test_masking_mse():
    _masking_function_test('mse', target_type=float)


@keras_test
def test_masking_mae():
    _masking_function_test('mae', target_type=float)


@keras_test
def test_masking_mape():
    _masking_function_test('mape', target_type=float)


@keras_test
def test_masking_msle():
    _masking_function_test('msle', target_type=float)


@keras_test
def test_masking_hinge():
    _masking_function_test('hinge', target_type=float)


@keras_test
def test_masking_squared_hinge():
    _masking_function_test('squared_hinge', target_type=float)


@keras_test
def test_masking_poisson():
    _masking_function_test('poisson', target_type=float)


@keras_test
def test_masking_cosine_proximity():
    _masking_function_test('cosine_proximity', target_type=float)


@keras_test
def test_precision():
    _masking_function_test('precision', output_shape=(1, 4, 5), target_shape=(1, 4, 5), target_type=bool)


@keras_test
def test_recall():
    _masking_function_test('recall', output_shape=(1, 4, 5), target_shape=(1, 4, 5), target_type=bool)


@keras_test
def test_fmeasure():
    _masking_function_test('fmeasure', output_shape=(1, 4, 5), target_shape=(1, 4, 5), target_type=bool)


@keras_test
def test_fbeta_score():
    _masking_function_test('fbeta_score', output_shape=(1, 4, 5), target_shape=(1, 4, 5), target_type=bool)


def _masking_function_test(fn_name, output_shape=(3, 4, 5), target_shape=(3, 4, 5), target_type=float):
    # test if masked_metric and original metric
    # give the same result
    fn = metrics.get(fn_name)
    fn_masked = masked_metric(fn)
    X1 = np.random.ranf(np.prod(output_shape)).reshape(output_shape)
    padding_shape = list(output_shape)
    padding_shape[1] = 1
    padding_shape = tuple(padding_shape)
    X2 = np.concatenate((X1, np.random.ranf(np.prod(padding_shape)).reshape(padding_shape)), axis=1)

    target_padding_shape = list(target_shape)
    target_padding_shape[1] = 1
    target_padding_shape = tuple(target_padding_shape)

    if target_type == int:
        Y1 = np.random.randint(0, target_shape[-1], np.prod(target_shape)).reshape(target_shape)
        Y2 = np.concatenate((Y1, np.random.randint(0, target_shape[-1], np.prod(target_padding_shape)).reshape(target_padding_shape)), axis=1)

    elif target_type == float:
        Y1 = np.random.random(np.prod(target_shape)).reshape(target_shape)
        Y2 = np.concatenate((Y1, np.random.ranf(np.prod(target_padding_shape)).reshape(target_padding_shape)), axis=1)

    elif target_type == bool:
        Y1 = np.random.randint(0, 2, np.prod(target_shape)).reshape(target_shape)
        Y2 = np.concatenate((Y1, np.random.randint(0, 2, np.prod(target_padding_shape)).reshape(target_padding_shape)), axis=1)

    mask = np.ones((target_shape[0], target_shape[1] + 1))
    mask[:, -1] = 0

    mse_X1_Y1 = K.eval(fn(K.variable(Y1), K.variable(X1)))

    mse_X1_Y1_mask_fn = K.eval(fn_masked(K.variable(Y1), K.variable(X1)))

    mse_X2_Y2 = K.eval(fn(K.variable(Y2), K.variable(X2)))

    mse_X2_Y2_mask_fn = K.eval(fn_masked(K.variable(Y2), K.variable(X2)))

    mse_X2_Y2_masked = K.eval(fn_masked(K.variable(Y2), K.variable(X2), K.variable(mask)))

    # without mask metric output should be independent of which function is used
    # use almost equal to account for float precision
    assert abs(mse_X1_Y1 - mse_X1_Y1_mask_fn) < 0.0001, "Masked value not computed correctly for metric %s" % fn
    assert abs(mse_X2_Y2 - mse_X2_Y2_mask_fn) < 0.0001, "Masked value not computed correctly for metric %s" % fn

    # masked mse X2-Y2 should be equal to mse X1-Y1
    # use almost equal to account for float precision
    assert abs(mse_X1_Y1 - mse_X2_Y2_masked) < 0.0001, "Masked value not computed correctly for metric %s" % fn

    assert abs(mse_X2_Y2 - mse_X1_Y1) > 0.0001, "Metric %s not sufficiently tested as masking did not result in different metric value" % fn_name


if __name__ == '__main__':
    np.random.seed(1989)
    pytest.main([__file__])
