import pytest
import json
from keras.utils.test_utils import keras_test
import keras


@keras_test
def test_dense_legacy_interface():
    old_layer = keras.layers.Dense(input_dim=3, output_dim=2, name='d')
    new_layer = keras.layers.Dense(2, input_shape=(3,), name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Dense(2, bias=False, init='normal',
                                   W_regularizer='l1',
                                   W_constraint='max_norm', name='d')
    new_layer = keras.layers.Dense(2, use_bias=False,
                                   kernel_initializer='normal',
                                   kernel_regularizer='l1',
                                   kernel_constraint='max_norm', name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Dense(2, bias=True,
                                   b_regularizer='l1',
                                   b_constraint='max_norm', name='d')
    new_layer = keras.layers.Dense(2, use_bias=True,
                                   bias_regularizer='l1',
                                   bias_constraint='max_norm', name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_maxpool1d_legacy_interface():
    old_layer = keras.layers.MaxPool1D(pool_length=2, border_mode='valid', name='maxpool1d')
    new_layer = keras.layers.MaxPool1D(pool_size=2, padding='valid', name='maxpool1d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


if __name__ == '__main__':
    pytest.main([__file__])
