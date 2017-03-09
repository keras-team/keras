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
def test_dropout_legacy_interface():
    old_layer = keras.layers.Dropout(p=3, name='drop')
    new_layer_1 = keras.layers.Dropout(rate=3, name='drop')
    new_layer_2 = keras.layers.Dropout(3, name='drop')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer_1.get_config())
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer_2.get_config())


@keras_test
def test_maxpooling1d_legacy_interface():
    old_layer = keras.layers.MaxPool1D(pool_length=2, border_mode='valid', name='maxpool1d')
    new_layer = keras.layers.MaxPool1D(pool_size=2, padding='valid', name='maxpool1d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPool1D(2, padding='valid', name='maxpool1d')
    new_layer = keras.layers.MaxPool1D(pool_size=2, padding='valid', name='maxpool1d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_simplernn_legacy_interface():
    old_layer = keras.layers.SimpleRNN(input_shape=[3, 5], output_dim=2, name='d')
    new_layer = keras.layers.SimpleRNN(2, input_shape=[3, 5], name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.SimpleRNN(2, init='normal',
                                       inner_init='glorot_uniform',
                                       W_regularizer='l1',
                                       U_regularizer='l1',
                                       b_regularizer='l1',
                                       dropout_W=0.1,
                                       dropout_U=0.1,
                                       name='SimpleRNN')
    new_layer = keras.layers.SimpleRNN(2, kernel_initializer='normal',
                                       recurrent_initializer='glorot_uniform',
                                       kernel_regularizer='l1',
                                       recurrent_regularizer='l1',
                                       bias_regularizer='l1',
                                       dropout=0.1,
                                       recurrent_dropout=0.1,
                                       name='SimpleRNN')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

@keras_test
def test_gru_legacy_interface():
    old_layer = keras.layers.GRU(input_shape=[3, 5], output_dim=2, name='d')
    new_layer = keras.layers.GRU(2, input_shape=[3, 5], name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GRU(2, init='normal',
                                 inner_init='glorot_uniform',
                                 inner_activation='hard_sigmoid',
                                 W_regularizer='l1',
                                 U_regularizer='l1',
                                 b_regularizer='l1',
                                 dropout_W=0.1,
                                 dropout_U=0.1,
                                 name='GRU')
    new_layer = keras.layers.GRU(2, kernel_initializer='normal',
                                 recurrent_initializer='glorot_uniform',
                                 recurrent_activation='hard_sigmoid',
                                 kernel_regularizer='l1',
                                 recurrent_regularizer='l1',
                                 bias_regularizer='l1',
                                 dropout=0.1,
                                 recurrent_dropout=0.1,
                                 name='GRU')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

@keras_test
def test_lstm_legacy_interface():
    old_layer = keras.layers.LSTM(input_shape=[3, 5], output_dim=2, name='d')
    new_layer = keras.layers.LSTM(2, input_shape=[3, 5], name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.LSTM(2, init='normal',
                                  inner_init='glorot_uniform',
                                  forget_bias_init='zero',
                                  inner_activation='hard_sigmoid',
                                  W_regularizer='l1',
                                  U_regularizer='l1',
                                  b_regularizer='l1',
                                  dropout_W=0.1,
                                  dropout_U=0.1,
                                  name='LSTM')
    new_layer = keras.layers.LSTM(2, kernel_initializer='normal',
                                  recurrent_initializer='glorot_uniform',
                                  bias_initializer='zero',
                                  recurrent_activation='hard_sigmoid',
                                  kernel_regularizer='l1',
                                  recurrent_regularizer='l1',
                                  bias_regularizer='l1',
                                  dropout=0.1,
                                  recurrent_dropout=0.1,
                                  name='LSTM')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

if __name__ == '__main__':
    pytest.main([__file__])
