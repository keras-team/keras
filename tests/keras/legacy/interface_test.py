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
def test_embedding_legacy_interface():
    old_layer = keras.layers.Embedding(4, 2, name='d')
    new_layer = keras.layers.Embedding(output_dim=2, input_dim=4, name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Embedding(input_dim=4, output_dim=2, name='d',
                                       init='normal',
                                       W_regularizer='l1',
                                       W_constraint='max_norm')
    new_layer = keras.layers.Embedding(input_dim=4, output_dim=2, name='d',
                                       embeddings_initializer='normal',
                                       embeddings_regularizer='l1',
                                       embeddings_constraint='max_norm')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Embedding(1, 1, dropout=0.0, name='d')
    new_layer = keras.layers.Embedding(1, 1, name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_maxpooling1d_legacy_interface():
    old_layer = keras.layers.MaxPool1D(pool_length=2,
                                       border_mode='valid',
                                       name='maxpool1d')
    new_layer = keras.layers.MaxPool1D(pool_size=2,
                                       padding='valid',
                                       name='maxpool1d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPool1D(2, padding='valid', name='maxpool1d')
    new_layer = keras.layers.MaxPool1D(pool_size=2,
                                       padding='valid',
                                       name='maxpool1d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_avgpooling1d_legacy_interface():
    old_layer = keras.layers.AvgPool1D(pool_length=2,
                                       border_mode='valid',
                                       name='d')
    new_layer = keras.layers.AvgPool1D(pool_size=2, padding='valid', name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AvgPool1D(2, padding='valid', name='d')
    new_layer = keras.layers.AvgPool1D(pool_size=2, padding='valid', name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_prelu_legacy_interface():
    old_layer = keras.layers.PReLU(init='zero', name='p')
    new_layer = keras.layers.PReLU('zero', name='p')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_gaussiannoise_legacy_interface():
    old_layer = keras.layers.GaussianNoise(sigma=0.5, name='gn')
    new_layer = keras.layers.GaussianNoise(stddev=0.5, name='gn')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_lstm_legacy_interface():
    old_layer = keras.layers.LSTM(input_shape=[3, 5], output_dim=2, name='d')
    new_layer = keras.layers.LSTM(2, input_shape=[3, 5], name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.LSTM(2, init='normal',
                                  inner_init='glorot_uniform',
                                  forget_bias_init='one',
                                  inner_activation='hard_sigmoid',
                                  W_regularizer='l1',
                                  U_regularizer='l1',
                                  b_regularizer='l1',
                                  dropout_W=0.1,
                                  dropout_U=0.1,
                                  name='LSTM')

    new_layer = keras.layers.LSTM(2, kernel_initializer='normal',
                                  recurrent_initializer='glorot_uniform',
                                  unit_forget_bias=True,
                                  recurrent_activation='hard_sigmoid',
                                  kernel_regularizer='l1',
                                  recurrent_regularizer='l1',
                                  bias_regularizer='l1',
                                  dropout=0.1,
                                  recurrent_dropout=0.1,
                                  name='LSTM')

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
                                  unit_forget_bias=True,
                                  recurrent_activation='hard_sigmoid',
                                  kernel_regularizer='l1',
                                  recurrent_regularizer='l1',
                                  bias_regularizer='l1',
                                  dropout=0.1,
                                  recurrent_dropout=0.1,
                                  name='LSTM')

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
def test_gaussiandropout_legacy_interface():
    old_layer = keras.layers.GaussianDropout(p=0.6, name='drop')
    new_layer_1 = keras.layers.GaussianDropout(rate=0.6, name='drop')
    new_layer_2 = keras.layers.GaussianDropout(0.6, name='drop')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer_1.get_config())
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer_2.get_config())


@keras_test
def test_maxpooling2d_legacy_interface():
    old_layer = keras.layers.MaxPooling2D(pool_size=(2, 2), border_mode='valid', name='maxpool2d')
    new_layer = keras.layers.MaxPool2D(pool_size=2, padding='valid', name='maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling2D((2, 2), 2, 'valid', name='maxpool2d')
    new_layer = keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid', name='maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling2D((2, 2), padding='valid', dim_ordering='tf', name='maxpool2d')
    new_layer = keras.layers.MaxPool2D(pool_size=2, padding='valid', data_format='channels_last', name='maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling2D((2, 2), padding='valid', dim_ordering='th', name='maxpool2d')
    new_layer = keras.layers.MaxPool2D(pool_size=2, padding='valid', data_format='channels_first', name='maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling2D((2, 2), padding='valid', dim_ordering='default', name='maxpool2d')
    new_layer = keras.layers.MaxPool2D(pool_size=2, padding='valid', name='maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_avgpooling2d_legacy_interface():
    old_layer = keras.layers.AveragePooling2D(pool_size=(2, 2), border_mode='valid', name='avgpooling2d')
    new_layer = keras.layers.AvgPool2D(pool_size=(2, 2), padding='valid', name='avgpooling2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling2D((2, 2), (2, 2), 'valid', name='avgpooling2d')
    new_layer = keras.layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='avgpooling2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling2D((2, 2), padding='valid', dim_ordering='tf', name='avgpooling2d')
    new_layer = keras.layers.AvgPool2D(pool_size=2, padding='valid', data_format='channels_last', name='avgpooling2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling2D((2, 2), padding='valid', dim_ordering='th', name='avgpooling2d')
    new_layer = keras.layers.AvgPool2D(pool_size=2, padding='valid', data_format='channels_first', name='avgpooling2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling2D((2, 2), padding='valid', dim_ordering='default', name='avgpooling2d')
    new_layer = keras.layers.AvgPool2D(pool_size=2, padding='valid', name='avgpooling2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_maxpooling3d_legacy_interface():
    old_layer = keras.layers.MaxPooling3D(pool_size=(2, 2, 2), border_mode='valid', name='maxpool3d')
    new_layer = keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='valid', name='maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling3D((2, 2, 2), (2, 2, 2), 'valid', name='maxpool3d')
    new_layer = keras.layers.MaxPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling3D((2, 2, 2), padding='valid', dim_ordering='tf', name='maxpool3d')
    new_layer = keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='valid', data_format='channels_last', name='maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling3D((2, 2, 2), padding='valid', dim_ordering='th', name='maxpool3d')
    new_layer = keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='valid', data_format='channels_first', name='maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling3D((2, 2, 2), padding='valid', dim_ordering='default', name='maxpool3d')
    new_layer = keras.layers.MaxPool3D(pool_size=(2, 2, 2), padding='valid', name='maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_avgpooling3d_legacy_interface():
    old_layer = keras.layers.AveragePooling3D(pool_size=(2, 2, 2), border_mode='valid', name='avgpooling3d')
    new_layer = keras.layers.AvgPool3D(pool_size=(2, 2, 2), padding='valid', name='avgpooling3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling3D((2, 2, 2), (2, 2, 2), 'valid', name='avgpooling3d')
    new_layer = keras.layers.AvgPool3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='avgpooling3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling3D((2, 2, 2), padding='valid', dim_ordering='tf', name='avgpooling3d')
    new_layer = keras.layers.AvgPool3D(pool_size=(2, 2, 2), padding='valid', data_format='channels_last', name='avgpooling3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling3D((2, 2, 2), padding='valid', dim_ordering='th', name='avgpooling3d')
    new_layer = keras.layers.AvgPool3D(pool_size=(2, 2, 2), padding='valid', data_format='channels_first', name='avgpooling3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling3D((2, 2, 2), padding='valid', dim_ordering='default', name='avgpooling3d')
    new_layer = keras.layers.AvgPool3D(pool_size=(2, 2, 2), padding='valid', name='avgpooling3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_global_maxpooling2d_legacy_interface():
    old_layer = keras.layers.GlobalMaxPooling2D(dim_ordering='tf', name='global_maxpool2d')
    new_layer = keras.layers.GlobalMaxPool2D(data_format='channels_last', name='global_maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalMaxPooling2D(dim_ordering='th', name='global_maxpool2d')
    new_layer = keras.layers.GlobalMaxPool2D(data_format='channels_first', name='global_maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalMaxPooling2D(dim_ordering='default', name='global_maxpool2d')
    new_layer = keras.layers.GlobalMaxPool2D(name='global_maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_global_avgpooling2d_legacy_interface():
    old_layer = keras.layers.GlobalAveragePooling2D(dim_ordering='tf', name='global_avgpool2d')
    new_layer = keras.layers.GlobalAvgPool2D(data_format='channels_last', name='global_avgpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalAveragePooling2D(dim_ordering='th', name='global_avgpool2d')
    new_layer = keras.layers.GlobalAvgPool2D(data_format='channels_first', name='global_avgpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalAveragePooling2D(dim_ordering='default', name='global_avgpool2d')
    new_layer = keras.layers.GlobalAvgPool2D(name='global_avgpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_global_maxpooling3d_legacy_interface():
    old_layer = keras.layers.GlobalMaxPooling3D(dim_ordering='tf', name='global_maxpool3d')
    new_layer = keras.layers.GlobalMaxPool3D(data_format='channels_last', name='global_maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalMaxPooling3D(dim_ordering='th', name='global_maxpool3d')
    new_layer = keras.layers.GlobalMaxPool3D(data_format='channels_first', name='global_maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalMaxPooling3D(dim_ordering='default', name='global_maxpool3d')
    new_layer = keras.layers.GlobalMaxPool3D(name='global_maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


@keras_test
def test_global_avgpooling3d_legacy_interface():
    old_layer = keras.layers.GlobalAveragePooling3D(dim_ordering='tf', name='global_avgpool3d')
    new_layer = keras.layers.GlobalAvgPool3D(data_format='channels_last', name='global_avgpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalAveragePooling3D(dim_ordering='th', name='global_avgpool3d')
    new_layer = keras.layers.GlobalAvgPool3D(data_format='channels_first', name='global_avgpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalAveragePooling3D(dim_ordering='default', name='global_avgpool3d')
    new_layer = keras.layers.GlobalAvgPool3D(name='global_avgpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


if __name__ == '__main__':
    pytest.main([__file__])
