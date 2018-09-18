import pytest
import json
import keras
import numpy as np


def test_dense_legacy_interface():
    old_layer = keras.layers.Dense(input_dim=3, output_dim=2, name='d')
    new_layer = keras.layers.Dense(2, input_shape=(3,), name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Dense(2, bias=False, init='normal',
                                   W_regularizer='l1',
                                   W_constraint='maxnorm', name='d')
    new_layer = keras.layers.Dense(2, use_bias=False,
                                   kernel_initializer='normal',
                                   kernel_regularizer='l1',
                                   kernel_constraint='max_norm', name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Dense(2, bias=True,
                                   b_regularizer='l1',
                                   b_constraint='maxnorm', name='d')
    new_layer = keras.layers.Dense(2, use_bias=True,
                                   bias_regularizer='l1',
                                   bias_constraint='max_norm', name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_dropout_legacy_interface():
    old_layer = keras.layers.Dropout(p=3, name='drop')
    new_layer1 = keras.layers.Dropout(rate=3, name='drop')
    new_layer2 = keras.layers.Dropout(3, name='drop')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer1.get_config())
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer2.get_config())


def test_embedding_legacy_interface():
    old_layer = keras.layers.Embedding(4, 2, name='d')
    new_layer = keras.layers.Embedding(output_dim=2, input_dim=4, name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Embedding(input_dim=4, output_dim=2, name='d',
                                       init='normal',
                                       W_regularizer='l1',
                                       W_constraint='maxnorm')
    new_layer = keras.layers.Embedding(input_dim=4, output_dim=2, name='d',
                                       embeddings_initializer='normal',
                                       embeddings_regularizer='l1',
                                       embeddings_constraint='max_norm')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Embedding(1, 1, dropout=0.0, name='d')
    new_layer = keras.layers.Embedding(1, 1, name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


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


def test_avgpooling1d_legacy_interface():
    old_layer = keras.layers.AvgPool1D(pool_length=2,
                                       border_mode='valid',
                                       name='d')
    new_layer = keras.layers.AvgPool1D(pool_size=2, padding='valid', name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AvgPool1D(2, padding='valid', name='d')
    new_layer = keras.layers.AvgPool1D(pool_size=2, padding='valid', name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_prelu_legacy_interface():
    old_layer = keras.layers.PReLU(init='zero', name='p')
    new_layer = keras.layers.PReLU('zero', name='p')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_gaussiannoise_legacy_interface():
    old_layer = keras.layers.GaussianNoise(sigma=0.5, name='gn')
    new_layer = keras.layers.GaussianNoise(stddev=0.5, name='gn')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_lstm_legacy_interface():
    old_layer = keras.layers.LSTM(input_shape=[3, 5], output_dim=2, name='d')
    new_layer = keras.layers.LSTM(2, input_shape=[3, 5], name='d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.LSTM(input_shape=[3, 5], output_dim=2, name='d',
                                  consume_less='mem')
    new_layer = keras.layers.LSTM(2, input_shape=[3, 5], name='d', implementation=1)
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.LSTM(input_dim=5, input_length=3,
                                  output_dim=2, name='d', consume_less='mem')
    new_layer = keras.layers.LSTM(2, input_shape=[3, 5], name='d', implementation=1)
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.LSTM(input_dim=5,
                                  output_dim=2, name='d', consume_less='mem')
    new_layer = keras.layers.LSTM(2, input_shape=[None, 5], name='d',
                                  implementation=1)
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.LSTM(input_shape=[3, 5], output_dim=2, name='d',
                                  consume_less='gpu')
    new_layer = keras.layers.LSTM(2, input_shape=[3, 5], name='d', implementation=2)
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


def test_gaussiandropout_legacy_interface():
    old_layer = keras.layers.GaussianDropout(p=0.6, name='drop')
    new_layer1 = keras.layers.GaussianDropout(rate=0.6, name='drop')
    new_layer2 = keras.layers.GaussianDropout(0.6, name='drop')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer1.get_config())
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer2.get_config())


def test_maxpooling2d_legacy_interface():
    old_layer = keras.layers.MaxPooling2D(
        pool_size=(2, 2), border_mode='valid', name='maxpool2d')
    new_layer = keras.layers.MaxPool2D(
        pool_size=2, padding='valid', name='maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling2D((2, 2), 2, 'valid', name='maxpool2d')
    new_layer = keras.layers.MaxPool2D(
        pool_size=2, strides=2, padding='valid', name='maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling2D(
        (2, 2), padding='valid', dim_ordering='tf', name='maxpool2d')
    new_layer = keras.layers.MaxPool2D(
        pool_size=2, padding='valid', data_format='channels_last', name='maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling2D(
        (2, 2), padding='valid', dim_ordering='th', name='maxpool2d')
    new_layer = keras.layers.MaxPool2D(
        pool_size=2, padding='valid', data_format='channels_first',
        name='maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling2D(
        (2, 2), padding='valid', dim_ordering='default', name='maxpool2d')
    new_layer = keras.layers.MaxPool2D(
        pool_size=2, padding='valid', name='maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_avgpooling2d_legacy_interface():
    old_layer = keras.layers.AveragePooling2D(
        pool_size=(2, 2), border_mode='valid', name='avgpooling2d')
    new_layer = keras.layers.AvgPool2D(
        pool_size=(2, 2), padding='valid', name='avgpooling2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling2D(
        (2, 2), (2, 2), 'valid', name='avgpooling2d')
    new_layer = keras.layers.AvgPool2D(
        pool_size=(2, 2), strides=(2, 2), padding='valid', name='avgpooling2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling2D(
        (2, 2), padding='valid', dim_ordering='tf', name='avgpooling2d')
    new_layer = keras.layers.AvgPool2D(
        pool_size=2, padding='valid', data_format='channels_last',
        name='avgpooling2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling2D(
        (2, 2), padding='valid', dim_ordering='th', name='avgpooling2d')
    new_layer = keras.layers.AvgPool2D(
        pool_size=2, padding='valid', data_format='channels_first',
        name='avgpooling2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling2D(
        (2, 2), padding='valid', dim_ordering='default', name='avgpooling2d')
    new_layer = keras.layers.AvgPool2D(
        pool_size=2, padding='valid', name='avgpooling2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_maxpooling3d_legacy_interface():
    old_layer = keras.layers.MaxPooling3D(
        pool_size=(2, 2, 2), border_mode='valid', name='maxpool3d')
    new_layer = keras.layers.MaxPool3D(
        pool_size=(2, 2, 2), padding='valid', name='maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling3D(
        (2, 2, 2), (2, 2, 2), 'valid', name='maxpool3d')
    new_layer = keras.layers.MaxPool3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid', name='maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling3D(
        (2, 2, 2), padding='valid', dim_ordering='tf', name='maxpool3d')
    new_layer = keras.layers.MaxPool3D(
        pool_size=(2, 2, 2), padding='valid', data_format='channels_last',
        name='maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling3D(
        (2, 2, 2), padding='valid', dim_ordering='th', name='maxpool3d')
    new_layer = keras.layers.MaxPool3D(
        pool_size=(2, 2, 2), padding='valid', data_format='channels_first',
        name='maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.MaxPooling3D(
        (2, 2, 2), padding='valid', dim_ordering='default', name='maxpool3d')
    new_layer = keras.layers.MaxPool3D(
        pool_size=(2, 2, 2), padding='valid', name='maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_avgpooling3d_legacy_interface():
    old_layer = keras.layers.AveragePooling3D(
        pool_size=(2, 2, 2), border_mode='valid', name='avgpooling3d')
    new_layer = keras.layers.AvgPool3D(
        pool_size=(2, 2, 2), padding='valid', name='avgpooling3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling3D(
        (2, 2, 2), (2, 2, 2), 'valid', name='avgpooling3d')
    new_layer = keras.layers.AvgPool3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding='valid',
        name='avgpooling3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling3D(
        (2, 2, 2), padding='valid', dim_ordering='tf', name='avgpooling3d')
    new_layer = keras.layers.AvgPool3D(
        pool_size=(2, 2, 2), padding='valid', data_format='channels_last',
        name='avgpooling3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling3D(
        (2, 2, 2), padding='valid', dim_ordering='th', name='avgpooling3d')
    new_layer = keras.layers.AvgPool3D(
        pool_size=(2, 2, 2), padding='valid', data_format='channels_first',
        name='avgpooling3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.AveragePooling3D(
        (2, 2, 2), padding='valid', dim_ordering='default', name='avgpooling3d')
    new_layer = keras.layers.AvgPool3D(
        pool_size=(2, 2, 2), padding='valid', name='avgpooling3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_global_maxpooling2d_legacy_interface():
    old_layer = keras.layers.GlobalMaxPooling2D(dim_ordering='tf',
                                                name='global_maxpool2d')
    new_layer = keras.layers.GlobalMaxPool2D(data_format='channels_last',
                                             name='global_maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalMaxPooling2D(dim_ordering='th',
                                                name='global_maxpool2d')
    new_layer = keras.layers.GlobalMaxPool2D(data_format='channels_first',
                                             name='global_maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalMaxPooling2D(dim_ordering='default',
                                                name='global_maxpool2d')
    new_layer = keras.layers.GlobalMaxPool2D(name='global_maxpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_global_avgpooling2d_legacy_interface():
    old_layer = keras.layers.GlobalAveragePooling2D(dim_ordering='tf',
                                                    name='global_avgpool2d')
    new_layer = keras.layers.GlobalAvgPool2D(data_format='channels_last',
                                             name='global_avgpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalAveragePooling2D(dim_ordering='th',
                                                    name='global_avgpool2d')
    new_layer = keras.layers.GlobalAvgPool2D(data_format='channels_first',
                                             name='global_avgpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalAveragePooling2D(dim_ordering='default',
                                                    name='global_avgpool2d')
    new_layer = keras.layers.GlobalAvgPool2D(name='global_avgpool2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_global_maxpooling3d_legacy_interface():
    old_layer = keras.layers.GlobalMaxPooling3D(dim_ordering='tf',
                                                name='global_maxpool3d')
    new_layer = keras.layers.GlobalMaxPool3D(data_format='channels_last',
                                             name='global_maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalMaxPooling3D(dim_ordering='th',
                                                name='global_maxpool3d')
    new_layer = keras.layers.GlobalMaxPool3D(data_format='channels_first',
                                             name='global_maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalMaxPooling3D(dim_ordering='default',
                                                name='global_maxpool3d')
    new_layer = keras.layers.GlobalMaxPool3D(name='global_maxpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_global_avgpooling3d_legacy_interface():
    old_layer = keras.layers.GlobalAveragePooling3D(dim_ordering='tf',
                                                    name='global_avgpool3d')
    new_layer = keras.layers.GlobalAvgPool3D(data_format='channels_last',
                                             name='global_avgpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalAveragePooling3D(dim_ordering='th',
                                                    name='global_avgpool3d')
    new_layer = keras.layers.GlobalAvgPool3D(data_format='channels_first',
                                             name='global_avgpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.GlobalAveragePooling3D(dim_ordering='default',
                                                    name='global_avgpool3d')
    new_layer = keras.layers.GlobalAvgPool3D(name='global_avgpool3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_upsampling1d_legacy_interface():
    old_layer = keras.layers.UpSampling1D(length=3, name='us1d')
    new_layer_1 = keras.layers.UpSampling1D(size=3, name='us1d')
    new_layer_2 = keras.layers.UpSampling1D(3, name='us1d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer_1.get_config())
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer_2.get_config())


def test_upsampling2d_legacy_interface():
    old_layer = keras.layers.UpSampling2D((2, 2), dim_ordering='tf', name='us2d')
    new_layer = keras.layers.UpSampling2D((2, 2), data_format='channels_last',
                                          name='us2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_upsampling3d_legacy_interface():
    old_layer = keras.layers.UpSampling3D((2, 2, 2),
                                          dim_ordering='tf',
                                          name='us3d')
    new_layer = keras.layers.UpSampling3D((2, 2, 2),
                                          data_format='channels_last',
                                          name='us3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_conv2d_legacy_interface():
    old_layer = keras.layers.Convolution2D(5, 3, 3, name='conv')
    new_layer = keras.layers.Conv2D(5, (3, 3), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Convolution2D(5, 3, nb_col=3, name='conv')
    new_layer = keras.layers.Conv2D(5, (3, 3), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Convolution2D(5, nb_row=3, nb_col=3, name='conv')
    new_layer = keras.layers.Conv2D(5, (3, 3), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Convolution2D(5, 3, 3,
                                           init='normal',
                                           subsample=(2, 2),
                                           border_mode='valid',
                                           dim_ordering='th',
                                           W_regularizer='l1',
                                           b_regularizer='l2',
                                           W_constraint='maxnorm',
                                           b_constraint='unitnorm',
                                           name='conv')
    new_layer = keras.layers.Conv2D(5, (3, 3),
                                    kernel_initializer='normal',
                                    strides=(2, 2),
                                    padding='valid',
                                    kernel_regularizer='l1',
                                    bias_regularizer='l2',
                                    kernel_constraint='max_norm',
                                    bias_constraint='unit_norm',
                                    data_format='channels_first',
                                    name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_deconv2d_legacy_interface():
    old_layer = keras.layers.Deconvolution2D(5, 3, 3, (6, 7, 5), name='deconv')
    new_layer = keras.layers.Conv2DTranspose(5, (3, 3), name='deconv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Deconvolution2D(5, 3, 3, output_shape=(6, 7, 5),
                                             name='deconv')
    new_layer = keras.layers.Conv2DTranspose(5, (3, 3), name='deconv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Deconvolution2D(5, 3, nb_col=3, output_shape=(6, 7, 5),
                                             name='deconv')
    new_layer = keras.layers.Conv2DTranspose(5, (3, 3), name='deconv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Deconvolution2D(5, nb_row=3, nb_col=3,
                                             output_shape=(6, 7, 5), name='deconv')
    new_layer = keras.layers.Conv2DTranspose(5, (3, 3), name='deconv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Deconvolution2D(5, 3, 3,
                                             output_shape=(6, 7, 5),
                                             init='normal',
                                             subsample=(2, 2),
                                             border_mode='valid',
                                             dim_ordering='th',
                                             W_regularizer='l1',
                                             b_regularizer='l2',
                                             W_constraint='maxnorm',
                                             b_constraint='unitnorm',
                                             name='conv')
    new_layer = keras.layers.Conv2DTranspose(
        5, (3, 3),
        kernel_initializer='normal',
        strides=(2, 2),
        padding='valid',
        kernel_regularizer='l1',
        bias_regularizer='l2',
        kernel_constraint='max_norm',
        bias_constraint='unit_norm',
        data_format='channels_first',
        name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_conv1d_legacy_interface():
    old_layer = keras.layers.Convolution1D(5,
                                           filter_length=3,
                                           input_dim=3,
                                           input_length=4,
                                           name='conv')
    new_layer = keras.layers.Conv1D(5, 3, name='conv', input_shape=(4, 3))
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Convolution1D(5, 3,
                                           init='normal',
                                           subsample_length=2,
                                           border_mode='valid',
                                           W_regularizer='l1',
                                           b_regularizer='l2',
                                           W_constraint='maxnorm',
                                           b_constraint='unitnorm',
                                           name='conv')
    new_layer = keras.layers.Conv1D(5, 3,
                                    kernel_initializer='normal',
                                    strides=2,
                                    padding='valid',
                                    kernel_regularizer='l1',
                                    bias_regularizer='l2',
                                    kernel_constraint='max_norm',
                                    bias_constraint='unit_norm',
                                    name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_separable_conv2d_legacy_interface():
    old_layer = keras.layers.SeparableConv2D(5, 3, 3, name='conv')
    new_layer = keras.layers.SeparableConv2D(5, (3, 3), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.SeparableConv2D(5, 3, nb_col=3, name='conv')
    new_layer = keras.layers.SeparableConv2D(5, (3, 3), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.SeparableConv2D(5, nb_row=3, nb_col=3, name='conv')
    new_layer = keras.layers.SeparableConv2D(5, (3, 3), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.SeparableConv2D(5, 3, 3,
                                             init='normal',
                                             subsample=(2, 2),
                                             border_mode='valid',
                                             dim_ordering='th',
                                             depthwise_regularizer='l1',
                                             b_regularizer='l2',
                                             depthwise_constraint='maxnorm',
                                             b_constraint='unitnorm',
                                             name='conv')
    new_layer = keras.layers.SeparableConv2D(5, (3, 3),
                                             depthwise_initializer='normal',
                                             pointwise_initializer='normal',
                                             strides=(2, 2),
                                             padding='valid',
                                             depthwise_regularizer='l1',
                                             bias_regularizer='l2',
                                             depthwise_constraint='max_norm',
                                             bias_constraint='unit_norm',
                                             data_format='channels_first',
                                             name='conv')
    old_config = json.dumps(old_layer.get_config())
    new_config = json.dumps(new_layer.get_config())
    assert old_config == new_config


def test_conv3d_legacy_interface():
    old_layer = keras.layers.Convolution3D(5, 3, 3, 4, name='conv')
    new_layer = keras.layers.Conv3D(5, (3, 3, 4), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Convolution3D(5, 3, 3, kernel_dim3=4, name='conv')
    new_layer = keras.layers.Conv3D(5, (3, 3, 4), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Convolution3D(5, 3,
                                           kernel_dim2=3,
                                           kernel_dim3=4,
                                           name='conv')
    new_layer = keras.layers.Conv3D(5, (3, 3, 4), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Convolution3D(5,
                                           kernel_dim1=3,
                                           kernel_dim2=3,
                                           kernel_dim3=4,
                                           name='conv')
    new_layer = keras.layers.Conv3D(5, (3, 3, 4), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.Convolution3D(5, 3, 3, 4,
                                           init='normal',
                                           subsample=(2, 2, 2),
                                           border_mode='valid',
                                           dim_ordering='th',
                                           W_regularizer='l1',
                                           b_regularizer='l2',
                                           W_constraint='maxnorm',
                                           b_constraint='unitnorm',
                                           name='conv')
    new_layer = keras.layers.Conv3D(5, (3, 3, 4),
                                    kernel_initializer='normal',
                                    strides=(2, 2, 2),
                                    padding='valid',
                                    kernel_regularizer='l1',
                                    bias_regularizer='l2',
                                    kernel_constraint='max_norm',
                                    bias_constraint='unit_norm',
                                    data_format='channels_first',
                                    name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_convlstm2d_legacy_interface():
    old_layer = keras.layers.ConvLSTM2D(5, 3, 3, name='conv')
    new_layer = keras.layers.ConvLSTM2D(5, (3, 3), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.ConvLSTM2D(5, 3, nb_col=3, name='conv')
    new_layer = keras.layers.ConvLSTM2D(5, (3, 3), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.ConvLSTM2D(5, nb_row=3, nb_col=3, name='conv')
    new_layer = keras.layers.ConvLSTM2D(5, (3, 3), name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.ConvLSTM2D(5, 3, 3,
                                        init='normal',
                                        inner_init='uniform',
                                        forget_bias_init='one',
                                        inner_activation='relu',
                                        subsample=(2, 2),
                                        border_mode='valid',
                                        dim_ordering='th',
                                        W_regularizer='l1',
                                        U_regularizer='l2',
                                        b_regularizer='l2',
                                        dropout_W=0.2,
                                        dropout_U=0.1,
                                        name='conv')
    new_layer = keras.layers.ConvLSTM2D(5, (3, 3),
                                        kernel_initializer='normal',
                                        recurrent_initializer='uniform',
                                        unit_forget_bias=True,
                                        recurrent_activation='relu',
                                        strides=(2, 2),
                                        padding='valid',
                                        kernel_regularizer='l1',
                                        recurrent_regularizer='l2',
                                        bias_regularizer='l2',
                                        data_format='channels_first',
                                        dropout=0.2,
                                        recurrent_dropout=0.1,
                                        name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_batchnorm_legacy_interface():
    old_layer = keras.layers.BatchNormalization(mode=0, name='bn')
    new_layer = keras.layers.BatchNormalization(name='bn')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())

    old_layer = keras.layers.BatchNormalization(mode=0,
                                                beta_init='one',
                                                gamma_init='uniform',
                                                name='bn')
    new_layer = keras.layers.BatchNormalization(beta_initializer='ones',
                                                gamma_initializer='uniform',
                                                name='bn')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_atrousconv1d_legacy_interface():
    old_layer = keras.layers.AtrousConvolution1D(5, 3,
                                                 init='normal',
                                                 subsample_length=2,
                                                 border_mode='valid',
                                                 W_regularizer='l1',
                                                 b_regularizer='l2',
                                                 W_constraint='maxnorm',
                                                 b_constraint='unitnorm',
                                                 atrous_rate=2,
                                                 name='conv')
    new_layer = keras.layers.Conv1D(5, 3,
                                    kernel_initializer='normal',
                                    strides=2,
                                    padding='valid',
                                    kernel_regularizer='l1',
                                    bias_regularizer='l2',
                                    kernel_constraint='max_norm',
                                    bias_constraint='unit_norm',
                                    dilation_rate=2,
                                    name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_atrousconv2d_legacy_interface():
    old_layer = keras.layers.AtrousConvolution2D(
        5, 3, 3,
        atrous_rate=(2, 2),
        init='normal',
        subsample=(2, 2),
        border_mode='valid',
        dim_ordering='th',
        W_regularizer='l1',
        b_regularizer='l2',
        W_constraint='maxnorm',
        b_constraint='unitnorm',
        name='conv')
    new_layer = keras.layers.Conv2D(5, (3, 3),
                                    kernel_initializer='normal',
                                    strides=(2, 2),
                                    padding='valid',
                                    kernel_regularizer='l1',
                                    bias_regularizer='l2',
                                    kernel_constraint='max_norm',
                                    bias_constraint='unit_norm',
                                    data_format='channels_first',
                                    dilation_rate=(2, 2),
                                    name='conv')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_zeropadding2d_legacy_interface():
    old_layer = keras.layers.ZeroPadding2D(padding={'right_pad': 4,
                                                    'bottom_pad': 2,
                                                    'top_pad': 1,
                                                    'left_pad': 3},
                                           dim_ordering='tf',
                                           name='zp2d')
    new_layer = keras.layers.ZeroPadding2D(((1, 2), (3, 4)),
                                           data_format='channels_last',
                                           name='zp2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_zeropadding3d_legacy_interface():
    old_layer = keras.layers.ZeroPadding3D((2, 2, 2),
                                           dim_ordering='tf',
                                           name='zp3d')
    new_layer = keras.layers.ZeroPadding3D((2, 2, 2),
                                           data_format='channels_last',
                                           name='zp3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_cropping2d_legacy_interface():
    old_layer = keras.layers.Cropping2D(dim_ordering='tf', name='c2d')
    new_layer = keras.layers.Cropping2D(data_format='channels_last', name='c2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_cropping3d_legacy_interface():
    old_layer = keras.layers.Cropping3D(dim_ordering='tf', name='c3d')
    new_layer = keras.layers.Cropping3D(data_format='channels_last', name='c3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer.get_config())


def test_generator_methods_interface():
    def train_generator():
        x = np.random.randn(2, 2)
        y = np.random.randint(0, 2, size=[2, 1])
        while True:
            yield (x, y)

    def val_generator():
        x = np.random.randn(2, 2)
        y = np.random.randint(0, 2, size=[2, 1])
        while True:
            yield (x, y)

    def pred_generator():
        x = np.random.randn(1, 2)
        while True:
            yield x

    x = keras.layers.Input(shape=(2, ))
    y = keras.layers.Dense(2)(x)

    model = keras.models.Model(inputs=x, outputs=y)
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit_generator(generator=train_generator(),
                        samples_per_epoch=1,
                        validation_data=val_generator(),
                        nb_val_samples=1,
                        nb_worker=1, pickle_safe=True, max_q_size=3)

    model.evaluate_generator(generator=train_generator(),
                             val_samples=2,
                             nb_worker=1, pickle_safe=False, max_q_size=3)
    model.predict_generator(generator=pred_generator(),
                            val_samples=2,
                            nb_worker=1, pickle_safe=False, max_q_size=3)


def test_spatialdropout1d_legacy_interface():
    old_layer = keras.layers.SpatialDropout1D(p=0.6, name='sd1d')
    new_layer_1 = keras.layers.SpatialDropout1D(rate=0.6, name='sd1d')
    new_layer_2 = keras.layers.SpatialDropout1D(0.6, name='sd1d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer_1.get_config())
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer_2.get_config())


def test_spatialdropout2d_legacy_interface():
    old_layer = keras.layers.SpatialDropout2D(p=0.5,
                                              dim_ordering='tf',
                                              name='sd2d')
    new_layer_1 = keras.layers.SpatialDropout2D(rate=0.5,
                                                data_format='channels_last',
                                                name='sd2d')
    new_layer_2 = keras.layers.SpatialDropout2D(0.5,
                                                data_format='channels_last',
                                                name='sd2d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer_1.get_config())
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer_2.get_config())


def test_spatialdropout3d_legacy_interface():
    old_layer = keras.layers.SpatialDropout3D(p=0.5,
                                              dim_ordering='tf',
                                              name='sd3d')
    new_layer_1 = keras.layers.SpatialDropout3D(rate=0.5,
                                                data_format='channels_last',
                                                name='sd3d')
    new_layer_2 = keras.layers.SpatialDropout3D(0.5,
                                                data_format='channels_last',
                                                name='sd3d')
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer_1.get_config())
    assert json.dumps(old_layer.get_config()) == json.dumps(new_layer_2.get_config())


def test_optimizer_get_updates_legacy_interface():
    for optimizer_cls in [keras.optimizers.RMSprop,
                          keras.optimizers.SGD,
                          keras.optimizers.Adadelta,
                          keras.optimizers.Adam,
                          keras.optimizers.Adagrad,
                          keras.optimizers.Nadam,
                          keras.optimizers.Adamax]:
        optimizer = optimizer_cls()
        param = keras.backend.variable(0.)
        loss = keras.backend.mean(param)
        constraints = {param: lambda x: x}
        params = [param]
        optimizer.get_updates(params, constraints, loss)
        optimizer.get_updates(params, constraints, loss=loss)
        optimizer.get_updates(loss, params)
        optimizer.get_updates(loss, params=params)
        optimizer.get_updates(loss=loss, params=params)


if __name__ == '__main__':
    pytest.main([__file__])
