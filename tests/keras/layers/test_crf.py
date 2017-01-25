import pytest
import numpy as np
from numpy.testing import assert_allclose
from keras.utils.test_utils import keras_test
from keras.layers import Embedding, CRF
from keras.models import Sequential, model_from_json

nb_samples, timesteps, embedding_dim, output_dim = 2, 10, 4, 5
embedding_num = 12


@keras_test
def test_CRF():
    # data
    x = np.random.randint(1, embedding_num, nb_samples * timesteps).reshape((nb_samples, timesteps))
    x[0, -4:] = 0  # right padding
    x[1, :5] = 0  # left padding
    y = np.random.randint(0, output_dim, nb_samples * timesteps).reshape((nb_samples, timesteps))
    y_onehot = np.eye(output_dim)[y]
    y = np.expand_dims(y, 2)  # .astype('float32')

    # test with no masking, onehot, fix length
    model = Sequential()
    model.add(Embedding(embedding_num, embedding_dim, input_length=timesteps))
    crf = CRF(output_dim)
    model.add(crf)
    model.compile(optimizer='rmsprop', loss=crf.loss_function)
    model.fit(x, y_onehot, nb_epoch=1, batch_size=10)

    # test with masking, sparse target, dynamic length; test crf.viterbi_acc, crf.marginal_acc

    model = Sequential()
    model.add(Embedding(embedding_num, embedding_dim, mask_zero=True))
    crf = CRF(output_dim, sparse_target=True)
    model.add(crf)
    model.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.viterbi_acc, crf.marginal_acc])
    model.fit(x, y, nb_epoch=1, batch_size=10)

    # check mask
    y_pred = model.predict(x).argmax(-1)
    assert (y_pred[0, -4:] == 0).all()  # right padding
    assert (y_pred[1, :5] == 0).all()  # left padding

    # test `viterbi_acc
    _, v_acc, _ = model.evaluate(x, y)
    np_acc = (y_pred[x > 0] == y[:, :, 0][x > 0]).astype('float32').mean()
    assert np.abs(v_acc - np_acc) < 1e-4

    # test config
    model.get_config()
    model = model_from_json(model.to_json())

    # test marginal learn mode, fix length, unroll

    model = Sequential()
    model.add(Embedding(embedding_num, embedding_dim, input_length=timesteps, mask_zero=True))
    crf = CRF(output_dim, learn_mode='marginal', unroll=True, input_length=timesteps)
    model.add(crf)
    model.compile(optimizer='rmsprop', loss=crf.loss_function)
    model.fit(x, y_onehot, nb_epoch=1, batch_size=10)

    # check mask (marginal output)
    y_pred = model.predict(x)
    assert_allclose(y_pred[0, -4:], 1. / output_dim, atol=1e-6)
    assert_allclose(y_pred[1, :5], 1. / output_dim, atol=1e-6)

    # test marginal learn mode, but with Viterbi test_mode
    model = Sequential()
    model.add(Embedding(embedding_num, embedding_dim, input_length=timesteps, mask_zero=True))
    crf = CRF(output_dim, learn_mode='marginal', test_mode='viterbi', unroll=True, input_length=timesteps)
    model.add(crf)
    model.compile(optimizer='rmsprop', loss=crf.loss_function, metrics=[crf.accuracy])
    model.fit(x, y_onehot, nb_epoch=1, batch_size=10)

    y_pred = model.predict(x)

    # check y_pred is onehot vector (output from 'viterbi' test mode)
    assert_allclose(np.eye(output_dim)[y_pred.argmax(-1)], y_pred, atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__])
