import pytest
import os
import sys
import tempfile
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_raises

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Dense, Lambda, RepeatVector, TimeDistributed
from keras.layers import Input
from keras import optimizers
from keras import losses
from keras import metrics

if sys.version_info[0] == 3:
    import pickle
else:
    import cPickle as pickle


skipif_no_tf_gpu = pytest.mark.skipif(
    (K.backend() != 'tensorflow') or
    (not K.tensorflow_backend._get_available_gpus()),
    reason='Requires TensorFlow backend and a GPU')


def test_sequential_model_pickling():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.add(RepeatVector(3))
    model.add(TimeDistributed(Dense(3)))
    model.compile(loss=losses.MSE,
                  optimizer=optimizers.RMSprop(lr=0.0001),
                  metrics=[metrics.categorical_accuracy],
                  sample_weight_mode='temporal')
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)

    state = pickle.dumps(model)

    new_model = pickle.loads(state)

    out2 = new_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)

    # test that new updates are the same with both models
    x = np.random.random((1, 3))
    y = np.random.random((1, 3, 3))
    model.train_on_batch(x, y)
    new_model.train_on_batch(x, y)
    out = model.predict(x)
    out2 = new_model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


def test_sequential_model_pickling_2():
    # test with custom optimizer, loss
    custom_opt = optimizers.rmsprop
    custom_loss = losses.mse
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.add(Dense(3))
    model.compile(loss=custom_loss, optimizer=custom_opt(), metrics=['acc'])

    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)

    state = pickle.dumps(model)
    model = pickle.loads(state)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


def test_functional_model_pickling():
    inputs = Input(shape=(3,))
    x = Dense(2)(inputs)
    outputs = Dense(3)(x)

    model = Model(inputs, outputs)
    model.compile(loss=losses.MSE,
                  optimizer=optimizers.Adam(),
                  metrics=[metrics.categorical_accuracy])
    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)
    state = pickle.dumps(model)

    model = pickle.loads(state)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


def test_pickling_multiple_metrics_outputs():
    inputs = Input(shape=(5,))
    x = Dense(5)(inputs)
    output1 = Dense(1, name='output1')(x)
    output2 = Dense(1, name='output2')(x)

    model = Model(inputs=inputs, outputs=[output1, output2])

    metrics = {'output1': ['mse', 'binary_accuracy'],
               'output2': ['mse', 'binary_accuracy']
               }
    loss = {'output1': 'mse', 'output2': 'mse'}

    model.compile(loss=loss, optimizer='sgd', metrics=metrics)

    # assure that model is working
    x = np.array([[1, 1, 1, 1, 1]])
    out = model.predict(x)

    model = pickle.loads(pickle.dumps(model))

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


def test_pickling_without_compilation():
    """Test pickling model without compiling.
    """
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.add(Dense(3))

    model = pickle.loads(pickle.dumps(model))


def test_pickling_right_after_compilation():
    model = Sequential()
    model.add(Dense(2, input_shape=(3,)))
    model.add(Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
    model._make_train_function()

    model = pickle.loads(pickle.dumps(model))


def test_pickling_unused_layers_is_ok():
    a = Input(shape=(256, 512, 6))
    b = Input(shape=(256, 512, 1))
    c = Lambda(lambda x: x[:, :, :, :1])(a)

    model = Model(inputs=[a, b], outputs=c)

    model = pickle.loads(pickle.dumps(model))


if __name__ == '__main__':
    pytest.main([__file__])
