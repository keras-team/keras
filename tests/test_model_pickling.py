import pytest
import sys
import numpy as np
from numpy.testing import assert_allclose

import keras
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

if sys.version_info[0] == 3:
    import pickle
else:
    import cPickle as pickle


def test_sequential_model_pickling():
    model = keras.Sequential()
    model.add(layers.Dense(2, input_shape=(3,)))
    model.add(layers.RepeatVector(3))
    model.add(layers.TimeDistributed(layers.Dense(3)))
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


def test_sequential_model_pickling_custom_objects():
    # test with custom optimizer, loss
    class CustomSGD(optimizers.SGD):
        pass

    def custom_mse(*args, **kwargs):
        return losses.mse(*args, **kwargs)

    model = keras.Sequential()
    model.add(layers.Dense(2, input_shape=(3,)))
    model.add(layers.Dense(3))
    model.compile(loss=custom_mse, optimizer=CustomSGD(), metrics=['acc'])

    x = np.random.random((1, 3))
    y = np.random.random((1, 3))
    model.train_on_batch(x, y)

    out = model.predict(x)

    state = pickle.dumps(model)

    with keras.utils.CustomObjectScope(
            {'CustomSGD': CustomSGD, 'custom_mse': custom_mse}):
        model = pickle.loads(state)

    out2 = model.predict(x)
    assert_allclose(out, out2, atol=1e-05)


def test_functional_model_pickling():
    inputs = keras.Input(shape=(3,))
    x = layers.Dense(2)(inputs)
    outputs = layers.Dense(3)(x)

    model = keras.Model(inputs, outputs)
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
    inputs = keras.Input(shape=(5,))
    x = layers.Dense(5)(inputs)
    output1 = layers.Dense(1, name='output1')(x)
    output2 = layers.Dense(1, name='output2')(x)

    model = keras.Model(inputs=inputs, outputs=[output1, output2])

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
    model = keras.Sequential()
    model.add(layers.Dense(2, input_shape=(3,)))
    model.add(layers.Dense(3))

    model = pickle.loads(pickle.dumps(model))


def test_pickling_right_after_compilation():
    model = keras.Sequential()
    model.add(layers.Dense(2, input_shape=(3,)))
    model.add(layers.Dense(3))
    model.compile(loss='mse', optimizer='sgd', metrics=['acc'])
    model._make_train_function()

    model = pickle.loads(pickle.dumps(model))


if __name__ == '__main__':
    pytest.main([__file__])
