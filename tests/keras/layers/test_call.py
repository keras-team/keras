"""Test keras.layers.core.Layer.__call__"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from keras import backend as K
from keras.layers.core import Dense
from keras.models import Sequential, Graph


def test_layer_call():
    """Test keras.layers.core.Layer.__call__"""
    nb_samples, input_dim, output_dim = 3, 10, 5
    layer = Dense(output_dim, input_dim=input_dim)
    W = np.asarray(K.eval(layer.W)).astype(K.floatx())
    X = K.placeholder(ndim=2)
    Y = layer(X)
    f = K.function([X], [Y])

    x = np.ones((nb_samples, input_dim)).astype(K.floatx())
    y = f([x])[0].astype(K.floatx())
    t = np.dot(x, W).astype(K.floatx())
    assert_allclose(t, y, rtol=.2)


def test_sequential_call():
    """Test keras.models.Sequential.__call__"""
    nb_samples, input_dim, output_dim = 3, 10, 5
    model = Sequential()
    model.add(Dense(output_dim=output_dim, input_dim=input_dim))
    model.compile('sgd', 'mse')

    # test flat model
    X = K.placeholder(ndim=2)
    Y = model(X)
    f = K.function([X], [Y])

    x = np.ones((nb_samples, input_dim)).astype(K.floatx())
    y1 = f([x])[0].astype(K.floatx())
    y2 = model.predict(x)
    # results of __call__ should match model.predict
    assert_allclose(y1, y2)

    # test nested model
    model2 = Sequential()
    model2.add(model)
    model2.compile('sgd', 'mse')

    Y2 = model2(X)
    f = K.function([X], [Y2])

    y1 = f([x])[0].astype(K.floatx())
    y2 = model2.predict(x)
    # results of __call__ should match model.predict
    assert_allclose(y1, y2)


def test_graph_call():
    """Test keras.models.Graph.__call__"""
    nb_samples, input_dim, output_dim = 3, 10, 5
    model = Graph()
    model.add_input('input', input_shape=(input_dim, ))
    model.add_node(Dense(output_dim=output_dim, input_dim=input_dim),
                   input='input', name='output', create_output=True)

    model.compile('sgd', {'output': 'mse'})

    # test flat model
    X = K.placeholder(ndim=2)
    Y = model(X)
    f = K.function([X], [Y])

    x = np.ones((nb_samples, input_dim)).astype(K.floatx())
    y1 = f([x])[0].astype(K.floatx())
    y2 = model.predict({'input': x})['output']
    # results of __call__ should match model.predict
    assert_allclose(y1, y2)

    # test nested Graph models
    model2 = Graph()
    model2.add_input('input', input_shape=(input_dim, ))
    model2.add_node(model, input='input', name='output', create_output=True)
    # need to turn off cache because we're reusing model
    model2.cache_enabled = False
    model2.compile('sgd', {'output': 'mse'})

    Y2 = model2(X)
    f = K.function([X], [Y2])

    y1 = f([x])[0].astype(K.floatx())
    y2 = model2.predict({'input': x})['output']
    # results of __call__ should match model.predict
    assert_allclose(y1, y2)


def test_graph_multiple_in_out_call():
    """Test keras.models.Graph.__call__ with multiple inputs"""
    nb_samples, input_dim, output_dim = 3, 10, 5
    model = Graph()
    model.add_input('input1', input_shape=(input_dim, ))
    model.add_input('input2', input_shape=(input_dim, ))
    model.add_node(Dense(output_dim=output_dim, input_dim=input_dim),
                   inputs=['input1', 'input2'], merge_mode='sum', name='output', create_output=True)

    model.compile('sgd', {'output': 'mse'})

    # test flat model
    X1 = K.placeholder(ndim=2)
    X2 = K.placeholder(ndim=2)
    Y = model({'input1': X1, 'input2': X2})['output']
    f = K.function([X1, X2], [Y])

    x1 = np.ones((nb_samples, input_dim)).astype(K.floatx())
    x2 = np.ones((nb_samples, input_dim)).astype(K.floatx()) * -2
    y1 = f([x1, x2])[0].astype(K.floatx())
    y2 = model.predict({'input1': x1, 'input2': x2})['output']
    # results of __call__ should match model.predict
    assert_allclose(y1, y2)

    # test with single input, multiple outputs
    model2 = Graph()
    model2.add_input('input', input_shape=(input_dim, ))
    model2.add_node(Dense(output_dim=output_dim, input_dim=input_dim),
                    input='input', name='output1', create_output=True)
    model2.add_node(Dense(output_dim=output_dim, input_dim=input_dim),
                    input='input', name='output2', create_output=True)

    model2.compile('sgd', {'output1': 'mse', 'output2': 'mse'})

    # test flat model
    X = K.placeholder(ndim=2)
    Y = model2(X)
    f = K.function([X], [Y['output1'], Y['output2']])

    x = np.ones((nb_samples, input_dim)).astype(K.floatx())
    out = f([x])
    y1a = out[0].astype(K.floatx())
    y1b = out[1].astype(K.floatx())
    y2 = model2.predict({'input': x})
    # results of __call__ should match model.predict
    assert_allclose(y1a, y2['output1'])
    assert_allclose(y1b, y2['output2'])

    # test with multiple inputs, multiple outputs
    model3 = Graph()
    model3.add_input('input1', input_shape=(input_dim, ))
    model3.add_input('input2', input_shape=(input_dim, ))
    model3.add_shared_node(Dense(output_dim=output_dim, input_dim=input_dim),
                           inputs=['input1', 'input2'], name='output',
                           outputs=['output1', 'output2'], create_output=True)
    model3.compile('sgd', {'output1': 'mse', 'output2': 'mse'})

    # test flat model
    Y = model3({'input1': X1, 'input2': X2})
    f = K.function([X1, X2], [Y['output1'], Y['output2']])

    x1 = np.ones((nb_samples, input_dim)).astype(K.floatx())
    x2 = np.ones((nb_samples, input_dim)).astype(K.floatx()) * -2
    out = f([x1, x2])
    y1a = out[0].astype(K.floatx())
    y1b = out[1].astype(K.floatx())
    y2 = model3.predict({'input1': x1, 'input2': x2})
    # results of __call__ should match model.predict
    assert_allclose(y1a, y2['output1'])
    assert_allclose(y1b, y2['output2'])


def test_nested_call():
    """Test nested Sequential and Graph models"""
    nb_samples, input_dim, output_dim = 3, 10, 5
    X = K.placeholder(ndim=2)
    x = np.ones((nb_samples, input_dim)).astype(K.floatx())

    # test Graph model nested inside Sequential model
    model = Graph()
    model.add_input('input', input_shape=(input_dim, ))
    model.add_node(Dense(output_dim=output_dim, input_dim=input_dim),
                   input='input', name='output', create_output=True)

    model2 = Sequential()
    model2.add(model)
    model2.compile('sgd', 'mse')

    Y2 = model2(X)
    f = K.function([X], [Y2])

    y1 = f([x])[0].astype(K.floatx())
    y2 = model2.predict(x)
    # results of __call__ should match model.predict
    assert_allclose(y1, y2)

    # test Sequential model inside Graph model
    model3 = Sequential()
    model3.add(Dense(output_dim=output_dim, input_dim=input_dim))

    model4 = Graph()
    model4.add_input('input', input_shape=(input_dim, ))
    model4.add_node(model3, input='input', name='output', create_output=True)
    model4.compile('sgd', {'output': 'mse'})

    Y2 = model4(X)
    f = K.function([X], [Y2])

    y1 = f([x])[0].astype(K.floatx())
    y2 = model4.predict({'input': x})['output']
    # results of __call__ should match model.predict
    assert_allclose(y1, y2)


if __name__ == '__main__':
    pytest.main([__file__])
