import numpy as np
from numpy.testing import assert_allclose
import inspect

from ..engine import Model, Input
from ..models import Sequential, model_from_json
from .. import backend as K


def get_test_data(nb_train=1000, nb_test=500, input_shape=(10,),
                  output_shape=(2,),
                  classification=True, nb_class=2):
    '''
        classification=True overrides output_shape
        (i.e. output_shape is set to (1,)) and the output
        consists in integers in [0, nb_class-1].

        Otherwise: float output with shape output_shape.
    '''
    nb_sample = nb_train + nb_test
    if classification:
        y = np.random.randint(0, nb_class, size=(nb_sample,))
        X = np.zeros((nb_sample,) + input_shape)
        for i in range(nb_sample):
            X[i] = np.random.normal(loc=y[i], scale=0.7, size=input_shape)
    else:
        y_loc = np.random.random((nb_sample,))
        X = np.zeros((nb_sample,) + input_shape)
        y = np.zeros((nb_sample,) + output_shape)
        for i in range(nb_sample):
            X[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=input_shape)
            y[i] = np.random.normal(loc=y_loc[i], scale=0.7, size=output_shape)

    return (X[:nb_train], y[:nb_train]), (X[nb_train:], y[nb_train:])


def layer_test(layer_cls, kwargs={}, input_shape=None, input_dtype=None,
               input_data=None, expected_output=None, expected_output_dtype=None):
    '''Test routine for a layer with a single input tensor
    and single output tensor.
    '''
    if input_data is None:
        assert input_shape
        if not input_dtype:
            input_dtype = K.floatx()
        input_data = (10 * np.random.random(input_shape)).astype(input_dtype)
    elif input_shape is None:
        input_shape = input_data.shape

    if expected_output_dtype is None:
        expected_output_dtype = input_dtype

    # instantiation
    layer = layer_cls(**kwargs)

    # test get_weights , set_weights
    weights = layer.get_weights()
    layer.set_weights(weights)

    # test and instantiation from weights
    if 'weights' in inspect.getargspec(layer_cls.__init__):
        kwargs['weights'] = weights
        layer = layer_cls(**kwargs)

    # test in functional API
    x = Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)
    assert K.dtype(y) == expected_output_dtype

    model = Model(input=x, output=y)
    model.compile('rmsprop', 'mse')

    expected_output_shape = layer.get_output_shape_for(input_shape)
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    assert expected_output_shape == actual_output_shape
    if expected_output is not None:
        assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test serialization
    model_config = model.get_config()
    model = Model.from_config(model_config)
    model.compile('rmsprop', 'mse')

    # test as first layer in Sequential API
    layer_config = layer.get_config()
    layer_config['batch_input_shape'] = input_shape
    layer = layer.__class__.from_config(layer_config)

    model = Sequential()
    model.add(layer)
    model.compile('rmsprop', 'mse')
    actual_output = model.predict(input_data)
    actual_output_shape = actual_output.shape
    assert expected_output_shape == actual_output_shape
    if expected_output is not None:
        assert_allclose(actual_output, expected_output, rtol=1e-3)

    # test JSON serialization
    json_model = model.to_json()
    model = model_from_json(json_model)

    # for further checks in the caller function
    return actual_output
