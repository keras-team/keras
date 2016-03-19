import numpy as np
from ..engine import Model, Input
from ..models import Sequential


def get_test_data(nb_train=1000, nb_test=500, input_shape=(10,), output_shape=(2,),
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


def test_layer(layer, input_shape, input_dtype=None):
    '''Test routine for a layer with a single input tensor
    and single output tensor.
    '''
    if not input_dtype:
        input_dtype = K.float()
    input_data = (10 * np.random.random(input_shape)).astype(input_dtype)

    # test basic serialization
    layer_config = layer.get_config()
    layer = layer.__class__.from_config(layer_config)

    # test in functional API
    x = Input(shape=input_shape[1:], dtype=input_dtype)
    y = layer(x)
    model = Model(x, y)
    model.compile('rmsprop', 'mse')

    expected_output_shape = layer.get_output_shape_for(input_shape)
    actual_output_shape = model.predict(input_data).shape
    assert expected_output_shape == actual_output_shape

    # test serialization
    model_config = model.get_config()
    model = Model.from_config(model_config)
    model.compile('rmsprop', 'mse')

    # test whether container recursion works
    x_outer = Input(shape=input_shape[1:], dtype=input_dtype)
    y_outer = model(x_outer)
    outer_model = Model(x_outer, y_outer)
    outer_model.compile('rmsprop', 'mse')

    actual_output_shape = outer_model.predict(input_data).shape
    assert expected_output_shape == actual_output_shape

    outer_model_config = outer_model.get_config()
    outer_model = Model.from_config(outer_model_config)
    outer_model.compile('rmsprop', 'mse')

    # test as first layer in Sequential API
    layer_config = layer.get_config()
    layer_config['batch_input_shape'] = input_shape
    layer = layer.__class__.from_config(layer_config)

    model = Sequential()
    model.add(layer)
    model.compile('rmsprop', 'mse')
    actual_output_shape = model.predict(input_data).shape
    assert expected_output_shape == actual_output_shape
