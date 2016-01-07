import pytest
import numpy as np
from keras import backend as K
from keras.layers import core
from keras.layers import noise

input_shape = (10, 10)
batch_input_shape = (10, 10, 10)


def test_GaussianNoise():
    layer = noise.GaussianNoise(sigma=1., input_shape=input_shape)
    _runner(layer)


def test_GaussianDropout():
    layer = noise.GaussianDropout(p=0.2, input_shape=input_shape)
    _runner(layer)


def _runner(layer):
    assert isinstance(layer, core.Layer)
    layer.build()
    conf = layer.get_config()
    assert (type(conf) == dict)

    param = layer.get_params()
    # Typically a list or a tuple, but may be any iterable
    assert hasattr(param, '__iter__')
    layer.input = K.variable(np.random.random(batch_input_shape))
    output = layer.get_output(train=False)
    output_np = K.eval(output)
    assert output_np.shape == batch_input_shape

    output = layer.get_output(train=True)
    output_np = K.eval(output)
    assert output_np.shape == batch_input_shape


if __name__ == '__main__':
    pytest.main([__file__])
