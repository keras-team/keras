import pytest
import numpy as np

from keras import initializations
from keras import backend as K

# 2D tensor test fixture
FC_SHAPE = (100, 100)

# 4D convolution in th order. This shape has the same effective shape as FC_SHAPE
CONV_SHAPE = (25, 25, 2, 2)

# The equivalent shape of both test fixtures
SHAPE = (100, 100)

def _runner(init, shape, target_mean=None, target_std=None,
            target_max=None, target_min=None):
    variable = init(shape)
    output = K.get_value(variable)
    lim = 1e-2
    if target_std is not None:
        assert abs(output.std() - target_std) < lim
    if target_mean is not None:
        assert abs(output.mean() - target_mean) < lim
    if target_max is not None:
        assert abs(output.max() - target_max) < lim
    if target_min is not None:
        assert abs(output.min() - target_min) < lim


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_uniform(tensor_shape):
    _runner(initializations.uniform, tensor_shape, target_mean=0.,
            target_max=0.05, target_min=-0.05)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_normal(tensor_shape):
    _runner(initializations.normal, tensor_shape, target_mean=0., target_std=0.05)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_lecun_uniform(tensor_shape):
    scale = np.sqrt(3. / SHAPE[0])
    _runner(initializations.lecun_uniform, tensor_shape,
            target_mean=0., target_max=scale, target_min=-scale)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_glorot_uniform(tensor_shape):
    scale = np.sqrt(6. / (SHAPE[0] + SHAPE[1]))
    _runner(initializations.glorot_uniform, tensor_shape, target_mean=0.,
            target_max=scale, target_min=-scale)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_glorot_normal(tensor_shape):
    scale = np.sqrt(2. / (SHAPE[0] + SHAPE[1]))
    _runner(initializations.glorot_normal, tensor_shape,
            target_mean=0., target_std=scale)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_he_uniform(tensor_shape):
    scale = np.sqrt(6. / SHAPE[0])
    _runner(initializations.he_uniform, tensor_shape, target_mean=0.,
            target_max=scale, target_min=-scale)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_he_normal(tensor_shape):
    scale = np.sqrt(2. / SHAPE[0])
    _runner(initializations.he_normal, tensor_shape,
            target_mean=0., target_std=scale)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_orthogonal(tensor_shape):
    _runner(initializations.orthogonal, tensor_shape,
            target_mean=0.)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_identity(tensor_shape):
    if len(tensor_shape) > 2:
        with pytest.raises(Exception):
            _runner(initializations.identity, tensor_shape,
                    target_mean=1./SHAPE[0], target_max=1.)
    else:
        _runner(initializations.identity, tensor_shape,
                target_mean=1./SHAPE[0], target_max=1.)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_zero(tensor_shape):
    _runner(initializations.zero, tensor_shape,
            target_mean=0., target_max=0.)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_one(tensor_shape):
    _runner(initializations.one, tensor_shape,
            target_mean=1., target_max=1.)


if __name__ == '__main__':
    pytest.main([__file__])
