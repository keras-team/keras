import pytest
import numpy as np

from keras import initializers
from keras import backend as K

# 2D tensor test fixture
FC_SHAPE = (200, 100)

# 4D convolution in th order. This shape has the same effective shape as FC_SHAPE
CONV_SHAPE = (25, 25, 20, 20)


def _runner(init, shape, target_mean=None, target_std=None,
            target_max=None, target_min=None):
    variable = K.variable(init(shape))
    output = K.get_value(variable)
    lim = 3e-2
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
    _runner(initializers.RandomUniform(minval=-1, maxval=1), tensor_shape,
            target_mean=0., target_max=1, target_min=-1)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_normal(tensor_shape):
    _runner(initializers.RandomNormal(mean=0, stddev=1), tensor_shape,
            target_mean=0., target_std=1)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_truncated_normal(tensor_shape):
    _runner(initializers.TruncatedNormal(mean=0, stddev=1), tensor_shape,
            target_mean=0., target_max=2, target_min=-2)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_constant(tensor_shape):
    _runner(initializers.Constant(2), tensor_shape,
            target_mean=2, target_max=2, target_min=2)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_lecun_uniform(tensor_shape):
    fan_in, _ = initializers._compute_fans(tensor_shape)
    std = np.sqrt(1. / fan_in)
    _runner(initializers.lecun_uniform(), tensor_shape,
            target_mean=0., target_std=std)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_glorot_uniform(tensor_shape):
    fan_in, fan_out = initializers._compute_fans(tensor_shape)
    std = np.sqrt(2. / (fan_in + fan_out))
    _runner(initializers.glorot_uniform(), tensor_shape,
            target_mean=0., target_std=std)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_he_uniform(tensor_shape):
    fan_in, _ = initializers._compute_fans(tensor_shape)
    std = np.sqrt(2. / fan_in)
    _runner(initializers.he_uniform(), tensor_shape,
            target_mean=0., target_std=std)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_lecun_normal(tensor_shape):
    fan_in, _ = initializers._compute_fans(tensor_shape)
    std = np.sqrt(1. / fan_in)
    _runner(initializers.lecun_normal(), tensor_shape,
            target_mean=0., target_std=std)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_glorot_normal(tensor_shape):
    fan_in, fan_out = initializers._compute_fans(tensor_shape)
    std = np.sqrt(2. / (fan_in + fan_out))
    _runner(initializers.glorot_normal(), tensor_shape,
            target_mean=0., target_std=std)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_he_normal(tensor_shape):
    fan_in, _ = initializers._compute_fans(tensor_shape)
    std = np.sqrt(2. / fan_in)
    _runner(initializers.he_normal(), tensor_shape,
            target_mean=0., target_std=std)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_orthogonal(tensor_shape):
    _runner(initializers.orthogonal(), tensor_shape,
            target_mean=0.)


@pytest.mark.parametrize('tensor_shape',
                         [(100, 100), (10, 20), (30, 80), (1, 2, 3, 4)],
                         ids=['FC', 'RNN', 'RNN_INVALID', 'CONV'])
def test_identity(tensor_shape):
    if len(tensor_shape) > 2 or max(tensor_shape) % min(tensor_shape) != 0:
        with pytest.raises(ValueError):
            _runner(initializers.identity(), tensor_shape,
                    target_mean=1. / tensor_shape[0], target_max=1.)
    else:
        _runner(initializers.identity(), tensor_shape,
                target_mean=1. / tensor_shape[0], target_max=1.)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_zero(tensor_shape):
    _runner(initializers.zeros(), tensor_shape,
            target_mean=0., target_max=0.)


@pytest.mark.parametrize('tensor_shape', [FC_SHAPE, CONV_SHAPE], ids=['FC', 'CONV'])
def test_one(tensor_shape):
    _runner(initializers.ones(), tensor_shape,
            target_mean=1., target_max=1.)


if __name__ == '__main__':
    pytest.main([__file__])
