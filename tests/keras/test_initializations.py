import pytest
import numpy as np

from keras import initializations
from keras import backend as K

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


def test_uniform():
    _runner(initializations.uniform, SHAPE, target_mean=0.,
            target_max=0.05, target_min=-0.05)


def test_normal():
    _runner(initializations.normal, SHAPE, target_mean=0., target_std=0.05)


def test_lecun_uniform():
    scale = np.sqrt(3. / SHAPE[0])
    _runner(initializations.lecun_uniform, SHAPE,
            target_mean=0., target_max=scale, target_min=-scale)


def test_glorot_uniform():
    scale = np.sqrt(6. / (SHAPE[0] + SHAPE[1]))
    _runner(initializations.glorot_uniform, SHAPE, target_mean=0.,
            target_max=scale, target_min=-scale)


def test_glorot_normal():
    scale = np.sqrt(2. / (SHAPE[0] + SHAPE[1]))
    _runner(initializations.glorot_normal, SHAPE,
            target_mean=0., target_std=scale)


def test_he_uniform():
    scale = np.sqrt(6. / SHAPE[0])
    _runner(initializations.he_uniform, SHAPE, target_mean=0.,
            target_max=scale, target_min=-scale)


def test_he_normal():
    scale = np.sqrt(2. / SHAPE[0])
    _runner(initializations.he_normal, SHAPE,
            target_mean=0., target_std=scale)


def test_orthogonal():
    _runner(initializations.orthogonal, SHAPE,
            target_mean=0.)


def test_identity():
    _runner(initializations.identity, SHAPE,
            target_mean=1./SHAPE[0], target_max=1.)


def test_zero():
    _runner(initializations.zero, SHAPE,
            target_mean=0., target_max=0.)


def test_one():
    _runner(initializations.one, SHAPE,
            target_mean=1., target_max=1.)


if __name__ == '__main__':
    pytest.main([__file__])
