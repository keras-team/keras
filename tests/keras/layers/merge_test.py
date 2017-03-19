import pytest
import numpy as np
from numpy.testing import assert_allclose
from keras import layers
from keras import models
from keras.utils.test_utils import layer_test
from keras.utils.test_utils import keras_test
from keras.layers import merge


@keras_test
def test_merge_add():
    i1 = layers.Input(shape=(4, 5))
    i2 = layers.Input(shape=(4, 5))
    i3 = layers.Input(shape=(4, 5))
    o = layers.add([i1, i2, i3])
    assert o._keras_shape == (None, 4, 5)
    model = models.Model([i1, i2, i3], o)

    add_layer = layers.Add()
    o2 = add_layer([i1, i2, i3])
    assert add_layer.output_shape == (None, 4, 5)

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    x3 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2, x3])
    assert out.shape == (2, 4, 5)
    assert_allclose(out, x1 + x2 + x3, atol=1e-4)


@keras_test
def test_merge_multiply():
    i1 = layers.Input(shape=(4, 5))
    i2 = layers.Input(shape=(4, 5))
    i3 = layers.Input(shape=(4, 5))
    o = layers.multiply([i1, i2, i3])
    assert o._keras_shape == (None, 4, 5)
    model = models.Model([i1, i2, i3], o)

    mul_layer = layers.Multiply()
    o2 = mul_layer([i1, i2, i3])
    assert mul_layer.output_shape == (None, 4, 5)

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    x3 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2, x3])
    assert out.shape == (2, 4, 5)
    assert_allclose(out, x1 * x2 * x3, atol=1e-4)


@keras_test
def test_merge_average():
    i1 = layers.Input(shape=(4, 5))
    i2 = layers.Input(shape=(4, 5))
    o = layers.average([i1, i2])
    assert o._keras_shape == (None, 4, 5)
    model = models.Model([i1, i2], o)

    avg_layer = layers.Average()
    o2 = avg_layer([i1, i2])
    assert avg_layer.output_shape == (None, 4, 5)

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2])
    assert out.shape == (2, 4, 5)
    assert_allclose(out, 0.5 * (x1 + x2), atol=1e-4)


@keras_test
def test_merge_maximum():
    i1 = layers.Input(shape=(4, 5))
    i2 = layers.Input(shape=(4, 5))
    o = layers.maximum([i1, i2])
    assert o._keras_shape == (None, 4, 5)
    model = models.Model([i1, i2], o)

    max_layer = layers.Maximum()
    o2 = max_layer([i1, i2])
    assert max_layer.output_shape == (None, 4, 5)

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2])
    assert out.shape == (2, 4, 5)
    assert_allclose(out, np.maximum(x1, x2), atol=1e-4)


@keras_test
def test_merge_concatenate():
    i1 = layers.Input(shape=(4, 5))
    i2 = layers.Input(shape=(4, 5))
    o = layers.concatenate([i1, i2], axis=1)
    assert o._keras_shape == (None, 8, 5)
    model = models.Model([i1, i2], o)

    concat_layer = layers.Concatenate(axis=1)
    o2 = concat_layer([i1, i2])
    assert concat_layer.output_shape == (None, 8, 5)

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2])
    assert out.shape == (2, 8, 5)
    assert_allclose(out, np.concatenate([x1, x2], axis=1), atol=1e-4)


@keras_test
def test_merge_dot():
    i1 = layers.Input(shape=(4,))
    i2 = layers.Input(shape=(4,))
    o = layers.dot([i1, i2], axes=1)
    assert o._keras_shape == (None, 1)
    model = models.Model([i1, i2], o)

    dot_layer = layers.Dot(axes=1)
    o2 = dot_layer([i1, i2])
    assert dot_layer.output_shape == (None, 1)

    x1 = np.random.random((2, 4))
    x2 = np.random.random((2, 4))
    out = model.predict([x1, x2])
    assert out.shape == (2, 1)
    expected = np.zeros((2, 1))
    expected[0, 0] = np.dot(x1[0], x2[0])
    expected[1, 0] = np.dot(x1[1], x2[1])
    assert_allclose(out, expected, atol=1e-4)

    # Test with negative tuple of axes.
    o = layers.dot([i1, i2], axes=(-1, -1))
    assert o._keras_shape == (None, 1)
    model = models.Model([i1, i2], o)
    out = model.predict([x1, x2])
    assert out.shape == (2, 1)
    assert_allclose(out, expected, atol=1e-4)


if __name__ == '__main__':
    pytest.main([__file__])
