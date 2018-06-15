import pytest
import numpy as np
from numpy.testing import assert_allclose
from keras import layers
from keras import models
from keras import backend as K
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

    assert add_layer.compute_mask([i1, i2, i3], [None, None, None]) is None
    assert np.all(K.eval(add_layer.compute_mask(
        [i1, i2, i3], [K.variable(x1), K.variable(x2), K.variable(x3)])))

    # Test invalid use case
    with pytest.raises(ValueError):
        add_layer.compute_mask([i1, i2, i3], x1)
    with pytest.raises(ValueError):
        add_layer.compute_mask(i1, [None, None, None])
    with pytest.raises(ValueError):
        add_layer.compute_mask([i1, i2, i3], [None, None])


@keras_test
def test_merge_subtract():
    i1 = layers.Input(shape=(4, 5))
    i2 = layers.Input(shape=(4, 5))
    i3 = layers.Input(shape=(4, 5))
    i4 = layers.Input(shape=(3, 5))
    o = layers.subtract([i1, i2])
    assert o._keras_shape == (None, 4, 5)
    model = models.Model([i1, i2], o)

    subtract_layer = layers.Subtract()
    o2 = subtract_layer([i1, i2])
    assert subtract_layer.output_shape == (None, 4, 5)

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2])
    assert out.shape == (2, 4, 5)
    assert_allclose(out, x1 - x2, atol=1e-4)

    assert subtract_layer.compute_mask([i1, i2], [None, None]) is None
    assert np.all(K.eval(subtract_layer.compute_mask(
        [i1, i2], [K.variable(x1), K.variable(x2)])))

    # Test invalid use case
    with pytest.raises(ValueError):
        subtract_layer.compute_mask([i1, i2], x1)
    with pytest.raises(ValueError):
        subtract_layer.compute_mask(i1, [None, None])
    with pytest.raises(ValueError):
        subtract_layer([i1, i2, i3])
    with pytest.raises(ValueError):
        subtract_layer([i1])


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
def test_merge_minimum():
    i1 = layers.Input(shape=(4, 5))
    i2 = layers.Input(shape=(4, 5))
    o = layers.minimum([i1, i2])
    assert o._keras_shape == (None, 4, 5)
    model = models.Model([i1, i2], o)

    max_layer = layers.Minimum()
    o2 = max_layer([i1, i2])
    assert max_layer.output_shape == (None, 4, 5)

    x1 = np.random.random((2, 4, 5))
    x2 = np.random.random((2, 4, 5))
    out = model.predict([x1, x2])
    assert out.shape == (2, 4, 5)
    assert_allclose(out, np.minimum(x1, x2), atol=1e-4)


@keras_test
def test_merge_concatenate():
    i1 = layers.Input(shape=(None, 5))
    i2 = layers.Input(shape=(None, 5))
    o = layers.concatenate([i1, i2], axis=1)
    assert o._keras_shape == (None, None, 5)
    model = models.Model([i1, i2], o)

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

    x3 = np.random.random((1, 1, 1))
    nb_layers = 4
    x_i = layers.Input(shape=(None, None))
    x_list = [x_i]
    x = x_i
    for i in range(nb_layers):
        x_list.append(x)
        x = layers.concatenate(x_list, axis=1)
    concat_model = models.Model(x_i, x)
    concat_out = concat_model.predict([x3])
    x3 = np.repeat(x3, 16, axis=1)
    assert concat_out.shape == (1, 16, 1)
    assert_allclose(concat_out, x3)

    assert concat_layer.compute_mask([i1, i2], [None, None]) is None
    assert np.all(K.eval(concat_layer.compute_mask(
        [i1, i2], [K.variable(x1), K.variable(x2)])).reshape(-1))

    # Test invalid use case
    with pytest.raises(ValueError):
        concat_layer.compute_mask([i1, i2], x1)
    with pytest.raises(ValueError):
        concat_layer.compute_mask(i1, [None, None])
    with pytest.raises(ValueError):
        concat_layer.compute_mask([i1, i2], [None])
    with pytest.raises(ValueError):
        concat_layer([i1])


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


@keras_test
def test_merge_broadcast():
    # shapes provided
    i1 = layers.Input(shape=(4, 5))
    i2 = layers.Input(shape=(5,))
    ops = [layers.add, layers.maximum]
    for op in ops:
        o = op([i1, i2])
        assert o._keras_shape == (None, 4, 5)
        model = models.Model([i1, i2], o)

        x1 = np.random.random((2, 4, 5))
        x2 = np.random.random((2, 5))
        out = model.predict([x1, x2])
        assert out.shape == (2, 4, 5)

    # shapes not provided
    i1 = layers.Input(shape=(None, None))
    i2 = layers.Input(shape=(None,))
    ops = [layers.add, layers.maximum]
    for op in ops:
        o = op([i1, i2])
        assert o._keras_shape == (None, None, None)
        model = models.Model([i1, i2], o)

        x1 = np.random.random((2, 4, 5))
        x2 = np.random.random((2, 5))
        out = model.predict([x1, x2])
        assert out.shape == (2, 4, 5)

    # ndim not provided
    if K.backend() == 'tensorflow':
        k_ndim = K.ndim
        K.ndim = lambda _: None

        i1 = layers.Input(shape=(None, None))
        i2 = layers.Input(shape=(None,))
        ops = [layers.add, layers.maximum]
        for op in ops:
            o = op([i1, i2])
            assert o._keras_shape == (None, None, None)
            model = models.Model([i1, i2], o)

            x1 = np.random.random((2, 4, 5))
            x2 = np.random.random((2, 5))
            out = model.predict([x1, x2])
            assert out.shape == (2, 4, 5)
        K.ndim = k_ndim


if __name__ == '__main__':
    pytest.main([__file__])
