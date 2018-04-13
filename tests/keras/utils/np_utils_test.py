"""Tests for functions in np_utils.py.
"""
import numpy as np
import pytest
from keras.utils import to_categorical
from keras.utils import to_channels_first


def test_to_categorical():
    num_classes = 5
    shapes = [(1,), (3,), (4, 3), (5, 4, 3), (3, 1), (3, 2, 1)]
    expected_shapes = [(1, num_classes),
                       (3, num_classes),
                       (4, 3, num_classes),
                       (5, 4, 3, num_classes),
                       (3, num_classes),
                       (3, 2, num_classes)]
    labels = [np.random.randint(0, num_classes, shape) for shape in shapes]
    one_hots = [to_categorical(label, num_classes) for label in labels]
    for label, one_hot, expected_shape in zip(labels,
                                              one_hots,
                                              expected_shapes):
        # Check shape
        assert one_hot.shape == expected_shape
        # Make sure there are only 0s and 1s
        assert np.array_equal(one_hot, one_hot.astype(bool))
        # Make sure there is exactly one 1 in a row
        assert np.all(one_hot.sum(axis=-1) == 1)
        # Get original labels back from one hots
        assert np.all(np.argmax(one_hot, -1).reshape(label.shape) == label)


def test_to_channels_first():
    shapes = [
        (10, 32, 32, 3),
        (10, 24, 128, 128, 1)
    ]

    expected_shapes = [
        (10, 3, 32, 32),
        (10, 1, 24, 128, 128)
    ]

    inputs = [np.random.randint(0, 255, shape) for shape in shapes]

    # Test for single numpy array
    for inp, exp_shape in zip(inputs, expected_shapes):
        inp = to_channels_first(inp)
        assert inp.shape == exp_shape

    # Test for list of numpy arrays
    inputs = to_channels_first(inputs)
    for inp, exp_shape in zip(inputs, expected_shapes):
        assert inp.shape == exp_shape

    shapes = [
        (10, 32, 32, 64, 64, 3),
        (10, 28, 28),
        (10, 28),
        (10)
    ]

    # Negative use case
    # Test for single numpy array
    inputs = [np.random.randint(0, 255, shape) for shape in shapes]
    for inp in inputs:
        with pytest.raises(ValueError) as excinfo:
            to_channels_first(inp)
            assert str(excinfo.typename) == 'ValueError'

    # Test for list of numpy arrays
    with pytest.raises(ValueError) as excinfo:
        to_channels_first(inputs)
        assert str(excinfo.typename) == 'ValueError'


if __name__ == '__main__':
    pytest.main([__file__])
