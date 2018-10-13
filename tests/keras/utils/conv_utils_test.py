import pytest
import numpy as np
from keras.utils import conv_utils
from keras import backend as K


def test_normalize_tuple():
    assert conv_utils.normalize_tuple(5, 2, 'kernel_size') == (5, 5)
    assert conv_utils.normalize_tuple([7, 9], 2, 'kernel_size') == (7, 9)

    with pytest.raises(ValueError):
        conv_utils.normalize_tuple(None, 2, 'kernel_size')
    with pytest.raises(ValueError):
        conv_utils.normalize_tuple([2, 3, 4], 2, 'kernel_size')
    with pytest.raises(ValueError):
        conv_utils.normalize_tuple(['str', 'impossible'], 2, 'kernel_size')


def test_invalid_data_format():
    with pytest.raises(ValueError):
        K.normalize_data_format('channels_middle')


def test_invalid_padding():
    with pytest.raises(ValueError):
        conv_utils.normalize_padding('diagonal')


def test_invalid_convert_kernel():
    with pytest.raises(ValueError):
        conv_utils.convert_kernel(np.zeros((10, 20)))


def test_conv_output_length():
    assert conv_utils.conv_output_length(None, 7, 'same', 1) is None
    assert conv_utils.conv_output_length(224, 7, 'same', 1) == 224
    assert conv_utils.conv_output_length(224, 7, 'same', 2) == 112
    assert conv_utils.conv_output_length(32, 5, 'valid', 1) == 28
    assert conv_utils.conv_output_length(32, 5, 'valid', 2) == 14
    assert conv_utils.conv_output_length(32, 5, 'causal', 1) == 32
    assert conv_utils.conv_output_length(32, 5, 'causal', 2) == 16
    assert conv_utils.conv_output_length(32, 5, 'full', 1) == 36
    assert conv_utils.conv_output_length(32, 5, 'full', 2) == 18

    with pytest.raises(AssertionError):
        conv_utils.conv_output_length(32, 5, 'diagonal', 2)


def test_conv_input_length():
    assert conv_utils.conv_input_length(None, 7, 'same', 1) is None
    assert conv_utils.conv_input_length(112, 7, 'same', 1) == 112
    assert conv_utils.conv_input_length(112, 7, 'same', 2) == 223
    assert conv_utils.conv_input_length(28, 5, 'valid', 1) == 32
    assert conv_utils.conv_input_length(14, 5, 'valid', 2) == 31
    assert conv_utils.conv_input_length(36, 5, 'full', 1) == 32
    assert conv_utils.conv_input_length(18, 5, 'full', 2) == 31

    with pytest.raises(AssertionError):
        conv_utils.conv_output_length(18, 5, 'diagonal', 2)


def test_deconv_length():
    assert conv_utils.deconv_length(None, 1, 7, 'same', None) is None
    assert conv_utils.deconv_length(224, 1, 7, 'same', None) == 224
    assert conv_utils.deconv_length(224, 2, 7, 'same', None) == 448
    assert conv_utils.deconv_length(32, 1, 5, 'valid', None) == 36
    assert conv_utils.deconv_length(32, 2, 5, 'valid', None) == 67
    assert conv_utils.deconv_length(32, 1, 5, 'full', None) == 28
    assert conv_utils.deconv_length(32, 2, 5, 'full', None) == 59
    assert conv_utils.deconv_length(224, 1, 7, 'same', 0) == 224
    assert conv_utils.deconv_length(224, 2, 7, 'same', 0) == 447
    assert conv_utils.deconv_length(224, 2, 7, 'same', 1) == 448
    assert conv_utils.deconv_length(32, 1, 5, 'valid', 0) == 36
    assert conv_utils.deconv_length(32, 2, 5, 'valid', 0) == 67
    assert conv_utils.deconv_length(32, 2, 5, 'valid', 1) == 68
    assert conv_utils.deconv_length(6, 1, 3, 'full', 0) == 4
    assert conv_utils.deconv_length(6, 2, 3, 'full', 1) == 10
    assert conv_utils.deconv_length(6, 2, 3, 'full', 2) == 11


if __name__ == '__main__':
    pytest.main([__file__])
