import pytest

from keras.utils.test_utils import layer_test, keras_test
from keras.layers import local


@keras_test
def test_locallyconnected_1d():
    num_samples = 2
    num_steps = 8
    input_dim = 5
    filter_length = 3
    filters = 4

    for border_mode in ['valid']:
        for subsample_length in [1]:
            if border_mode == 'same' and subsample_length != 1:
                continue
            layer_test(local.LocallyConnected1D,
                       kwargs={'filters': filters,
                               'filter_length': filter_length,
                               'border_mode': border_mode,
                               'subsample_length': subsample_length},
                       input_shape=(num_samples, num_steps, input_dim))

            layer_test(local.LocallyConnected1D,
                       kwargs={'filters': filters,
                               'filter_length': filter_length,
                               'border_mode': border_mode,
                               'W_regularizer': 'l2',
                               'b_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'subsample_length': subsample_length},
                       input_shape=(num_samples, num_steps, input_dim))


@keras_test
def test_locallyconnected_2d():
    num_samples = 8
    filters = 3
    stack_size = 4
    num_row = 6
    num_col = 10

    for border_mode in ['valid']:
        for subsample in [(1, 1), (2, 2)]:
            if border_mode == 'same' and subsample != (1, 1):
                continue

            layer_test(local.LocallyConnected2D,
                       kwargs={'filters': filters,
                               'num_row': 3,
                               'num_col': 3,
                               'border_mode': border_mode,
                               'W_regularizer': 'l2',
                               'b_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'subsample': subsample,
                               'data_format': 'channels_last'},
                       input_shape=(num_samples, num_row, num_col, stack_size))

            layer_test(local.LocallyConnected2D,
                       kwargs={'filters': filters,
                               'num_row': 3,
                               'num_col': 3,
                               'border_mode': border_mode,
                               'W_regularizer': 'l2',
                               'b_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'subsample': subsample,
                               'data_format': 'channels_first'},
                       input_shape=(num_samples, stack_size, num_row, num_col))


if __name__ == '__main__':
    pytest.main([__file__])
