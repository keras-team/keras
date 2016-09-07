import pytest

from keras.utils.test_utils import layer_test, keras_test
from keras.layers import local


@keras_test
def test_locallyconnected_1d():
    nb_samples = 2
    nb_steps = 8
    input_dim = 5
    filter_length = 3
    nb_filter = 4

    for border_mode in ['valid']:
        for subsample_length in [1]:
            if border_mode == 'same' and subsample_length != 1:
                continue
            layer_test(local.LocallyConnected1D,
                       kwargs={'nb_filter': nb_filter,
                               'filter_length': filter_length,
                               'border_mode': border_mode,
                               'subsample_length': subsample_length},
                       input_shape=(nb_samples, nb_steps, input_dim))

            layer_test(local.LocallyConnected1D,
                       kwargs={'nb_filter': nb_filter,
                               'filter_length': filter_length,
                               'border_mode': border_mode,
                               'W_regularizer': 'l2',
                               'b_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'subsample_length': subsample_length},
                       input_shape=(nb_samples, nb_steps, input_dim))


@keras_test
def test_locallyconnected_2d():
    nb_samples = 8
    nb_filter = 3
    stack_size = 4
    nb_row = 6
    nb_col = 10

    for border_mode in ['valid']:
        for subsample in [(1, 1), (2, 2)]:
            if border_mode == 'same' and subsample != (1, 1):
                continue

            layer_test(local.LocallyConnected2D,
                       kwargs={'nb_filter': nb_filter,
                               'nb_row': 3,
                               'nb_col': 3,
                               'border_mode': border_mode,
                               'W_regularizer': 'l2',
                               'b_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'subsample': subsample,
                               'dim_ordering': 'tf'},
                       input_shape=(nb_samples, nb_row, nb_col, stack_size))

            layer_test(local.LocallyConnected2D,
                       kwargs={'nb_filter': nb_filter,
                               'nb_row': 3,
                               'nb_col': 3,
                               'border_mode': border_mode,
                               'W_regularizer': 'l2',
                               'b_regularizer': 'l2',
                               'activity_regularizer': 'activity_l2',
                               'subsample': subsample,
                               'dim_ordering': 'th'},
                       input_shape=(nb_samples, stack_size, nb_row, nb_col))


if __name__ == '__main__':
    pytest.main([__file__])
