import pytest

from keras.utils.test_utils import layer_test, keras_test
from keras.layers import discrimination


@keras_test
def test_minibatchdiscrimination():
    nb_kernels = 2
    kernel_dim = 2
    nb_samples = 4
    input_dim = 5

    layer_test(discrimination.MinibatchDiscrimination,
               kwargs={'nb_kernels': nb_kernels,
                       'kernel_dim': kernel_dim},
               input_shape=(nb_samples, input_dim))

    layer_test(discrimination.MinibatchDiscrimination,
               kwargs={'nb_kernels': nb_kernels,
                       'kernel_dim': kernel_dim,
                       'W_regularizer': 'l2',
                       'activity_regularizer': 'activity_l2',
                       'input_dim': input_dim},
               input_shape=(nb_samples, input_dim))


if __name__ == '__main__':
    pytest.main([__file__])
