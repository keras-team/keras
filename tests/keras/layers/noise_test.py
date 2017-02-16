import pytest
from keras.utils.test_utils import layer_test
from keras.utils.test_utils import keras_test
from keras.layers import noise


@keras_test
def test_GaussianNoise():
    layer_test(noise.GaussianNoise,
               kwargs={'stddev': 1.},
               input_shape=(3, 2, 3))


@keras_test
def test_GaussianDropout():
    layer_test(noise.GaussianDropout,
               kwargs={'rate': 0.5},
               input_shape=(3, 2, 3))


if __name__ == '__main__':
    pytest.main([__file__])
