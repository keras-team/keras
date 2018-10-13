import pytest
from keras.utils.test_utils import layer_test
from keras.layers import noise
from keras import backend as K


@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support it yet")
def test_GaussianNoise():
    layer_test(noise.GaussianNoise,
               kwargs={'stddev': 1.},
               input_shape=(3, 2, 3))


@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support it yet")
def test_GaussianDropout():
    layer_test(noise.GaussianDropout,
               kwargs={'rate': 0.5},
               input_shape=(3, 2, 3))


@pytest.mark.skipif((K.backend() == 'cntk'),
                    reason="cntk does not support it yet")
def test_AlphaDropout():
    layer_test(noise.AlphaDropout,
               kwargs={'rate': 0.1},
               input_shape=(3, 2, 3))


if __name__ == '__main__':
    pytest.main([__file__])
