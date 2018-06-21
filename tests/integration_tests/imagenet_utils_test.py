import pytest
import numpy as np

from keras.applications import imagenet_utils as utils


def test_imagenet_utils_available():
    x = np.random.uniform(0, 255, (2, 10, 10, 3))
    assert utils.preprocess_input(x).shape == x.shape


if __name__ == '__main__':
    pytest.main([__file__])
