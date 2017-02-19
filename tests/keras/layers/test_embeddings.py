import pytest
import numpy as np

from keras.utils.test_utils import layer_test, keras_test
from keras.layers.embeddings import Embedding, OneHot
import keras.backend as K


@keras_test
def test_embedding():
    layer_test(Embedding,
               kwargs={'output_dim': 4, 'input_dim': 10, 'input_length': 2},
               input_shape=(3, 2),
               input_dtype='int32',
               expected_output_dtype=K.floatx())


@keras_test
def test_one_hot():
    layer_test(OneHot,
               kwargs={'input_dim': 3, 'input_length': 1},
               input_data=np.array([[0], [1], [2]]),
               input_dtype='int32',
               expected_output_dtype=K.floatx(),
               expected_output=np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]], [[0.0, 0.0, 1.0]]]))

if __name__ == '__main__':
    pytest.main([__file__])
