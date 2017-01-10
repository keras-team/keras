import pytest
from keras.utils.test_utils import layer_test, keras_test
from keras.layers.embeddings import Embedding
import keras.backend as K
from keras.models import Sequential
from keras.optimizers import Adam
import numpy as np


@keras_test
def test_embedding():
    layer_test(Embedding,
               kwargs={'output_dim': 4, 'input_dim': 10, 'input_length': 2},
               input_shape=(3, 2),
               input_dtype='int32',
               expected_output_dtype=K.floatx())


@pytest.mark.skipif(K.backend() != 'tensorflow', reason='Requires TF backend')
@keras_test
def test_embedding_with_clipnorm():
    model = Sequential()
    model.add(Embedding(input_dim=1, output_dim=1))
    model.compile(optimizer=Adam(clipnorm=1.0), loss='mse')
    model.fit(np.array([[0]]), np.array([[[0.5]]]), nb_epoch=1)


if __name__ == '__main__':
    pytest.main([__file__])
