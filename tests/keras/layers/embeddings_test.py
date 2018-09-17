import pytest
from keras.utils.test_utils import layer_test
from keras.layers.embeddings import Embedding
from keras.models import Sequential
import keras.backend as K


def test_embedding():
    layer_test(Embedding,
               kwargs={'output_dim': 4, 'input_dim': 10, 'input_length': 2},
               input_shape=(3, 2),
               input_dtype='int32',
               expected_output_dtype=K.floatx())
    layer_test(Embedding,
               kwargs={'output_dim': 4, 'input_dim': 10, 'mask_zero': True},
               input_shape=(3, 2),
               input_dtype='int32',
               expected_output_dtype=K.floatx())
    layer_test(Embedding,
               kwargs={'output_dim': 4, 'input_dim': 10, 'mask_zero': True},
               input_shape=(3, 2, 5),
               input_dtype='int32',
               expected_output_dtype=K.floatx())
    layer_test(Embedding,
               kwargs={'output_dim': 4, 'input_dim': 10, 'mask_zero': True,
                       'input_length': (None, 5)},
               input_shape=(3, 2, 5),
               input_dtype='int32',
               expected_output_dtype=K.floatx())


def test_embedding_invalid():

    # len(input_length) should be equal to len(input_shape) - 1
    with pytest.raises(ValueError):
        model = Sequential([Embedding(
            input_dim=10,
            output_dim=4,
            input_length=2,
            input_shape=(3, 4, 5))])

    # input_length should be equal to input_shape[1:]
    with pytest.raises(ValueError):
        model = Sequential([Embedding(
            input_dim=10,
            output_dim=4,
            input_length=2,
            input_shape=(3, 5))])


if __name__ == '__main__':
    pytest.main([__file__])
