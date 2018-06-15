import pytest

from keras.utils.test_utils import keras_test
from keras.utils.test_utils import layer_test
from keras.legacy import layers as legacy_layers
from keras import regularizers
from keras import constraints


@keras_test
def test_highway():
    layer_test(legacy_layers.Highway,
               kwargs={},
               input_shape=(3, 2))

    layer_test(legacy_layers.Highway,
               kwargs={'W_regularizer': regularizers.l2(0.01),
                       'b_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.l2(0.01),
                       'W_constraint': constraints.MaxNorm(1),
                       'b_constraint': constraints.MaxNorm(1)},
               input_shape=(3, 2))


@keras_test
def test_maxout_dense():
    layer_test(legacy_layers.MaxoutDense,
               kwargs={'output_dim': 3},
               input_shape=(3, 2))

    layer_test(legacy_layers.MaxoutDense,
               kwargs={'output_dim': 3,
                       'W_regularizer': regularizers.l2(0.01),
                       'b_regularizer': regularizers.l1(0.01),
                       'activity_regularizer': regularizers.l2(0.01),
                       'W_constraint': constraints.MaxNorm(1),
                       'b_constraint': constraints.MaxNorm(1)},
               input_shape=(3, 2))


if __name__ == '__main__':
    pytest.main([__file__])
