from __future__ import print_function
import pytest
import numpy as np
from keras.utils.test_utils import get_test_data
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.utils.test_utils import layer_test
from keras import backend as K

seed = 1224

def test_learning_rate_multipliers_maxout_dense():
    from keras.layers.core import MaxoutDense
    
    layer_test(MaxoutDense,
               kwargs={'output_dim': 3,
                       'W_learning_rate_multiplier': 0.1,
                       'b_learning_rate_multiplier': 0.1},
               input_shape=(3, 2))

    with pytest.raises(Exception) as e_info:
        layer_test(MaxoutDense,
                   kwargs={'output_dim': 3,
                           'bias': False,
                           'W_learning_rate_multiplier': 0.1,
                           'b_learning_rate_multiplier': 0.1},
                   input_shape=(3, 2))

def test_learning_rate_multipliers_conv1d():
    from keras.layers.convolutional import Convolution1D

    layer_test(Convolution1D,
               kwargs={'nb_filter': 4,
                       'filter_length': 3,
                       'W_learning_rate_multiplier': 0.1,
                       'b_learning_rate_multiplier': 0.1},
               input_shape=(2, 8, 5))

    with pytest.raises(Exception) as e_info:
        layer_test(Convolution1D,
                   kwargs={'nb_filter': 4,
                           'filter_length': 3,
                           'bias': False,
                           'W_learning_rate_multiplier': 0.1,
                           'b_learning_rate_multiplier': 0.1},
                   input_shape=(2, 8, 5))

@pytest.mark.skipif((K._BACKEND != 'theano'),
                    reason="Requires theano backend or be able to set random seed in tensorflow")
def test_learning_rate_multipliers_dense():
    from keras.layers.core import Dense
    from keras.optimizers import SGD
    
    layer_test(Dense,
               kwargs={'output_dim': 3,
                       'W_learning_rate_multiplier': 0.1,
                       'b_learning_rate_multiplier': 0.1},
               input_shape=(3, 2))

    # This should raise an error
    with pytest.raises(Exception) as e_info:
        layer_test(Dense,
                   kwargs={'output_dim': 3,
                           'bias': False,
                           'W_learning_rate_multiplier': 0.1,
                           'b_learning_rate_multiplier': 0.1},
                   input_shape=(3, 2))

    np.random.seed(seed)
    (X_train, y_train), (X_test, y_test) = get_test_data(nb_train=10,
                                                         nb_test=1,
                                                         input_shape=(5,),
                                                         classification=True,
                                                         nb_class=2)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    np.random.seed(seed)
    model0 = Sequential()
    model0.add(Dense(output_dim=2, input_dim=5))
    sgd = SGD(lr=0.4, momentum=0.,decay=0.)
    model0.compile(loss='mse', optimizer=sgd)
    (m0w0_ini,m0b0_ini) = model0.layers[0].get_weights()
    model0.train_on_batch(X_train, y_train)
    (m0w0_end,m0b0_end) = model0.layers[0].get_weights() 
    
    np.random.seed(seed)
    model1 = Sequential()
    model1.add(Dense(output_dim=2, input_dim=5,
                     W_learning_rate_multiplier=0.5, b_learning_rate_multiplier=0.5))
    sgd = SGD(lr=0.4, momentum=0.,decay=0.)
    model1.compile(loss='mse', optimizer=sgd)
    (m1w0_ini,m1b0_ini) = model1.layers[0].get_weights()
    model1.train_on_batch(X_train, y_train)
    (m1w0_end,m1b0_end) = model1.layers[0].get_weights() 

    # This should be ~0.5
    np.testing.assert_almost_equal(np.mean((m1w0_end - m1w0_ini)/(m0w0_end - m0w0_ini)), 0.5, decimal=2)
    np.testing.assert_almost_equal(np.mean((m1b0_end - m1b0_ini)/(m0b0_end - m0b0_ini)), 0.5, decimal=2)

@pytest.mark.skipif((K._BACKEND != 'theano'),
                    reason="Requires theano backend or be able to set random seed in tensorflow")
def test_learning_rate_multipliers_conv2d():
    from keras.layers.convolutional import Convolution2D

    layer_test(Convolution2D,
               kwargs={'nb_filter': 3,
                       'nb_row': 3,
                       'nb_col': 3,
                       'W_learning_rate_multiplier': 0.1,
                       'b_learning_rate_multiplier': 0.1},
               input_shape=(8, 4, 10, 6))

    with pytest.raises(Exception) as e_info:
        layer_test(Convolution2D,
                   kwargs={'nb_filter': 3,
                           'nb_row': 3,
                           'nb_col': 3,
                           'bias': False,
                           'W_learning_rate_multiplier': 0.1,
                           'b_learning_rate_multiplier': 0.1},
                   input_shape=(8, 4, 10, 6))

    np.random.seed(seed)
    X_train = np.random.rand(10,3,10,10)
    y_train = np.random.rand(10,2,8,8)

    np.random.seed(seed)
    model0 = Sequential()
    model0.add(Convolution2D(2,3,3,
                             input_shape=(3,10,10), 
                             border_mode='valid'))
    model0.compile(loss='mse', optimizer='sgd')
    (m0w0_ini,m0b0_ini) = model0.layers[0].get_weights()
    model0.train_on_batch(X_train, y_train)
    (m0w0_end,m0b0_end) = model0.layers[0].get_weights() 
    
    np.random.seed(seed)
    model1 = Sequential()
    model1.add(Convolution2D(2,3,3,
                             input_shape=(3,10,10), 
                             border_mode='valid', 
                             W_learning_rate_multiplier=0.5, b_learning_rate_multiplier=0.5))
    model1.compile(loss='mse', optimizer='sgd')
    (m1w0_ini,m1b0_ini) = model1.layers[0].get_weights()
    model1.train_on_batch(X_train, y_train)
    (m1w0_end,m1b0_end) = model1.layers[0].get_weights() 
        
    # This should be ~0.5
    np.testing.assert_almost_equal(np.mean((m1w0_end - m1w0_ini)/(m0w0_end - m0w0_ini)), 0.5, decimal=2)
    np.testing.assert_almost_equal(np.mean((m1b0_end - m1b0_ini)/(m0b0_end - m0b0_ini)), 0.5, decimal=2)

if __name__ == '__main__':
    pytest.main([__file__])
