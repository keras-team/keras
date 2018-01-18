import pytest
import numpy as np
import nose
from numpy.testing import assert_allclose

from keras.utils.test_utils import layer_test
from keras.utils.test_utils import keras_test
from keras import backend as K
from keras.engine.topology import InputLayer
from keras.layers import convolutional
from keras.layers import pooling
from keras.models import Sequential
from keras.layers import demo
from keras.layers import core
'''
特殊说明如果自己自定义层需要另外在keras/keras/layers下面创建一个新的py文件那么就需要在layers文件夹下的
init py文件下引这个新创py文件的所有类，详情可看init py文件

'''

# TensorFlow does not support full convolution.

#该py文件可对自定义类进行测试，类似于单元测试，构造一个输入的tensor看输出是否正确，这里可利用Keras的layer_test函数进行测试。

if K.backend() == 'theano':
    _convolution_paddings = ['valid', 'same', 'full']
else:
    _convolution_paddings = ['valid', 'same']


@keras_test  #这个一定要加上好用来测试
def test_L2Normalize():
    layer_test(demo.MyLayer,
               kwargs={},
               input_shape=(3,2))

@keras_test
def test_dropout():
    layer_test(core.Dropout,
               kwargs={'rate': 0.5},
               input_shape=(3, 2))

    layer_test(core.Dropout,
               kwargs={'rate': 0.5, 'noise_shape': [3, 1]},
               input_shape=(3, 2))

    layer_test(core.Dropout,
               kwargs={'rate': 0.5, 'noise_shape': [None, 1]},
               input_shape=(3, 2))

    layer_test(core.SpatialDropout1D,
               kwargs={'rate': 0.5},
               input_shape=(2, 3, 4))

    for data_format in ['channels_last', 'channels_first']:
        for shape in [(4, 5), (4, 5, 6)]:
            if data_format == 'channels_last':
                input_shape = (2,) + shape + (3,)
            else:
                input_shape = (2, 3) + shape
            layer_test(core.SpatialDropout2D if len(shape) == 2 else core.SpatialDropout3D,
                       kwargs={'rate': 0.5,
                               'data_format': data_format},
                       input_shape=input_shape)

            # Test invalid use cases,利用python的上下文管理机制测试错误输入或者不合法输入
            with pytest.raises(ValueError):
                layer_test(core.SpatialDropout2D if len(shape) == 2 else core.SpatialDropout3D,
                           kwargs={'rate': 0.5,
                                   'data_format': 'channels_middle'},
                           input_shape=input_shape)

