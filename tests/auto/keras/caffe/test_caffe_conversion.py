from __future__ import print_function
import unittest
import numpy as np
np.random.seed(1337)
from keras.models import Graph, Sequential
from keras.layers import containers
from keras.layers.core import Dense, Activation
from keras.utils.test_utils import get_test_data

from keras.caffe.convert import CaffeToKeras

X = np.random.random((100, 32))
X2 = np.random.random((100, 32))
y = np.random.random((100, 4))
y2 = np.random.random((100, 4))

(X_train, y_train), (X_test, y_test) = get_test_data(nb_train=1000, nb_test=200, input_shape=(32,),
    classification=False, output_shape=(4,))
(X2_train, y2_train), (X2_test, y2_test) = get_test_data(nb_train=1000, nb_test=200, input_shape=(32,),
    classification=False, output_shape=(4,))

class TestCaffeIntegration(unittest.TestCase):
    def test_proto_non_deploy(self):
        print('test a minimal inception module for correct topology')
        
        model = CaffeToKeras(prototext='./minimal_inception.prototxt')
        assert(len(model('inputs')) == 1)
        assert(len(model('outputs')) == 3)
        assert(model.outputs == ['loss1', 'loss2', 'loss3'])
        assert(model.inputs == ['data'])

        network = model('network')
        network.compile('rmsprop', {'loss1': 'mse', 'loss2': 'mse', 'loss3': 'mse'})

        datam = np.random.random((1, 3, 224, 224))
        outputs = network.predict({'data': datam})
        assert(len(outputs) == 3)
        assert(outputs['loss1'].shape == (1, 64, 112, 112))
        assert(outputs['loss2'].shape == (1, 192, 27, 27))
        assert(outputs['loss3'].shape == (1, 256, 27, 27))


if __name__ == '__main__':
    print('Test graph model')
    unittest.main()