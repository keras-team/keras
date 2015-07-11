from __future__ import print_function
import unittest
import numpy as np

import os

from keras.caffe.convert import CaffeToKeras

class TestCaffeIntegration(unittest.TestCase):
    def test_proto(self):
        print('test a minimal inception module for correct topology construction from prototext')
        
        #add fetching option here
        model = CaffeToKeras(prototext='./minimal_inception.prototxt')
        assert(model('outputs') == ['loss1', 'loss2', 'loss3'])
        assert(model('inputs') == ['data'])

        network = model('network')
        network.compile('rmsprop', {'loss1': 'mse', 'loss2': 'mse', 'loss3': 'mse'})
        datam = np.random.random((1, 3, 224, 224))
        outputs = network.predict({'data': datam})

        assert(len(outputs) == 3)
        assert(outputs['loss1'].shape == (1, 64, 112, 112))
        assert(outputs['loss2'].shape == (1, 192, 27, 27))
        assert(outputs['loss3'].shape == (1, 256, 27, 27))

        network.get_config(verbose=1)


    def test_param(self):
        print('test a complete parameterized model for correct param conversion, grouped convolutions')
        
        #add fetching option here
        model = CaffeToKeras(caffemodel='./hybridCNN_iter_700000_upgraded.caffemodel')
        assert(model('outputs') == ['prob'])
        assert(model('inputs') == ['conv1'])

        network = model('network')
        network.compile('rmsprop', {'prob': 'mse'})
        datam = np.random.random((1, 3, 227, 227))
        outputs = network.predict({'conv1': datam})

        assert(len(outputs) == 1)
        assert(outputs['prob'].shape == (1, 1183))

        network.get_config(verbose=1)


if __name__ == '__main__':
    print('Test caffe model conversion')
    unittest.main()