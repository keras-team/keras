from __future__ import print_function
import unittest
import numpy as np
import os

from keras.datasets.data_utils import get_file
from keras.caffe.convert import CaffeToKeras


class TestCaffeIntegration(unittest.TestCase):
    def test_proto(self):
        print('test a correct topology construction from prototext')

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
        print('test a parameterized model for param conversion, grouped convolutions')

        dirname = "MIT_scenes"
        origin = "http://places.csail.mit.edu/model/hybridCNN_upgraded.tar.gz"
        path = get_file(dirname, origin=origin, untar=True)
        caffemodel = os.path.join(path, 'hybridCNN_iter_700000_upgraded.caffemodel')

        model = CaffeToKeras(caffemodel=caffemodel)
        assert(model('outputs') == ['loss'])
        assert(model('inputs') == ['data'])

        network = model('network')
        network.compile('rmsprop', {'loss': 'mse'})
        datam = np.random.random((1, 3, 227, 227))
        outputs = network.predict({'data': datam})

        result = outputs['loss'].tolist()[0]
        result_prob = max(result)
        result_class = result.index(result_prob)

        assert(len(outputs) == 1)
        assert(outputs['loss'].shape == (1, 1183))
        assert(result_class == 779)
        assert(result_prob < 0.003)

        network.get_config(verbose=1)


if __name__ == '__main__':
    print('Test caffe model conversion')
    unittest.main()
