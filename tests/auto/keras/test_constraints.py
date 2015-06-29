import unittest
import numpy as np
from theano import tensor as T

class TestConstraints(unittest.TestCase):
    def setUp(self):
        self.some_values = [0.1,0.5,3,8,1e-7]
        self.example_array = np.random.random((100,100))*100. - 50.
        self.example_array[0,0] = 0. # 0 could possibly cause trouble

    def test_maxnorm(self):
        from keras.constraints import maxnorm

        for m in self.some_values:
            norm_instance = maxnorm(m)
            normed = norm_instance(self.example_array)
            assert(np.all(normed.eval() < m))

if __name__ == '__main__':
    unittest.main()
