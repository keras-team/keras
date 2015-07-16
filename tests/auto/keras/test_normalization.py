import unittest
import numpy as np
from numpy.testing import assert_allclose
from theano import tensor as T
from keras.layers import normalization

class TestBatchNormalization(unittest.TestCase):
    def setUp(self):
        self.input_1 = np.arange(10)
        self.input_2 = np.zeros(10)
        self.input_3 = np.ones((10))

        self.input_shapes = [np.ones((10,10)), np.ones((10,10,10))]

    def test_setup(self):
        norm_m0 = normalization.BatchNormalization((10,10))
        norm_m1 = normalization.BatchNormalization((10,10), mode=1)

        # mode 3 does not exist
        self.assertRaises(Exception,normalization.BatchNormalization((10,10),mode=3))

    def test_mode_0(self):
        """
        Test the function of mode 0. Need to be somewhat lenient with the
        equality assertions because of the epsilon trick used to avoid NaNs.
        """
        norm_m0 = normalization.BatchNormalization((10,), momentum=0.5)

        norm_m0.input = self.input_1
        out = (norm_m0.get_output(train=True) - norm_m0.beta)/norm_m0.gamma
        self.assertAlmostEqual(out.mean().eval(), 0.0)
        self.assertAlmostEqual(out.std().eval(), 1.0, places=2)

        self.assertAlmostEqual(norm_m0.running_mean, 4.5)
        self.assertAlmostEqual(norm_m0.running_std.eval(), np.arange(10).std(), places=2)

        norm_m0.input = self.input_2
        out = (norm_m0.get_output(train=True) - norm_m0.beta)/norm_m0.gamma
        self.assertAlmostEqual(out.mean().eval(), 0.0)
        self.assertAlmostEqual(out.std().eval(), 0.0, places=2)

        #Values calculated by hand
        self.assertAlmostEqual(norm_m0.running_mean, 2.25)
        self.assertAlmostEqual(norm_m0.running_std.eval(), 0.5*np.arange(10).std(), places=2)

        out_test = (norm_m0.get_output(train=False) - norm_m0.beta)/norm_m0.gamma
        self.assertAlmostEqual(out_test.mean().eval(), -2.25 / (0.5*np.arange(10).std()),places=2)
        self.assertAlmostEqual(out_test.std().eval(), 0.0, places=2)

        norm_m0.input = self.input_3
        out = (norm_m0.get_output(train=True) - norm_m0.beta)/norm_m0.gamma
        self.assertAlmostEqual(out.mean().eval(), 0.0)
        self.assertAlmostEqual(out.std().eval(), 0.0, places=2)

    def test_mode_1(self):
        norm_m1 = normalization.BatchNormalization((10,), mode=1)

        for inp in [self.input_1, self.input_2, self.input_3]:
            norm_m1.input = inp
            out = (norm_m1.get_output(train=True) - norm_m1.beta)/norm_m1.gamma
            self.assertAlmostEqual(out.mean().eval(), 0.0)
            if inp.std() > 0.:
                self.assertAlmostEqual(out.std().eval(), 1.0, places=2)
            else:
                self.assertAlmostEqual(out.std().eval(), 0.0, places=2)

    def test_shapes(self):
        """
        Test batch normalization with various input shapes
        """
        for inp in self.input_shapes:
            norm_m0 = normalization.BatchNormalization(inp.shape, mode=0)
            norm_m0.input = inp
            out = (norm_m0.get_output(train=True) - norm_m0.beta)/norm_m0.gamma

            norm_m1 = normalization.BatchNormalization(inp.shape, mode=1)
            norm_m1.input = inp
            out = (norm_m1.get_output(train=True) - norm_m1.beta)/norm_m1.gamma

    def test_weight_init(self):
        """
        Test weight initialization
        """

        norm_m1 = normalization.BatchNormalization((10,), mode=1, weights=[np.ones(10),np.ones(10)])

        for inp in [self.input_1, self.input_2, self.input_3]:
            norm_m1.input = inp
            out = (norm_m1.get_output(train=True) - np.ones(10))/1.
            self.assertAlmostEqual(out.mean().eval(), 0.0)
            if inp.std() > 0.:
                self.assertAlmostEqual(out.std().eval(), 1.0, places=2)
            else:
                self.assertAlmostEqual(out.std().eval(), 0.0, places=2)

        assert_allclose(norm_m1.gamma.eval(),np.ones(10))
        assert_allclose(norm_m1.beta.eval(),np.ones(10))

        #Weights must be an iterable of gamma AND beta.
        self.assertRaises(Exception,normalization.BatchNormalization(10,), weights = np.ones(10))


    def test_config(self):
        norm = normalization.BatchNormalization((10,10), mode=1, epsilon=0.1)
        conf = norm.get_config()
        conf_target = {"input_shape": (10,10), "name": normalization.BatchNormalization.__name__,
                       "epsilon":0.1, "mode": 1}

        self.assertDictEqual(conf, conf_target)


if __name__ == '__main__':
    unittest.main()
