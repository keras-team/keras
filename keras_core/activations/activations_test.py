import numpy as np

from keras_core import activations
from keras_core import backend
from keras_core import testing


def _ref_softmax(values):
    m = np.max(values)
    e = np.exp(values - m)
    return e / np.sum(e)


def _ref_softplus(x):
    return np.log(np.ones_like(x) + np.exp(x))


class ActivationsTest(testing.TestCase):
    def test_softmax(self):
        x = np.random.random((2, 5))

        result = activations.softmax(x[np.newaxis, :])[0]
        expected = _ref_softmax(x[0])
        self.assertAllClose(result[0], expected, rtol=1e-05)

    def test_softmax_2d_axis_0(self):
        x = np.random.random((2, 5))
        result = activations.softmax(x[np.newaxis, :], axis=1)[0]
        expected = np.zeros((2, 5))
        for i in range(5):
            expected[:, i] = _ref_softmax(x[:, i])
        self.assertAllClose(result, expected, rtol=1e-05)

    # TODO: Fails on Tuple Axis
    # ops/nn_ops.py:3824: TypeError:
    # '<=' not supported between instances of 'int' and 'tuple'
    # def test_softmax_3d_axis_tuple(self):
    #     x = np.random.random((2, 3, 5))
    #     result = activations.softmax([x], axis=(1, 2))[0]
    #     expected = np.zeros((2, 3, 5))
    #     for i in range(2):
    #         expected[i, :, :] = _ref_softmax(x[i, :, :])
    #     self.assertAllClose(result, expected, rtol=1e-05)

    def test_temporal_softmax(self):
        x = np.random.random((2, 2, 3)) * 10
        result = activations.softmax(x[np.newaxis, :])[0]
        expected = _ref_softmax(x[0, 0])
        self.assertAllClose(result[0, 0], expected, rtol=1e-05)

    def test_selu(self):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        positive_values = np.array([[1, 2]], dtype=backend.floatx())
        result = activations.selu(positive_values[np.newaxis, :])[0]
        self.assertAllClose(result, positive_values * scale, rtol=1e-05)

        negative_values = np.array([[-1, -2]], dtype=backend.floatx())
        result = activations.selu(negative_values[np.newaxis, :])[0]
        true_result = (np.exp(negative_values) - 1) * scale * alpha
        self.assertAllClose(result, true_result)

    def test_softplus(self):
        x = np.random.random((2, 5))
        result = activations.softplus(x[np.newaxis, :])[0]
        expected = _ref_softplus(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_softsign(self):
        def softsign(x):
            return np.divide(x, np.ones_like(x) + np.absolute(x))

        x = np.random.random((2, 5))
        result = activations.softsign(x[np.newaxis, :])[0]
        expected = softsign(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_sigmoid(self):
        def ref_sigmoid(x):
            if x >= 0:
                return 1 / (1 + np.exp(-x))
            else:
                z = np.exp(x)
                return z / (1 + z)

        sigmoid = np.vectorize(ref_sigmoid)

        x = np.random.random((2, 5))
        result = activations.sigmoid(x[np.newaxis, :])[0]
        expected = sigmoid(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_hard_sigmoid(self):
        def ref_hard_sigmoid(x):
            x = (x / 6.0) + 0.5
            z = 0.0 if x <= 0 else (1.0 if x >= 1 else x)
            return z

        hard_sigmoid = np.vectorize(ref_hard_sigmoid)
        x = np.random.random((2, 5))
        result = activations.hard_sigmoid(x[np.newaxis, :])[0]
        expected = hard_sigmoid(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_relu(self):
        positive_values = np.random.random((2, 5))
        result = activations.relu(positive_values[np.newaxis, :])[0]
        self.assertAllClose(result, positive_values, rtol=1e-05)

        negative_values = np.random.uniform(-1, 0, (2, 5))
        result = activations.relu(negative_values[np.newaxis, :])[0]
        expected = np.zeros((2, 5))
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_gelu(self):
        def gelu(x, approximate=False):
            if approximate:
                return (
                    0.5
                    * x
                    * (
                        1.0
                        + np.tanh(
                            np.sqrt(2.0 / np.pi)
                            * (x + 0.044715 * np.power(x, 3))
                        )
                    )
                )
            else:
                from scipy.stats import norm

                return x * norm.cdf(x)

        x = np.random.random((2, 5))
        result = activations.gelu(x[np.newaxis, :])[0]
        expected = gelu(x)
        self.assertAllClose(result, expected, rtol=1e-05)

        x = np.random.random((2, 5))
        result = activations.gelu(x[np.newaxis, :], approximate=True)[0]
        expected = gelu(x, True)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_elu(self):
        x = np.random.random((2, 5))
        result = activations.elu(x[np.newaxis, :])[0]
        self.assertAllClose(result, x, rtol=1e-05)
        negative_values = np.array([[-1, -2]], dtype=backend.floatx())
        result = activations.elu(negative_values[np.newaxis, :])[0]
        true_result = np.exp(negative_values) - 1
        self.assertAllClose(result, true_result)

    def test_tanh(self):
        x = np.random.random((2, 5))
        result = activations.tanh(x[np.newaxis, :])[0]
        expected = np.tanh(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_exponential(self):
        x = np.random.random((2, 5))
        result = activations.exponential(x[np.newaxis, :])[0]
        expected = np.exp(x)
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_mish(self):
        x = np.random.random((2, 5))
        result = activations.mish(x[np.newaxis, :])[0]
        expected = x * np.tanh(_ref_softplus(x))
        self.assertAllClose(result, expected, rtol=1e-05)

    def test_linear(self):
        x = np.random.random((10, 5))
        self.assertAllClose(x, activations.linear(x))

    def test_get_method(self):
        obj = activations.get("relu")
        self.assertEqual(obj, activations.relu)

        obj = activations.get(None)
        self.assertEqual(obj, activations.linear)

        with self.assertRaises(ValueError):
            activations.get("typo")
