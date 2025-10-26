import numpy as np
import tensorflow as tf
from keras import ops

def test_quantile_graph_mode():
    @tf.function
    def run_quantile():
        x = np.array([[1, 2, 3], [4, 5, 6]])
        q = [0.5]
        return ops.quantile(x, q, axis=1)

    result = run_quantile()
    expected = np.array([[2, 5]])
    np.testing.assert_allclose(result, expected)
