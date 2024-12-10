import numpy as np

from keras.src import ops
from keras.src import testing
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.common.symbolic_scope import in_symbolic_scope


class TestSymbolicScope(testing.TestCase):
    def test_basic_flow(self):
        # Define a function that behaves differently according to
        # `in_symbolic_scope`.
        def compute_loss(y, y_pred):
            if in_symbolic_scope():
                return ops.zeros_like(y)
            return ops.add(y, y_pred)

        y = ops.ones(shape=(2,))
        y_pred = ops.ones(shape=(2,))
        with SymbolicScope():
            loss = compute_loss(y, y_pred)
        self.assertAllClose(loss, np.zeros((2,)))

        loss = compute_loss(y, y_pred)
        self.assertAllClose(loss, 2 * np.ones((2,)))
