from keras_core import testing
from keras_core import backend
from keras_core import operations as ops
import numpy as np
from keras_core.backend.stateless_scope import StatelessScope


class TestStatelessScope(testing.TestCase):
    def test_basic_flow(self):
        var1 = backend.Variable(np.zeros((2,)))
        var2 = backend.Variable(np.zeros((2,)))
        var_out = backend.Variable(np.zeros((2,)))

        value1 = ops.ones(shape=(2,))
        value2 = ops.ones(shape=(2,))
        with StatelessScope(state_mapping=[(var1, value1), (var2, value2)]) as scope:
            out = var1 + var2
            var_out.assign(out)
            var_out_value = var_out + 0.
            # Inside scope: new value is used.
            self.assertAllClose(var_out_value, 2 * np.ones((2,)))
        
        # Out of scope: old value is used.
        var_out_value = var_out + 0.
        self.assertAllClose(var_out_value, np.zeros((2,)))

        # Updates are tracked.
        var_out_value = scope.get_current_value(var_out)
        self.assertAllClose(var_out_value, 2 * np.ones((2,)))

        # Updates can be reapplied.
        var_out.assign(scope.get_current_value(var_out))
        self.assertAllClose(var_out_value, 2 * np.ones((2,)))
