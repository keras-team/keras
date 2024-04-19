import numpy as np

from keras.src import backend
from keras.src import ops
from keras.src import testing
from keras.src.backend.common.stateless_scope import StatelessScope


class TestStatelessScope(testing.TestCase):
    def test_basic_flow(self):
        var1 = backend.Variable(np.zeros((2,)))
        var2 = backend.Variable(np.zeros((2,)))
        var_out = backend.Variable(np.zeros((2,)))

        value1 = ops.ones(shape=(2,))
        value2 = ops.ones(shape=(2,))
        with StatelessScope(
            state_mapping=[(var1, value1), (var2, value2)]
        ) as scope:
            out = var1 + var2
            var_out.assign(out)
            var_out_value = var_out + 0.0
            # Inside scope: new value is used.
            self.assertAllClose(var_out_value, 2 * np.ones((2,)))

        # Out of scope: old value is used.
        var_out_value = var_out + 0.0
        self.assertAllClose(var_out_value, np.zeros((2,)))

        # Updates are tracked.
        var_out_value = scope.get_current_value(var_out)
        self.assertAllClose(var_out_value, 2 * np.ones((2,)))

        # Updates can be reapplied.
        var_out.assign(scope.get_current_value(var_out))
        self.assertAllClose(var_out_value, 2 * np.ones((2,)))

    def test_invalid_key_in_state_mapping(self):
        # var1 = backend.Variable(np.zeros((2,)))
        invalid_key = "not_a_keras_variable"
        value1 = ops.ones(shape=(2,))

        with self.assertRaisesRegex(
            ValueError, "all keys in argument `mapping` must be KerasVariable"
        ):
            StatelessScope(state_mapping=[(invalid_key, value1)])

    def test_invalid_value_shape_in_state_mapping(self):
        var1 = backend.Variable(np.zeros((2,)))
        invalid_value = ops.ones(shape=(3,))  # Incorrect shape

        with self.assertRaisesRegex(
            ValueError, "all values in argument `mapping` must be tensors with"
        ):
            StatelessScope(state_mapping=[(var1, invalid_value)])
