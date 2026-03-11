import numpy as np
import pytest

from keras.src import backend
from keras.src import testing


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX-specific LSTM tests.",
)
class JaxLSTMTest(testing.TestCase):
    def test_cudnn_ok_standard(self):
        from jax import numpy as jnp

        from keras.src import activations
        from keras.src import ops
        from keras.src.backend.jax.rnn import cudnn_ok

        # These only return True when GPU is available, so on CPU
        # we just verify they return a bool and don't crash.
        result = cudnn_ok(activations.tanh, activations.sigmoid, False)
        self.assertIsInstance(result, (bool, np.bool_))

        result = cudnn_ok(jnp.tanh, activations.sigmoid, False)
        self.assertIsInstance(result, (bool, np.bool_))

        result = cudnn_ok(ops.tanh, ops.sigmoid, False)
        self.assertIsInstance(result, (bool, np.bool_))

    def test_cudnn_ok_rejects_unroll(self):
        from keras.src import activations
        from keras.src.backend.jax.rnn import cudnn_ok

        self.assertFalse(cudnn_ok(activations.tanh, activations.sigmoid, True))

    def test_cudnn_ok_rejects_no_bias(self):
        from keras.src import activations
        from keras.src.backend.jax.rnn import cudnn_ok

        self.assertFalse(
            cudnn_ok(
                activations.tanh, activations.sigmoid, False, use_bias=False
            )
        )

    def test_cudnn_ok_rejects_wrong_activation(self):
        from keras.src import activations
        from keras.src.backend.jax.rnn import cudnn_ok

        self.assertFalse(cudnn_ok(activations.relu, activations.sigmoid, False))
        self.assertFalse(cudnn_ok(activations.tanh, activations.tanh, False))

    def test_assert_valid_mask_right_padded(self):
        from jax import numpy as jnp

        from keras.src.backend.jax.rnn import _assert_valid_mask

        mask = jnp.array(
            [[True, True, True, False], [True, True, False, False]]
        )
        # Should not raise.
        _assert_valid_mask(mask)

    def test_assert_valid_mask_all_true(self):
        from jax import numpy as jnp

        from keras.src.backend.jax.rnn import _assert_valid_mask

        mask = jnp.ones((2, 5), dtype=jnp.bool_)
        _assert_valid_mask(mask)

    def test_assert_valid_mask_not_right_padded(self):
        from jax import numpy as jnp

        from keras.src.backend.jax.rnn import _assert_valid_mask

        mask = jnp.array(
            [[True, False, True, False], [True, True, False, False]]
        )
        with self.assertRaises(ValueError):
            _assert_valid_mask(mask)

    def test_assert_valid_mask_fully_masked(self):
        from jax import numpy as jnp

        from keras.src.backend.jax.rnn import _assert_valid_mask

        mask = jnp.array([[False, False, False], [True, True, False]])
        with self.assertRaises(ValueError):
            _assert_valid_mask(mask)

    def test_lstm_raises_on_cpu(self):
        """On CPU, lstm() should raise NotImplementedError."""
        from keras.src.backend.jax.rnn import lstm

        batch, seq_len, input_size, hidden_size = 2, 5, 4, 3
        rng = np.random.RandomState(42)
        inputs = rng.randn(batch, seq_len, input_size).astype("float32")
        h_0 = np.zeros((batch, hidden_size), dtype="float32")
        c_0 = np.zeros((batch, hidden_size), dtype="float32")
        kernel = rng.randn(input_size, 4 * hidden_size).astype("float32")
        recurrent_kernel = rng.randn(hidden_size, 4 * hidden_size).astype(
            "float32"
        )
        bias = rng.randn(4 * hidden_size).astype("float32")

        from keras.src import activations

        # On CPU, cudnn_ok returns False, so this should raise.
        with self.assertRaises(NotImplementedError):
            lstm(
                inputs,
                h_0,
                c_0,
                None,
                kernel,
                recurrent_kernel,
                bias,
                activations.tanh,
                activations.sigmoid,
            )

    def test_lstm_raises_unroll(self):
        from keras.src.backend.jax.rnn import lstm

        batch, seq_len, input_size, hidden_size = 2, 5, 4, 3
        rng = np.random.RandomState(42)
        inputs = rng.randn(batch, seq_len, input_size).astype("float32")
        h_0 = np.zeros((batch, hidden_size), dtype="float32")
        c_0 = np.zeros((batch, hidden_size), dtype="float32")
        kernel = rng.randn(input_size, 4 * hidden_size).astype("float32")
        recurrent_kernel = rng.randn(hidden_size, 4 * hidden_size).astype(
            "float32"
        )
        bias = rng.randn(4 * hidden_size).astype("float32")

        from keras.src import activations

        with self.assertRaises(NotImplementedError):
            lstm(
                inputs,
                h_0,
                c_0,
                None,
                kernel,
                recurrent_kernel,
                bias,
                activations.tanh,
                activations.sigmoid,
                unroll=True,
            )

    def test_layer_correctness(self):
        """Verify LSTM layer produces correct output (falls back on CPU)."""
        from keras.src import initializers
        from keras.src import layers

        sequence = np.arange(72).reshape((3, 6, 4)).astype("float32")
        layer = layers.LSTM(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.6288687, 0.6288687, 0.6288687],
                    [0.86899155, 0.86899155, 0.86899155],
                    [0.9460773, 0.9460773, 0.9460773],
                ]
            ),
            output,
            atol=1e-5,
        )

    def test_layer_go_backwards(self):
        from keras.src import initializers
        from keras.src import layers

        sequence = np.arange(72).reshape((3, 6, 4)).astype("float32")
        layer = layers.LSTM(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.35622165, 0.35622165, 0.35622165],
                    [0.74789524, 0.74789524, 0.74789524],
                    [0.8872726, 0.8872726, 0.8872726],
                ]
            ),
            output,
            atol=1e-5,
        )

    def test_layer_return_state(self):
        from keras.src import initializers
        from keras.src import layers

        sequence = np.arange(24).reshape((2, 4, 3)).astype("float32")
        layer = layers.LSTM(
            2,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            return_state=True,
        )
        output, state_h, state_c = layer(sequence)
        self.assertAllClose(output, state_h, atol=1e-5)
        self.assertEqual(state_c.shape, (2, 2))
