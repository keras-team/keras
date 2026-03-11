import numpy as np
import pytest

from keras.src import backend
from keras.src import testing


def _np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _np_tanh(x):
    return np.tanh(x)


def _gru_reference(
    inputs,
    initial_state,
    kernel,
    recurrent_kernel,
    bias,
    go_backwards=False,
    return_sequences=False,
):
    """Pure NumPy GRU reference implementation (reset_after=True)."""
    batch, timesteps, _ = inputs.shape
    hidden_size = recurrent_kernel.shape[0]

    if bias is not None:
        input_bias = bias[0]
        recurrent_bias = bias[1]
    else:
        input_bias = np.zeros(3 * hidden_size)
        recurrent_bias = np.zeros(3 * hidden_size)

    h = initial_state.copy()
    all_outputs = []

    indices = range(timesteps)
    if go_backwards:
        indices = reversed(indices)

    for t in indices:
        x_t = inputs[:, t, :]
        x_all = x_t @ kernel + input_bias
        h_all = h @ recurrent_kernel + recurrent_bias

        x_z, x_r, x_h = np.split(x_all, 3, axis=-1)
        h_z, h_r, h_h = np.split(h_all, 3, axis=-1)

        z = _np_sigmoid(x_z + h_z)
        r = _np_sigmoid(x_r + h_r)
        hh = _np_tanh(x_h + r * h_h)
        h = z * h + (1 - z) * hh
        all_outputs.append(h.copy())

    if go_backwards:
        all_outputs = list(reversed(all_outputs))

    outputs = np.stack(all_outputs, axis=1)
    last_output = h
    if not return_sequences:
        outputs = last_output[:, np.newaxis, :]
    return last_output, outputs, [h]


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX-specific optimized GRU tests.",
)
class JaxGRUTest(testing.TestCase):
    def _get_activations(self):
        """Return JAX-compatible activation functions."""
        from jax import numpy as jnp
        from jax.nn import sigmoid

        return jnp.tanh, sigmoid

    def _get_test_weights(self, input_size, hidden_size, use_bias=True):
        rng = np.random.RandomState(42)
        kernel = rng.randn(input_size, 3 * hidden_size).astype("float32") * 0.1
        recurrent_kernel = (
            rng.randn(hidden_size, 3 * hidden_size).astype("float32") * 0.1
        )
        if use_bias:
            bias = rng.randn(2, 3 * hidden_size).astype("float32") * 0.1
        else:
            bias = None
        return kernel, recurrent_kernel, bias

    def test_forward(self):
        from keras.src.backend.jax.rnn import gru

        tanh, sigmoid = self._get_activations()
        batch, seq_len, input_size, hidden_size = 2, 5, 4, 3
        kernel, recurrent_kernel, bias = self._get_test_weights(
            input_size, hidden_size
        )

        rng = np.random.RandomState(0)
        inputs = rng.randn(batch, seq_len, input_size).astype("float32")
        h_0 = np.zeros((batch, hidden_size), dtype="float32")

        last_output, outputs, states = gru(
            inputs,
            h_0,
            None,
            kernel,
            recurrent_kernel,
            bias,
            tanh,
            sigmoid,
        )

        ref_last, _, _ = _gru_reference(
            inputs,
            h_0,
            kernel,
            recurrent_kernel,
            bias,
        )

        self.assertAllClose(last_output, ref_last, atol=1e-5)
        self.assertEqual(outputs.shape, (batch, 1, hidden_size))

    def test_return_sequences(self):
        from keras.src.backend.jax.rnn import gru

        tanh, sigmoid = self._get_activations()
        batch, seq_len, input_size, hidden_size = 2, 5, 4, 3
        kernel, recurrent_kernel, bias = self._get_test_weights(
            input_size, hidden_size
        )

        rng = np.random.RandomState(0)
        inputs = rng.randn(batch, seq_len, input_size).astype("float32")
        h_0 = np.zeros((batch, hidden_size), dtype="float32")

        last_output, outputs, states = gru(
            inputs,
            h_0,
            None,
            kernel,
            recurrent_kernel,
            bias,
            tanh,
            sigmoid,
            return_sequences=True,
        )

        ref_last, ref_out, _ = _gru_reference(
            inputs,
            h_0,
            kernel,
            recurrent_kernel,
            bias,
            return_sequences=True,
        )

        self.assertAllClose(last_output, ref_last, atol=1e-5)
        self.assertAllClose(outputs, ref_out, atol=1e-5)
        self.assertEqual(outputs.shape, (batch, seq_len, hidden_size))

    def test_go_backwards(self):
        from keras.src.backend.jax.rnn import gru

        tanh, sigmoid = self._get_activations()
        batch, seq_len, input_size, hidden_size = 2, 5, 4, 3
        kernel, recurrent_kernel, bias = self._get_test_weights(
            input_size, hidden_size
        )

        rng = np.random.RandomState(0)
        inputs = rng.randn(batch, seq_len, input_size).astype("float32")
        h_0 = np.zeros((batch, hidden_size), dtype="float32")

        last_output, outputs, _ = gru(
            inputs,
            h_0,
            None,
            kernel,
            recurrent_kernel,
            bias,
            tanh,
            sigmoid,
            go_backwards=True,
        )

        ref_last, _, _ = _gru_reference(
            inputs,
            h_0,
            kernel,
            recurrent_kernel,
            bias,
            go_backwards=True,
        )

        self.assertAllClose(last_output, ref_last, atol=1e-5)

    def test_go_backwards_return_sequences(self):
        from keras.src.backend.jax.rnn import gru

        tanh, sigmoid = self._get_activations()
        batch, seq_len, input_size, hidden_size = 2, 5, 4, 3
        kernel, recurrent_kernel, bias = self._get_test_weights(
            input_size, hidden_size
        )

        rng = np.random.RandomState(0)
        inputs = rng.randn(batch, seq_len, input_size).astype("float32")
        h_0 = np.zeros((batch, hidden_size), dtype="float32")

        last_output, outputs, _ = gru(
            inputs,
            h_0,
            None,
            kernel,
            recurrent_kernel,
            bias,
            tanh,
            sigmoid,
            go_backwards=True,
            return_sequences=True,
        )

        ref_last, ref_out, _ = _gru_reference(
            inputs,
            h_0,
            kernel,
            recurrent_kernel,
            bias,
            go_backwards=True,
            return_sequences=True,
        )

        self.assertAllClose(last_output, ref_last, atol=1e-5)
        self.assertAllClose(outputs, ref_out, atol=1e-5)

    def test_nonzero_initial_state(self):
        from keras.src.backend.jax.rnn import gru

        tanh, sigmoid = self._get_activations()
        batch, seq_len, input_size, hidden_size = 2, 5, 4, 3
        kernel, recurrent_kernel, bias = self._get_test_weights(
            input_size, hidden_size
        )

        rng = np.random.RandomState(0)
        inputs = rng.randn(batch, seq_len, input_size).astype("float32")
        h_0 = rng.randn(batch, hidden_size).astype("float32") * 0.5

        last_output, _, _ = gru(
            inputs,
            h_0,
            None,
            kernel,
            recurrent_kernel,
            bias,
            tanh,
            sigmoid,
        )

        ref_last, _, _ = _gru_reference(
            inputs,
            h_0,
            kernel,
            recurrent_kernel,
            bias,
        )

        self.assertAllClose(last_output, ref_last, atol=1e-5)

    def test_no_bias(self):
        from keras.src.backend.jax.rnn import gru

        tanh, sigmoid = self._get_activations()
        batch, seq_len, input_size, hidden_size = 2, 5, 4, 3
        kernel, recurrent_kernel, _ = self._get_test_weights(
            input_size, hidden_size, use_bias=False
        )

        rng = np.random.RandomState(0)
        inputs = rng.randn(batch, seq_len, input_size).astype("float32")
        h_0 = np.zeros((batch, hidden_size), dtype="float32")

        last_output, outputs, _ = gru(
            inputs,
            h_0,
            None,
            kernel,
            recurrent_kernel,
            None,
            tanh,
            sigmoid,
            return_sequences=True,
        )

        ref_last, ref_out, _ = _gru_reference(
            inputs,
            h_0,
            kernel,
            recurrent_kernel,
            None,
            return_sequences=True,
        )

        self.assertAllClose(last_output, ref_last, atol=1e-5)
        self.assertAllClose(outputs, ref_out, atol=1e-5)

    def test_fallback_reset_after_false(self):
        from keras.src.backend.jax.rnn import gru

        tanh, sigmoid = self._get_activations()
        batch, seq_len, input_size, hidden_size = 2, 5, 4, 3
        kernel, recurrent_kernel, bias = self._get_test_weights(
            input_size, hidden_size
        )

        inputs = np.zeros((batch, seq_len, input_size), dtype="float32")
        h_0 = np.zeros((batch, hidden_size), dtype="float32")

        with self.assertRaises(NotImplementedError):
            gru(
                inputs,
                h_0,
                None,
                kernel,
                recurrent_kernel,
                bias,
                tanh,
                sigmoid,
                reset_after=False,
            )

    def test_fallback_unroll(self):
        from keras.src.backend.jax.rnn import gru

        tanh, sigmoid = self._get_activations()
        batch, seq_len, input_size, hidden_size = 2, 5, 4, 3
        kernel, recurrent_kernel, bias = self._get_test_weights(
            input_size, hidden_size
        )

        inputs = np.zeros((batch, seq_len, input_size), dtype="float32")
        h_0 = np.zeros((batch, hidden_size), dtype="float32")

        with self.assertRaises(NotImplementedError):
            gru(
                inputs,
                h_0,
                None,
                kernel,
                recurrent_kernel,
                bias,
                tanh,
                sigmoid,
                unroll=True,
            )

    def test_fallback_mask(self):
        from keras.src.backend.jax.rnn import gru

        tanh, sigmoid = self._get_activations()
        batch, seq_len, input_size, hidden_size = 2, 5, 4, 3
        kernel, recurrent_kernel, bias = self._get_test_weights(
            input_size, hidden_size
        )

        inputs = np.zeros((batch, seq_len, input_size), dtype="float32")
        h_0 = np.zeros((batch, hidden_size), dtype="float32")
        mask = np.ones((batch, seq_len), dtype="bool")

        with self.assertRaises(NotImplementedError):
            gru(
                inputs,
                h_0,
                mask,
                kernel,
                recurrent_kernel,
                bias,
                tanh,
                sigmoid,
            )

    def test_matches_layer_output(self):
        """Verify the optimized path matches the layer's output."""
        from keras.src import initializers
        from keras.src import layers

        sequence = np.arange(72).reshape((3, 6, 4)).astype("float32")
        layer = layers.GRU(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
        )
        layer_output = layer(sequence)

        self.assertAllClose(
            np.array(
                [
                    [0.5217289, 0.5217289, 0.5217289],
                    [0.6371659, 0.6371659, 0.6371659],
                    [0.39384964, 0.39384964, 0.3938496],
                ]
            ),
            layer_output,
            atol=1e-5,
        )

    def test_matches_layer_go_backwards(self):
        from keras.src import initializers
        from keras.src import layers

        sequence = np.arange(72).reshape((3, 6, 4)).astype("float32")
        layer = layers.GRU(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
        )
        layer_output = layer(sequence)

        self.assertAllClose(
            np.array(
                [
                    [0.24406259, 0.24406259, 0.24406259],
                    [0.611516, 0.611516, 0.611516],
                    [0.3928808, 0.3928808, 0.3928808],
                ]
            ),
            layer_output,
            atol=1e-5,
        )
