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


@pytest.mark.skipif(
    backend.backend() != "jax",
    reason="JAX-specific LSTM cuDNN path tests.",
)
class JaxLSTMCuDNNPathTest(testing.TestCase):
    """Tests that exercise the cuDNN code path by mocking cudnn_ok and
    jax.experimental.rnn.lstm so the weight conversion, mask handling,
    and output reshaping logic runs on CPU."""

    def _make_inputs(
        self, batch=2, seq_len=5, input_size=4, hidden_size=3, with_mask=False
    ):
        from jax import numpy as jnp

        rng = np.random.RandomState(42)
        inputs = rng.randn(batch, seq_len, input_size).astype("float32")
        h_0 = rng.randn(batch, hidden_size).astype("float32")
        c_0 = rng.randn(batch, hidden_size).astype("float32")
        kernel = rng.randn(input_size, 4 * hidden_size).astype("float32")
        rec_kernel = rng.randn(hidden_size, 4 * hidden_size).astype("float32")
        bias = rng.randn(4 * hidden_size).astype("float32")
        mask = None
        if with_mask:
            mask = np.array(
                [
                    [True, True, True, False, False],
                    [True, True, True, True, False],
                ]
            )

        def fake_jax_lstm(inputs, h_0, c_0, weights, seq_lengths, **kwargs):
            hs = kwargs["hidden_size"]
            b = inputs.shape[0]
            sl = inputs.shape[1]
            y = jnp.ones((b, sl, hs))
            h_n = jnp.ones((1, b, hs)) * 0.5
            c_n = jnp.ones((1, b, hs)) * 0.25
            return y, h_n, c_n

        return dict(
            inputs=inputs,
            h_0=h_0,
            c_0=c_0,
            kernel=kernel,
            rec_kernel=rec_kernel,
            bias=bias,
            mask=mask,
            fake_jax_lstm=fake_jax_lstm,
        )

    def _call_lstm(
        self,
        d,
        mask=None,
        return_sequences=False,
        go_backwards=False,
    ):
        import importlib
        from unittest.mock import patch

        from keras.src import activations

        rnn_module = importlib.import_module("keras.src.backend.jax.rnn")

        with (
            patch.object(rnn_module, "cudnn_ok", return_value=True),
            patch("jax.experimental.rnn.lstm", d["fake_jax_lstm"]),
        ):
            return rnn_module.lstm(
                d["inputs"],
                d["h_0"],
                d["c_0"],
                mask if mask is not None else d["mask"],
                d["kernel"],
                d["rec_kernel"],
                d["bias"],
                activations.tanh,
                activations.sigmoid,
                return_sequences=return_sequences,
                go_backwards=go_backwards,
            )

    def test_basic_no_mask(self):
        d = self._make_inputs()
        last_output, outputs, states = self._call_lstm(d)
        self.assertEqual(last_output.shape, (2, 3))
        # return_sequences=False -> (batch, 1, hidden)
        self.assertEqual(outputs.shape, (2, 1, 3))
        self.assertEqual(states[0].shape, (2, 3))
        self.assertEqual(states[1].shape, (2, 3))

    def test_return_sequences(self):
        d = self._make_inputs()
        last_output, outputs, states = self._call_lstm(d, return_sequences=True)
        # return_sequences=True -> (batch, seq_len, hidden)
        self.assertEqual(outputs.shape, (2, 5, 3))

    def test_go_backwards_return_sequences(self):
        d = self._make_inputs()
        last_output, outputs, states = self._call_lstm(
            d, return_sequences=True, go_backwards=True
        )
        self.assertEqual(outputs.shape, (2, 5, 3))

    def test_with_mask(self):
        d = self._make_inputs(with_mask=True)
        last_output, outputs, states = self._call_lstm(d, mask=d["mask"])
        # With mask, last_output comes from h_n.
        self.assertAllClose(last_output, np.full((2, 3), 0.5))
        self.assertEqual(outputs.shape, (2, 1, 3))

    def test_with_3d_mask(self):
        d = self._make_inputs(with_mask=True)
        # Expand mask to 3D: (batch, seq_len, 1)
        mask_3d = np.expand_dims(d["mask"], axis=-1)
        last_output, outputs, states = self._call_lstm(d, mask=mask_3d)
        self.assertEqual(last_output.shape, (2, 3))

    def test_no_bias(self):
        d = self._make_inputs()
        d["bias"] = None
        # no bias triggers the cudnn_ok check (use_bias=False), but
        # since we mock cudnn_ok, it still proceeds through the
        # zero-bias branch.
        last_output, outputs, states = self._call_lstm(d)
        self.assertEqual(last_output.shape, (2, 3))

    def test_jax_lstm_runtime_error_raises(self):
        """When the cuDNN call itself fails, it should re-raise as
        NotImplementedError."""
        import importlib
        from unittest.mock import patch

        from keras.src import activations

        rnn_module = importlib.import_module("keras.src.backend.jax.rnn")
        d = self._make_inputs()

        def failing_jax_lstm(*args, **kwargs):
            raise RuntimeError("cuDNN not available")

        with (
            patch.object(rnn_module, "cudnn_ok", return_value=True),
            patch("jax.experimental.rnn.lstm", failing_jax_lstm),
        ):
            with self.assertRaisesRegex(
                NotImplementedError, "cuDNN LSTM failed"
            ):
                rnn_module.lstm(
                    d["inputs"],
                    d["h_0"],
                    d["c_0"],
                    None,
                    d["kernel"],
                    d["rec_kernel"],
                    d["bias"],
                    activations.tanh,
                    activations.sigmoid,
                )

    def test_import_error_raises(self):
        """When jax.experimental.rnn is not importable, it should raise
        NotImplementedError."""
        import importlib
        from unittest.mock import patch

        from keras.src import activations

        rnn_module = importlib.import_module("keras.src.backend.jax.rnn")
        d = self._make_inputs()

        with (
            patch.object(rnn_module, "cudnn_ok", return_value=True),
            patch.dict("sys.modules", {"jax.experimental.rnn": None}),
        ):
            with self.assertRaises(NotImplementedError):
                rnn_module.lstm(
                    d["inputs"],
                    d["h_0"],
                    d["c_0"],
                    None,
                    d["kernel"],
                    d["rec_kernel"],
                    d["bias"],
                    activations.tanh,
                    activations.sigmoid,
                )
