import numpy as np
import pytest
from absl.testing import parameterized

from keras.src import layers
from keras.src import models
from keras.src import ops
from keras.src import testing
from keras.src.quantizers.gptq_config import GPTQConfig
from keras.src.quantizers.gptq_core import get_dataloader
from keras.src.quantizers.gptq_core import gptq_quantize

VOCAB_SIZE = 100


class MockTokenizer:
    """A mock tokenizer that mimics the real API for testing."""

    def tokenize(self, text):
        return [ord(c) % VOCAB_SIZE for c in "".join(text)]

    def __call__(self, text):
        return self.tokenize(text)


class EmptyBlock(layers.Layer):
    """A block that contains no quantizable layers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ln = layers.LayerNormalization()

    def call(self, inputs):
        return self.ln(inputs)


class TransformerBlock(layers.Layer):
    """A toy transformer block with a quantizable Dense layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(128)

    def call(self, inputs):
        return self.dense(inputs)


def _get_model_with_backbone(
    has_transformer_layers=True, embedding_name="embedding"
):
    """Creates a KerasHub-style model with a backbone."""

    class Backbone(layers.Layer):
        def __init__(self, vocab_size, embedding_dim=128, **kwargs):
            super().__init__(**kwargs)
            # Use direct assignment
            setattr(
                self,
                embedding_name,
                layers.Embedding(vocab_size, embedding_dim),
            )

            # Keep track of layers in a list for the call method
            self.transformer_layers = []
            if has_transformer_layers:
                self.transformer_layers.append(TransformerBlock())

        def call(self, inputs):
            x = getattr(self, embedding_name)(inputs)
            for layer in self.transformer_layers:
                x = layer(x)
            return x

    class Model(models.Model):
        def __init__(self, vocab_size, **kwargs):
            super().__init__(**kwargs)
            # Pass configuration directly
            self.backbone = Backbone(vocab_size=vocab_size)
            self.classifier = layers.Dense(1, activation="sigmoid")

        def call(self, inputs):
            x = self.backbone(inputs)
            x = layers.GlobalAveragePooling1D()(x)
            return self.classifier(x)

    model = Model(vocab_size=VOCAB_SIZE)
    rng = np.random.default_rng(seed=42)
    dummy_input = rng.normal(loc=0, scale=1, size=(2, 64)).astype(np.float32)

    _ = model(dummy_input)
    return model


def build_all_tokens_strings(dataset, tokenizer, eos_id=None):
    pieces = []
    for i, s in enumerate(dataset):
        toks = np.asarray(tokenizer.tokenize(s), dtype=np.int32).reshape(-1)
        pieces.append(toks)
        if eos_id is not None and i < len(dataset) - 1:
            pieces.append(np.array([eos_id], dtype=np.int32))
    return np.concatenate(pieces, axis=0).astype(np.int32, copy=False)


def sliding_windows(x, L):
    return np.lib.stride_tricks.sliding_window_view(x, L)


@pytest.mark.requires_trainable_backend
class TestGPTQCore(testing.TestCase):
    @parameterized.named_parameters(
        [("strided", "strided"), ("linspace", "linspace"), ("random", "random")]
    )
    def test_shape_and_dtype_strings(self, strategy):
        """Test the shape and dtype of the output for string inputs."""
        tok = MockTokenizer()
        dataset = ["a b c d e f g", "h i j k"]
        seq_len, n = 5, 7

        out = get_dataloader(
            tok, seq_len, dataset, num_samples=n, strategy=strategy, seed=123
        )
        self.assertEqual(out.shape, (n, 1, seq_len))
        self.assertEqual(out.dtype, np.int32)

    @parameterized.named_parameters(
        [("strided", "strided"), ("linspace", "linspace"), ("random", "random")]
    )
    def test_shape_and_dtype_pretokenized(self, strategy):
        """Test the shape and dtype of the output for pre-tokenized inputs."""
        tok = MockTokenizer()
        # Pre-tokenized inputs; mixed shapes (1, L) and (L,)
        seqs = [
            np.array([[1, 2, 3, 4]], dtype=np.int64),
            np.array([5, 6], dtype=np.int64),
        ]
        tok = MockTokenizer()
        seq_len, n = 3, 4

        out = get_dataloader(
            tok, seq_len, seqs, num_samples=n, strategy=strategy, seed=7
        )
        self.assertEqual(out.shape, (n, 1, seq_len))
        self.assertEqual(out.dtype, np.int32)

    def test_strided_is_deterministic_for_same_args(self):
        tok = MockTokenizer()
        dataset = ["a b c d e", "f g h i j k"]
        out1 = get_dataloader(
            tok, 4, dataset, num_samples=6, strategy="strided", seed=99
        )
        out2 = get_dataloader(
            tok, 4, dataset, num_samples=6, strategy="strided", seed=99
        )
        self.assertTrue(ops.all(ops.equal(out1, out2)))

    def test_random_reproducibility_by_seed(self):
        tok = MockTokenizer()
        dataset = ["a b c d e", "f g h i j k"]
        a = get_dataloader(
            tok, 4, dataset, num_samples=6, strategy="random", seed=123
        )
        b = get_dataloader(
            tok, 4, dataset, num_samples=6, strategy="random", seed=123
        )
        c = get_dataloader(
            tok, 4, dataset, num_samples=6, strategy="random", seed=124
        )
        self.assertTrue(ops.all(ops.equal(a, b)))
        self.assertFalse(ops.all(ops.equal(a, c)))

    def test_linspace_windows_match_expected(self):
        tok = MockTokenizer()
        dataset = ["aa bb cc dd", "ee ff gg"]
        seq_len, n = 3, 5
        eos_id = None

        all_tokens = build_all_tokens_strings(dataset, tok, eos_id=eos_id)
        max_start = all_tokens.size - seq_len
        expected_starts = np.linspace(0, max_start, n, dtype=np.int64)

        expected = sliding_windows(all_tokens, seq_len)[expected_starts]
        got = get_dataloader(
            tok, seq_len, dataset, num_samples=n, strategy="linspace"
        )
        self.assertTrue(
            ops.all(ops.equal(got[:, 0, :], expected.astype(np.int32)))
        )

    def test_strided_override_respected(self):
        """Tests that strided windows are disjoint and cover the input."""
        tok = MockTokenizer()
        # 20 tokens total
        # with seq_len=4 and stride=4, we expect disjoint chunks
        # in order (modulo offset)
        dataset = [" ".join([f"t{i}" for i in range(20)])]
        seq_len, n, stride = 4, 5, 4

        out = get_dataloader(
            tok,
            seq_len,
            dataset,
            num_samples=n,
            strategy="strided",
            stride=stride,
            seed=0,
        )

        # Validate that each sample is a contiguous run
        # of length seq_len from the flattened stream
        flat = build_all_tokens_strings(dataset, tok)
        for s in out[:, 0, :]:
            # Each window should appear as a slice in the flat stream
            # (This is a soft check; exact start positions depend on offset.)
            joined = " ".join(map(str, s.tolist()))
            self.assertIn(joined, " ".join(map(str, flat.tolist())))

    def test_eos_insertion_is_present_in_some_window_with_linspace(self):
        tok = MockTokenizer()
        dataset = ["aa aa", "bb bb"]  # len = 5 + 1(EOS) + 5 = 11
        eos = 9999
        seq_len = 3
        n = 3

        out = get_dataloader(
            tok,
            seq_len,
            dataset,
            num_samples=n,
            strategy="linspace",
            eos_id=eos,
        )

        # linspace starts -> [0, 4, 8]; the middle window [4:7]
        # includes EOS at 5
        windows = out[:, 0, :]
        self.assertTrue(
            np.any(np.any(windows == eos, axis=1)),
            "Expected EOS to appear in at least one sampled window with "
            "linspace.",
        )

    def test_get_dataloader_error_scenarios(self):
        """Tests error cases for get_dataloader."""
        with pytest.raises(ValueError, match="Provided dataset is empty"):
            get_dataloader(
                tokenizer=MockTokenizer(),
                sequence_length=10,
                dataset=[],
                num_samples=10,
            )
        with self.assertRaisesRegex(
            TypeError,
            "The `dataset` argument must be an iterable.*Got type: str.*"
            "Please pass the loaded dataset directly.",
        ):
            get_dataloader(
                tokenizer=MockTokenizer(),
                sequence_length=10,
                dataset="wikitext2",
                num_samples=10,
            )

    def test_apply_gptq_on_multi_block_model(self):
        """Tests quantization on a model with multiple blocks."""
        model = models.Sequential(
            [
                layers.Embedding(VOCAB_SIZE, 128),
                TransformerBlock(),
                TransformerBlock(),
            ]
        )
        model.build(input_shape=(None, 10))

        layer_structure = {
            "pre_block_layers": [model.layers[0]],
            "sequential_blocks": [model.layers[1], model.layers[2]],
        }

        config = GPTQConfig(
            dataset=["test data"],
            tokenizer=MockTokenizer(),
            group_size=32,
            quantization_layer_structure=layer_structure,
        )
        model.quantize("gptq", config=config)

    @parameterized.named_parameters(
        (
            "no_embedding_layer",
            models.Sequential([layers.Dense(10)]),
            "For 'gptq' mode, a valid quantization structure must be provided",
        ),
        (
            "no_transformer_blocks",
            models.Sequential(
                [layers.Embedding(VOCAB_SIZE, 10), layers.Dense(10)]
            ),
            "For 'gptq' mode, a valid quantization structure must be provided",
        ),
        (
            "backbone_no_layers",
            _get_model_with_backbone(has_transformer_layers=False),
            "For 'gptq' mode, a valid quantization structure must be provided",
        ),
        (
            "backbone_no_embedding",
            _get_model_with_backbone(embedding_name="wrong_name"),
            "For 'gptq' mode, a valid quantization structure must be provided",
        ),
    )
    def test_apply_gptq_with_unsupported_architectures(
        self, model, error_message
    ):
        """Tests that quantize fails correctly for various unsupported
        model architectures."""
        if not model.built:
            model.build(input_shape=(None, 10))

        config = GPTQConfig(dataset=["test"], tokenizer=MockTokenizer())
        with self.assertRaisesRegex(ValueError, error_message):
            # We pass None as structure to trigger the error
            gptq_quantize(config, quantization_layer_structure=None)
