import pytest
from absl import logging

from keras.src import layers
from keras.src import models
from keras.src.quantizers import gptq_core
from keras.src.quantizers.gptq_config import GPTQConfig

VOCAB_SIZE = 100


class MockTokenizer:
    """A mock tokenizer that mimics the real API for testing."""

    def tokenize(self, text):
        return [ord(c) % VOCAB_SIZE for c in "".join(text)]

    def __call__(self, text):
        return self.tokenize(text)


class MockEmptyBlock(layers.Layer):
    """A mock block that contains no quantizable layers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ln = layers.LayerNormalization()

    def call(self, inputs):
        return self.ln(inputs)


class MockTransformerBlock(layers.Layer):
    """A mock transformer block with a quantizable Dense layer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = layers.Dense(128)

    def call(self, inputs):
        return self.dense(inputs)


def _get_model_with_backbone(
    has_transformer_layers=True, embedding_name="embedding"
):
    """Creates a mock KerasNLP-style model with a backbone."""

    class MockBackbone(layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            if has_transformer_layers:
                self.transformer_layers = [MockTransformerBlock()]
            setattr(self, embedding_name, layers.Embedding(VOCAB_SIZE, 128))

    class MockModel(models.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.backbone = MockBackbone()

        def call(self, inputs):
            return self.backbone(inputs)

    model = MockModel()
    model.build(input_shape=(None, 10))
    return model


@pytest.mark.requires_trainable_backend
class TestGPTQCore:
    def test_get_dataloader_error_scenarios(self):
        """Tests error cases for get_dataloader."""
        with pytest.raises(ValueError, match="Provided dataset is empty"):
            gptq_core.get_dataloader(
                tokenizer=MockTokenizer(),
                sequence_length=10,
                dataset=[],
                num_samples=10,
            )
        with pytest.raises(
            TypeError,
            match=(
                "The `dataset` argument must be an iterable.*Got type: str.*"
                "Please pass the loaded dataset directly."
            ),
        ):
            gptq_core.get_dataloader(
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
                MockTransformerBlock(),
                MockTransformerBlock(),
            ]
        )
        model.build(input_shape=(None, 10))
        config = GPTQConfig(
            dataset=["test data"], tokenizer=MockTokenizer(), group_size=32
        )
        try:
            model.quantize("gptq", config=config)
        except Exception as e:
            pytest.fail(f"Multi-block quantization failed unexpectedly: {e}")

    def test_apply_gptq_with_empty_block(self, caplog):
        """Tests that a block with no quantizable layers is skipped
        correctly."""
        caplog.set_level(logging.INFO)
        model = models.Sequential(
            [layers.Embedding(VOCAB_SIZE, 10), MockEmptyBlock()]
        )
        model.build(input_shape=(None, 10))
        config = GPTQConfig(dataset=["test data"], tokenizer=MockTokenizer())
        model.quantize("gptq", config=config)
        assert "No Dense or EinsumDense layers found" in caplog.text

    architecture_test_cases = [
        (
            models.Sequential([layers.Dense(10)]),
            "Could not automatically find an embedding layer",
            "no_embedding_layer",
        ),
        (
            models.Sequential(
                [layers.Embedding(VOCAB_SIZE, 10), layers.Dense(10)]
            ),
            "Could not automatically find any transformer-like blocks",
            "no_transformer_blocks",
        ),
        (
            _get_model_with_backbone(has_transformer_layers=False),
            "backbone does not have a 'transformer_layers' attribute",
            "backbone_no_layers",
        ),
        (
            _get_model_with_backbone(embedding_name="wrong_name"),
            "Could not automatically find an embedding layer in the model",
            "backbone_no_embedding",
        ),
    ]

    @pytest.mark.parametrize(
        "model, match_message, test_id",
        architecture_test_cases,
        ids=[case[-1] for case in architecture_test_cases],
    )
    def test_apply_gptq_with_unsupported_architectures(
        self, model, match_message, test_id
    ):
        """Tests that quantize fails correctly for various unsupported
        model architectures."""
        if not model.built:
            model.build(input_shape=(None, 10))

        config = GPTQConfig(dataset=["test"], tokenizer=MockTokenizer())
        with pytest.raises(ValueError, match=match_message):
            model.quantize("gptq", config=config)
