import pytest

from keras.src import layers
from keras.src import models
from keras.src.quantizers import gptq_core
from keras.src.quantizers.gptq_config import GPTQConfig


def _get_model_no_embedding():
    """Returns a simple model that lacks an Embedding layer."""
    return models.Sequential([layers.Dense(10, input_shape=(5,))])


def _get_model_no_blocks():
    """Returns a model with an embedding layer but no subsequent container
    layers."""
    return models.Sequential([layers.Embedding(100, 10, input_shape=(5,))])


class MockTokenizer:
    """A mock tokenizer that mimics the real API for testing."""

    def tokenize(self, text):
        return [ord(c) for c in "".join(text)]

    def __call__(self, text):
        return self.tokenize(text)


architecture_test_cases = [
    (
        _get_model_no_embedding(),
        "Could not automatically find an embedding layer",
        "no_embedding_layer",
    ),
    (
        _get_model_no_blocks(),
        "Could not automatically find any transformer-like blocks",
        "no_transformer_blocks",
    ),
]


@pytest.mark.requires_trainable_backend
class TestGPTQCore:
    def test_get_dataloader_with_empty_dataset(self):
        """
        Tests that get_dataloader raises a ValueError for an empty dataset.
        """
        with pytest.raises(ValueError, match="Provided dataset is empty"):
            gptq_core.get_dataloader(
                tokenizer=MockTokenizer(), seqlen=10, dataset=[], nsamples=10
            )

    @pytest.mark.parametrize(
        "model, match_message, test_id",
        architecture_test_cases,
        ids=[case[-1] for case in architecture_test_cases],
    )
    def test_apply_gptq_with_unsupported_architectures(
        self, model, match_message, test_id
    ):
        """
        Tests that quantize fails correctly for various unsupported model
        architectures.
        """
        config = GPTQConfig(dataset=["test"], tokenizer=MockTokenizer())

        with pytest.raises(ValueError, match=match_message):
            model.quantize("gptq", config=config)
