"""Fixtures shared by the quantization tests.

`calibrate_layer` is the only place in the tests that constructs a
`Calibrator`, so a change to how a calibration mode quantizes one layer
is one edit here. `tiny_calibration_model` is the model the calibration
tests run `Model.quantize` on.
"""

import numpy as np

from keras.src import layers
from keras.src import models
from keras.src.quantizers import strategy_registry

# --- Calibration ----------------------------------------------------------


def calibration_config(mode, **kwargs):
    """The config of calibration `mode`, without a dataset by default.

    Args:
        mode: `"gptq"` or `"awq"`.
        **kwargs: Arguments of the mode's config class.
    """
    kwargs.setdefault("dataset", None)
    kwargs.setdefault("tokenizer", None)
    return strategy_registry.get_strategy(mode).config_cls(**kwargs)


def calibrate_layer(layer, config, *batches, solve=True):
    """Calibrates a float layer in `config.mode` on `batches` of its inputs.

    This is the only place in the tests that constructs a `Calibrator`.
    The calibrator observes the batches. With `solve=True`, it then solves
    for the codes and swaps the float layer for the quantized one in one
    step. With `solve=False`, the layer stays float and the calibrator
    only observes, so a test can read the statistics or solve later.

    Args:
        layer: The float layer to calibrate.
        config: The config of a calibration mode.
        *batches: Inputs of the layer, observed in order.
        solve: Whether to solve and quantize the layer.

    Returns:
        The calibrator.
    """
    strategy = strategy_registry.get_strategy(config.mode)
    calibrator = strategy.calibrator_cls(layer, config)
    for batch in batches:
        calibrator.observe(batch)
    if solve:
        calibrator.quantize()
    return calibrator


def calibrate_on_random_rows(layer, config, num_rows=16):
    """Calibrates a float layer in `config.mode` on random input rows.

    The calibrator observes `num_rows` rows of the contracted features,
    drawn from a standard normal distribution, and its solve swaps the
    float layer for the quantized one.

    Args:
        layer: The float layer to calibrate.
        config: The config of a calibration mode.
        num_rows: The number of input rows to observe.

    Returns:
        The calibrator.
    """
    calibrator = calibrate_layer(layer, config, solve=False)
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((num_rows, calibrator.rows))
    calibrator.observe(rows.astype("float32"))
    calibrator.quantize()
    return calibrator


def assert_calibrated(test, layer, mode):
    """`layer` holds `mode`'s codes, no float kernel and no config."""
    test.assertEqual(layer.quantization_mode, mode)
    test.assertFalse(hasattr(layer, "_kernel"))
    test.assertIsNone(layer.quantization_config)


def assert_float(test, layer):
    """`layer` is float, with its kernel and no config."""
    test.assertIsNone(layer.quantization_mode)
    test.assertFalse(layer._is_quantized)
    test.assertTrue(hasattr(layer, "_kernel"))
    test.assertIsNone(layer.quantization_config)


def calibration_statistic(calibrator):
    """The statistic a calibrator accumulates from the layer's inputs.

    GPTQ's Hessian, or AWQ's mean activation magnitudes.
    """
    if calibrator.mode == "gptq":
        return calibrator.hessian
    return calibrator.activation_magnitudes


def token_dataset(num_samples, sequence_length, vocab_size, rng):
    """`num_samples` random token batches of shape `(1, sequence_length)`."""
    return [
        rng.integers(0, vocab_size, (1, sequence_length), dtype=np.int32)
        for _ in range(num_samples)
    ]


def tiny_calibration_model(
    block_layers,
    *,
    vocab_size=48,
    sequence_length=16,
    embed_dim=8,
    head_units=4,
    dtype=None,
):
    """Embedding, one sequential block, mean pooling and a `Dense` head.

    Args:
        block_layers: The layers of the block, in order. The first one
            takes the `(batch, sequence_length, embed_dim)` embeddings.
        vocab_size: The number of tokens.
        sequence_length: The length of an input sequence.
        embed_dim: The embedding width.
        head_units: The number of outputs of the head.
        dtype: The dtype policy of every layer the function creates.

    Returns:
        `(model, structure)`, where `structure` is the model's
        `quantization_layer_structure`.
    """
    inputs = layers.Input((sequence_length,), dtype="int32")
    embedding = layers.Embedding(vocab_size, embed_dim, dtype=dtype)
    block = models.Sequential(block_layers)
    x = block(embedding(inputs))
    x = layers.GlobalAveragePooling1D(data_format="channels_last", dtype=dtype)(
        x
    )
    model = models.Model(inputs, layers.Dense(head_units, dtype=dtype)(x))
    structure = {
        "pre_block_layers": [embedding],
        "sequential_blocks": [block],
    }
    return model, structure
