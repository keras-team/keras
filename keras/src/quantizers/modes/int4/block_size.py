"""What an int4 block size implies: per-channel or grouped, and the scheme."""

from keras.src.quantizers.quantized_weight import WeightScheme


def is_per_channel(block_size):
    """Whether `block_size` selects per-channel (ungrouped) quantization.

    `block_size` is validated to be `None`, `-1`, or a positive integer by
    both `Int4QuantizationConfig` and the policy-string codec, so `None`
    and `-1` are the two spellings of per-channel.
    """
    return block_size is None or block_size == -1


def is_grouped(block_size):
    """Whether `block_size` selects sub-channel (grouped) quantization."""
    return not is_per_channel(block_size)


def int4_scheme(block_size):
    """The int4 scheme for a block size: per-channel or grouped."""
    if is_per_channel(block_size):
        # Symmetric codes with a per-channel divisor scale.
        return WeightScheme(code_range=(-8, 7), scale_form="divisor")
    # Asymmetric codes with a multiplier scale and a zero point per group.
    return WeightScheme(
        code_range=(-8, 7),
        scale_form="multiplier",
        has_zero_point=True,
        group_size=block_size,
    )
