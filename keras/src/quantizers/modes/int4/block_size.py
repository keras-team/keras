"""The two spellings of an int4 block size: per-channel or grouped."""


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
