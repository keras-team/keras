import numpy as np

from keras.src.api_export import keras_export


@keras_export(
    [
        "keras.utils.pad_sequences",
        "keras.preprocessing.sequence.pad_sequences",
    ]
)
def pad_sequences(
    sequences,
    maxlen=None,
    dtype="int32",
    padding="pre",
    truncating="pre",
    value=0.0,
):
    """Pads sequences to the same length.

    This function transforms a list (of length `num_samples`)
    of sequences (lists of integers)
    into a 2D NumPy array of shape `(num_samples, num_timesteps)`.
    `num_timesteps` is either the `maxlen` argument if provided,
    or the length of the longest sequence in the list.

    Sequences that are shorter than `num_timesteps`
    are padded with `value` until they are `num_timesteps` long.

    Sequences longer than `num_timesteps` are truncated
    so that they fit the desired length.

    The position where padding or truncation happens is determined by
    the arguments `padding` and `truncating`, respectively.
    Pre-padding or removing values from the beginning of the sequence is the
    default.

    >>> sequence = [[1], [2, 3], [4, 5, 6]]
    >>> keras.utils.pad_sequences(sequence)
    array([[0, 0, 1],
           [0, 2, 3],
           [4, 5, 6]], dtype=int32)

    >>> keras.utils.pad_sequences(sequence, value=-1)
    array([[-1, -1,  1],
           [-1,  2,  3],
           [ 4,  5,  6]], dtype=int32)

    >>> keras.utils.pad_sequences(sequence, padding='post')
    array([[1, 0, 0],
           [2, 3, 0],
           [4, 5, 6]], dtype=int32)

    >>> keras.utils.pad_sequences(sequence, maxlen=2)
    array([[0, 1],
           [2, 3],
           [5, 6]], dtype=int32)

    Args:
        sequences: List of sequences (each sequence is a list of integers).
        maxlen: Optional Int, maximum length of all sequences. If not provided,
            sequences will be padded to the length of the longest individual
            sequence.
        dtype: (Optional, defaults to `"int32"`). Type of the output sequences.
            To pad sequences with variable length strings, you can use `object`.
        padding: String, "pre" or "post" (optional, defaults to `"pre"`):
            pad either before or after each sequence.
        truncating: String, "pre" or "post" (optional, defaults to `"pre"`):
            remove values from sequences larger than
            `maxlen`, either at the beginning or at the end of the sequences.
        value: Float or String, padding value. (Optional, defaults to `0.`)

    Returns:
        NumPy array with shape `(len(sequences), maxlen)`
    """
    if not hasattr(sequences, "__len__"):
        raise ValueError("`sequences` must be iterable.")
    num_samples = len(sequences)

    lengths = []
    sample_shape = ()
    flag = True

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.

    for x in sequences:
        try:
            lengths.append(len(x))
            if flag and len(x):
                sample_shape = np.asarray(x).shape[1:]
                flag = False
        except TypeError as e:
            raise ValueError(
                "`sequences` must be a list of iterables. "
                f"Found non-iterable: {str(x)}"
            ) from e

    if maxlen is None:
        maxlen = np.max(lengths)

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(
        dtype, np.str_
    )
    if isinstance(value, str) and dtype is not object and not is_dtype_str:
        raise ValueError(
            f"`dtype` {dtype} is not compatible with `value`'s type: "
            f"{type(value)}\nYou should set `dtype=object` for variable length "
            "strings."
        )

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError(f'Truncating type "{truncating}" not understood')

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                f"Shape of sample {trunc.shape[1:]} of sequence at "
                f"position {idx} is different from expected shape "
                f"{sample_shape}"
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc) :] = trunc
        else:
            raise ValueError(f'Padding type "{padding}" not understood')
    return x
