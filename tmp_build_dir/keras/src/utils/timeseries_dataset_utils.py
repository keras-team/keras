import numpy as np

from keras.src.api_export import keras_export
from keras.src.utils.module_utils import tensorflow as tf


@keras_export(
    [
        "keras.utils.timeseries_dataset_from_array",
        "keras.preprocessing.timeseries_dataset_from_array",
    ]
)
def timeseries_dataset_from_array(
    data,
    targets,
    sequence_length,
    sequence_stride=1,
    sampling_rate=1,
    batch_size=128,
    shuffle=False,
    seed=None,
    start_index=None,
    end_index=None,
):
    """Creates a dataset of sliding windows over a timeseries provided as array.

    This function takes in a sequence of data-points gathered at
    equal intervals, along with time series parameters such as
    length of the sequences/windows, spacing between two sequence/windows, etc.,
    to produce batches of timeseries inputs and targets.

    Args:
        data: Numpy array or eager tensor
            containing consecutive data points (timesteps).
            Axis 0 is expected to be the time dimension.
        targets: Targets corresponding to timesteps in `data`.
            `targets[i]` should be the target
            corresponding to the window that starts at index `i`
            (see example 2 below).
            Pass `None` if you don't have target data (in this case the dataset
            will only yield the input data).
        sequence_length: Length of the output sequences
            (in number of timesteps).
        sequence_stride: Period between successive output sequences.
            For stride `s`, output samples would
            start at index `data[i]`, `data[i + s]`, `data[i + 2 * s]`, etc.
        sampling_rate: Period between successive individual timesteps
            within sequences. For rate `r`, timesteps
            `data[i], data[i + r], ... data[i + sequence_length]`
            are used for creating a sample sequence.
        batch_size: Number of timeseries samples in each batch
            (except maybe the last one). If `None`, the data will not be batched
            (the dataset will yield individual samples).
        shuffle: Whether to shuffle output samples,
            or instead draw them in chronological order.
        seed: Optional int; random seed for shuffling.
        start_index: Optional int; data points earlier (exclusive)
            than `start_index` will not be used
            in the output sequences. This is useful to reserve part of the
            data for test or validation.
        end_index: Optional int; data points later (exclusive) than `end_index`
            will not be used in the output sequences.
            This is useful to reserve part of the data for test or validation.

    Returns:

    A `tf.data.Dataset` instance. If `targets` was passed, the dataset yields
    tuple `(batch_of_sequences, batch_of_targets)`. If not, the dataset yields
    only `batch_of_sequences`.

    Example 1:

    Consider indices `[0, 1, ... 98]`.
    With `sequence_length=10,  sampling_rate=2, sequence_stride=3`,
    `shuffle=False`, the dataset will yield batches of sequences
    composed of the following indices:

    ```
    First sequence:  [0  2  4  6  8 10 12 14 16 18]
    Second sequence: [3  5  7  9 11 13 15 17 19 21]
    Third sequence:  [6  8 10 12 14 16 18 20 22 24]
    ...
    Last sequence:   [78 80 82 84 86 88 90 92 94 96]
    ```

    In this case the last 2 data points are discarded since no full sequence
    can be generated to include them (the next sequence would have started
    at index 81, and thus its last step would have gone over 98).

    Example 2: Temporal regression.

    Consider an array `data` of scalar values, of shape `(steps,)`.
    To generate a dataset that uses the past 10
    timesteps to predict the next timestep, you would use:

    ```python
    input_data = data[:-10]
    targets = data[10:]
    dataset = timeseries_dataset_from_array(
        input_data, targets, sequence_length=10)
    for batch in dataset:
      inputs, targets = batch
      assert np.array_equal(inputs[0], data[:10])  # First sequence: steps [0-9]
      # Corresponding target: step 10
      assert np.array_equal(targets[0], data[10])
      break
    ```

    Example 3: Temporal regression for many-to-many architectures.

    Consider two arrays of scalar values `X` and `Y`,
    both of shape `(100,)`. The resulting dataset should consist samples with
    20 timestamps each. The samples should not overlap.
    To generate a dataset that uses the current timestamp
    to predict the corresponding target timestep, you would use:

    ```python
    X = np.arange(100)
    Y = X*2

    sample_length = 20
    input_dataset = timeseries_dataset_from_array(
        X, None, sequence_length=sample_length, sequence_stride=sample_length)
    target_dataset = timeseries_dataset_from_array(
        Y, None, sequence_length=sample_length, sequence_stride=sample_length)

    for batch in zip(input_dataset, target_dataset):
        inputs, targets = batch
        assert np.array_equal(inputs[0], X[:sample_length])

        # second sample equals output timestamps 20-40
        assert np.array_equal(targets[1], Y[sample_length:2*sample_length])
        break
    ```
    """
    if start_index:
        if start_index < 0:
            raise ValueError(
                "`start_index` must be 0 or greater. Received: "
                f"start_index={start_index}"
            )
        if start_index >= len(data):
            raise ValueError(
                "`start_index` must be lower than the length of the "
                f"data. Received: start_index={start_index}, for data "
                f"of length {len(data)}"
            )
    if end_index:
        if start_index and end_index <= start_index:
            raise ValueError(
                "`end_index` must be higher than `start_index`. "
                f"Received: start_index={start_index}, and "
                f"end_index={end_index} "
            )
        if end_index >= len(data):
            raise ValueError(
                "`end_index` must be lower than the length of the "
                f"data. Received: end_index={end_index}, for data of "
                f"length {len(data)}"
            )
        if end_index <= 0:
            raise ValueError(
                "`end_index` must be higher than 0. "
                f"Received: end_index={end_index}"
            )

    # Validate strides
    if sampling_rate <= 0:
        raise ValueError(
            "`sampling_rate` must be higher than 0. Received: "
            f"sampling_rate={sampling_rate}"
        )
    if sampling_rate >= len(data):
        raise ValueError(
            "`sampling_rate` must be lower than the length of the "
            f"data. Received: sampling_rate={sampling_rate}, for data "
            f"of length {len(data)}"
        )
    if sequence_stride <= 0:
        raise ValueError(
            "`sequence_stride` must be higher than 0. Received: "
            f"sequence_stride={sequence_stride}"
        )
    if sequence_stride >= len(data):
        raise ValueError(
            "`sequence_stride` must be lower than the length of the "
            f"data. Received: sequence_stride={sequence_stride}, for "
            f"data of length {len(data)}"
        )

    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(data)

    # Determine the lowest dtype to store start positions (to lower memory
    # usage).
    num_seqs = end_index - start_index - (sequence_length - 1) * sampling_rate
    if targets is not None:
        num_seqs = min(num_seqs, len(targets))
    if num_seqs < 2147483647:
        index_dtype = "int32"
    else:
        index_dtype = "int64"

    # Generate start positions
    start_positions = np.arange(0, num_seqs, sequence_stride, dtype=index_dtype)
    if shuffle:
        if seed is None:
            seed = np.random.randint(1e6)
        rng = np.random.RandomState(seed)
        rng.shuffle(start_positions)

    sequence_length = tf.cast(sequence_length, dtype=index_dtype)
    sampling_rate = tf.cast(sampling_rate, dtype=index_dtype)

    positions_ds = tf.data.Dataset.from_tensors(start_positions).repeat()

    # For each initial window position, generates indices of the window elements
    indices = tf.data.Dataset.zip(
        (tf.data.Dataset.range(len(start_positions)), positions_ds)
    ).map(
        lambda i, positions: tf.range(
            positions[i],
            positions[i] + sequence_length * sampling_rate,
            sampling_rate,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = sequences_from_indices(data, indices, start_index, end_index)
    if targets is not None:
        indices = tf.data.Dataset.zip(
            (tf.data.Dataset.range(len(start_positions)), positions_ds)
        ).map(
            lambda i, positions: positions[i],
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        target_ds = sequences_from_indices(
            targets, indices, start_index, end_index
        )
        dataset = tf.data.Dataset.zip((dataset, target_ds))
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    if batch_size is not None:
        if shuffle:
            # Shuffle locally at each iteration
            dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
        dataset = dataset.batch(batch_size)
    else:
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1024, seed=seed)
    return dataset


def sequences_from_indices(array, indices_ds, start_index, end_index):
    dataset = tf.data.Dataset.from_tensors(array[start_index:end_index])
    dataset = tf.data.Dataset.zip((dataset.repeat(), indices_ds)).map(
        lambda steps, inds: tf.gather(steps, inds),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset
