import random
import time
import warnings

import numpy as np

from keras_core.api_export import keras_core_export
from keras_core.utils.module_utils import tensorflow as tf


@keras_core_export("keras_core.utils.split_dataset")
def split_dataset(
    dataset, left_size=None, right_size=None, shuffle=False, seed=None
):
    """Splits a dataset into a left half and a right half (e.g. train / test).

    Args:
        dataset:
            A `tf.data.Dataset`, a `torch.utils.data.Dataset` object,
            or a list/tuple of arrays with the same length.
        left_size: If float (in the range `[0, 1]`), it signifies
            the fraction of the data to pack in the left dataset. If integer, it
            signifies the number of samples to pack in the left dataset. If
            `None`, defaults to the complement to `right_size`.
            Defaults to `None`.
        right_size: If float (in the range `[0, 1]`), it signifies
            the fraction of the data to pack in the right dataset.
            If integer, it signifies the number of samples to pack
            in the right dataset.
            If `None`, defaults to the complement to `left_size`.
            Defaults to `None`.
        shuffle: Boolean, whether to shuffle the data before splitting it.
        seed: A random seed for shuffling.

    Returns:
        A tuple of two `tf.data.Dataset` objects:
        the left and right splits.

    Example:

    >>> data = np.random.random(size=(1000, 4))
    >>> left_ds, right_ds = keras_core.utils.split_dataset(data, left_size=0.8)
    >>> int(left_ds.cardinality())
    800
    >>> int(right_ds.cardinality())
    200
    """
    dataset_type_spec = _get_type_spec(dataset)

    if dataset_type_spec is None:
        raise TypeError(
            "The `dataset` argument must be either"
            "a `tf.data.Dataset`, a `torch.utils.data.Dataset`"
            "object, or a list/tuple of arrays. "
            f"Received: dataset={dataset} of type {type(dataset)}"
        )

    if right_size is None and left_size is None:
        raise ValueError(
            "At least one of the `left_size` or `right_size` "
            "must be specified. Received: left_size=None and "
            "right_size=None"
        )

    dataset_as_list = _convert_dataset_to_list(dataset, dataset_type_spec)

    if shuffle:
        if seed is None:
            seed = random.randint(0, int(1e6))
        random.seed(seed)
        random.shuffle(dataset_as_list)

    total_length = len(dataset_as_list)

    left_size, right_size = _rescale_dataset_split_sizes(
        left_size, right_size, total_length
    )
    left_split = list(dataset_as_list[:left_size])
    right_split = list(dataset_as_list[-right_size:])

    left_split = _restore_dataset_from_list(
        left_split, dataset_type_spec, dataset
    )
    right_split = _restore_dataset_from_list(
        right_split, dataset_type_spec, dataset
    )

    left_split = tf.data.Dataset.from_tensor_slices(left_split)
    right_split = tf.data.Dataset.from_tensor_slices(right_split)

    # apply batching to the splits if the dataset is batched
    if dataset_type_spec is tf.data.Dataset and is_batched(dataset):
        batch_size = get_batch_size(dataset)
        if batch_size is not None:
            left_split = left_split.batch(batch_size)
            right_split = right_split.batch(batch_size)

    left_split = left_split.prefetch(tf.data.AUTOTUNE)
    right_split = right_split.prefetch(tf.data.AUTOTUNE)
    return left_split, right_split


def _convert_dataset_to_list(
    dataset,
    dataset_type_spec,
    data_size_warning_flag=True,
    ensure_shape_similarity=True,
):
    """Convert `dataset` object to a list of samples.

    Args:
        dataset: A `tf.data.Dataset`, a `torch.utils.data.Dataset` object,
            or a list/tuple of arrays.
        dataset_type_spec: the type of the dataset.
        data_size_warning_flag: If set to `True`, a warning will
            be issued if the dataset takes longer than 10 seconds to iterate.
            Defaults to `True`.
        ensure_shape_similarity: If set to `True`, the shape of
            the first sample will be used to validate the shape of rest of the
            samples. Defaults to `True`.

    Returns:
        List: A list of samples.
    """
    dataset_iterator = _get_data_iterator_from_dataset(
        dataset, dataset_type_spec
    )
    dataset_as_list = []

    start_time = time.time()
    for sample in _get_next_sample(
        dataset_iterator,
        ensure_shape_similarity,
        data_size_warning_flag,
        start_time,
    ):
        if dataset_type_spec in [tuple, list]:
            # The try-except here is for NumPy 1.24 compatibility, see:
            # https://numpy.org/neps/nep-0034-infer-dtype-is-object.html
            try:
                arr = np.array(sample)
            except ValueError:
                arr = np.array(sample, dtype=object)
            dataset_as_list.append(arr)
        else:
            dataset_as_list.append(sample)

    return dataset_as_list


def _get_data_iterator_from_dataset(dataset, dataset_type_spec):
    """Get the iterator from a dataset.

    Args:
        dataset: A `tf.data.Dataset`, a `torch.utils.data.Dataset` object,
            or a list/tuple of arrays.
        dataset_type_spec: The type of the dataset.

    Returns:
        iterator: An `iterator` object.
    """
    if dataset_type_spec == list:
        if len(dataset) == 0:
            raise ValueError(
                "Received an empty list dataset. "
                "Please provide a non-empty list of arrays."
            )

        if _get_type_spec(dataset[0]) is np.ndarray:
            expected_shape = dataset[0].shape
            for i, element in enumerate(dataset):
                if np.array(element).shape[0] != expected_shape[0]:
                    raise ValueError(
                        "Received a list of NumPy arrays with different "
                        f"lengths. Mismatch found at index {i}, "
                        f"Expected shape={expected_shape} "
                        f"Received shape={np.array(element).shape}."
                        "Please provide a list of NumPy arrays with "
                        "the same length."
                    )
        else:
            raise ValueError(
                "Expected a list of `numpy.ndarray` objects,"
                f"Received: {type(dataset[0])}"
            )

        return iter(zip(*dataset))
    elif dataset_type_spec == tuple:
        if len(dataset) == 0:
            raise ValueError(
                "Received an empty list dataset."
                "Please provide a non-empty tuple of arrays."
            )

        if _get_type_spec(dataset[0]) is np.ndarray:
            expected_shape = dataset[0].shape
            for i, element in enumerate(dataset):
                if np.array(element).shape[0] != expected_shape[0]:
                    raise ValueError(
                        "Received a tuple of NumPy arrays with different "
                        f"lengths. Mismatch found at index {i}, "
                        f"Expected shape={expected_shape} "
                        f"Received shape={np.array(element).shape}."
                        "Please provide a tuple of NumPy arrays with "
                        "the same length."
                    )
        else:
            raise ValueError(
                "Expected a tuple of `numpy.ndarray` objects, "
                f"Received: {type(dataset[0])}"
            )

        return iter(zip(*dataset))
    elif dataset_type_spec == tf.data.Dataset:
        if is_batched(dataset):
            dataset = dataset.unbatch()
        return iter(dataset)
    # torch dataset iterator might be required to change
    elif is_torch_dataset(dataset):
        return iter(dataset)
    elif dataset_type_spec == np.ndarray:
        return iter(dataset)
    raise ValueError(f"Invalid dataset_type_spec: {dataset_type_spec}")


def _get_next_sample(
    dataset_iterator,
    ensure_shape_similarity,
    data_size_warning_flag,
    start_time,
):
    """Yield data samples from the `dataset_iterator`.

    Args:
        dataset_iterator: An `iterator` object.
        ensure_shape_similarity: If set to `True`, the shape of
            the first sample will be used to validate the shape of rest of the
            samples. Defaults to `True`.
        data_size_warning_flag: If set to `True`, a warning will
            be issued if the dataset takes longer than 10 seconds to iterate.
            Defaults to `True`.
        start_time (float): the start time of the dataset iteration. this is
            used only if `data_size_warning_flag` is set to true.

    Yields:
        data_sample: The next sample.
    """
    try:
        dataset_iterator = iter(dataset_iterator)
        first_sample = next(dataset_iterator)
        if isinstance(first_sample, (tf.Tensor, np.ndarray)) or is_torch_tensor(
            first_sample
        ):
            first_sample_shape = np.array(first_sample).shape
        else:
            first_sample_shape = None
            ensure_shape_similarity = False
        yield first_sample
    except StopIteration:
        raise ValueError(
            "Received an empty dataset. Argument `dataset` must "
            "be a non-empty list/tuple of `numpy.ndarray` objects "
            "or `tf.data.Dataset` objects."
        )

    for i, sample in enumerate(dataset_iterator):
        if ensure_shape_similarity:
            if first_sample_shape != np.array(sample).shape:
                raise ValueError(
                    "All `dataset` samples must have same shape, "
                    f"Expected shape: {np.array(first_sample).shape} "
                    f"Received shape: {np.array(sample).shape} at index "
                    f"{i}."
                )
        if data_size_warning_flag:
            if i % 10 == 0:
                cur_time = time.time()
                # warns user if the dataset is too large to iterate within 10s
                if int(cur_time - start_time) > 10 and data_size_warning_flag:
                    warnings.warn(
                        "The dataset is taking longer than 10 seconds to "
                        "iterate over. This may be due to the size of the "
                        "dataset. Keep in mind that the `split_dataset` "
                        "utility is only for small in-memory dataset "
                        "(e.g. < 10,000 samples).",
                        category=ResourceWarning,
                        source="split_dataset",
                    )
                    data_size_warning_flag = False
        yield sample


def is_torch_tensor(value):
    if hasattr(value, "__class__"):
        for parent in value.__class__.__mro__:
            if parent.__name__ == "Tensor" and str(parent.__module__).endswith(
                "torch"
            ):
                return True
    return False


def is_torch_dataset(dataset):
    if hasattr(dataset, "__class__"):
        for parent in dataset.__class__.__mro__:
            if parent.__name__ == "Dataset" and str(
                parent.__module__
            ).startswith("torch.utils.data"):
                return True
    return False


def _rescale_dataset_split_sizes(left_size, right_size, total_length):
    """Rescale the dataset split sizes.

    We want to ensure that the sum of
    the split sizes is equal to the total length of the dataset.

    Args:
        left_size: The size of the left dataset split.
        right_size: The size of the right dataset split.
        total_length: The total length of the dataset.

    Returns:
        tuple: A tuple of rescaled `left_size` and `right_size` integers.
    """
    left_size_type = type(left_size)
    right_size_type = type(right_size)

    # check both left_size and right_size are integers or floats
    if (left_size is not None and left_size_type not in [int, float]) and (
        right_size is not None and right_size_type not in [int, float]
    ):
        raise TypeError(
            "Invalid `left_size` and `right_size` Types. Expected: "
            "integer or float or None, Received: type(left_size)="
            f"{left_size_type} and type(right_size)={right_size_type}"
        )

    # check left_size is a integer or float
    if left_size is not None and left_size_type not in [int, float]:
        raise TypeError(
            "Invalid `left_size` Type. Expected: int or float or None, "
            f"Received: type(left_size)={left_size_type}.  "
        )

    # check right_size is a integer or float
    if right_size is not None and right_size_type not in [int, float]:
        raise TypeError(
            "Invalid `right_size` Type. "
            "Expected: int or float or None,"
            f"Received: type(right_size)={right_size_type}."
        )

    # check left_size and right_size are non-zero
    if left_size == 0 and right_size == 0:
        raise ValueError(
            "Both `left_size` and `right_size` are zero. "
            "At least one of the split sizes must be non-zero."
        )

    # check left_size is non-negative and less than 1 and less than total_length
    if (
        left_size_type == int
        and (left_size <= 0 or left_size >= total_length)
        or left_size_type == float
        and (left_size <= 0 or left_size >= 1)
    ):
        raise ValueError(
            "`left_size` should be either a positive integer "
            f"smaller than {total_length}, or a float "
            "within the range `[0, 1]`. Received: left_size="
            f"{left_size}"
        )

    # check right_size is non-negative and less than 1 and less than
    # total_length
    if (
        right_size_type == int
        and (right_size <= 0 or right_size >= total_length)
        or right_size_type == float
        and (right_size <= 0 or right_size >= 1)
    ):
        raise ValueError(
            "`right_size` should be either a positive integer "
            f"and smaller than {total_length} or a float "
            "within the range `[0, 1]`. Received: right_size="
            f"{right_size}"
        )

    # check sum of left_size and right_size is less than or equal to
    # total_length
    if (
        right_size_type == left_size_type == float
        and right_size + left_size > 1
    ):
        raise ValueError(
            "The sum of `left_size` and `right_size` is greater "
            "than 1. It must be less than or equal to 1."
        )

    if left_size_type == float:
        left_size = round(left_size * total_length)
    elif left_size_type == int:
        left_size = float(left_size)

    if right_size_type == float:
        right_size = round(right_size * total_length)
    elif right_size_type == int:
        right_size = float(right_size)

    if left_size is None:
        left_size = total_length - right_size
    elif right_size is None:
        right_size = total_length - left_size

    if left_size + right_size > total_length:
        raise ValueError(
            "The sum of `left_size` and `right_size` should "
            "be smaller than the {total_length}. "
            f"Received: left_size + right_size = {left_size+right_size}"
            f"and total_length = {total_length}"
        )

    for split, side in [(left_size, "left"), (right_size, "right")]:
        if split == 0:
            raise ValueError(
                f"With `dataset` of length={total_length}, `left_size`="
                f"{left_size} and `right_size`={right_size}."
                f"Resulting {side} side dataset split will be empty. "
                "Adjust any of the aforementioned parameters"
            )

    left_size, right_size = int(left_size), int(right_size)
    return left_size, right_size


def _restore_dataset_from_list(
    dataset_as_list, dataset_type_spec, original_dataset
):
    """Restore the dataset from the list of arrays."""
    if dataset_type_spec in [tuple, list]:
        return tuple(np.array(sample) for sample in zip(*dataset_as_list))
    elif dataset_type_spec == tf.data.Dataset:
        if isinstance(original_dataset.element_spec, dict):
            restored_dataset = {}
            for d in dataset_as_list:
                for k, v in d.items():
                    if k not in restored_dataset:
                        restored_dataset[k] = [v]
                    else:
                        restored_dataset[k].append(v)
            return restored_dataset
        else:
            return tuple(np.array(sample) for sample in zip(*dataset_as_list))

    elif is_torch_dataset(original_dataset):
        return tuple(np.array(sample) for sample in zip(*dataset_as_list))
    return dataset_as_list


def is_batched(dataset):
    """Check if the `tf.data.Dataset` is batched."""
    return hasattr(dataset, "_batch_size")


def get_batch_size(dataset):
    """Get the batch size of the dataset."""
    if is_batched(dataset):
        return dataset._batch_size
    else:
        return None


def _get_type_spec(dataset):
    """Get the type spec of the dataset."""
    if isinstance(dataset, tuple):
        return tuple
    elif isinstance(dataset, list):
        return list
    elif isinstance(dataset, np.ndarray):
        return np.ndarray
    elif isinstance(dataset, dict):
        return dict
    elif isinstance(dataset, tf.data.Dataset):
        return tf.data.Dataset
    elif is_torch_dataset(dataset):
        from torch.utils.data import Dataset as torchDataset

        return torchDataset
    else:
        return None


@keras_core_export(
    [
        "keras_core.utils.image_dataset_from_directory",
        "keras_core.preprocessing.image_dataset_from_directory",
    ]
)
def image_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    color_mode="rgb",
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=False,
):
    """Generates a `tf.data.Dataset` from image files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_image_1.jpg
    ......a_image_2.jpg
    ...class_b/
    ......b_image_1.jpg
    ......b_image_2.jpg
    ```

    Then calling `image_dataset_from_directory(main_directory,
    labels='inferred')` will return a `tf.data.Dataset` that yields batches of
    images from the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Supported image formats: `.jpeg`, `.jpg`, `.png`, `.bmp`, `.gif`.
    Animated gifs are truncated to the first frame.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain
            subdirectories, each containing images for a class.
            Otherwise, the directory structure is ignored.
        labels: Either `"inferred"`
            (labels are generated from the directory structure),
            `None` (no labels),
            or a list/tuple of integer labels of the same size as the number of
            image files found in the directory. Labels should be sorted
            according to the alphanumeric order of the image file paths
            (obtained via `os.walk(directory)` in Python).
        label_mode: String describing the encoding of `labels`. Options are:
            - `"int"`: means that the labels are encoded as integers
                (e.g. for `sparse_categorical_crossentropy` loss).
            - `"categorical"` means that the labels are
                encoded as a categorical vector
                (e.g. for `categorical_crossentropy` loss).
            - `"binary"` means that the labels (there can be only 2)
                are encoded as `float32` scalars with values 0 or 1
                (e.g. for `binary_crossentropy`).
            - `None` (no labels).
        class_names: Only valid if `labels` is `"inferred"`.
            This is the explicit list of class names
            (must match names of subdirectories). Used to control the order
            of the classes (otherwise alphanumerical order is used).
        color_mode: One of `"grayscale"`, `"rgb"`, `"rgba"`.
            Defaults to `"rgb"`. Whether the images will be converted to
            have 1, 3, or 4 channels.
        batch_size: Size of the batches of data. Defaults to 32.
            If `None`, the data will not be batched
            (the dataset will yield individual samples).
        image_size: Size to resize images to after they are read from disk,
            specified as `(height, width)`. Defaults to `(256, 256)`.
            Since the pipeline processes batches of images that must all have
            the same size, this must be provided.
        shuffle: Whether to shuffle the data. Defaults to `True`.
            If set to `False`, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: Subset of the data to return.
            One of `"training"`, `"validation"`, or `"both"`.
            Only used if `validation_split` is set.
            When `subset="both"`, the utility returns a tuple of two datasets
            (the training and validation datasets respectively).
        interpolation: String, the interpolation method used when
            resizing images. Defaults to `"bilinear"`.
            Supports `"bilinear"`, `"nearest"`, `"bicubic"`, `"area"`,
            `"lanczos3"`, `"lanczos5"`, `"gaussian"`, `"mitchellcubic"`.
        follow_links: Whether to visit subdirectories pointed to by symlinks.
            Defaults to `False`.
        crop_to_aspect_ratio: If `True`, resize the images without aspect
            ratio distortion. When the original aspect ratio differs from the
            target aspect ratio, the output image will be cropped so as to
            return the largest possible window in the image
            (of size `image_size`) that matches the target aspect ratio. By
            default (`crop_to_aspect_ratio=False`), aspect ratio may not be
            preserved.

    Returns:

    A `tf.data.Dataset` object.

    - If `label_mode` is `None`, it yields `float32` tensors of shape
        `(batch_size, image_size[0], image_size[1], num_channels)`,
        encoding images (see below for rules regarding `num_channels`).
    - Otherwise, it yields a tuple `(images, labels)`, where `images` has
        shape `(batch_size, image_size[0], image_size[1], num_channels)`,
        and `labels` follows the format described below.

    Rules regarding labels format:

    - if `label_mode` is `"int"`, the labels are an `int32` tensor of shape
        `(batch_size,)`.
    - if `label_mode` is `"binary"`, the labels are a `float32` tensor of
        1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `"categorical"`, the labels are a `float32` tensor
        of shape `(batch_size, num_classes)`, representing a one-hot
        encoding of the class index.

    Rules regarding number of channels in the yielded images:

    - if `color_mode` is `"grayscale"`,
        there's 1 channel in the image tensors.
    - if `color_mode` is `"rgb"`,
        there are 3 channels in the image tensors.
    - if `color_mode` is `"rgba"`,
        there are 4 channels in the image tensors.
    """
    # TODO: long-term, port implementation.
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        color_mode=color_mode,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        interpolation=interpolation,
        follow_links=follow_links,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
    )


@keras_core_export(
    [
        "keras_core.utils.timeseries_dataset_from_array",
        "keras_core.preprocessing.timeseries_dataset_from_array",
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
    # TODO: long-term, port implementation.
    return tf.keras.utils.timeseries_dataset_from_array(
        data,
        targets,
        sequence_length,
        sequence_stride=sequence_stride,
        sampling_rate=sampling_rate,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        start_index=start_index,
        end_index=end_index,
    )


@keras_core_export(
    [
        "keras_core.utils.text_dataset_from_directory",
        "keras_core.preprocessing.text_dataset_from_directory",
    ]
)
def text_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    max_length=None,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
):
    """Generates a `tf.data.Dataset` from text files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_text_1.txt
    ......a_text_2.txt
    ...class_b/
    ......b_text_1.txt
    ......b_text_2.txt
    ```

    Then calling `text_dataset_from_directory(main_directory,
    labels='inferred')` will return a `tf.data.Dataset` that yields batches of
    texts from the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Only `.txt` files are supported at this time.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain
            subdirectories, each containing text files for a class.
            Otherwise, the directory structure is ignored.
        labels: Either `"inferred"`
            (labels are generated from the directory structure),
            `None` (no labels),
            or a list/tuple of integer labels of the same size as the number of
            text files found in the directory. Labels should be sorted according
            to the alphanumeric order of the text file paths
            (obtained via `os.walk(directory)` in Python).
        label_mode: String describing the encoding of `labels`. Options are:
            - `"int"`: means that the labels are encoded as integers
                (e.g. for `sparse_categorical_crossentropy` loss).
            - `"categorical"` means that the labels are
                encoded as a categorical vector
                (e.g. for `categorical_crossentropy` loss).
            - `"binary"` means that the labels (there can be only 2)
                are encoded as `float32` scalars with values 0 or 1
                (e.g. for `binary_crossentropy`).
            - `None` (no labels).
        class_names: Only valid if `"labels"` is `"inferred"`.
            This is the explicit list of class names
            (must match names of subdirectories). Used to control the order
            of the classes (otherwise alphanumerical order is used).
        batch_size: Size of the batches of data. Defaults to 32.
            If `None`, the data will not be batched
            (the dataset will yield individual samples).
        max_length: Maximum size of a text string. Texts longer than this will
            be truncated to `max_length`.
        shuffle: Whether to shuffle the data. Defaults to `True`.
            If set to `False`, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1,
            fraction of data to reserve for validation.
        subset: Subset of the data to return.
            One of `"training"`, `"validation"` or `"both"`.
            Only used if `validation_split` is set.
            When `subset="both"`, the utility returns a tuple of two datasets
            (the training and validation datasets respectively).
        follow_links: Whether to visits subdirectories pointed to by symlinks.
            Defaults to `False`.

    Returns:

    A `tf.data.Dataset` object.

    - If `label_mode` is `None`, it yields `string` tensors of shape
        `(batch_size,)`, containing the contents of a batch of text files.
    - Otherwise, it yields a tuple `(texts, labels)`, where `texts`
        has shape `(batch_size,)` and `labels` follows the format described
        below.

    Rules regarding labels format:

    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
        `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
        1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorical`, the labels are a `float32` tensor
        of shape `(batch_size, num_classes)`, representing a one-hot
        encoding of the class index.
    """
    # TODO: long-term, port implementation.
    return tf.keras.utils.text_dataset_from_directory(
        directory,
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        batch_size=batch_size,
        max_length=max_length,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        follow_links=follow_links,
    )


@keras_core_export("keras_core.utils.audio_dataset_from_directory")
def audio_dataset_from_directory(
    directory,
    labels="inferred",
    label_mode="int",
    class_names=None,
    batch_size=32,
    sampling_rate=None,
    output_sequence_length=None,
    ragged=False,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False,
):
    """Generates a `tf.data.Dataset` from audio files in a directory.

    If your directory structure is:

    ```
    main_directory/
    ...class_a/
    ......a_audio_1.wav
    ......a_audio_2.wav
    ...class_b/
    ......b_audio_1.wav
    ......b_audio_2.wav
    ```

    Then calling `audio_dataset_from_directory(main_directory,
    labels='inferred')`
    will return a `tf.data.Dataset` that yields batches of audio files from
    the subdirectories `class_a` and `class_b`, together with labels
    0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

    Only `.wav` files are supported at this time.

    Args:
        directory: Directory where the data is located.
            If `labels` is `"inferred"`, it should contain subdirectories,
            each containing audio files for a class. Otherwise, the directory
            structure is ignored.
        labels: Either "inferred" (labels are generated from the directory
            structure), `None` (no labels), or a list/tuple of integer labels
            of the same size as the number of audio files found in
            the directory. Labels should be sorted according to the
            alphanumeric order of the audio file paths
            (obtained via `os.walk(directory)` in Python).
        label_mode: String describing the encoding of `labels`. Options are:
            - `"int"`: means that the labels are encoded as integers (e.g. for
              `sparse_categorical_crossentropy` loss).
            - `"categorical"` means that the labels are encoded as a categorical
              vector (e.g. for `categorical_crossentropy` loss)
            - `"binary"` means that the labels (there can be only 2)
              are encoded as `float32` scalars with values 0
              or 1 (e.g. for `binary_crossentropy`).
            - `None` (no labels).
        class_names: Only valid if "labels" is `"inferred"`.
            This is the explicit list of class names
            (must match names of subdirectories). Used to control the order
            of the classes (otherwise alphanumerical order is used).
        batch_size: Size of the batches of data. Default: 32. If `None`,
            the data will not be batched
            (the dataset will yield individual samples).
        sampling_rate: Audio sampling rate (in samples per second).
        output_sequence_length: Maximum length of an audio sequence. Audio files
            longer than this will be truncated to `output_sequence_length`.
            If set to `None`, then all sequences in the same batch will
            be padded to the
            length of the longest sequence in the batch.
        ragged: Whether to return a Ragged dataset (where each sequence has its
            own length). Defaults to `False`.
        shuffle: Whether to shuffle the data. Defaults to `True`.
            If set to `False`, sorts the data in alphanumeric order.
        seed: Optional random seed for shuffling and transformations.
        validation_split: Optional float between 0 and 1, fraction of data to
            reserve for validation.
        subset: Subset of the data to return. One of `"training"`,
            `"validation"` or `"both"`. Only used if `validation_split` is set.
        follow_links: Whether to visits subdirectories pointed to by symlinks.
            Defaults to `False`.

    Returns:

    A `tf.data.Dataset` object.

    - If `label_mode` is `None`, it yields `string` tensors of shape
      `(batch_size,)`, containing the contents of a batch of audio files.
    - Otherwise, it yields a tuple `(audio, labels)`, where `audio`
      has shape `(batch_size, sequence_length, num_channels)` and `labels`
      follows the format described
      below.

    Rules regarding labels format:

    - if `label_mode` is `int`, the labels are an `int32` tensor of shape
      `(batch_size,)`.
    - if `label_mode` is `binary`, the labels are a `float32` tensor of
      1s and 0s of shape `(batch_size, 1)`.
    - if `label_mode` is `categorical`, the labels are a `float32` tensor
      of shape `(batch_size, num_classes)`, representing a one-hot
      encoding of the class index.
    """
    # TODO: long-term, port implementation.
    return tf.keras.utils.audio_dataset_from_directory(
        directory,
        labels=labels,
        label_mode=label_mode,
        class_names=class_names,
        batch_size=batch_size,
        sampling_rate=sampling_rate,
        output_sequence_length=output_sequence_length,
        ragged=ragged,
        shuffle=shuffle,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        follow_links=follow_links,
    )
