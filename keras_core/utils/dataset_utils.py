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
