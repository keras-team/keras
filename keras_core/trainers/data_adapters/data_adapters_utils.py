import numpy as np
import tensorflow as tf


def handle_partial_sample_weights(
    outputs, sample_weights, sample_weight_modes, check_all_flat=False
):
    """Adds 1.0 as sample weights for the outputs for which there is no weight.

    Args:
      outputs: List of model outputs.
      sample_weights: List of sample weight inputs.
      sample_weight_modes: List of sample weight modes or None.
      check_all_flat: Ensure that inputs are not nested structures. This is not
        a free check, so we may not want to run it eagerly every iteration.

    Returns:
      Tuple of sample weights, one sample weight for every output, and booleans
      describing the raw sample weights.
    """
    if not isinstance(sample_weights, (list, tuple)):
        any_sample_weight = sample_weights is not None
        partial_sample_weight = any_sample_weight and sample_weights is None
    else:
        any_sample_weight = sample_weights is not None and any(
            w is not None for w in sample_weights
        )
        partial_sample_weight = any_sample_weight and any(
            w is None for w in sample_weights
        )

    if not any_sample_weight:
        return None, any_sample_weight, partial_sample_weight

    if not partial_sample_weight:
        return sample_weights, any_sample_weight, partial_sample_weight

    if check_all_flat:
        tf.nest.assert_same_structure(
            list_to_tuple(sample_weights),
            list_to_tuple(tf.nest.flatten(sample_weights)),
        )
        tf.nest.assert_same_structure(
            list_to_tuple(outputs), list_to_tuple(tf.nest.flatten(outputs))
        )
        if sample_weight_modes is not None:
            tf.nest.assert_same_structure(
                sample_weight_modes, tf.nest.flatten(sample_weight_modes)
            )

    new_sample_weights = []
    for i, sw in enumerate(sample_weights):
        if sw is None:
            as_numpy = isinstance(outputs[i], np.ndarray)
            output = outputs[i]
            output_shape = output.shape if as_numpy else tf.shape(output)

            is_temporal = (
                sample_weight_modes is not None
                and sample_weight_modes[i] == "temporal"
            )
            sw_shape = (
                (output_shape[0], output_shape[1])
                if is_temporal
                else (output_shape[0],)
            )

            new_sample_weights.append(
                np.ones(sw_shape) if as_numpy else tf.ones(sw_shape)
            )

        else:
            new_sample_weights.append(sw)
    return (
        list_to_tuple(new_sample_weights),
        any_sample_weight,
        partial_sample_weight,
    )


def slice_tf_tensors(arrays, indices, contiguous=True):
    """Slices batches out of provided arrays (workaround for eager TF tensors).

    Unfortunately eager tensors don't have the same slicing behavior as
    Numpy arrays (they follow the same slicing behavior as symbolic TF tensors),
    hence we cannot use `generic_utils.slice_arrays` directly
    and we have to implement this workaround based on `concat`. This has a
    performance cost.

    Args:
        arrays: Single array or list of arrays.
        indices: List of indices in the array that should be included in the
            output batch.
        contiguous: Boolean flag indicating whether the indices are contiguous.

    Returns:
        Slice of data (either single array or list of arrays).
    """
    converted_to_list = False
    if not isinstance(arrays, list):
        converted_to_list = True
        arrays = [arrays]
    if any(tf.is_tensor(x) for x in arrays):
        if not contiguous:
            entries = [[x[i : i + 1] for i in indices] for x in arrays]
            slices = [tf.concat(x, axis=0) for x in entries]
        else:
            slices = [x[indices[0] : indices[-1] + 1] for x in arrays]
    else:
        slices = slice_arrays(arrays, indices)

    if converted_to_list:
        slices = slices[0]
    return slices


def list_to_tuple(maybe_list):
    """Datasets will stack the list of tensor, so switch them to tuples."""
    if isinstance(maybe_list, list):
        return tuple(maybe_list)
    return maybe_list


def slice_arrays(arrays, start=None, stop=None):
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `slice_arrays(x, indices)`

    Args:
        arrays: Single array or list of arrays.
        start: can be an integer index (start index) or a list/array of indices
        stop: integer (stop index); should be None if `start` was a list.

    Returns:
        A slice of the array(s).

    Raises:
        ValueError: If the value of start is a list and stop is not None.
    """
    if arrays is None:
        return [None]
    if isinstance(start, list) and stop is not None:
        raise ValueError(
            "The stop argument has to be None if the value of start "
            f"is a list. Received start={start}, stop={stop}"
        )
    elif isinstance(arrays, list):
        if hasattr(start, "__len__"):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, "shape"):
                start = start.tolist()
            return [None if x is None else x[start] for x in arrays]
        return [
            None
            if x is None
            else None
            if not hasattr(x, "__getitem__")
            else x[start:stop]
            for x in arrays
        ]
    else:
        if hasattr(start, "__len__"):
            if hasattr(start, "shape"):
                start = start.tolist()
            return arrays[start]
        if hasattr(start, "__getitem__"):
            return arrays[start:stop]
        return [None]


def check_data_cardinality(data):
    num_samples = set(int(i.shape[0]) for i in tf.nest.flatten(data))
    if len(num_samples) > 1:
        msg = (
            "Data cardinality is ambiguous. "
            "Make sure all arrays contain the same number of samples."
        )
        for label, single_data in zip(["x", "y", "sample_weight"], data):
            sizes = ", ".join(
                str(i.shape[0]) for i in tf.nest.flatten(single_data)
            )
            msg += f"'{label}' sizes: {sizes}\n"
        raise ValueError(msg)


def sync_shuffle(data, num_samples=None):
    if num_samples is None:
        num_samples_set = set(int(i.shape[0]) for i in tf.nest.flatten(data))
        assert len(num_samples_set) == 1
        num_samples = num_samples_set.pop()
    p = np.random.permutation(num_samples)
    return tf.nest.map_structure(lambda x: x[p], data)
