import functools
import math

import numpy as np

from keras.src import tree
from keras.src.trainers.data_adapters import array_slicing
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter


class ArrayDataAdapter(DataAdapter):
    """Adapter for array-like objects, e.g. TF/JAX Tensors, NumPy arrays."""

    def __init__(
        self,
        x,
        y=None,
        sample_weight=None,
        batch_size=None,
        steps=None,
        shuffle=False,
        class_weight=None,
    ):
        if not can_convert_arrays((x, y, sample_weight)):
            raise ValueError(
                "Expected all elements of `x` to be array-like. "
                f"Received invalid types: x={x}"
            )

        if sample_weight is not None:
            if class_weight is not None:
                raise ValueError(
                    "You cannot `class_weight` and `sample_weight` "
                    "at the same time."
                )
            if tree.is_nested(y):
                if isinstance(sample_weight, (list, tuple, dict)):
                    try:
                        tree.assert_same_structure(y, sample_weight)
                    except ValueError:
                        raise ValueError(
                            "You should provide one `sample_weight` array per "
                            "output in `y`. The two structures did not match:\n"
                            f"- y: {y}\n"
                            f"- sample_weight: {sample_weight}\n"
                        )
                else:
                    is_samplewise = len(sample_weight.shape) == 1 or (
                        len(sample_weight.shape) == 2
                        and sample_weight.shape[1] == 1
                    )
                    if not is_samplewise:
                        raise ValueError(
                            "For a model with multiple outputs, when providing "
                            "a single `sample_weight` array, it should only "
                            "have one scalar score per sample "
                            "(i.e. shape `(num_samples,)`). If you want to use "
                            "non-scalar sample weights, pass a `sample_weight` "
                            "argument with one array per model output."
                        )
                    # Replicate the same sample_weight array on all outputs.
                    sample_weight = tree.map_structure(
                        lambda _: sample_weight, y
                    )
        if class_weight is not None:
            if tree.is_nested(y):
                raise ValueError(
                    "`class_weight` is only supported for Models with a single "
                    "output."
                )
            sample_weight = data_adapter_utils.class_weight_to_sample_weights(
                y, class_weight
            )

        inputs = data_adapter_utils.pack_x_y_sample_weight(x, y, sample_weight)

        data_adapter_utils.check_data_cardinality(inputs)
        num_samples = set(
            i.shape[0] for i in tree.flatten(inputs) if i is not None
        ).pop()
        self._num_samples = num_samples
        self._inputs = inputs

        # If batch_size is not passed but steps is, calculate from the input
        # data.  Defaults to `32` for backwards compatibility.
        if not batch_size:
            batch_size = int(math.ceil(num_samples / steps)) if steps else 32

        self._size = int(math.ceil(num_samples / batch_size))
        self._batch_size = batch_size
        self._partial_batch_size = num_samples % batch_size
        self._shuffle = shuffle

    def get_numpy_iterator(self):
        inputs = array_slicing.convert_to_sliceable(
            self._inputs, target_backend="numpy"
        )

        def slice_and_convert_to_numpy(sliceable, indices=None):
            x = sliceable[indices]
            x = sliceable.convert_to_numpy(x)
            return x

        return self._get_iterator(slice_and_convert_to_numpy, inputs)

    def get_tf_dataset(self):
        from keras.src.utils.module_utils import tensorflow as tf

        shuffle = self._shuffle
        batch_size = self._batch_size
        num_samples = self._num_samples
        num_full_batches = int(self._num_samples // batch_size)

        # Vectorized version of shuffle.
        # This is a performance improvement over using `from_tensor_slices`.
        # The indices of the data are shuffled and batched, and these indices
        # are then zipped with the data and used to extract a batch of the data
        # at each step. The performance improvements here come from:
        # 1. vectorized batch using gather
        # 2. parallelized map
        # 3. pipelined permutation generation
        # 4. optimized permutation batching
        # 5. disabled static optimizations

        indices_dataset = tf.data.Dataset.range(1)

        def permutation(_):
            # It turns out to be more performant to make a new set of indices
            # rather than reusing the same range Tensor. (presumably because of
            # buffer forwarding.)
            indices = tf.range(num_samples, dtype=tf.int64)
            if shuffle and shuffle != "batch":
                indices = tf.random.shuffle(indices)
            return indices

        # We prefetch a single element. Computing large permutations can take
        # quite a while so we don't want to wait for prefetching over an epoch
        # boundary to trigger the next permutation. On the other hand, too many
        # simultaneous shuffles can contend on a hardware level and degrade all
        # performance.
        indices_dataset = indices_dataset.map(permutation).prefetch(1)

        def slice_batch_indices(indices):
            """Convert a Tensor of indices into a dataset of batched indices.

            This step can be accomplished in several ways. The most natural is
            to slice the Tensor in a Dataset map. (With a condition on the upper
            index to handle the partial batch.) However it turns out that
            coercing the Tensor into a shape which is divisible by the batch
            size (and handling the last partial batch separately) allows for a
            much more favorable memory access pattern and improved performance.

            Args:
                indices: Tensor which determines the data order for an entire
                    epoch.

            Returns:
                A Dataset of batched indices.
            """
            num_in_full_batch = num_full_batches * batch_size
            first_k_indices = tf.slice(indices, [0], [num_in_full_batch])
            first_k_indices = tf.reshape(
                first_k_indices, [num_full_batches, batch_size]
            )

            flat_dataset = tf.data.Dataset.from_tensor_slices(first_k_indices)
            if self._partial_batch_size:
                index_remainder = tf.data.Dataset.from_tensors(
                    tf.slice(
                        indices, [num_in_full_batch], [self._partial_batch_size]
                    )
                )
                flat_dataset = flat_dataset.concatenate(index_remainder)

            return flat_dataset

        def slice_inputs(indices_dataset, inputs):
            """Slice inputs into a Dataset of batches.

            Given a Dataset of batch indices and the unsliced inputs,
            this step slices the inputs in a parallelized fashion
            and produces a dataset of input batches.

            Args:
                indices_dataset: A Dataset of batched indices.
                inputs: A python data structure that contains the inputs,
                    targets, and possibly sample weights.

            Returns:
                A Dataset of input batches matching the batch indices.
            """
            inputs = array_slicing.convert_to_sliceable(
                self._inputs, target_backend="tensorflow"
            )
            inputs = tree.lists_to_tuples(inputs)

            dataset = tf.data.Dataset.zip(
                (indices_dataset, tf.data.Dataset.from_tensors(inputs).repeat())
            )

            def grab_batch(i, data):
                def grab_one(x):
                    if isinstance(x, array_slicing.TensorflowSparseWrapper):
                        return array_slicing.slice_tensorflow_sparse_wrapper(
                            x, i
                        )
                    if isinstance(x, (list, tuple, dict)):
                        return None
                    if tf.is_tensor(x):
                        return tf.gather(x, i, axis=0)
                    return x

                return tree.traverse(grab_one, data)

            dataset = dataset.map(
                grab_batch, num_parallel_calls=tf.data.AUTOTUNE
            )

            # Default optimizations are disabled to avoid the overhead of
            # (unnecessary) input pipeline graph serialization & deserialization
            options = tf.data.Options()
            options.experimental_optimization.apply_default_optimizations = (
                False
            )
            if self._shuffle:
                options.experimental_external_state_policy = (
                    tf.data.experimental.ExternalStatePolicy.IGNORE
                )
            dataset = dataset.with_options(options)
            return dataset

        indices_dataset = indices_dataset.flat_map(slice_batch_indices)
        if shuffle == "batch":
            indices_dataset = indices_dataset.map(tf.random.shuffle)

        dataset = slice_inputs(indices_dataset, self._inputs)

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = (
            tf.data.experimental.AutoShardPolicy.DATA
        )
        dataset = dataset.with_options(options)
        return dataset.prefetch(tf.data.AUTOTUNE)

    def get_jax_iterator(self):
        inputs = array_slicing.convert_to_sliceable(
            self._inputs, target_backend="jax"
        )

        def slice_and_convert_to_jax(sliceable, indices=None):
            x = sliceable[indices]
            x = sliceable.convert_to_jax_compatible(x)
            return x

        return self._get_iterator(slice_and_convert_to_jax, inputs)

    def get_torch_dataloader(self):
        import torch

        from keras.src.backend.torch.core import convert_to_tensor

        class ArrayDataset(torch.utils.data.Dataset):
            def __init__(self, array):
                self.array = array

            def __getitems__(self, indices):
                def slice_and_convert(sliceable):
                    x = sliceable[indices]
                    x = sliceable.convert_to_torch_compatible(x)
                    x = convert_to_tensor(x)
                    return x

                return tree.map_structure(
                    slice_and_convert, self.array, none_is_leaf=False
                )

            def __len__(self):
                return len(self.array[0])

        class RandomBatchSampler(torch.utils.data.Sampler):
            def __init__(self, sampler):
                self.sampler = sampler

            def __iter__(self):
                for batch in self.sampler:
                    yield [batch[i] for i in torch.randperm(len(batch))]

            def __len__(self):
                return len(self.sampler)

        if self._shuffle == "batch":
            batch_sampler = RandomBatchSampler(
                torch.utils.data.BatchSampler(
                    range(self._num_samples),
                    batch_size=self._batch_size,
                    drop_last=False,
                )
            )
        elif self._shuffle:
            batch_sampler = torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(range(self._num_samples)),
                batch_size=self._batch_size,
                drop_last=False,
            )
        else:
            batch_sampler = torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(range(self._num_samples)),
                batch_size=self._batch_size,
                drop_last=False,
            )

        # Because ArrayDataset.__getitems__ returns full batches organized in
        # the expected structure, there is nothing to collate.
        def no_op_collate(batch):
            return batch

        inputs = array_slicing.convert_to_sliceable(
            self._inputs, target_backend="torch"
        )
        dataset = ArrayDataset(inputs)
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=batch_sampler, collate_fn=no_op_collate
        )

    def _get_iterator(self, slice_and_convert_fn, inputs):
        global_permutation = None
        if self._shuffle and self._shuffle != "batch":
            global_permutation = np.random.permutation(self._num_samples)

        for i in range(self._size):
            start = i * self._batch_size
            stop = min((i + 1) * self._batch_size, self._num_samples)
            if self._shuffle == "batch":
                indices = np.random.permutation(stop - start) + start
            elif self._shuffle:
                indices = global_permutation[start:stop]
            else:
                indices = slice(start, stop)

            slice_indices_and_convert_fn = functools.partial(
                slice_and_convert_fn, indices=indices
            )
            yield tree.map_structure(
                slice_indices_and_convert_fn, inputs, none_is_leaf=False
            )

    @property
    def num_batches(self):
        return self._size

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def has_partial_batch(self):
        return self._partial_batch_size > 0

    @property
    def partial_batch_size(self):
        return self._partial_batch_size or None


def can_convert_arrays(arrays):
    """Check if array like-inputs can be handled by `ArrayDataAdapter`

    Args:
        inputs: Structure of `Tensor`s, NumPy arrays, or tensor-like.

    Returns:
        `True` if `arrays` can be handled by `ArrayDataAdapter`, `False`
        otherwise.
    """

    return all(
        tree.flatten(tree.map_structure(array_slicing.can_slice_array, arrays))
    )
