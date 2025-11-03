import itertools

import numpy as np

from keras.src import tree
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter
from keras.src.utils.module_utils import grain
from keras.src.utils.module_utils import tensorflow as tf


class GrainDatasetAdapter(DataAdapter):
    """Adapter that handles `grain.DataLoader`, `grain.MapDataset` and
    `grain.IterDataset`.
    """

    def __init__(self, dataset):
        """Initialize the GrainDatasetAdapter.

        Args:
            dataset: A Grain dataset instance. Must be one of
                `grain.DataLoader`, `grain.MapDataset`, or `grain.IterDataset`.
        """

        if not isinstance(
            dataset, (grain.MapDataset, grain.IterDataset, grain.DataLoader)
        ):
            raise ValueError(
                "Expected `dataset` to be a grain.MapDataset, "
                "grain.IterDataset or grain.DataLoader. "
                f"Received: {dataset} of type {type(dataset)}"
            )

        self._dataset = dataset

        batch_size, output_signature = self._get_dataset_info(dataset)
        self._batch_size = batch_size
        self._output_signature = output_signature
        self._output_tf_signature = None

    def _get_dataset_info(self, dataset):
        """Get the `batch_size` and `output_signature` from the dataset.

        We use a small list of batches to infer the `batch_size` and
        `output_signature`.
        """
        batches = list(
            itertools.islice(
                dataset, data_adapter_utils.NUM_BATCHES_FOR_TENSOR_SPEC
            )
        )
        output_signature = data_adapter_utils.get_keras_tensor_spec(batches)
        flat_output_signature = tree.flatten(output_signature)
        batch_size = flat_output_signature[0].shape[0]
        if batch_size is not None:
            batch_size = int(batch_size)
        return batch_size, output_signature

    def get_numpy_iterator(self):
        from grain._src.python.shared_memory_array import (
            SharedMemoryArrayMetadata,
        )

        def convert_to_numpy(x):
            if isinstance(x, (np.ndarray, SharedMemoryArrayMetadata)):
                return x
            else:
                # Using `__array__` should handle `tf.Tensor`, `jax.np.ndarray`,
                # `torch.Tensor`, as well as any other tensor-like object that
                # has added numpy support.
                if hasattr(x, "__array__"):
                    if data_adapter_utils.is_torch_tensor(x):
                        x = x.cpu()
                    x = np.asarray(x)
                return x

        class ConvertToNumpy(grain.transforms.Map):
            def map(self, x):
                return tree.map_structure(
                    convert_to_numpy, x, none_is_leaf=False
                )

        if isinstance(self._dataset, (grain.MapDataset, grain.IterDataset)):
            dataset = self._dataset.map(ConvertToNumpy())
        else:
            # Instantiate a new `DataLoader`.
            dataset = grain.DataLoader(
                data_source=self._dataset._data_source,
                sampler=self._dataset._sampler,
                # Append `ConvertToNumpy`.
                operations=list(self._dataset._operations) + [ConvertToNumpy()],
                worker_count=self._dataset._multiprocessing_options.num_workers,
                worker_buffer_size=self._dataset._multiprocessing_options.per_worker_buffer_size,
                shard_options=self._dataset._shard_options,
                read_options=self._dataset._read_options,
                enable_profiling=self._dataset._multiprocessing_options.enable_profiling,
            )
        return dataset

    def get_jax_iterator(self):
        def convert_to_jax_compatible(x):
            if data_adapter_utils.is_scipy_sparse(x):
                x = data_adapter_utils.scipy_sparse_to_jax_sparse(x)
            elif data_adapter_utils.is_tensorflow_sparse(x):
                x = data_adapter_utils.tf_sparse_to_jax_sparse(x)
            return x

        class ConvertToJaxCompatible(grain.transforms.Map):
            def map(self, x):
                return tree.map_structure(
                    convert_to_jax_compatible, x, none_is_leaf=False
                )

        if isinstance(self._dataset, (grain.MapDataset, grain.IterDataset)):
            dataset = self._dataset.map(ConvertToJaxCompatible())
        else:
            # Instantiate a new `DataLoader`.
            dataset = grain.DataLoader(
                data_source=self._dataset._data_source,
                sampler=self._dataset._sampler,
                # Append `ConvertToJaxCompatible`.
                operations=list(self._dataset._operations)
                + [ConvertToJaxCompatible()],
                worker_count=self._dataset._multiprocessing_options.num_workers,
                worker_buffer_size=self._dataset._multiprocessing_options.per_worker_buffer_size,
                shard_options=self._dataset._shard_options,
                read_options=self._dataset._read_options,
                enable_profiling=self._dataset._multiprocessing_options.enable_profiling,
            )
        return dataset

    def get_tf_dataset(self):
        def convert_to_tf(x):
            if x is None:
                return tf.experimental.Optional.empty(None)
            if data_adapter_utils.is_scipy_sparse(x):
                x = data_adapter_utils.scipy_sparse_to_tf_sparse(x)
            elif data_adapter_utils.is_jax_sparse(x):
                x = data_adapter_utils.jax_sparse_to_tf_sparse(x)
            return x

        class ConvertToTF(grain.transforms.Map):
            def map(self, x):
                return tree.map_structure(convert_to_tf, x)

        # `tf.data.Dataset.from_generator` does not support lists as output.
        # We convert lists to tuples.
        class ListToTuple(grain.transforms.Map):
            def map(self, x):
                return tree.lists_to_tuples(x)

        if isinstance(self._dataset, (grain.MapDataset, grain.IterDataset)):
            dataset = self._dataset.map(ConvertToTF())
            dataset = dataset.map(ListToTuple())
        else:
            # Instantiate a new `DataLoader`.
            dataset = grain.DataLoader(
                data_source=self._dataset._data_source,
                sampler=self._dataset._sampler,
                # Append `ConvertToTF` and `ListToTuple`.
                operations=list(self._dataset._operations)
                + [ConvertToTF(), ListToTuple()],
                worker_count=self._dataset._multiprocessing_options.num_workers,
                worker_buffer_size=self._dataset._multiprocessing_options.per_worker_buffer_size,
                shard_options=self._dataset._shard_options,
                read_options=self._dataset._read_options,
                enable_profiling=self._dataset._multiprocessing_options.enable_profiling,
            )

        if self._output_tf_signature is None:
            self._output_tf_signature = tree.map_structure(
                data_adapter_utils.convert_to_tf_tensor_spec,
                self._output_signature,
            )

        return tf.data.Dataset.from_generator(
            lambda: dataset, output_signature=self._output_tf_signature
        )

    def get_torch_dataloader(self):
        import torch.utils.data as torch_data

        class ConverterIterableDataset(torch_data.IterableDataset):
            def __init__(self, iterable):
                super().__init__()
                self.iterable = iterable

            def __iter__(self):
                return iter(self.iterable)

        # `batch_size=None` indicates that we should not re-batch
        return torch_data.DataLoader(
            ConverterIterableDataset(self._dataset), batch_size=None
        )

    @property
    def builtin_prefetch(self):
        return True

    @property
    def num_batches(self):
        return None

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def has_partial_batch(self):
        return None

    @property
    def partial_batch_size(self):
        return None
