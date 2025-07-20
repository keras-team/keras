import itertools
import math
import sys

import numpy as np

from keras.src import tree
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter


class GrainDatasetAdapter(DataAdapter):
    """Adapter that Grain dataset."""

    def __init__(self, dataset):
        import grain

        if not isinstance(
            dataset, (grain.MapDataset, grain.IterDataset, grain.DataLoader)
        ):
            raise ValueError(
                "Expected argument `dataset` to be a grain.MapDataset, "
                "grain.IterDataset or grain.DataLoader. "
                f"Received: {dataset} of type {type(dataset)}"
            )

        self._dataset = dataset

        # Retrieve the information.
        num_batches = None
        batch_size = None
        if isinstance(self._dataset, (grain.MapDataset, grain.IterDataset)):
            from grain._src.python.dataset.transformations import batch

            current_dataset = self._dataset
            while True:
                if isinstance(current_dataset, batch.BatchMapDataset):
                    num_batches = len(current_dataset)
                    batch_size = current_dataset._batch_size
                    break
                elif isinstance(current_dataset, batch.BatchIterDataset):
                    batch_size = current_dataset._batch_size
                    break
                else:
                    try:
                        # We may receive a non-batched dataset.
                        num_batches = len(current_dataset)
                        break
                    except TypeError:
                        pass
                if not hasattr(current_dataset, "_parent"):
                    break
                try:
                    current_dataset = current_dataset._parent
                except AssertionError:
                    break
        else:
            drop_remainder = False
            operations = self._dataset._operations
            # Retrieve the `batch_size`.
            for op in reversed(operations):
                if isinstance(op, grain.transforms.Batch):
                    batch_size = op.batch_size
                    drop_remainder = op.drop_remainder
                    break
            if batch_size is not None:
                data_source = self._dataset._data_source
                if drop_remainder:
                    num_batches = len(data_source) // batch_size
                else:
                    num_batches = math.ceil(len(data_source) / batch_size)
        self._num_batches = num_batches
        self._batch_size = batch_size
        self._output_signature = None

    def get_numpy_iterator(self):
        import grain
        from grain._src.python.dataset.transformations import map
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
                return tree.map_structure(convert_to_numpy, x)

        if isinstance(self._dataset, grain.MapDataset):
            dataset = map.MapMapDataset(self._dataset, ConvertToNumpy())
        elif isinstance(self._dataset, grain.IterDataset):
            dataset = map.MapIterDataset(self._dataset, ConvertToNumpy())
        else:

            class ConvertToNumpy(grain.transforms.Map):
                def map(self, x):
                    return tree.map_structure(convert_to_numpy, x)

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
        import grain
        from grain._src.python.dataset.transformations import map

        def convert_to_jax_compatible(x):
            if data_adapter_utils.is_scipy_sparse(x):
                x = data_adapter_utils.scipy_sparse_to_jax_sparse(x)
            elif data_adapter_utils.is_tensorflow_sparse(x):
                x = data_adapter_utils.tf_sparse_to_jax_sparse(x)
            return x

        class ConvertToJaxCompatible(grain.transforms.Map):
            def map(self, x):
                return tree.map_structure(convert_to_jax_compatible, x)

        if isinstance(self._dataset, grain.MapDataset):
            dataset = map.MapMapDataset(self._dataset, ConvertToJaxCompatible())
        elif isinstance(self._dataset, grain.IterDataset):
            dataset = map.MapIterDataset(
                self._dataset, ConvertToJaxCompatible()
            )
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
        import grain
        import tensorflow as tf
        from grain._src.python.dataset.transformations import map

        def convert_to_tf(x):
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

        if isinstance(self._dataset, grain.MapDataset):
            dataset = map.MapMapDataset(self._dataset, ConvertToTF())
            dataset = map.MapMapDataset(dataset, ListToTuple())
        elif isinstance(self._dataset, grain.IterDataset):
            dataset = map.MapIterDataset(self._dataset, ConvertToTF())
            dataset = map.MapIterDataset(dataset, ListToTuple())
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

        if self._output_signature is None:
            batches = list(
                itertools.islice(
                    self._dataset,
                    data_adapter_utils.NUM_BATCHES_FOR_TENSOR_SPEC,
                )
            )
            self._output_signature = data_adapter_utils.get_tensor_spec(batches)

        return tf.data.Dataset.from_generator(
            lambda: dataset, output_signature=self._output_signature
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
        # After calling `repeat()`, the number of batches becomes `sys.maxsize`
        # which means infinite.
        if self._num_batches == sys.maxsize:
            return None
        return self._num_batches

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def has_partial_batch(self):
        return None

    @property
    def partial_batch_size(self):
        return None
