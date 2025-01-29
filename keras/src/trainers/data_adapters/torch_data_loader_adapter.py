import itertools

import numpy as np

from keras.src import tree
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter


class TorchDataLoaderAdapter(DataAdapter):
    """Adapter that handles `torch.utils.data.DataLoader`."""

    def __init__(self, dataloader):
        import torch

        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise ValueError(
                f"Expected argument `dataloader` to be an instance of"
                f"`torch.utils.data.DataLoader`. Received: {dataloader}"
            )

        self._dataloader = dataloader
        self._output_signature = None
        self._batch_size = dataloader.batch_size
        self._num_batches = None
        self._partial_batch_size = None
        if hasattr(dataloader.dataset, "__len__"):
            self._num_batches = len(dataloader)
            if self._batch_size is not None:
                self._partial_batch_size = (
                    len(dataloader.dataset) % self._batch_size
                )

    def get_numpy_iterator(self):
        for batch in self._dataloader:
            # shared memory using `np.asarray`
            yield tuple(
                tree.map_structure(lambda x: np.asarray(x.cpu()), batch)
            )

    def get_jax_iterator(self):
        # We use numpy as an intermediary because it is faster.
        return self.get_numpy_iterator()

    def get_tf_dataset(self):
        from keras.src.utils.module_utils import tensorflow as tf

        if self._output_signature is None:
            batches = list(
                itertools.islice(
                    self._dataloader,
                    data_adapter_utils.NUM_BATCHES_FOR_TENSOR_SPEC,
                )
            )
            self._output_signature = tuple(
                data_adapter_utils.get_tensor_spec(batches)
            )
        return tf.data.Dataset.from_generator(
            self.get_numpy_iterator,
            output_signature=self._output_signature,
        )

    def get_torch_dataloader(self):
        return self._dataloader

    @property
    def num_batches(self):
        return self._num_batches

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def has_partial_batch(self):
        if self._partial_batch_size:
            return self._partial_batch_size > 0
        else:
            return None

    @property
    def partial_batch_size(self):
        return self._partial_batch_size
