import numpy as np
import tree

from keras.trainers.data_adapters.data_adapter import DataAdapter


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
        self._batch_size = dataloader.batch_size
        self._size = len(dataloader)
        self._partial_batch_size = len(dataloader.dataset) % self._batch_size

    def get_numpy_iterator(self):
        for batch in self._dataloader:
            # shared memory using `np.asarray`
            yield tuple(
                tree.map_structure(lambda x: np.asarray(x.cpu()), batch)
            )

    def get_torch_dataloader(self):
        return self._dataloader

    def get_tf_dataset(self):
        from keras.utils.module_utils import tensorflow as tf

        output_signature = self.peek_and_get_tensor_spec()
        return tf.data.Dataset.from_generator(
            self.get_numpy_iterator,
            output_signature=output_signature,
        )

    def peek_and_get_tensor_spec(self):
        from keras.utils.module_utils import tensorflow as tf

        batch_data = next(iter(self._dataloader))

        def get_tensor_spec(x):
            shape = x.shape
            if len(shape) < 1:
                raise ValueError(
                    "When passing a Pytorch DataLoader to a Keras model, "
                    "the arrays returned by the generator "
                    "must be at least rank 1. Received: "
                    f"{x} of rank {len(x.shape)}"
                )
            shape = list(shape)
            shape[0] = None  # The batch size is not guaranteed to be static.

            # No easy way to get string representation of dtype in torch
            # TODO: Figure out a better way to achieve this
            dtype = str(x.dtype).replace("torch.", "")
            return tf.TensorSpec(shape=shape, dtype=dtype)

        return tuple(tree.map_structure(get_tensor_spec, batch_data))

    @property
    def num_batches(self):
        return self._size

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
