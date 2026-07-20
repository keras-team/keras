import itertools

import numpy as np

from keras.src import tree
from keras.src.distribution import distribution_lib
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

        dist = distribution_lib.distribution()
        self._num_data_shards = 1
        self._data_shard_id = 0
        if dist is not None and getattr(dist, "auto_shard_dataset", False):
            self._num_data_shards = dist.num_data_shards
            self._data_shard_id = dist.data_shard_id

        if self._num_data_shards > 1:
            dataset = dataloader.dataset
            if isinstance(dataset, torch.utils.data.IterableDataset):

                class ShardedIterableDataset(torch.utils.data.IterableDataset):
                    def __init__(self, dataset, num_data_shards, data_shard_id):
                        self.dataset = dataset
                        self.num_data_shards = num_data_shards
                        self.data_shard_id = data_shard_id

                    def __iter__(self):
                        for i, item in enumerate(self.dataset):
                            if i % self.num_data_shards == self.data_shard_id:
                                yield item

                if hasattr(dataset, "__len__"):

                    def __len__(self):
                        return (
                            len(self.dataset)
                            - self.data_shard_id
                            + self.num_data_shards
                            - 1
                        ) // self.num_data_shards

                    ShardedIterableDataset.__len__ = __len__

                dataloader = torch.utils.data.DataLoader(
                    ShardedIterableDataset(
                        dataset, self._num_data_shards, self._data_shard_id
                    ),
                    batch_size=dataloader.batch_size,
                    num_workers=dataloader.num_workers,
                    collate_fn=dataloader.collate_fn,
                    pin_memory=dataloader.pin_memory,
                    drop_last=dataloader.drop_last,
                    timeout=dataloader.timeout,
                    worker_init_fn=dataloader.worker_init_fn,
                    multiprocessing_context=dataloader.multiprocessing_context,
                    generator=dataloader.generator,
                    prefetch_factor=dataloader.prefetch_factor,
                    persistent_workers=dataloader.persistent_workers,
                )
            else:
                # Detect shuffle from dataloader
                shuffle = False
                if hasattr(dataloader, "sampler") and isinstance(
                    dataloader.sampler, torch.utils.data.RandomSampler
                ):
                    shuffle = True

                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataset,
                    num_replicas=self._num_data_shards,
                    rank=self._data_shard_id,
                    shuffle=shuffle,
                )
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=dataloader.batch_size,
                    sampler=sampler,
                    num_workers=dataloader.num_workers,
                    collate_fn=dataloader.collate_fn,
                    pin_memory=dataloader.pin_memory,
                    drop_last=dataloader.drop_last,
                    timeout=dataloader.timeout,
                    worker_init_fn=dataloader.worker_init_fn,
                    multiprocessing_context=dataloader.multiprocessing_context,
                    generator=dataloader.generator,
                    prefetch_factor=dataloader.prefetch_factor,
                    persistent_workers=dataloader.persistent_workers,
                )

        self._dataloader = dataloader
        self._epoch = 0
        self._batch_size = dataloader.batch_size
        self._output_signature = None
        self._num_batches = None
        self._partial_batch_size = None
        if hasattr(self._dataloader.dataset, "__len__"):
            self._num_batches = len(self._dataloader)
            if self._batch_size is not None:
                self._partial_batch_size = (
                    len(self._dataloader.dataset) % self._batch_size
                )

    def on_epoch_begin(self):
        if hasattr(self._dataloader, "sampler") and hasattr(
            self._dataloader.sampler, "set_epoch"
        ):
            # DistributedSampler requires `set_epoch` to be called at the
            # beginning of each epoch to ensure that the data is shuffled
            # differently across epochs.
            self._dataloader.sampler.set_epoch(self._epoch)

    def on_epoch_end(self):
        self._epoch += 1

    def get_numpy_iterator(self):
        for batch in self._dataloader:
            # shared memory using `np.asarray`
            yield tree.map_structure(
                lambda x: np.asarray(x.cpu()), batch, none_is_leaf=False
            )

    def get_jax_iterator(self):
        # We use numpy as an intermediary because it is faster.
        return self.get_numpy_iterator()

    def get_tf_dataset(self):
        from keras.src.utils.module_utils import tensorflow as tf

        def get_tf_iterator():
            for batch in self.get_numpy_iterator():
                yield tree.lists_to_tuples(batch)

        if self._output_signature is None:
            batches = list(
                itertools.islice(
                    self._dataloader,
                    data_adapter_utils.NUM_BATCHES_FOR_TENSOR_SPEC,
                )
            )
            self._output_signature = tree.lists_to_tuples(
                data_adapter_utils.get_tensor_spec(batches)
            )
        return tf.data.Dataset.from_generator(
            get_tf_iterator,
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
