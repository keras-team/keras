import itertools

import numpy as np

from keras.src import tree
from keras.src.trainers.data_adapters import data_adapter_utils
from keras.src.trainers.data_adapters.data_adapter import DataAdapter


class TorchDataLoaderAdapter(DataAdapter):
    """Adapter that handles `torch.utils.data.DataLoader`."""

    def __init__(self, dataloader):
        import torch
        from keras.src.distribution import distribution_lib as dist_lib

        if not isinstance(dataloader, torch.utils.data.DataLoader):
            raise ValueError(
                f"Expected argument `dataloader` to be an instance of"
                f"`torch.utils.data.DataLoader`. Received: {dataloader}"
            )

        dist = dist_lib.distribution()
        if dist is not None and dist.auto_shard_dataset:
            num_replicas = None
            rank = None
            if isinstance(dist, dist_lib.DataParallel):
                num_replicas = torch.distributed.get_world_size()
                rank = torch.distributed.get_rank()
            elif isinstance(dist, dist_lib.ModelParallel):
                mesh_batch_dim_index = dist.device_mesh.axis_names.index(
                    dist.batch_dim_name
                )
                num_model_replicas = dist.device_mesh.shape[
                    mesh_batch_dim_index
                ]
                if num_model_replicas > 1:
                    num_process = torch.distributed.get_world_size()
                    process_id = torch.distributed.get_rank()
                    if num_model_replicas >= num_process:
                        num_replicas = num_process
                        rank = process_id
                    else:
                        num_replicas = num_model_replicas
                        processes_per_replica = (
                            num_process // num_model_replicas
                        )
                        rank = process_id // processes_per_replica

            if num_replicas is not None and rank is not None:
                # Reconstruct the DataLoader with a DistributedSampler
                sampler = torch.utils.data.distributed.DistributedSampler(
                    dataloader.dataset,
                    num_replicas=num_replicas,
                    rank=rank,
                    shuffle=True,
                )
                dataloader = torch.utils.data.DataLoader(
                    dataloader.dataset,
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
                    pin_memory_device=dataloader.pin_memory_device,
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
                tree.map_structure(
                    lambda x: np.asarray(x.cpu()), batch, none_is_leaf=False
                )
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
