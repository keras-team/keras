class DataAdapter(object):
    """Base class for input data adapters.

    The purpose of a DataAdapter is to provide a unfied interface to
    iterate over input data provided in a variety of formats -- such as
    NumPy arrays, tf.Tensors, tf.data.Datasets, Keras PyDatasets, etc.
    """

    def get_numpy_iterator(self):
        """Get a Python iterable for the `DataAdapter`, that yields NumPy
        arrays.

        Returns:
            A Python iterator.
        """
        raise NotImplementedError

    def get_tf_dataset(self):
        """Get a `tf.data.Dataset` instance for the DataAdapter.

        Note that the dataset returned does not repeat for epoch, so caller
        might need to create new iterator for the same dataset at the beginning
        of the epoch. This behavior might change in the future.

        Returns:
            A `tf.data.Dataset`. Caller might use the dataset in different
            context, e.g. iter(dataset) in eager to get the value directly, or
            in graph mode, provide the iterator tensor to Keras model function.
        """
        raise NotImplementedError

    def get_jax_iterator(self):
        """Get a Python iterable for the `DataAdapter`, that yields JAX arrays.

        Returns:
            A Python iterator.
        """
        raise NotImplementedError

    def get_torch_dataloader(self):
        """Get a Torch `DataLoader` for the `DataAdapter`.

        Returns:
            A Torch `DataLoader`.
        """
        raise NotImplementedError

    @property
    def num_batches(self):
        """Return the size (number of batches) for the dataset created.

        For certain type of the data input, the number of batches is known, eg
        for Numpy data, the size is same as (number_of_element / batch_size).
        Whereas for dataset or python generator, the size is unknown since it
        may or may not have an end state.

        Returns:
            int, the number of batches for the dataset, or None if it is
            unknown.  The caller could use this to control the loop of training,
            show progress bar, or handle unexpected StopIteration error.
        """
        raise NotImplementedError

    @property
    def batch_size(self):
        """Return the batch size of the dataset created.

        For certain type of the data input, the batch size is known, and even
        required, like numpy array. Whereas for dataset, the batch is unknown
        unless we take a peek.

        Returns:
          int, the batch size of the dataset, or None if it is unknown.
        """
        raise NotImplementedError

    @property
    def has_partial_batch(self):
        """Whether the dataset has partial batch at the end."""
        raise NotImplementedError

    @property
    def partial_batch_size(self):
        """The size of the final partial batch for dataset.

        Will return None if has_partial_batch is False or batch_size is None.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """A hook called after each epoch."""
        pass
