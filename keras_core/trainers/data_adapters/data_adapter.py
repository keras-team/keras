class DataAdapter(object):
    """Base class for input data adapters.

    The purpose of a DataAdapter is to provide a unfied interface to
    iterate over input data provided in a variety of formats -- such as
    NumPy arrays, tf.Tensors, tf.data.Datasets, Keras PyDatasets, etc.
    """

    @staticmethod
    def can_handle(x, y=None):
        """Whether the current DataAdapter could handle the input x and y.

        Structure wise, x and y can be single object, or list of objects if
        there multiple input/output, or dictionary of objects when the
        input/output are named.

        Args:
            x: input features.
            y: target labels. Note that y could be None in the case of prediction.

        Returns:
            boolean
        """
        raise NotImplementedError

    def __init__(self, x, y=None, **kwargs):
        """Create a DataAdapter based on data inputs.

        The caller must make sure to call `can_handle()` first before invoking
        this method. Provide unsupported data type will result into unexpected
        behavior.

        Args:
            x: input features.
            y: target labels. Note that y could be None in the case of prediction.
            **kwargs: Other keyword arguments for DataAdapter during the
                construction of the tf.dataset.Dataset. For example:
                - Numpy data might have `sample_weights` which will be used for
                weighting the loss function during training.
                - Numpy data might need to have `batch_size` parameter when
                constructing the dataset and iterator.
                - Certain input might need to be distribution strategy aware. When
                `distribution_strategy` is passed, the created dataset need to
                respect the strategy.
                DataAdapter might choose to ignore any keyword argument if it
                doesn't use it, or raise exception if any required argument is not
                provided.
        """
        if not self.can_handle(x, y):
            raise ValueError(f"{self.__class__} cannot handle input {x}, {y}")

    def get_numpy_iterator(self):
        """Get a Python iterable for the DataAdapter, that yields NumPy arrays.

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
            context, e.g. iter(dataset) in eager to get the value directly, or in
            graph mode, provide the iterator tensor to Keras model function.
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
            int, the number of batches for the dataset, or None if it is unknown.
            The caller could use this to control the loop of training, show
            progress bar, or handle unexpected StopIteration error.
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

    def has_partial_batch(self):
        """Whether the dataset has partial batch at the end."""
        raise NotImplementedError

    def partial_batch_size(self):
        """The size of the final partial batch for dataset.

        Will return None if has_partial_batch is False or batch_size is None.
        """
        raise NotImplementedError

    def should_recreate_iterator(self):
        """Returns whether a new iterator should be created every epoch."""
        raise NotImplementedError

    def get_samples(self):
        """Returns number of samples in the data, or `None`."""
        if not self.get_size() or not self.batch_size():
            return None
        total_sample = self.get_size() * self.batch_size()
        if self.has_partial_batch():
            total_sample -= self.batch_size() - self.partial_batch_size()
        return total_sample

    def on_epoch_end(self):
        """A hook called after each epoch."""
        pass
