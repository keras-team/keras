from keras.src import backend
from keras.src import utils
from keras.src.api_export import keras_export


@keras_export("keras.callbacks.Callback")
class Callback:
    """Base class used to build new callbacks.

    Callbacks can be passed to keras methods such as `fit()`, `evaluate()`, and
    `predict()` in order to hook into the various stages of the model training,
    evaluation, and inference lifecycle.

    To create a custom callback, subclass `keras.callbacks.Callback` and
    override the method associated with the stage of interest.

    Example:

    >>> training_finished = False
    >>> class MyCallback(Callback):
    ...   def on_train_end(self, logs=None):
    ...     global training_finished
    ...     training_finished = True
    >>> model = Sequential([
    ...     layers.Dense(1, input_shape=(1,))])
    >>> model.compile(loss='mean_squared_error')
    >>> model.fit(np.array([[1.0]]), np.array([[1.0]]),
    ...           callbacks=[MyCallback()])
    >>> assert training_finished == True

    If you want to use `Callback` objects in a custom training loop:

    1. You should pack all your callbacks into a single `callbacks.CallbackList`
       so they can all be called together.
    2. You will need to manually call all the `on_*` methods at the appropriate
       locations in your loop. Like this:

    Example:

    ```python
    callbacks =  keras.callbacks.CallbackList([...])
    callbacks.append(...)
    callbacks.on_train_begin(...)
    for epoch in range(EPOCHS):
        callbacks.on_epoch_begin(epoch)
        for i, data in dataset.enumerate():
        callbacks.on_train_batch_begin(i)
        batch_logs = model.train_step(data)
        callbacks.on_train_batch_end(i, batch_logs)
        epoch_logs = ...
        callbacks.on_epoch_end(epoch, epoch_logs)
    final_logs=...
    callbacks.on_train_end(final_logs)
    ```

    Attributes:
        params: Dict. Training parameters
            (eg. verbosity, batch size, number of epochs...).
        model: Instance of `Model`.
            Reference of the model being trained.

    The `logs` dictionary that callback methods
    take as argument will contain keys for quantities relevant to
    the current batch or epoch (see method-specific docstrings).
    """

    def __init__(self):
        self.params = None
        self._model = None

    def set_params(self, params):
        self.params = params

    def set_model(self, model):
        self._model = model

    @property
    def model(self):
        if backend.backend() == "torch":
            from torch.nn.parallel import DistributedDataParallel

            if isinstance(self._model, DistributedDataParallel):
                # Keras Callbacks expect to work with Keras models. e.g
                # ModelCheckpoint and EarlyStopping both attempt to call
                # keras-specific APIs on the value returned from this
                # property. If this callback was created against a DDP
                # wrapper instead of the underlying keras.Model, it is
                # likely to fail. Return self._model.module for DDP
                # instances instead.
                return self._model.module

        if backend.backend() == "jax" and hasattr(
            self._model, "jax_state_sync"
        ):
            # With JAX, by default the model state is not
            # attached to the model in the middle of an
            # epoch. We have to force a sync before
            # accessing model state for e.g. checkpointing.
            self._model.jax_state_sync()
        return self._model

    @utils.default
    def on_batch_begin(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_begin`."""

    @utils.default
    def on_batch_end(self, batch, logs=None):
        """A backwards compatibility alias for `on_train_batch_end`."""

    @utils.default
    def on_epoch_begin(self, epoch, logs=None):
        """Called at the start of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    @utils.default
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of an epoch.

        Subclasses should override for any actions to run. This function should
        only be called during TRAIN mode.

        Args:
            epoch: Integer, index of epoch.
            logs: Dict, metric results for this training epoch, and for the
              validation epoch if validation is performed. Validation result
              keys are prefixed with `val_`. For training epoch, the values of
              the `Model`'s metrics are returned. Example:
              `{'loss': 0.2, 'accuracy': 0.7}`.
        """

    @utils.default
    def on_train_batch_begin(self, batch, logs=None):
        """Called at the beginning of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
        # For backwards compatibility.
        self.on_batch_begin(batch, logs=logs)

    @utils.default
    def on_train_batch_end(self, batch, logs=None):
        """Called at the end of a training batch in `fit` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """
        # For backwards compatibility.
        self.on_batch_end(batch, logs=logs)

    @utils.default
    def on_test_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `evaluate` methods.

        Also called at the beginning of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    @utils.default
    def on_test_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `evaluate` methods.

        Also called at the end of a validation batch in the `fit`
        methods, if validation data is provided.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    @utils.default
    def on_predict_batch_begin(self, batch, logs=None):
        """Called at the beginning of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    @utils.default
    def on_predict_batch_end(self, batch, logs=None):
        """Called at the end of a batch in `predict` methods.

        Subclasses should override for any actions to run.

        Note that if the `steps_per_execution` argument to `compile` in
        `Model` is set to `N`, this method will only be called every
        `N` batches.

        Args:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Aggregated metric results up until this batch.
        """

    @utils.default
    def on_train_begin(self, logs=None):
        """Called at the beginning of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    @utils.default
    def on_train_end(self, logs=None):
        """Called at the end of training.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_epoch_end()` is passed to this argument for this method but
              that may change in the future.
        """

    @utils.default
    def on_test_begin(self, logs=None):
        """Called at the beginning of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    @utils.default
    def on_test_end(self, logs=None):
        """Called at the end of evaluation or validation.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently the output of the last call to
              `on_test_batch_end()` is passed to this argument for this method
              but that may change in the future.
        """

    @utils.default
    def on_predict_begin(self, logs=None):
        """Called at the beginning of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """

    @utils.default
    def on_predict_end(self, logs=None):
        """Called at the end of prediction.

        Subclasses should override for any actions to run.

        Args:
            logs: Dict. Currently no data is passed to this argument for this
              method but that may change in the future.
        """
