import platform
import warnings

from keras_core import backend
from keras_core import metrics as metrics_module
from keras_core import operations as ops
from keras_core import optimizers
from keras_core.saving import serialization_lib
from keras_core.trainers.compile_utils import CompileLoss
from keras_core.trainers.compile_utils import CompileMetrics
from keras_core.utils import traceback_utils
from keras_core.utils import tracking


class Trainer:
    def __init__(self):
        self._lock = False
        self._run_eagerly = False
        self._jit_compile = None
        self.compiled = False
        self.steps_per_execution = 1

    @traceback_utils.filter_traceback
    @tracking.no_automatic_dependency_tracking
    def compile(
        self,
        optimizer="rmsprop",
        loss=None,
        loss_weights=None,
        metrics=None,
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=1,
        jit_compile="auto",
    ):
        self.optimizer = optimizers.get(optimizer)
        if hasattr(self, "output_names"):
            output_names = self.output_names
        else:
            output_names = None
        if loss is not None:
            self._compile_loss = CompileLoss(
                loss, loss_weights, output_names=output_names
            )
        else:
            self._compile_loss = None
        if metrics is not None:
            self._compile_metrics = CompileMetrics(
                metrics, weighted_metrics, output_names=output_names
            )
        else:
            self._compile_metrics = None
        if jit_compile == "auto":
            if not run_eagerly and model_supports_jit(self):
                jit_compile = True
            else:
                jit_compile = False
        if jit_compile and run_eagerly:
            jit_compile = False
            warnings.warn(
                "If `run_eagerly` is True, then `jit_compile` "
                "cannot also be True. Disabling `jit_compile`.",
                stacklevel=2,
            )
        self.jit_compile = jit_compile
        self.run_eagerly = run_eagerly
        self.stop_training = False
        self.compiled = True
        self._loss_tracker = metrics_module.Mean(name="loss")
        self.steps_per_execution = steps_per_execution

        self._compile_config = serialization_lib.SerializableDict(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
        )

    @property
    def jit_compile(self):
        if self._jit_compile is None:
            # Value was never set. Resolve it now.
            jit_compile = model_supports_jit(self)
            self._jit_compile = jit_compile
        return self._jit_compile

    @jit_compile.setter
    def jit_compile(self, value):
        self._jit_compile = value

    @property
    def run_eagerly(self):
        return self._run_eagerly

    @run_eagerly.setter
    def run_eagerly(self, value):
        self._run_eagerly = value

    @property
    def metrics(self):
        metrics = [self._loss_tracker]
        metrics.extend(self._metrics[:])
        if self._compile_metrics is not None:
            metrics += [self._compile_metrics]
        return metrics

    @property
    def metrics_names(self):
        return [m.name for m in self.metrics]

    @property
    def metrics_variables(self):
        vars = []
        for metric in self.metrics:
            vars.extend(metric.variables)
        return vars

    def reset_metrics(self):
        for m in self.metrics:
            m.reset_state()

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        """Compute the total loss, validate it, and return it.

        Subclasses can optionally override this method to provide custom loss
        computation logic.

        Example:

        ```python
        class MyModel(Model):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.loss_tracker = metrics.Mean(name='loss')

            def compute_loss(self, x, y, y_pred, sample_weight):
                loss = ops.means((y_pred - y) ** 2)
                loss += ops.sum(self.losses)
                self.loss_tracker.update_state(loss)
                return loss

            def reset_metrics(self):
                self.loss_tracker.reset_state()

            @property
            def metrics(self):
                return [self.loss_tracker]

        inputs = layers.Input(shape=(10,), name='my_input')
        outputs = layers.Dense(10)(inputs)
        model = MyModel(inputs, outputs)
        model.add_loss(ops.sum(outputs))

        optimizer = SGD()
        model.compile(optimizer, loss='mse', steps_per_execution=10)
        dataset = ...
        model.fit(dataset, epochs=2, steps_per_epoch=10)
        print(f"Custom loss: {model.loss_tracker.result()}")
        ```

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the model (output of `model(x)`)
            sample_weight: Sample weights for weighting the loss function.
            allow_empty: If `False`, the method will error out if
                no loss has been computed by the model. If `True`, then
                if no loss is computed, the method returns 0.

        Returns:
            The total loss as a scalar tensor, or `None` if no loss results
            (which is the case when called by `Model.test_step`).
        """
        del x  # The default implementation does not use `x`.
        losses = []
        if self._compile_loss is not None:
            loss = self._compile_loss(y, y_pred, sample_weight)
            if loss is not None:
                losses.append(loss)
        for loss in self.losses:
            losses.append(ops.cast(loss, dtype=backend.floatx()))
        if not allow_empty and len(losses) == 0:
            raise ValueError(
                "No loss to compute. Provide a `loss` argument in `compile()`."
            )
        if len(losses) == 1:
            total_loss = losses[0]
        elif len(losses) == 0:
            total_loss = ops.zeros(())
        else:
            total_loss = ops.sum(losses)
        return total_loss

    def compute_metrics(self, x, y, y_pred, sample_weight=None):
        """Update metric states and collect all metrics to be returned.

        Subclasses can optionally override this method to provide custom metric
        updating and collection logic.

        Example:

        ```python
        class MyModel(Sequential):
            def compute_metrics(self, x, y, y_pred, sample_weight):
                # This super call updates `self.compiled_metrics` and returns
                # results for all metrics listed in `self.metrics`.
                metric_results = super().compute_metrics(
                    x, y, y_pred, sample_weight)

                # Note that `self.custom_metric` is not listed
                # in `self.metrics`.
                self.custom_metric.update_state(x, y, y_pred, sample_weight)
                metric_results['metric_name'] = self.custom_metric.result()
                return metric_results
        ```

        Args:
            x: Input data.
            y: Target data.
            y_pred: Predictions returned by the model output of `model.call(x)`.
            sample_weight: Sample weights for weighting the loss function.

        Returns:
            A `dict` containing values that will be passed to
            `keras_core.callbacks.CallbackList.on_train_batch_end()`. Typically,
            the values of the metrics listed in `self.metrics` are returned.
            Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        del x  # The default implementation does not use `x`.
        if self._compile_metrics is not None:
            self._compile_metrics.update_state(y, y_pred, sample_weight)
        return self.get_metrics_result()

    def get_metrics_result(self):
        """Returns the model's metrics values as a dict.

        If any of the metric result is a dict (containing multiple metrics),
        each of them gets added to the top level returned dict of this method.

        Returns:
            A `dict` containing values of the metrics listed in `self.metrics`.
            Example: `{'loss': 0.2, 'accuracy': 0.7}`.
        """
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return self._pythonify_logs(return_metrics)

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
    ):
        """Trains the model for a fixed number of epochs (dataset iterations).

        Args:
            x: Input data. It could be:
              - A NumPy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
              - A tensor, or a list of tensors
                (in case the model has multiple inputs).
              - A dict mapping input names to the corresponding array/tensors,
                if the model has named inputs.
              - A `tf.data.Dataset`. Should return a tuple
                of either `(inputs, targets)` or
                `(inputs, targets, sample_weights)`.
              - A `keras_core.utils.PyDataset` returning `(inputs,
                targets)` or `(inputs, targets, sample_weights)`.
            y: Target data. Like the input data `x`,
                it could be either NumPy array(s) or backend-native tensor(s).
                If `x` is a dataset, generator,
                or `keras_core.utils.PyDataset` instance, `y` should
                not be specified (since targets will be obtained from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of datasets, generators, or `keras_core.utils.PyDataset`
                instances (since they generate batches).
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided
                (unless the `steps_per_epoch` flag is set to
                something other than None).
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                "auto" becomes 1 for most cases.
                Note that the progress bar is not
                particularly useful when logged to a file,
                so `verbose=2` is recommended when not running interactively
                (e.g., in a production environment). Defaults to `"auto"`.
            callbacks: List of `keras_core.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `keras_core.callbacks`. Note
                `keras_core.callbacks.ProgbarLogger` and
                `keras_core.callbacks.History` callbacks are created
                automatically and need not be passed to `model.fit()`.
                `keras_core.callbacks.ProgbarLogger` is created
                or not based on the `verbose` argument in `model.fit()`.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This
                argument is not supported when `x` is a dataset, generator or
                `keras_core.utils.PyDataset` instance.
                If both `validation_data` and `validation_split` are provided,
                `validation_data` will override `validation_split`.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data. Thus, note the fact
                that the validation loss of data provided using
                `validation_split` or `validation_data` is not affected by
                regularization layers like noise and dropout.
                `validation_data` will override `validation_split`.
                `validation_data` could be:
                  - A tuple `(x_val, y_val)` of NumPy arrays or tensors.
                  - A tuple `(x_val, y_val, val_sample_weights)` of NumPy
                    arrays.
                  - A `tf.data.Dataset`.
                  - A Python generator or `keras_core.utils.PyDataset` returning
                  `(inputs, targets)` or `(inputs, targets, sample_weights)`.
            shuffle: Boolean, whether to shuffle the training data
                before each epoch. This argument is
                ignored when `x` is a generator or a `tf.data.Dataset`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class. When `class_weight` is specified
                and targets have a rank of 2 or greater, either `y` must be
                one-hot encoded, or an explicit final dimension of `1` must
                be included for sparse class labels.
            sample_weight: Optional NumPy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                NumPy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                This argument is not supported when `x` is a dataset, generator,
                or `keras_core.utils.PyDataset` instance, instead provide the
                sample_weights as the third element of `x`.
                Note that sample weighting does not apply to metrics specified
                via the `metrics` argument in `compile()`. To apply sample
                weighting to your metrics, you can specify them via the
                `weighted_metrics` in `compile()` instead.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                backend-native tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined. If `x` is a
                `tf.data.Dataset`, and `steps_per_epoch`
                is `None`, the epoch will run until the input dataset is
                exhausted.  When passing an infinitely repeating dataset, you
                must specify the `steps_per_epoch` argument. If
                `steps_per_epoch=-1` the training will run indefinitely with an
                infinitely repeating dataset.
            validation_steps: Only relevant if `validation_data` is provided.
                Total number of steps (batches of
                samples) to draw before stopping when performing validation
                at the end of every epoch. If `validation_steps` is `None`,
                validation will run until the `validation_data` dataset is
                exhausted. In the case of an infinitely repeated dataset, it
                will run into an infinite loop. If `validation_steps` is
                specified and only part of the dataset will be consumed, the
                evaluation will start from the beginning of the dataset at each
                epoch. This ensures that the same validation samples are used
                every time.
            validation_batch_size: Integer or `None`.
                Number of samples per validation batch.
                If unspecified, will default to `batch_size`.
                Do not specify the `validation_batch_size` if your data is in
                the form of datasets or `keras_core.utils.PyDataset`
                instances (since they generate batches).
            validation_freq: Only relevant if validation data is provided.
              Specifies how many training epochs to run
              before a new validation run is performed, e.g. `validation_freq=2`
              runs validation every 2 epochs.

        Unpacking behavior for iterator-like inputs:
            A common pattern is to pass an iterator like object such as a
            `tf.data.Dataset` or a `keras_core.utils.PyDataset` to `fit()`,
            which will in fact yield not only features (`x`)
            but optionally targets (`y`) and sample weights (`sample_weight`).
            Keras requires that the output of such iterator-likes be
            unambiguous. The iterator should return a tuple
            of length 1, 2, or 3, where the optional second and third elements
            will be used for `y` and `sample_weight` respectively.
            Any other type provided will be wrapped in
            a length-one tuple, effectively treating everything as `x`. When
            yielding dicts, they should still adhere to the top-level tuple
            structure,
            e.g. `({"x0": x0, "x1": x1}, y)`. Keras will not attempt to separate
            features, targets, and weights from the keys of a single dict.
            A notable unsupported data type is the `namedtuple`. The reason is
            that it behaves like both an ordered datatype (tuple) and a mapping
            datatype (dict). So given a namedtuple of the form:
            `namedtuple("example_tuple", ["y", "x"])`
            it is ambiguous whether to reverse the order of the elements when
            interpreting the value. Even worse is a tuple of the form:
            `namedtuple("other_tuple", ["x", "y", "z"])`
            where it is unclear if the tuple was intended to be unpacked
            into `x`, `y`, and `sample_weight` or passed through
            as a single element to `x`.

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).
        """
        raise NotImplementedError

    def evaluate(
        self,
        x=None,
        y=None,
        batch_size=None,
        verbose="auto",
        sample_weight=None,
        steps=None,
        callbacks=None,
        return_dict=False,
        **kwargs,
    ):
        """Returns the loss value & metrics values for the model in test mode.

        Computation is done in batches (see the `batch_size` arg.)

        Args:
            x: Input data. It could be:
                - A NumPy array (or array-like), or a list of arrays
                    (in case the model has multiple inputs).
                - A tensor, or a list of tensors
                    (in case the model has multiple inputs).
                - A dict mapping input names to the corresponding array/tensors,
                    if the model has named inputs.
                - A `tf.data.Dataset`. Should return a tuple
                    of either `(inputs, targets)` or
                    `(inputs, targets, sample_weights)`.
                - A generator or `keras_core.utils.PyDataset` returning
                    `(inputs, targets)` or `(inputs, targets, sample_weights)`.
            y: Target data. Like the input data `x`, it could be either NumPy
                array(s) or backend-native tensor(s).
                If `x` is a `tf.data.Dataset` or `keras_core.utils.PyDataset`
                instance, `y` should not be specified
                (since targets will be obtained from the iterator/dataset).
            batch_size: Integer or `None`. Number of samples per batch of
                computation. If unspecified, `batch_size` will default to 32. Do
                not specify the `batch_size` if your data is in the form of a
                dataset, generators, or `keras_core.utils.PyDataset` instances
                (since they generate batches).
            verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = single line.
                `"auto"` becomes 1 for most cases.
                Note that the progress bar is not
                particularly useful when logged to a file, so `verbose=2` is
                recommended when not running interactively
                (e.g. in a production environment). Defaults to `"auto"`.
            sample_weight: Optional NumPy array of weights for the test samples,
                used for weighting the loss function. You can either pass a flat
                (1D) NumPy array with the same length as the input samples
                (1:1 mapping between weights and samples), or in the case of
                temporal data, you can pass a 2D array with shape `(samples,
                sequence_length)`, to apply a different weight to every
                timestep of every sample. This argument is not supported when
                `x` is a dataset, instead pass sample weights as the third
                element of `x`.
            steps: Integer or `None`. Total number of steps (batches of samples)
                before declaring the evaluation round finished. Ignored with the
                default value of `None`. If `x` is a `tf.data.Dataset` and
                `steps` is `None`, evaluation will run until the dataset
                is exhausted.
            callbacks: List of `keras_core.callbacks.Callback` instances.
                List of callbacks to apply during evaluation.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric.
                If `False`, they are returned as a list.

        Returns:
            Scalar test loss (if the model has a single output and no metrics)
            or list of scalars (if the model has multiple outputs
            and/or metrics). The attribute `model.metrics_names` will give you
            the display labels for the scalar outputs.
        """
        raise NotImplementedError

    def predict(
        self, x, batch_size=None, verbose="auto", steps=None, callbacks=None
    ):
        """Generates output predictions for the input samples.

        Computation is done in batches. This method is designed for batch
        processing of large numbers of inputs. It is not intended for use inside
        of loops that iterate over your data and process small numbers of inputs
        at a time.

        For small numbers of inputs that fit in one batch,
        directly use `__call__()` for faster execution, e.g.,
        `model(x)`, or `model(x, training=False)` if you have layers such as
        `BatchNormalization` that behave differently during
        inference.

        Note: See [this FAQ entry](
        https://keras.io/getting_started/faq/#whats-the-difference-between-model-methods-predict-and-call)
        for more details about the difference between `Model` methods
        `predict()` and `__call__()`.

        Args:
            x: Input samples. It could be:
                - A NumPy array (or array-like), or a list of arrays
                    (in case the model has multiple inputs).
                - A tensor, or a list of tensors
                    (in case the model has multiple inputs).
                - A `tf.data.Dataset`.
                - A `keras_core.utils.PyDataset` instance.
            batch_size: Integer or `None`.
                Number of samples per batch.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of dataset, generators, or `keras_core.utils.PyDataset`
                instances (since they generate batches).
            verbose: `"auto"`, 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = single line.
                `"auto"` becomes 1 for most cases. Note that the progress bar
                is not particularly useful when logged to a file,
                so `verbose=2` is recommended when not running interactively
                (e.g. in a production environment). Defaults to `"auto"`.
            steps: Total number of steps (batches of samples)
                before declaring the prediction round finished.
                Ignored with the default value of `None`.
                If `x` is a `tf.data.Dataset` and `steps` is `None`,
                `predict()` will run until the input dataset is exhausted.
            callbacks: List of `keras_core.callbacks.Callback` instances.
                List of callbacks to apply during prediction.

        Returns:
            NumPy array(s) of predictions.
        """
        raise NotImplementedError

    def train_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        class_weight=None,
        return_dict=False,
    ):
        """Runs a single gradient update on a single batch of data.

        Args:
            x: Input data. Must be array-like.
            y: Target data. Must be array-like.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape `(samples, sequence_length)`, to apply a different
                weight to every timestep of every sample.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) to apply to the model's loss for the samples
                from this class during training. This can be useful to tell the
                model to "pay more attention" to samples from an
                under-represented class. When `class_weight` is specified
                and targets have a rank of 2 or greater, either `y` must
                be one-hot encoded, or an explicit final dimension of 1
                must be included for sparse class labels.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric. If `False`,
                they are returned as a list.

        Returns:
            A scalar loss value (when no metrics and `return_dict=False`),
            a list of loss and metric values
            (if there are metrics and `return_dict=False`), or a dict of
            metric and loss values (if `return_dict=True`).
        """
        raise NotImplementedError

    def test_on_batch(
        self,
        x,
        y=None,
        sample_weight=None,
        return_dict=False,
    ):
        """Test the model on a single batch of samples.

        Args:
            x: Input data. Must be array-like.
            y: Target data. Must be array-like.
            sample_weight: Optional array of the same length as x, containing
                weights to apply to the model's loss for each sample.
                In the case of temporal data, you can pass a 2D array
                with shape `(samples, sequence_length)`, to apply a different
                weight to every timestep of every sample.
            return_dict: If `True`, loss and metric results are returned as a
                dict, with each key being the name of the metric. If `False`,
                they are returned as a list.

        Returns:
            A scalar loss value (when no metrics and `return_dict=False`),
            a list of loss and metric values
            (if there are metrics and `return_dict=False`), or a dict of
            metric and loss values (if `return_dict=True`).
        """
        raise NotImplementedError

    def predict_on_batch(self, x):
        """Returns predictions for a single batch of samples.

        Args:
            x: Input data. It must be array-like.

        Returns:
            NumPy array(s) of predictions.
        """
        raise NotImplementedError

    def get_compile_config(self):
        """Returns a serialized config with information for compiling the model.

        This method returns a config dictionary containing all the information
        (optimizer, loss, metrics, etc.) with which the model was compiled.

        Returns:
            A dict containing information for compiling the model.
        """
        if self.compiled and hasattr(self, "_compile_config"):
            return self._compile_config.serialize()

    def compile_from_config(self, config):
        """Compiles the model with the information given in config.

        This method uses the information in the config (optimizer, loss,
        metrics, etc.) to compile the model.

        Args:
            config: Dict containing information for compiling the model.
        """
        has_overridden_compile = self.__class__.compile != Trainer.compile
        if has_overridden_compile:
            warnings.warn(
                "`compile()` was not called as part of model loading "
                "because the model's `compile()` method is custom. "
                "All subclassed Models that have `compile()` "
                "overridden should also override "
                "`get_compile_config()` and `compile_from_config(config)`. "
                "Alternatively, you can "
                "call `compile()` manually after loading.",
                stacklevel=2,
            )
            return
        config = serialization_lib.deserialize_keras_object(config)
        self.compile(**config)
        if hasattr(self, "optimizer") and self.built:
            # Create optimizer variables.
            self.optimizer.build(self.trainable_variables)

    def _should_eval(self, epoch, validation_freq):
        epoch = epoch + 1  # one-index the user-facing epoch.
        if isinstance(validation_freq, int):
            return epoch % validation_freq == 0
        elif isinstance(validation_freq, list):
            return epoch in validation_freq
        else:
            raise ValueError(
                "Expected `validation_freq` to be a list or int. "
                f"Received: validation_freq={validation_freq} of the "
                f"type {type(validation_freq)}."
            )

    def _pythonify_logs(self, logs):
        result = {}
        for key, value in sorted(logs.items()):
            if isinstance(value, dict):
                result.update(self._pythonify_logs(value))
            else:
                try:
                    value = float(value)
                except:
                    pass
                result[key] = value
        return result

    def _flatten_metrics_in_order(self, logs):
        """Turns `logs` dict into a list as per key order of `metrics_names`."""
        metric_names = [m.name for m in self.metrics]
        results = []
        for name in metric_names:
            if name in logs:
                results.append(logs[name])
        for key in sorted(logs.keys()):
            if key not in metric_names:
                results.append(logs[key])
        if len(results) == 1:
            return results[0]
        return results

    def _assert_compile_called(self, method_name=None):
        if not self.compiled:
            msg = "You must call `compile()` before "
            if metrics_module:
                msg += "using the model."
            else:
                msg += f"calling `{method_name}()`."
            raise ValueError(msg)


def model_supports_jit(model):
    if platform.system() == "Darwin" and "arm" in platform.processor().lower():
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            if tf.config.list_physical_devices("GPU"):
                return False
    if all(x.supports_jit for x in model._flatten_layers()):
        return True
    return False
