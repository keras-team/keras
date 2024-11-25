import logging
import os
import sys
import time
import warnings

from keras.src import backend
from keras.src import ops
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.callbacks.callback import Callback
from keras.src.layers import Embedding
from keras.src.optimizers import Optimizer
from keras.src.utils import file_utils


@keras_export("keras.callbacks.TensorBoard")
class TensorBoard(Callback):
    """Enable visualizations for TensorBoard.

    TensorBoard is a visualization tool provided with TensorFlow. A TensorFlow
    installation is required to use this callback.

    This callback logs events for TensorBoard, including:

    * Metrics summary plots
    * Training graph visualization
    * Weight histograms
    * Sampled profiling

    When used in `model.evaluate()` or regular validation
    in addition to epoch summaries, there will be a summary that records
    evaluation metrics vs `model.optimizer.iterations` written. The metric names
    will be prepended with `evaluation`, with `model.optimizer.iterations` being
    the step in the visualized TensorBoard.

    If you have installed TensorFlow with pip, you should be able
    to launch TensorBoard from the command line:

    ```
    tensorboard --logdir=path_to_your_logs
    ```

    You can find more information about TensorBoard
    [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

    Args:
        log_dir: the path of the directory where to save the log files to be
            parsed by TensorBoard. e.g.,
            `log_dir = os.path.join(working_dir, 'logs')`.
            This directory should not be reused by any other callbacks.
        histogram_freq: frequency (in epochs) at which to compute
            weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must be
            specified for histogram visualizations.
        write_graph:  (Not supported at this time)
            Whether to visualize the graph in TensorBoard.
            Note that the log file can become quite large
            when `write_graph` is set to `True`.
        write_images: whether to write model weights to visualize as image in
            TensorBoard.
        write_steps_per_second: whether to log the training steps per second
            into TensorBoard. This supports both epoch and batch frequency
            logging.
        update_freq: `"batch"` or `"epoch"` or integer. When using `"epoch"`,
            writes the losses and metrics to TensorBoard after every epoch.
            If using an integer, let's say `1000`, all metrics and losses
            (including custom ones added by `Model.compile`) will be logged to
            TensorBoard every 1000 batches. `"batch"` is a synonym for 1,
            meaning that they will be written every batch.
            Note however that writing too frequently to TensorBoard can slow
            down your training, especially when used with distribution
            strategies as it will incur additional synchronization overhead.
            Batch-level summary writing is also available via `train_step`
            override. Please see
            [TensorBoard Scalars tutorial](
                https://www.tensorflow.org/tensorboard/scalars_and_keras#batch-level_logging)
            for more details.
        profile_batch: Profile the batch(es) to sample compute characteristics.
            profile_batch must be a non-negative integer or a tuple of integers.
            A pair of positive integers signify a range of batches to profile.
            By default, profiling is disabled.
        embeddings_freq: frequency (in epochs) at which embedding layers will be
            visualized. If set to 0, embeddings won't be visualized.
        embeddings_metadata: Dictionary which maps embedding layer names to the
            filename of a file in which to save metadata for the embedding layer.
            In case the same metadata file is to be
            used for all embedding layers, a single filename can be passed.

    Examples:

    ```python
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./logs")
    model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
    # Then run the tensorboard command to view the visualizations.
    ```

    Custom batch-level summaries in a subclassed Model:

    ```python
    class MyModel(keras.Model):

        def build(self, _):
            self.dense = keras.layers.Dense(10)

        def call(self, x):
            outputs = self.dense(x)
            tf.summary.histogram('outputs', outputs)
            return outputs

    model = MyModel()
    model.compile('sgd', 'mse')

    # Make sure to set `update_freq=N` to log a batch-level summary every N
    # batches.  In addition to any `tf.summary` contained in `model.call()`,
    # metrics added in `Model.compile` will be logged every N batches.
    tb_callback = keras.callbacks.TensorBoard('./logs', update_freq=1)
    model.fit(x_train, y_train, callbacks=[tb_callback])
    ```

    Custom batch-level summaries in a Functional API Model:

    ```python
    def my_summary(x):
        tf.summary.histogram('x', x)
        return x

    inputs = keras.Input(10)
    x = keras.layers.Dense(10)(inputs)
    outputs = keras.layers.Lambda(my_summary)(x)
    model = keras.Model(inputs, outputs)
    model.compile('sgd', 'mse')

    # Make sure to set `update_freq=N` to log a batch-level summary every N
    # batches. In addition to any `tf.summary` contained in `Model.call`,
    # metrics added in `Model.compile` will be logged every N batches.
    tb_callback = keras.callbacks.TensorBoard('./logs', update_freq=1)
    model.fit(x_train, y_train, callbacks=[tb_callback])
    ```

    Profiling:

    ```python
    # Profile a single batch, e.g. the 5th batch.
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='./logs', profile_batch=5)
    model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])

    # Profile a range of batches, e.g. from 10 to 20.
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir='./logs', profile_batch=(10,20))
    model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
    ```
    """  # noqa: E501

    def __init__(
        self,
        log_dir="logs",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq="epoch",
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
    ):
        super().__init__()

        self.log_dir = str(log_dir)
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_images = write_images
        self.write_steps_per_second = write_steps_per_second
        self.update_freq = 1 if update_freq == "batch" else update_freq
        self.embeddings_freq = embeddings_freq
        self.embeddings_metadata = embeddings_metadata
        if profile_batch:
            if backend.backend() not in ("jax", "tensorflow"):
                # TODO: profiling not available in torch, numpy
                raise ValueError(
                    "Profiling is not yet available with the "
                    f"{backend.backend()} backend. Please open a PR "
                    "if you'd like to add this feature. Received: "
                    f"profile_batch={profile_batch} (must be 0)"
                )
            elif backend.backend() == "jax":
                if sys.version_info[1] < 12:
                    warnings.warn(
                        "Profiling with the "
                        f"{backend.backend()} backend requires python >= 3.12."
                    )
                    profile_batch = 0

        self._init_profile_batch(profile_batch)
        self._global_train_batch = 0
        self._global_test_batch = 0
        self._previous_epoch_iterations = 0
        self._train_accumulated_time = 0
        self._batch_start_time = 0
        self._summary_module = None

        # Lazily initialized in order to avoid creating event files when
        # not needed.
        self._writers = {}

        # Used to restore any existing `SummaryWriter` after training ends.
        self._prev_summary_state = []

    def set_model(self, model):
        """Sets Keras model and writes graph if specified."""
        self._model = model
        self._log_write_dir = self.log_dir

        self._train_dir = os.path.join(self._log_write_dir, "train")
        self._val_dir = os.path.join(self._log_write_dir, "validation")
        self._writers = {}  # Resets writers.

        self._should_write_train_graph = False
        if self.write_graph:
            self._write_keras_model_summary()
            self._should_write_train_graph = True
        if self.embeddings_freq:
            self._configure_embeddings()

    @property
    def summary(self):
        if self._summary_module is None:
            import tensorflow.summary as summary

            self._summary_module = summary
        return self._summary_module

    @property
    def _train_writer(self):
        if "train" not in self._writers:
            self._writers["train"] = self.summary.create_file_writer(
                self._train_dir
            )
        return self._writers["train"]

    @property
    def _val_writer(self):
        if "val" not in self._writers:
            self._writers["val"] = self.summary.create_file_writer(
                self._val_dir
            )
        return self._writers["val"]

    def _write_keras_model_train_graph(self):
        """Writes Keras model train_function graph to TensorBoard."""
        with self._train_writer.as_default():
            train_fn = self.model.train_function
            # If the train_function is a `tf.function`, we can write out a
            # graph
            if hasattr(train_fn, "function_spec"):
                # TODO(b/243822285): Use _variable_creation_fn directly.
                if hasattr(train_fn, "_concrete_stateful_fn"):
                    self.summary.graph(train_fn._concrete_stateful_fn.graph)
                else:
                    self.summary.graph(
                        train_fn._concrete_variable_creation_fn.graph
                    )

    def _write_keras_model_summary(self):
        """Writes Keras graph network summary to TensorBoard."""
        with self._train_writer.as_default():
            if (
                self.model.__class__.__name__ == "Functional"
                or self.model.__class__.__name__ == "Sequential"
            ):
                keras_model_summary("keras", self.model, step=0)

    def _configure_embeddings(self):
        """Configure the Projector for embeddings."""
        from google.protobuf import text_format
        from tensorboard.plugins import projector

        config = projector.ProjectorConfig()
        for layer in self.model.layers:
            if isinstance(layer, Embedding):
                embedding = config.embeddings.add()
                # Embeddings are always the first layer, so this naming should
                # be consistent in any keras models checkpoints.
                name = (
                    "layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE"
                )
                embedding.tensor_name = name

                if self.embeddings_metadata is not None:
                    if isinstance(self.embeddings_metadata, str):
                        embedding.metadata_path = self.embeddings_metadata
                    else:
                        if layer.name in self.embeddings_metadata.keys():
                            embedding.metadata_path = (
                                self.embeddings_metadata.pop(layer.name)
                            )

        if self.embeddings_metadata and not isinstance(
            self.embeddings_metadata, str
        ):
            raise ValueError(
                "Unrecognized `Embedding` layer names passed to "
                "`keras.callbacks.TensorBoard` `embeddings_metadata` "
                f"argument: {self.embeddings_metadata.keys()}"
            )

        config_pbtxt = text_format.MessageToString(config)
        path = os.path.join(self._log_write_dir, "projector_config.pbtxt")
        with file_utils.File(path, "w") as f:
            f.write(config_pbtxt)

    def _push_writer(self, writer, step):
        """Sets the default writer for custom batch-level summaries."""
        if self.update_freq == "epoch":
            return

        def should_record():
            return step % self.update_freq == 0

        summary_context = (
            writer.as_default(step),
            self.summary.record_if(should_record),
        )
        self._prev_summary_state.append(summary_context)
        summary_context[0].__enter__()
        summary_context[1].__enter__()

    def _pop_writer(self):
        """Pops the current writer."""
        if self.update_freq == "epoch":
            return

        # See _push_writer for the content of the previous_context, which is
        # pair of context.
        previous_context = self._prev_summary_state.pop()
        previous_context[1].__exit__(*sys.exc_info())
        previous_context[0].__exit__(*sys.exc_info())

    def _close_writers(self):
        for writer in self._writers.values():
            writer.close()

    def _init_profile_batch(self, profile_batch):
        """Validate profile_batch value and set the range of batches to profile.

        Sets values of _start_batch and _stop_batch attributes,
        specifying the start and stop batch to profile.
        Setting `profile_batch=0` disables profiling.

        Args:
          profile_batch: The range of batches to profile. Should be a
            non-negative integer or a comma separated string of pair of positive
            integers. A pair of positive integers signify a range of batches to
            profile.

        Raises:
          ValueError: If profile_batch is not an integer or a comma separated
            pair of positive integers.

        """
        profile_batch_error_message = (
            "profile_batch must be a non-negative integer or "
            "2-tuple of positive "
            "integers. A pair of positive integers "
            "signifies a range of batches "
            f"to profile. Found: {profile_batch}"
        )

        # Support legacy way of specifying "start,stop" or "start" as str.
        if isinstance(profile_batch, str):
            profile_batch = str(profile_batch).split(",")
            profile_batch = tree.map_structure(int, profile_batch)

        if isinstance(profile_batch, int):
            self._start_batch = profile_batch
            self._stop_batch = profile_batch
        elif (
            isinstance(profile_batch, (tuple, list)) and len(profile_batch) == 2
        ):
            self._start_batch, self._stop_batch = profile_batch
        else:
            raise ValueError(profile_batch_error_message)

        if self._start_batch < 0 or self._stop_batch < self._start_batch:
            raise ValueError(profile_batch_error_message)

        # True when the profiler was successfully started by this callback.
        # We track the status here to make sure callbacks do not interfere with
        # each other. The callback will only stop the profiler it started.
        self._profiler_started = False
        self._batch_trace_context = None

        if self._start_batch > 0:
            # Warm up and improve the profiling accuracy.
            self._start_profiler(logdir="")
            self._stop_profiler(save=False)
        # True when a trace is running.
        self._is_tracing = False

        # Setting `profile_batch=0` disables profiling.
        self._should_trace = not (
            self._start_batch == 0 and self._stop_batch == 0
        )

    def on_train_begin(self, logs=None):
        self._global_train_batch = 0
        self._previous_epoch_iterations = 0
        self._push_writer(self._train_writer, self._global_train_batch)

    def on_train_end(self, logs=None):
        self._pop_writer()

        if self._is_tracing:
            self._stop_trace()

        self._close_writers()

    def on_test_begin(self, logs=None):
        self._push_writer(self._val_writer, self._global_test_batch)

    def on_test_end(self, logs=None):
        if self.model.optimizer and hasattr(self.model.optimizer, "iterations"):
            with self._val_writer.as_default():
                for name, value in logs.items():
                    self.summary.scalar(
                        "evaluation_" + name + "_vs_iterations",
                        value,
                        step=self.model.optimizer.iterations,
                    )
        self._pop_writer()

    def on_train_batch_begin(self, batch, logs=None):
        self._global_train_batch += 1
        if self.write_steps_per_second:
            self._batch_start_time = time.time()
        if not self._should_trace:
            return

        if self._global_train_batch == self._start_batch:
            self._start_trace()
        if self._profiler_started:
            self._batch_trace_context = backend.tensorboard.start_batch_trace(
                batch
            )

    def on_train_batch_end(self, batch, logs=None):
        if self._should_write_train_graph:
            self._write_keras_model_train_graph()
            self._should_write_train_graph = False
        if self.write_steps_per_second:
            batch_run_time = time.time() - self._batch_start_time
            self.summary.scalar(
                "batch_steps_per_second",
                1.0 / batch_run_time,
                step=self._global_train_batch,
            )

        # `logs` isn't necessarily always a dict
        if isinstance(logs, dict):
            for name, value in logs.items():
                self.summary.scalar(
                    "batch_" + name, value, step=self._global_train_batch
                )

        if not self._should_trace:
            return

        if self._is_tracing:
            if self._profiler_started and self._batch_trace_context is not None:
                backend.tensorboard.stop_batch_trace(self._batch_trace_context)
                self._batch_trace_context = None
            if self._global_train_batch >= self._stop_batch:
                self._stop_trace()

    def on_test_batch_begin(self, batch, logs=None):
        self._global_test_batch += 1

    def on_epoch_begin(self, epoch, logs=None):
        # Keeps track of epoch for profiling.
        if self.write_steps_per_second:
            self._previous_epoch_iterations = ops.convert_to_tensor(
                self.model.optimizer.iterations, "float32"
            )
            self._epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        self._log_epoch_metrics(epoch, logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)

    def _start_trace(self):
        self.summary.trace_on(graph=True, profiler=False)
        self._start_profiler(logdir=self._train_dir)
        self._is_tracing = True

    def _stop_trace(self, batch=None):
        """Logs the trace graph to TensorBoard."""
        if batch is None:
            batch = self._stop_batch
        with self._train_writer.as_default():
            # TODO(b/126388999): Remove step info in the summary name.
            self.summary.trace_export(name="batch_%d" % batch, step=batch)
        self._stop_profiler()
        self._is_tracing = False

    def _collect_learning_rate(self, logs):
        if isinstance(self.model.optimizer, Optimizer):
            logs["learning_rate"] = float(
                ops.convert_to_numpy(self.model.optimizer.learning_rate)
            )
        return logs

    def _compute_steps_per_second(self):
        current_iteration = self.model.optimizer.iterations
        time_since_epoch_begin = time.time() - self._epoch_start_time
        current_iteration = ops.convert_to_tensor(current_iteration, "float32")
        time_since_epoch_begin = ops.convert_to_tensor(
            time_since_epoch_begin, "float32"
        )

        steps_per_second = (
            current_iteration - self._previous_epoch_iterations
        ) / time_since_epoch_begin
        return float(steps_per_second)

    def _log_epoch_metrics(self, epoch, logs):
        """Writes epoch metrics out as scalar summaries.

        Args:
            epoch: Int. The global step to use for TensorBoard.
            logs: Dict. Keys are scalar summary names, values are scalars.
        """
        if not logs:
            return

        train_logs = {k: v for k, v in logs.items() if not k.startswith("val_")}
        val_logs = {k: v for k, v in logs.items() if k.startswith("val_")}
        train_logs = self._collect_learning_rate(train_logs)
        if self.write_steps_per_second:
            train_logs["steps_per_second"] = self._compute_steps_per_second()

        if train_logs:
            with self._train_writer.as_default():
                for name, value in train_logs.items():
                    self.summary.scalar("epoch_" + name, value, step=epoch)
        if val_logs:
            with self._val_writer.as_default():
                for name, value in val_logs.items():
                    name = name[4:]  # Remove 'val_' prefix.
                    self.summary.scalar("epoch_" + name, value, step=epoch)

    def _log_weights(self, epoch):
        """Logs the weights of the Model to TensorBoard."""
        with self._train_writer.as_default():
            for layer in self.model.layers:
                for weight in layer.weights:
                    weight_name = weight.name.replace(":", "_")
                    # Add a suffix to prevent summary tag name collision.
                    histogram_weight_name = weight_name + "/histogram"
                    self.summary.histogram(
                        histogram_weight_name, weight, step=epoch
                    )
                    if self.write_images:
                        # Add a suffix to prevent summary tag name
                        # collision.
                        image_weight_name = weight_name + "/image"
                        self._log_weight_as_image(
                            weight, image_weight_name, epoch
                        )
            self._train_writer.flush()

    def _log_weight_as_image(self, weight, weight_name, epoch):
        """Logs a weight as a TensorBoard image."""
        w_img = ops.squeeze(weight)
        shape = w_img.shape
        if len(shape) == 1:  # Bias case
            w_img = ops.reshape(w_img, [1, shape[0], 1, 1])
        elif len(shape) == 2:  # Dense layer kernel case
            if shape[0] > shape[1]:
                w_img = ops.transpose(w_img)
                shape = w_img.shape
            w_img = ops.reshape(w_img, [1, shape[0], shape[1], 1])
        elif len(shape) == 3:  # ConvNet case
            if backend.image_data_format() == "channels_last":
                # Switch to channels_first to display every kernel as a separate
                # image.
                w_img = ops.transpose(w_img, [2, 0, 1])
                shape = w_img.shape
            w_img = ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])

        w_img = backend.convert_to_numpy(w_img)
        shape = w_img.shape
        # Not possible to handle 3D convnets etc.
        if len(shape) == 4 and shape[-1] in [1, 3, 4]:
            self.summary.image(weight_name, w_img, step=epoch)

    def _log_embeddings(self, epoch):
        embeddings_ckpt = os.path.join(
            self._log_write_dir,
            "train",
            f"keras_embedding.ckpt-{epoch}.weights.h5",
        )
        self.model.save_weights(embeddings_ckpt)

    def _start_profiler(self, logdir):
        """Starts the profiler if currently inactive.

        Args:
          logdir: Directory where profiler results will be saved.
        """
        if self._profiler_started:
            return
        try:
            backend.tensorboard.start_trace(logdir)
            self._profiler_started = True
        except Exception as e:
            # Profiler errors should not be fatal.
            logging.error("Failed to start profiler: %s", e)

    def _stop_profiler(self, save=True):
        """Stops the profiler if currently active.

        Args:
          save: Whether to save the profiler results to TensorBoard.
        """
        if not self._profiler_started:
            return
        try:
            backend.tensorboard.stop_trace(save=save)
        except Exception as e:
            # Profiler errors should not be fatal.
            logging.error("Failed to stop profiler: %s", e)
        finally:
            self._profiler_started = False


def keras_model_summary(name, data, step=None):
    """Writes a Keras model as JSON to as a Summary.

    Writing the Keras model configuration allows the TensorBoard graph plugin to
    render a conceptual graph, as opposed to graph of ops. In case the model
    fails to serialize as JSON, it ignores and returns False.

    Args:
        name: A name for this summary. The summary tag used for TensorBoard will
            be this name prefixed by any active name scopes.
        data: A Keras Model to write.
        step: Explicit `int64`-castable monotonic step value for this summary.
            If omitted, this defaults to `tf.summary.experimental.get_step()`,
            which must not be `None`.

    Returns:
        True on success, or False if no summary was written because no default
        summary writer was available.

    Raises:
        ValueError: if a default writer exists, but no step was provided and
            `tf.summary.experimental.get_step()` is `None`.
    """
    import tensorflow.summary as summary
    from tensorflow.compat.v1 import SummaryMetadata

    summary_metadata = SummaryMetadata()
    # Hard coding a plugin name. Please refer to go/tb-plugin-name-hardcode for
    # the rationale.
    summary_metadata.plugin_data.plugin_name = "graph_keras_model"
    # version number = 1
    summary_metadata.plugin_data.content = b"1"

    try:
        json_string = data.to_json()
    except Exception as exc:
        # An exception should not break a model code.
        warnings.warn(f"Model failed to serialize as JSON. Ignoring... {exc}")
        return False

    with summary.experimental.summary_scope(
        name, "graph_keras_model", [data, step]
    ) as (tag, _):
        return summary.write(
            tag=tag, tensor=json_string, step=step, metadata=summary_metadata
        )
