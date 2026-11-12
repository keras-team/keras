import collections
import contextlib
import io
import os
import random
import sys
from unittest import mock

import numpy as np
import pytest
from tensorboard.backend.event_processing import event_file_loader
from tensorboard.compat.proto import event_pb2

from keras.src import backend
from keras.src import callbacks
from keras.src import layers
from keras.src import losses
from keras.src import models
from keras.src import ops
from keras.src import optimizers
from keras.src import testing
from keras.src.optimizers import schedules
from keras.src.utils import file_utils
from keras.src.utils.module_utils import tensorflow

# A summary that was emitted during a test. Fields:
#   logdir: str. The logdir of the FileWriter to which the summary was
#     written.
#   tag: str. The name of the summary.
_ObservedSummary = collections.namedtuple("_ObservedSummary", ("logdir", "tag"))


class _SummaryIterator:
    """Yields `Event` protocol buffers from a given path."""

    def __init__(self, path):
        self._record_iterator = event_file_loader.RawEventFileLoader(
            path
        ).Load()

    def __iter__(self):
        return self

    def __next__(self):
        r = next(self._record_iterator)
        return event_pb2.Event.FromString(r)

    next = __next__


class _SummaryFile:
    """A record of summary tags and the files to which they were written.

    Fields `scalars`, `images`, `histograms`, and `tensors` are sets
    containing `_ObservedSummary` values.
    """

    def __init__(self):
        self.scalars = set()
        self.images = set()
        self.histograms = set()
        self.tensors = set()
        self.graph_defs = []
        self.convert_from_v2_summary_proto = False


def list_summaries(logdir):
    """Read all summaries under the logdir into a `_SummaryFile`.

    Args:
      logdir: A path to a directory that contains zero or more event
        files, either as direct children or in transitive subdirectories.
        Summaries in these events must only contain old-style scalars,
        images, and histograms. Non-summary events, like `graph_def`s, are
        ignored.

    Returns:
      A `_SummaryFile` object reflecting all summaries written to any
      event files in the logdir or any of its descendant directories.

    Raises:
      ValueError: If an event file contains an summary of unexpected kind.
    """
    result = _SummaryFile()
    for dirpath, _, filenames in os.walk(logdir):
        for filename in filenames:
            if not filename.startswith("events.out."):
                continue
            path = os.path.join(dirpath, filename)
            for event in _SummaryIterator(path):
                if event.graph_def:
                    result.graph_defs.append(event.graph_def)
                if not event.summary:  # (e.g., it's a `graph_def` event)
                    continue
                for value in event.summary.value:
                    tag = value.tag
                    # Case on the `value` rather than the summary metadata
                    # because the Keras callback uses `summary_ops_v2` to emit
                    # old-style summaries. See b/124535134.
                    kind = value.WhichOneof("value")
                    container = {
                        "simple_value": result.scalars,
                        "image": result.images,
                        "histo": result.histograms,
                        "tensor": result.tensors,
                    }.get(kind)
                    if container is None:
                        raise ValueError(
                            "Unexpected summary kind %r in event file %s:\n%r"
                            % (kind, path, event)
                        )
                    elif kind == "tensor" and tag != "keras":
                        # Convert the tf2 summary proto to old style for type
                        # checking.
                        plugin_name = value.metadata.plugin_data.plugin_name
                        container = {
                            "images": result.images,
                            "histograms": result.histograms,
                            "scalars": result.scalars,
                        }.get(plugin_name)
                        if container is not None:
                            result.convert_from_v2_summary_proto = True
                        else:
                            container = result.tensors
                    container.add(_ObservedSummary(logdir=dirpath, tag=tag))
    return result


class TestTensorBoardV2(testing.TestCase):
    def _get_log_dirs(self):
        logdir = os.path.join(
            self.get_temp_dir(), str(random.randint(1, int(1e7))), "tb"
        )
        train_dir = os.path.join(logdir, "train")
        validation_dir = os.path.join(logdir, "validation")
        return logdir, train_dir, validation_dir

    def _get_model(self, compile_model=True):
        model = models.Sequential(
            [
                layers.Input((10, 10, 1)),
                layers.Flatten(),
                layers.Dense(1),
            ]
        )
        if compile_model:
            model.compile("sgd", "mse")
        return model

    @pytest.mark.requires_trainable_backend
    def test_TensorBoard_basic(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        logdir, train_dir, validation_dir = self._get_log_dirs()
        tb_cbk = callbacks.TensorBoard(logdir)

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        summary_file = list_summaries(logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=validation_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=train_dir, tag="epoch_learning_rate"),
                _ObservedSummary(
                    logdir=validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )

    @pytest.mark.requires_trainable_backend
    def test_TensorBoard_evaluate(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        logdir, train_dir, validation_dir = self._get_log_dirs()
        tb_cbk = callbacks.TensorBoard(logdir)

        model.predict(x, batch_size=2, callbacks=[tb_cbk])
        self.assertEqual(tb_cbk._writers, {})
        model.evaluate(x, y, batch_size=2, callbacks=[tb_cbk])

        summary_file = list_summaries(logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(
                    logdir=validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )
        self.assertEqual(
            summary_file.tensors,
            {_ObservedSummary(logdir=train_dir, tag="keras")},
        )

    @pytest.mark.skipif(
        tensorflow.available, reason="Uses tensorboardX without TensorFlow."
    )
    @pytest.mark.requires_trainable_backend
    def test_TensorBoard_remote_log_dir(self):
        class RemoteStream(io.IOBase):
            def __init__(self, f):
                self._f = f

            def write(self, b):
                return self._f.write(b)

            def flush(self):
                self._f.flush()

            def close(self):
                super().close()
                self._f.close()

        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        logdir, train_dir, validation_dir = self._get_log_dirs()
        remote_logdir = "gs://bucket/tb"

        def to_local(path):
            self.assertTrue(path.startswith(remote_logdir), msg=path)
            suffix = path[len(remote_logdir) :].strip("/")
            return (
                os.path.join(logdir, *suffix.split("/")) if suffix else logdir
            )

        tb_cbk = callbacks.TensorBoard(remote_logdir)
        with (
            mock.patch.object(
                file_utils,
                "exists",
                side_effect=lambda path: os.path.exists(to_local(path)),
            ),
            mock.patch.object(
                file_utils,
                "makedirs",
                side_effect=lambda path: os.makedirs(
                    to_local(path), exist_ok=True
                ),
            ),
            mock.patch.object(
                file_utils,
                "File",
                side_effect=lambda path, mode="r": RemoteStream(
                    open(to_local(path), mode)
                ),
            ),
        ):
            model.fit(
                x,
                y,
                batch_size=2,
                epochs=2,
                validation_data=(x, y),
                callbacks=[tb_cbk],
            )

        summary_file = list_summaries(logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=validation_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=train_dir, tag="epoch_learning_rate"),
                _ObservedSummary(
                    logdir=validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )
        self.assertEqual(
            summary_file.tensors,
            {_ObservedSummary(logdir=train_dir, tag="keras")},
        )

    @pytest.mark.requires_trainable_backend
    def test_TensorBoard_across_invocations(self):
        """Regression test for summary writer resource use-after-free."""
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        logdir, train_dir, validation_dir = self._get_log_dirs()
        tb_cbk = callbacks.TensorBoard(logdir)

        for _ in (1, 2):
            model.fit(
                x,
                y,
                batch_size=2,
                epochs=2,
                validation_data=(x, y),
                callbacks=[tb_cbk],
            )
        model.evaluate(x, y, batch_size=2, callbacks=[tb_cbk])

        summary_file = list_summaries(logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=validation_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=train_dir, tag="epoch_learning_rate"),
                _ObservedSummary(
                    logdir=validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )

    @pytest.mark.requires_trainable_backend
    def test_TensorBoard_no_spurious_event_files(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        logdir, train_dir, _ = self._get_log_dirs()
        tb_cbk = callbacks.TensorBoard(logdir)
        model.fit(x, y, batch_size=2, epochs=2, callbacks=[tb_cbk])

        events_file_run_basenames = set()
        for dirpath, _, filenames in os.walk(train_dir):
            if any(fn.startswith("events.out.") for fn in filenames):
                events_file_run_basenames.add(os.path.basename(dirpath))
        self.assertEqual(events_file_run_basenames, {"train"})

    @pytest.mark.requires_trainable_backend
    def test_TensorBoard_batch_metrics(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        logdir, train_dir, validation_dir = self._get_log_dirs()
        tb_cbk = callbacks.TensorBoard(logdir, update_freq=1)

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        summary_file = list_summaries(logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=train_dir, tag="batch_loss"),
                _ObservedSummary(logdir=train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=validation_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=train_dir, tag="epoch_learning_rate"),
                _ObservedSummary(
                    logdir=validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )

    @pytest.mark.requires_trainable_backend
    def test_TensorBoard_learning_rate_schedules(self):
        model = self._get_model(compile_model=False)
        opt = optimizers.SGD(schedules.CosineDecay(0.01, 1))
        model.compile(opt, "mse")
        logdir, train_dir, _ = self._get_log_dirs()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            callbacks=[callbacks.TensorBoard(logdir)],
        )

        summary_file = list_summaries(logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=train_dir, tag="epoch_learning_rate"),
            },
        )

    @pytest.mark.requires_trainable_backend
    def test_TensorBoard_global_step(self):
        model = self._get_model(compile_model=False)
        opt = optimizers.SGD(schedules.CosineDecay(0.01, 1))
        model.compile(opt, "mse")
        logdir, train_dir, _ = self._get_log_dirs()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            verbose=0,
            callbacks=[
                callbacks.TensorBoard(
                    logdir,
                    update_freq=1,
                    profile_batch=0,
                    write_steps_per_second=True,
                )
            ],
        )

        summary_file = list_summaries(logdir)
        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=train_dir, tag="batch_loss"),
                _ObservedSummary(logdir=train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=train_dir, tag="epoch_learning_rate"),
                _ObservedSummary(
                    logdir=train_dir, tag="epoch_steps_per_second"
                ),
                _ObservedSummary(
                    logdir=train_dir, tag="batch_steps_per_second"
                ),
            },
        )

    @pytest.mark.requires_trainable_backend
    def test_TensorBoard_weight_histograms(self):
        model = self._get_model()
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        logdir, train_dir, validation_dir = self._get_log_dirs()
        tb_cbk = callbacks.TensorBoard(logdir, histogram_freq=1)

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(logdir)

        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=validation_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=train_dir, tag="epoch_learning_rate"),
                _ObservedSummary(
                    logdir=validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )
        self.assertEqual(
            self._strip_layer_names(summary_file.histograms, "sequential"),
            {_ObservedSummary(logdir=train_dir, tag="histogram")},
        )

    @pytest.mark.skipif(tensorflow.available, reason="Requires no TensorFlow.")
    def test_TensorBoard_weight_histograms_without_tensorflow(self):
        with self.assertRaisesRegex(ImportError, "require TensorFlow"):
            callbacks.TensorBoard(self.get_temp_dir(), histogram_freq=1)

    @pytest.mark.requires_trainable_backend
    def test_TensorBoard_weight_images(self):
        if backend.config.image_data_format() == "channels_last":
            input_shape = (10, 10, 1)
            x_shape = (10, 10, 10, 1)
        else:
            input_shape = (1, 10, 10)
            x_shape = (10, 1, 10, 10)
        x, y = np.ones(x_shape), np.ones((10, 1))
        logdir, train_dir, validation_dir = self._get_log_dirs()
        tb_cbk = callbacks.TensorBoard(
            logdir, histogram_freq=1, write_images=True
        )
        model_type = "sequential"
        model = models.Sequential(
            [
                layers.Input(input_shape),
                layers.Conv2D(3, 10),
                layers.GlobalAveragePooling2D(),
                layers.Dense(1),
            ]
        )
        model.compile("sgd", "mse")
        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(logdir)

        self.assertEqual(
            summary_file.scalars,
            {
                _ObservedSummary(logdir=train_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=validation_dir, tag="epoch_loss"),
                _ObservedSummary(logdir=train_dir, tag="epoch_learning_rate"),
                _ObservedSummary(
                    logdir=validation_dir,
                    tag="evaluation_loss_vs_iterations",
                ),
            },
        )
        self.assertEqual(
            self._strip_layer_names(summary_file.histograms, model_type),
            {
                _ObservedSummary(logdir=train_dir, tag="histogram"),
            },
        )
        expected_image_summaries = {
            _ObservedSummary(logdir=train_dir, tag="image"),
        }
        self.assertEqual(
            self._strip_layer_names(summary_file.images, model_type),
            expected_image_summaries,
        )

    @pytest.mark.requires_trainable_backend
    def test_TensorBoard_projector_callback(self):
        model = models.Sequential(
            [
                layers.Input((10,)),
                layers.Embedding(10, 10, name="test_embedding"),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss=losses.BinaryCrossentropy(from_logits=True)
        )
        x, y = np.ones((10, 10)), np.ones((10, 10))
        logdir, _, _ = self._get_log_dirs()
        tb_cbk = callbacks.TensorBoard(
            logdir,
            embeddings_freq=1,
            embeddings_metadata={"test_embedding": "metadata.tsv"},
        )

        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )

        with open(os.path.join(logdir, "projector_config.pbtxt")) as f:
            self.assertEqual(
                f.readlines(),
                [
                    "embeddings {\n",
                    "  tensor_name: "
                    '"layer_with_weights-0/embeddings/.ATTRIBUTES/'
                    'VARIABLE_VALUE"\n',
                    '  metadata_path: "metadata.tsv"\n',
                    "}\n",
                ],
            )

    @pytest.mark.requires_trainable_backend
    @pytest.mark.skipif(
        not tensorflow.available, reason="Custom summaries require TensorFlow."
    )
    def test_custom_summary(self):
        default_writers = []
        written = []

        class FakeWriter:
            def __init__(self, logdir):
                self.logdir = logdir

            @contextlib.contextmanager
            def as_default(self, step=None):
                default_writers.append(self)
                yield
                default_writers.pop()

            def flush(self):
                pass

            def close(self):
                pass

        summary = mock.MagicMock()
        summary.create_file_writer.side_effect = FakeWriter

        class LayerWithSummary(layers.Layer):
            def call(self, x):
                if default_writers:
                    written.append(default_writers[-1].logdir)
                return x

        model = models.Sequential(
            [
                layers.Input((5,)),
                LayerWithSummary(),
            ]
        )

        # summary ops not compatible with XLA
        model.compile("sgd", "mse", jit_compile=False)
        logdir, train_dir, validation_dir = self._get_log_dirs()
        tb_cbk = callbacks.TensorBoard(logdir, update_freq=1, write_graph=False)
        x, y = np.ones((10, 5)), np.ones((10, 5))
        with mock.patch.object(
            callbacks.TensorBoard,
            "summary",
            new_callable=mock.PropertyMock,
            return_value=summary,
        ):
            model.fit(
                x, y, batch_size=2, validation_data=(x, y), callbacks=[tb_cbk]
            )
        self.assertEqual(set(written), {train_dir, validation_dir})
        summary.record_if.assert_called()

    def _strip_layer_names(self, summaries, model_type):
        """Deduplicate summary names modulo layer prefix.

        This removes the first slash-component of each tag name: for
        instance, "foo/bar/baz" becomes "bar/baz".

        Args:
            summaries: A `set` of `_ObservedSummary` values.
            model_type: The model type currently being tested.

        Returns:
            A new `set` of `_ObservedSummary` values with layer prefixes
            removed.
        """
        result = set()
        for s in summaries:
            if "/" not in s.tag:
                raise ValueError(f"tag has no layer name: {s.tag!r}")
            new_tag = s.tag.split("/")[-1]
            result.add(s._replace(tag=new_tag))
        return result

    def _strip_variable_names(self, summaries):
        """Remove `variable_n` from summary tag

        `variable_n` tag names are added with random numbers. Removing them
        ensures deterministic tag names.

        Args:
            summaries: A `set` of `_ObservedSummary` values.

        Returns:
            A new `set` of `_ObservedSummary` values with layer prefixes
            removed.
        """
        result = set()
        for s in summaries:
            if "/" not in s.tag:
                result.add(s)
            else:
                split_tag = s.tag.split("/")
                if "variable" in split_tag[0]:
                    result.add(s._replace(tag=split_tag[-1]))
                else:
                    result.add(s)
        return result

    @pytest.mark.skipif(
        backend.backend() == "torch",
        reason="Torch backend requires blocking numpy conversion.",
    )
    @pytest.mark.requires_trainable_backend
    def test_TensorBoard_non_blocking(self):
        logdir, _, _ = self._get_log_dirs()
        model = models.Sequential([layers.Dense(1)])
        model.optimizer = optimizers.Adam()
        tb = callbacks.TensorBoard(logdir)
        cb_list = callbacks.CallbackList(
            [tb], model=model, epochs=1, steps=100, verbose=0
        )
        tensor = ops.convert_to_tensor(1.0)

        def mock_numpy():
            raise RuntimeError(
                "If this error is seen, TensorBoard is causing a blocking "
                "NumPy conversion."
            )

        tensor.numpy = mock_numpy

        logs = {"metric": tensor}

        cb_list.on_train_begin(logs)
        cb_list.on_epoch_begin(0, logs)
        cb_list.on_train_batch_begin(0, logs)
        cb_list.on_train_batch_end(0, logs)
        cb_list.on_epoch_end(0, logs)
        cb_list.on_train_end(logs)

        cb_list.on_test_begin(logs)
        cb_list.on_test_batch_begin(0, logs)
        cb_list.on_test_batch_end(0, logs)
        cb_list.on_test_end(logs)

        cb_list.on_predict_begin(logs)
        cb_list.on_predict_batch_begin(logs)
        cb_list.on_predict_batch_end(logs)
        cb_list.on_predict_end(logs)

    def _count_xplane_file(self, logdir):
        profile_dir = os.path.join(logdir, "plugins", "profile")
        count = 0
        for dirpath, dirnames, filenames in os.walk(profile_dir):
            del dirpath  # unused
            del dirnames  # unused
            for filename in filenames:
                if filename.endswith(".xplane.pb"):
                    count += 1
        return count

    def fitModelAndAssertKerasModelWritten(self, model):
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        logdir, train_dir, validation_dir = self._get_log_dirs()
        tb_cbk = callbacks.TensorBoard(
            logdir, write_graph=True, profile_batch=0
        )
        model.fit(
            x,
            y,
            batch_size=2,
            epochs=3,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(logdir)
        self.assertEqual(
            summary_file.tensors,
            {
                _ObservedSummary(logdir=train_dir, tag="keras"),
            },
        )
        if not model.run_eagerly:
            # There should be one train graph
            self.assertLen(summary_file.graph_defs, 1)
            for graph_def in summary_file.graph_defs:
                graph_def_str = str(graph_def)

                # All the model layers should appear in the graphs
                for layer in model.layers:
                    if "input" not in layer.name:
                        self.assertIn(layer.name, graph_def_str)

    def test_TensorBoard_write_sequential_model_no_input_shape(self):
        # TODO: Requires to_json implementation in trainer
        # model = models.Sequential(
        #     [
        #         Conv2D(8, (3, 3)),
        #         Flatten(),
        #         Dense(1),
        #     ]
        # )
        # model.compile("sgd", "mse")
        # self.fitModelAndAssertKerasModelWritten(model)
        pass

    def test_TensorBoard_write_sequential_model_with_input_shape(self):
        # TODO: Requires to_json implementation in trainer
        # model = models.Sequential(
        #     [
        #         Input(input_shape=(10, 10, 1)),
        #         Conv2D(8, (3, 3)),
        #         Flatten(),
        #         Dense(1),
        #     ]
        # )
        # model.compile("sgd", "mse")
        # self.fitModelAndAssertKerasModelWritten(model)
        pass

    def test_TensorBoard_write_model(self):
        # TODO: Requires to_json implementation in trainer
        # See https://github.com/keras-team/keras/blob/ \
        # a8d4a7f1ffc9de3c5932828a107e4e95e8803fb4/ \
        # keras/engine/training.py#L3313
        # inputs = Input([10, 10, 1])
        # x = Conv2D(8, (3, 3), activation="relu")(inputs)
        # x = Flatten()(x)
        # x = Dense(1)(x)
        # model = models.Model(inputs=inputs, outputs=[x])
        # model.compile("sgd", "mse")
        # breakpoint()
        # self.fitModelAndAssertKerasModelWritten(model)
        pass

    @pytest.mark.skipif(
        backend.backend() not in ("jax", "tensorflow"),
        reason="The profiling test can only run with TF and JAX backends.",
    )
    def test_TensorBoard_auto_trace(self):
        logdir, train_dir, validation_dir = self._get_log_dirs()
        model = models.Sequential(
            [
                layers.Input((10, 10, 1)),
                layers.Flatten(),
                layers.Dense(1),
            ]
        )
        x, y = np.ones((10, 10, 10, 1)), np.ones((10, 1))
        if backend.backend() == "jax" and sys.version_info[1] < 12:
            with pytest.warns(match="backend requires python >= 3.12"):
                callbacks.TensorBoard(
                    logdir, histogram_freq=1, profile_batch=1, write_graph=False
                )
            self.skipTest(
                "Profiling with JAX and python < 3.12 "
                "raises segmentation fault."
            )

        tb_cbk = callbacks.TensorBoard(
            logdir, histogram_freq=1, profile_batch=1, write_graph=False
        )
        model.compile("sgd", "mse")
        model.fit(
            x,
            y,
            batch_size=2,
            epochs=2,
            validation_data=(x, y),
            callbacks=[tb_cbk],
        )
        summary_file = list_summaries(logdir)

        self.assertEqual(
            summary_file.tensors,
            {
                _ObservedSummary(logdir=train_dir, tag="batch_1"),
            },
        )
        self.assertEqual(1, self._count_xplane_file(logdir=train_dir))
        pass
